import os
import re
import json
import warnings
from functools import cache
from typing import Iterable

import pyogrio
import pandas as pd
import numpy as np
import geopandas as gpd
from osgeo import gdal, ogr, osr
import xarray as xr
from shapely.geometry import box
import pyarrow.parquet as pq

from ._constants import STREAM_BOUNDS_FILE, STORAGE_OPTIONS
from .logger import LOG

gdal.UseExceptions()
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["AWS_S3_ENDPOINT"] = "s3.amazonaws.com"

STREAM_BOUNDS = None
FDC_DS = None
RP_DS = None
FC_DS = None

def _get_stream_bounds() -> dict[str, list[float]]:
    global STREAM_BOUNDS
    if os.path.exists(STREAM_BOUNDS_FILE):
        with open(STREAM_BOUNDS_FILE, 'r') as f:
            STREAM_BOUNDS = json.load(f)
    else:
        STREAM_BOUNDS = {}

    return STREAM_BOUNDS

def _get_fdc() -> xr.Dataset:
    global FDC_DS
    if FDC_DS is None:
        FDC_DS = xr.open_zarr('s3://geoglows-v2/retrospective/fdc.zarr', storage_options=STORAGE_OPTIONS)
    return FDC_DS

def _get_rp() -> xr.Dataset:
    global RP_DS
    if RP_DS is None:
        RP_DS = xr.open_zarr("s3://geoglows-v2/retrospective/return-periods.zarr", storage_options=STORAGE_OPTIONS)
    return RP_DS

def _get_forecast(date: str) -> xr.Dataset:
    global FC_DS
    if FC_DS is None:
        date = date.strip()
        if len(date) != 12:
            raise ValueError("Date must be in YYYYMMDDHH format.")
        
        FC_DS = xr.open_zarr(f's3://geoglows-v2-forecasts/{date}.zarr', storage_options=STORAGE_OPTIONS)
    return FC_DS

def opens_right(path: str, read: bool = False) -> bool:
    if path.startswith(('s3://', '/vsis3/')):
        return True
    
    if not os.path.exists(path):
        return False
    
    if path.endswith('.parquet'):
        try:
            pd.read_parquet(path, columns=['COMID'])
            return True
        except:
            return False
        
    if path.endswith('.csv'):
        try:
            pd.read_csv(path)
            return True
        except:
            return False
        
    try:
        ds: gdal.Dataset = gdal.Open(path)
        if ds is None:
            return False
        gt = ds.GetGeoTransform()
        if gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0) or sum(gt) == 0:
            return False
        return True
    except:
        return False
    
def rewrite_file_as_parquet_with_covering_bbox(geometry_file: str) -> None:
    """Rewrite a GeoParquet file with covering-bbox metadata for faster bbox reads."""
    read_any_geom(geometry_file).to_parquet(geometry_file, index=False, compression='brotli', write_covering_bbox=True)
    
def clean_stream_raster(stream_raster: str, num_passes: int = 2) -> bool:
    """
    This function comes from Mike Follum's ARC at https://github.com/MikeFHS/automated-rating-curve
    Returns whether there was any stream data in the raster
    """
    assert num_passes > 0, "num_passes must be greater than 0"
    
    # Get stream raster
    stream_ds: gdal.Dataset = gdal.Open(stream_raster, gdal.GA_Update)
    array = np.empty((stream_ds.RasterYSize + 2, stream_ds.RasterXSize + 2), dtype=np.int64)
    # array[1:-1, 1:-1] = stream_ds.ReadAsArray()
    stream_ds.ReadAsArray(buf_obj=array[1:-1, 1:-1])
    
    # Create an array that is slightly larger than the STRM Raster Array
    # array = np.pad(array, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    
    row_indices, col_indices = array.nonzero()
    num_nonzero = len(row_indices)
    saw = False
    
    for _ in range(num_passes):
        # First pass is just to get rid of single cells hanging out not doing anything
        n=0
        for x in range(num_nonzero):
            r = row_indices[x]
            c = col_indices[x]
            if array[r,c] <= 0:
                continue

            saw = True
            # Left and Right cells are zeros
            if array[r,c + 1] == 0 and array[r, c - 1] == 0:
                # The bottom cells are all zeros as well, but there is a cell directly above that is legit
                if (array[r+1,c-1]+array[r+1,c]+array[r+1,c+1])==0 and array[r-1,c]>0:
                    array[r,c] = 0
                    n=n+1
                # The top cells are all zeros as well, but there is a cell directly below that is legit
                elif (array[r-1,c-1]+array[r-1,c]+array[r-1,c+1])==0 and array[r+1,c]>0:
                    array[r,c] = 0
                    n=n+1
            # top and bottom cells are zeros
            if array[r,c]>0 and array[r+1,c]==0 and array[r-1,c]==0:
                # All cells on the right are zero, but there is a cell to the left that is legit
                if (array[r+1,c+1]+array[r,c+1]+array[r-1,c+1])==0 and array[r,c-1]>0:
                    array[r,c] = 0
                    n=n+1
                elif (array[r+1,c-1]+array[r,c-1]+array[r-1,c-1])==0 and array[r,c+1]>0:
                    array[r,c] = 0
                    n=n+1
        
        
        # This pass is to remove all the redundant cells
        n = 0
        for x in range(num_nonzero):
            r = row_indices[x]
            c = col_indices[x]
            value = array[r,c]
            if value<=0:
                continue

            saw = True
            if array[r+1,c] == value and (array[r+1, c+1] == value or array[r+1, c-1] == value):
                if array[r+1,c-1:c+2].max() == 0:
                    array[r+ 1 , c] = 0
                    n = n + 1
            elif array[r-1,c] == value and (array[r-1, c+1] == value or array[r-1, c-1] == value):
                if array[r-1,c-1:c+2].max() == 0:
                    array[r- 1 , c] = 0
                    n = n + 1
            elif array[r,c+1] == value and (array[r+1, c+1] == value or array[r-1, c+1] == value):
                if array[r-1:r+1,c+2].max() == 0:
                    array[r, c + 1] = 0
                    n = n + 1
            elif array[r,c-1] == value and (array[r+1, c-1] == value or array[r-1, c-1] == value):
                if array[r-1:r+1,c-2].max() == 0:
                        array[r, c - 1] = 0
                        n = n + 1
    
    # Write the cleaned array to the raster
    stream_ds.WriteArray(array[1:-1, 1:-1])
    stream_ds.FlushCache()

    return saw

def get_fabdem_in_extent(minx: float, 
                         miny: float, 
                         maxx: float, 
                         maxy: float,
                         dems: list[str],
                         pattern: re.Pattern = re.compile(r'([NS])(\d+)([EW])(\d+)')) -> list[str]:
    output = []

    for file in dems:
        basename = os.path.basename(file)
        match = pattern.search(basename)
        if match:
            ns = 1 if match.group(1) == 'N' else -1
            ew = 1 if match.group(3) == 'E' else -1
            lat = ns * int(match.group(2))
            lon = ew * int(match.group(4))
            if minx < lon + 1 and maxx > lon and miny < lat + 1 and maxy > lat:
                output.append(file)        
    return output

def get_dem_in_extent(minx: float, 
                      miny: float, 
                      maxx: float,
                      maxy: float,
                      dems: list[str],
                      dem_type: str) -> list[str]:
    if dem_type == 'fabdem':
        return get_fabdem_in_extent(minx, miny, maxx, maxy, dems)
    
    return filter_files_in_extent(minx, miny, maxx, maxy, dems)

def filter_files_in_extent(minx: float,
                           miny: float,
                           maxx: float,
                           maxy: float,
                           files: list[str]) -> list[str]:
    output = []
    for file in files:
        basename = os.path.basename(file)
        numbers = re.findall(r'-?\d+', basename)
        numbers = list(map(int, numbers))
        if len(numbers) !=  2:
            raise ValueError(f"File {basename} does not have exactly two numbers in its name")
        
        x, y = numbers
        if minx < x + 1 and maxx > x and miny < y + 1 and maxy > y:
            output.append(file)
    return output

def get_rasters_in_extent(bounds: list[float], rasters: list[str]) -> list[str]:
    output = []
    for raster in rasters:
        raster_bounds = get_raster_bbox(raster)
        if bounds_intersect(bounds, raster_bounds):
            output.append(raster)

    return output

def filter_files_in_extent_by_lat_lon_dirs(minx: float,
                                           miny: float,
                                           maxx: float,
                                           maxy: float,
                                           files: list[str]) -> list[str]:
    output = []
    for file in files:
        split = file.split(os.sep)
        dir_lat = [d for d in split if d.startswith('lat=')][0]
        dir_lon = [d for d in split if d.startswith('lon=')][0]

        y = int(dir_lat.replace('lat=', ''))
        x = int(dir_lon.replace('lon=', ''))
        
        if minx < x + 1 and maxx > x and miny < y + 1 and maxy > y:
            output.append(file)
    return output


def dem_to_dir(dem: str, output_dir: str) -> str:
    try:
        y = int(os.path.basename(dem)[1:3]) * (1 if os.path.basename(dem)[0] == 'N' else -1)
        x = int(os.path.basename(dem)[4:7]) * (1 if os.path.basename(dem)[3] == 'E' else -1)
        return os.path.join(output_dir, f"lon={x}", f"lat={y}")
    except ValueError:
        x, y = os.path.basename(dem).replace('.tif', '').split('_')[1:3]
        return os.path.join(output_dir, f"lon={x}", f"lat={y}")

def _dir(path: str, n_levels: int = 1) -> str:
    assert n_levels >= 1
    return _dir(os.path.dirname(path), n_levels - 1) if n_levels > 1 else os.path.dirname(path)

def get_linknos(stream_raster: str,) -> np.ndarray:
    if not isinstance(stream_raster, str):
        raise ValueError(f"stream_raster must be a string, got {stream_raster}")
    ds: gdal.Dataset = gdal.Open(stream_raster)
    array = ds.ReadAsArray()
    linknos = np.unique(array)
    return linknos[linknos > 0]

def get_dataset_info(path: str):
    ds: gdal.Dataset = gdal.Open(path)
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    projection = ds.GetProjection()
    return width, height, gt, projection

def get_bbox_from_ds_data(gt: tuple[float, ...], width: int, height: int, projection: str = None) -> tuple[float, float, float, float]:
    """
    Get bounds of a GDAL dataset as (minx, miny, maxx, maxy) in EPSG:4326`
    """
    minx = gt[0]
    maxx = gt[0] + width * gt[1]
    miny = gt[3] + height * gt[5]
    maxy = gt[3]
    if miny > maxy:
        miny, maxy = maxy, miny

    if projection:
        minx, miny, maxx, maxy = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs=projection).to_crs("EPSG:4326").total_bounds

    return (minx, miny, maxx, maxy)

def get_ds_bbox(ds: gdal.Dataset) -> tuple[float, float, float, float]:
    """
    Get bounds of a GDAL dataset as (minx, miny, maxx, maxy) in EPSG:4326
    """
    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize
    projection = ds.GetProjection()

    return get_bbox_from_ds_data(gt, width, height, projection)

@cache
def get_raster_bbox(raster_path: str) -> tuple[float, float, float, float]:
    ds: gdal.Dataset = gdal.Open(raster_path)
    return get_ds_bbox(ds)

@cache
def get_raster_res(raster_path: str) -> tuple[float, float]:
    ds: gdal.Dataset = gdal.Open(raster_path)
    gt = ds.GetGeoTransform()
    return (abs(gt[1]), abs(gt[5]))

def bounds_intersect(bounds1: tuple[float, float, float, float], bounds2: tuple[float, float, float, float]) -> bool:
    minx1, miny1, maxx1, maxy1 = bounds1
    minx2, miny2, maxx2, maxy2 = bounds2

    return not (maxx1 <= minx2 or maxx2 <= minx1 or maxy1 <= miny2 or maxy2 <= miny1)

def save_any_geom(gdf: gpd.GeoDataFrame, path: str, **kwargs) -> None:
    if path.lower().endswith(('.parquet', '.geoparquet')):
        gdf.to_parquet(path, **kwargs)
    else:
        gdf.to_file(path, **kwargs)
    
def is_tile_in_valid_tiles(minx: float, miny: float, valid_tiles: list[list[int]]) -> bool:
    if valid_tiles is None:
        return True
    return [round(minx), round(miny)] in valid_tiles

def _get_files_helper(bbox: list[float], all_files: list[str]) -> list[str]:
    minx, miny, maxx, maxy = bbox
    files = filter_files_in_extent_by_lat_lon_dirs(minx, miny, maxx, maxy, all_files)

    if len(files) == 0:
        raise ValueError("No files found for area")
    
    return files

def convert_area_to_single_map(bbox: list[float], all_files: list[str], output_file: str) -> tuple[list[float], int, int]:
    files = _get_files_helper(bbox, all_files)

    options = gdal.WarpOptions(format='GTiff',
                        dstNodata=0,
                        outputType=gdal.GDT_Byte,
                        creationOptions=['COMPRESS=ZSTD', 'PREDICTOR=2', 'ZSTD_LEVEL=9'])
    gdal.Warp(output_file, files, options=options)

def convert_area_to_single_vrt(bbox: list[float], all_files: list[str], output_file: str) -> tuple[list[float], int, int]:
    files = _get_files_helper(bbox, all_files)

    gdal.BuildVRT(output_file, files, options=gdal.BuildVRTOptions(srcNodata=255))

def lon_to_x(lon: float, z: int = 14) -> int:
    return round((lon + 180) * (2 ** z) / 360)

def lat_to_y(lat: float, z: int = 14) -> int:
    return round((1 - (np.log(np.tan(np.radians(lat)) + (1 / np.cos(np.radians(lat)))) / np.pi)) * (2 ** z) / 2)

def generate_bounding_args(minx: float,
                           miny: float,
                           maxx: float,
                           maxy: float,
                           valid_tiles: list[list[int]] = None,
                           number_of_tiles: int = None,
                           offset: int = 0) -> list[tuple[float, float, float, float]]:
    args = []
    if number_of_tiles is None:
        number_of_tiles = float('inf')
    
    start_offset = 0
    for x in range(minx-1, maxx+1):
        for y in range(miny-1, maxy+1):
            if is_tile_in_valid_tiles(x, y, valid_tiles):
                if start_offset < offset:
                    start_offset += 1
                else:
                    args.append((x, y, x + 1, y + 1))
                    if len(args) >= number_of_tiles:
                        return args

    return args

def get_s3_fabdem_path(x, y):
    # Determine 10° tile bounds
    lat0 = int(y // 10 * 10)
    lon0 = int(x // 10 * 10)

    lat1 = lat0 + 10
    lon1 = lon0 + 10

    # Hemisphere prefixes
    lat_prefix0 = "N" if lat0 >= 0 else "S"
    lon_prefix0 = "E" if lon0 >= 0 else "W"
    lat_prefix1 = "N" if lat1 >= 0 else "S"
    lon_prefix1 = "E" if lon1 >= 0 else "W"

    # Individual tile (1° tile assumed)
    lat_tile = int(y)
    lon_tile = int(x)
    lat_prefix_tile = "N" if lat_tile >= 0 else "S"
    lon_prefix_tile = "E" if lon_tile >= 0 else "W"

    # Format with zero padding
    def fmt_lat(val, prefix): return f"{prefix}{abs(val):02d}"
    def fmt_lon(val, prefix): return f"{prefix}{abs(val):03d}"

    folder = f"{fmt_lat(lat0, lat_prefix0)}{fmt_lon(lon0, lon_prefix0)}-" \
             f"{fmt_lat(lat1, lat_prefix1)}{fmt_lon(lon1, lon_prefix1)}_FABDEM_V1-2"
    filename = f"{fmt_lat(lat_tile, lat_prefix_tile)}{fmt_lon(lon_tile, lon_prefix_tile)}_FABDEM_V1-2.tif"

    return 'global-floodmaps', f'dems/fabdem/{folder}/{filename}'

def are_there_non_zero_in_raster(raster_path: str) -> bool:
    ds: gdal.Dataset = gdal.Open(raster_path)
    # Iterate on blocksize to avoid loading entire raster into memory
    band = ds.GetRasterBand(1)
    block_sizes = band.GetBlockSize()
    x_block_size = block_sizes[0]
    y_block_size = block_sizes[1]
    x_size = band.XSize
    y_size = band.YSize

    for y in range(0, y_size, y_block_size):
        rows = y_block_size if y + y_block_size < y_size else y_size - y
        for x in range(0, x_size, x_block_size):
            cols = x_block_size if x + x_block_size < x_size else x_size - x
            data = band.ReadAsArray(x, y, cols, rows)
            if np.any(data != 0):
                return True
            
    return False

def extract_base_path(path: str) -> str:
    match = re.search(r'(lon=[^\\\/]+[\\\/].*)', path)
    if not match:
        raise ValueError(f"Could not extract base path from {path}")
    return match.group(1).replace('\\', '/')

@cache
def pyogrio_read_info(path: str):
    info = pyogrio.read_info(path)
    data_crs = info['crs']
    if data_crs is None:
        parquet_file = pq.ParquetFile(path)

        # Retrieve the file metadata
        metadata = parquet_file.metadata
        geo_metadata = json.loads(metadata.metadata[b'geo'].decode('utf-8'))
        data_crs = ":".join(map(str,geo_metadata['columns']['geometry']['crs']['id'].values()))
        info['crs'] = data_crs

    return info

def read_any_geom(path: str, bbox: list[float] = None, columns: list[str] = None) -> gpd.GeoDataFrame:
    """
    Read any geometry file (shapefile, geojson, geoparquet) into a GeoDataFrame. Bbox in 4326.
    """
    if bbox is not None:
        # If bbox is provided, we need to make sure it's in the same CRS as the data. We can check the CRS of the data by reading just the metadata with pyogrio, and then reprojecting the bbox if necessary.
        info = pyogrio_read_info(path)
        data_crs = info['crs']
        if data_crs is not None and data_crs != 'EPSG:4326':
            minx, miny, maxx, maxy = bbox
            gdf_bbox = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs="EPSG:4326").to_crs(data_crs)
            bbox = gdf_bbox.total_bounds

    if path.lower().endswith(('.parquet', '.geoparquet')):
        try:
            return gpd.read_parquet(path, bbox=bbox, columns=columns)
        except ValueError as e:
            if "Specifying 'bbox' not supported for this Parquet file" in str(e):
                warnings.warn(f"Could not read {path} with bbox. Consider adding covering bbox to parquet file for faster reading.", stacklevel=2)
                gdf = gpd.read_parquet(path, columns=columns)
                minx, miny, maxx, maxy = bbox
                gdf = gdf.cx[minx:maxx, miny:maxy]
                return gdf
            raise e
    
    return gpd.read_file(path, use_arrow=True, bbox=bbox, columns=columns)

def _streamline_is_in_dem_bounds(stream: str, dem_bounds: tuple[float, float, float, float]) -> bool:
    all_stream_bounds = _get_stream_bounds()
    stream_bounds = all_stream_bounds.get(os.path.basename(stream))
    if stream_bounds is None:
        info = pyogrio_read_info(stream)
        stream_bounds = info['bbox']
        all_stream_bounds[os.path.basename(stream)] = stream_bounds

    return bounds_intersect(stream_bounds, dem_bounds)

def get_streamlines_in_dem_extent(dem: str, streamlines: list[str]) -> list[str]:
    """Return the stream parquet files whose stored bounds intersect a DEM tile."""
    dem_bounds = get_raster_bbox(dem)

    streamlines_to_clip = []
    for stream in streamlines:
        if _streamline_is_in_dem_bounds(stream, dem_bounds):
            streamlines_to_clip.append(stream)

    return streamlines_to_clip

def clip_streamlines_to_dem(dem: str, streamlines: list[str], output: str):
    """Clip one or more stream parquet files to a DEM footprint and save the result."""
    dem_bounds = get_raster_bbox(dem)
    gdf = pd.concat([read_any_geom(stream, bbox=dem_bounds) for stream in streamlines], ignore_index=True)
    save_any_geom(gdf, output, compression='brotli', write_covering_bbox=True)

    return
        
def _get_return_period_flows_for_linknos(linknos: Iterable[int], rps: list[float], flow_file: str):
    rp_ds = _get_rp()
    existing = set(rp_ds['river_id'].values)
    linknos = list(set(linknos) & existing)
    try:
        df = rp_ds.sel(river_id=linknos, return_period=rps).to_dataframe()
    except KeyError:
        LOG.error(f"Return period {rps} not found in the dataset. Available return periods are {', '.join(rp_ds['return_period'].values.astype(str))}")
        return
    
    if not df.empty:
        df['logpearson3'] = df['logpearson3'].fillna(df['gumbel'])
        df['logpearson3'].unstack(level='return_period').round().to_csv(flow_file)
    else:
        LOG.warning("No matching linknos found in return period dataset")

def get_return_period_flows_from_stream_raster(stream_raster: str, rps: list[float], flow_file: str):
    linknos = get_linknos(stream_raster)
    _get_return_period_flows_for_linknos(linknos, rps, flow_file)



def get_return_period_flows_in_dem_extent(dem: str, streamline: str | list[str], rps: list[float], flow_file: str, river_id_field: str = 'LINKNO'):
    """Write a GEOGLOWS return-period flow CSV for all stream IDs inside a DEM tile."""
    dem_bounds = get_raster_bbox(dem)
    
    if isinstance(streamline, str):
        streamlines = [streamline]
    else:
        streamlines = streamline

    linknos = set()
    for stream_geom in streamlines:
        linknos.update(read_any_geom(stream_geom, bbox=dem_bounds)[river_id_field].unique())

    if len(linknos) == 0:
        LOG.warning(f"No linknos found in {stream_geom} for {dem}, writing empty file.")
        pd.DataFrame(columns=["river_id"] + rps).to_csv(flow_file, index=False)
        return
    
    _get_return_period_flows_for_linknos(linknos, rps, flow_file)

def buffer_dem(dem: str, output_dem: str, all_dems: list[str], buffer_distance: float = 0.1, as_vrt: bool = True) -> str:
    """Expand a DEM tile by ``buffer_distance`` degrees using neighboring rasters."""
    minx, miny, maxx, maxy = get_raster_bbox(dem)
    
    minx -= buffer_distance
    maxx += buffer_distance
    miny -= buffer_distance
    maxy += buffer_distance

    surrounding_dems = get_rasters_in_extent((minx, miny, maxx, maxy), all_dems)
    if dem not in surrounding_dems:
        raise ValueError("The original DEM is not in the candidates list.")
    
    if as_vrt and not output_dem.lower().endswith('.vrt'):
        LOG.warning("Output file does not have .vrt extension, but as_vrt is True. Proceeding to create a VRT file regardless.")
    if not as_vrt and output_dem.lower().endswith('.vrt'):
        LOG.warning("Output file has .vrt extension, but as_vrt is False. Proceeding to create a non-VRT file regardless.")
    
    xres, yres = get_raster_res(dem)
    if as_vrt:
        vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest',
                                                outputBounds=(minx, miny, maxx, maxy),
                                                targetAlignedPixels=True,
                                                xRes=xres,
                                                yRes=yres)
        gdal.BuildVRT(output_dem, surrounding_dems, options=vrt_options)
    else:
        warp_options = gdal.WarpOptions(resampleAlg='nearest',
                                        outputBounds=(minx, miny, maxx, maxy),
                                        targetAlignedPixels=True,
                                        xRes=xres,
                                        yRes=yres)
        gdal.Warp(output_dem, surrounding_dems, options=warp_options)
    
    return output_dem

def get_oceans_array_in_area(bbox: list[float], oceans_pq: str, width: int, height: int, gt: tuple[float, ...], proj: str) -> np.ndarray | None:
    gdf = gpd.read_parquet(oceans_pq, columns=['geometry'], bbox=bbox)
    if gdf.empty:
        return
    
    gdf = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    if gdf.empty:
        return
    
    # Step 1: Convert GeoDataFrame to OGR Layer (in memory)
    vector_ds: gdal.Dataset = (ogr.GetDriverByName('MEM') or ogr.GetDriverByName('Memory')).CreateDataSource('temp')
    spatial_ref = osr.SpatialReference(wkt=gdf.crs.to_wkt())

    layer: ogr.Layer = vector_ds.CreateLayer('layer', srs=spatial_ref, geom_type=ogr.wkbPolygon)

    # Add dummy feature to layer
    for geom in gdf.geometry:
        geom = ogr.CreateGeometryFromWkb(geom.wkb)
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(geom)
        layer.CreateFeature(feature)
        feature = None

    # Step 2: Create output raster in memory
    oceans_ds: gdal.Dataset = gdal.GetDriverByName("MEM").Create('', width, height, 1, gdal.GDT_Byte)
    oceans_ds.SetGeoTransform(gt)
    oceans_ds.SetProjection(proj)

    # Step 3: Rasterize with a burn value of 1
    gdal.RasterizeLayer(
        oceans_ds,
        [1],            # band 1
        layer,
        burn_values=[1], # fill with 1 where geometry exists,
        options=['ALL_TOUCHED=TRUE']
    )
    oceans_ds.FlushCache()
    oceans_array = oceans_ds.ReadAsArray()

    return oceans_array

def unbuffer_and_mask_oceans(unbuffered_dem: str, floodmap: str, land_use: str = None, oceans_pq: str = None, flood_value: np.uint8 = 100):
    """Crop a buffered flood map back to the source DEM extent and remove ocean cells."""
    ds: gdal.Dataset = gdal.Open(floodmap)
    width = ds.RasterXSize
    height = ds.RasterYSize
    ds = None

    ds: gdal.Dataset = gdal.Open(unbuffered_dem)
    unbuffered_width = ds.RasterXSize
    unbuffered_height = ds.RasterYSize
    if unbuffered_width == width and unbuffered_height == height:
        return

    gt = ds.GetGeoTransform()
    minx = gt[0]
    maxx = gt[0] + ds.RasterXSize * gt[1]
    miny = gt[3] + ds.RasterYSize * gt[5]
    maxy = gt[3]
    ds = None

    options = gdal.WarpOptions(format='MEM',
                                    outputBounds=(minx, maxy, maxx, miny),
                                    width=unbuffered_width,
                                    height=unbuffered_height,
                                    )
    unbuffered_ds: gdal.Dataset = gdal.Warp('', floodmap, options=options)

    gt = unbuffered_ds.GetGeoTransform()
    proj = unbuffered_ds.GetProjection()
    flood_array: np.ndarray = unbuffered_ds.ReadAsArray()

    if land_use:
        lu_ds: gdal.Dataset = gdal.Warp('', land_use, options=options)
        lu_array: np.ndarray = lu_ds.ReadAsArray()
        lu_ds = None
        flood_array[lu_array == 80] = flood_value

    if oceans_pq:
        oceans_array = get_oceans_array_in_area((minx, miny, maxx, maxy), oceans_pq, unbuffered_width, unbuffered_height, gt, proj)
        if oceans_array is not None:
            flood_array[oceans_array == 1] = 0

    # Create a new MEM dataset to hold the modified array
    mem_ds: gdal.Dataset = gdal.GetDriverByName('MEM').Create('', unbuffered_width, unbuffered_height, 1, gdal.GDT_Byte)
    mem_ds.SetGeoTransform(gt)
    mem_ds.SetProjection(proj)
    mem_ds.GetRasterBand(1).WriteArray(flood_array)

    # Now safely write to COG
    gdal.GetDriverByName("COG").CreateCopy(floodmap, mem_ds, options=['COMPRESS=DEFLATE', 'PREDICTOR=2'])
