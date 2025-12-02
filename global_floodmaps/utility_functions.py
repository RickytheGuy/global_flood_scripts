import os
import re

import pandas as pd
import numpy as np
from osgeo import gdal

gdal.UseExceptions()
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["AWS_S3_ENDPOINT"] = "s3.amazonaws.com"

def opens_right(path: str, read: bool = False) -> bool:
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
        if gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0) or sum(ds.GetGeoTransform()) == 0:
            return False
        return True
    except:
        return False
    
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
    ds: gdal.Dataset = gdal.Open(stream_raster)
    array = ds.ReadAsArray()
    linknos = np.unique(array)
    if linknos[0] == 0:
        return linknos[1:]
    return linknos

def get_dataset_info(path: str):
    ds: gdal.Dataset = gdal.Open(path)
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    projection = ds.GetProjection()
    return width, height, gt, projection

def convert_gt_to_bbox(gt: tuple[float, ...], width: int, height: int) -> tuple[float, float, float, float]:
    minx = gt[0]
    maxx = gt[0] + width * gt[1]
    miny = gt[3] + height * gt[5]
    maxy = gt[3]
    if miny > maxy:
        miny, maxy = maxy, miny
    return (minx, miny, maxx, maxy)
    
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
