import os
import glob
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import psutil
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from osgeo import gdal, ogr, osr
from shapely.geometry import box

from arc import Arc
from curve2flood import Curve2Flood_MainFunction

from .utility_functions import (
    opens_right, get_dataset_info, convert_gt_to_bbox, is_tile_in_valid_tiles, 
    get_dem_in_extent, dem_to_dir, _dir, clean_stream_raster, get_linknos,
    no_leave_pbar
)

from ._constants import ESA_TILES_FILE, STORAGE_OPTIONS

gdal.UseExceptions()
gdal.SetConfigOption('AWS_NO_SIGN_REQUEST', 'YES')

FDC_DS = None
RP_DS = None
FC_DS = None

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

def unthrottled_map(executor: ProcessPoolExecutor, func, items):
    """
    Yield results from submitting func(item) for each item to the given executor,
    submitting all tasks immediately (i.e., unthrottled) and yielding results as
    they complete.

    Parameters
    ----------
    executor : concurrent.futures.Executor
        An executor (typically concurrent.futures.ProcessPoolExecutor or
        ThreadPoolExecutor) used to submit tasks. The executor must be running
        and should not be shut down while iteration is in progress.
    func : Callable[[Any], Any]
        A callable that accepts a single argument (an element from `items`) and
        returns a result. If you need to call a function with multiple
        arguments, wrap them in a tuple and use a small wrapper callable.
    items : Iterable[Any]
        An iterable of inputs to pass to `func`. All items will be submitted to
        the executor immediately; `items` may be consumed eagerly.

    Yields
    ------
    Any
        The results returned by `func(item)`, yielded in the order that the
        corresponding submitted tasks complete (completion order), not the
        original order of `items`.

    Raises
    ------
    Exception
        If `func` raises an exception in a worker, that exception will be raised
        when the corresponding future's result is retrieved during iteration.
        Other exceptions from the executor/future machinery may also be raised
        during iteration.

    Notes
    -----
    - This function uses an "unthrottled" submission strategy: it submits a
      future for every item immediately. For very large iterables this may
      overload the executor or exhaust system resources. Consider a throttled
      approach (submitting a bounded number of tasks at a time) for large
      workloads.
    - Results are yielded as they become available (completion order). If you
      require results in the same order as `items`, use a different mapping
      utility that preserves order.
    - The executor is not shut down by this function; managing the executor's
      lifecycle is the caller's responsibility.

    Example
    -------
    with concurrent.futures.ProcessPoolExecutor() as exe:
        for result in unthrottled_map(exe, my_function, my_items):
            handle(result)
    """
    futures = {executor.submit(func, x): x for x in items}
    for future in as_completed(futures):
        yield future.result()

def throttled_map(executor: ProcessPoolExecutor, func, items: list, limit: int = os.cpu_count()):
    """
    Throttle the submission of tasks to a ProcessPoolExecutor and yield once per completed task.

    This generator submits at most `limit` tasks from `items` to `executor` at a time and yields a single
    None value each time one of the submitted tasks completes. As each task finishes, if there are
    remaining items, a new task is submitted so that up to `limit` tasks are in-flight until all items
    have been submitted and completed.

    Parameters
    ----------
    executor : concurrent.futures.ProcessPoolExecutor
        Executor used to submit the tasks. Must support the submit() method.
    func : Callable[[Any], Any]
        Function to run in the executor for each item. It is called as func(item) for each element of
        `items`. Note: this generator does not return the results of `func`.
    items : Sequence
        Sequence (e.g., list) of inputs; each element will be passed to `func` in turn.
    limit : int, optional
        Maximum number of tasks to have running concurrently. Defaults to os.cpu_count().

    Yields
    ------
    None
        One None is yielded for each completed task (i.e., once per item). Yields occur in the order
        tasks complete, not in the order of `items`.

    Notes
    -----
    - This helper is intended as a lightweight concurrency throttle; it does not return or expose task
      results. If you need results or to observe exceptions raised by tasks, collect and call
      future.result() on the futures yourself or adapt this implementation.
    - Submissions (executor.submit) may raise (e.g., if the executor has been shut down); such errors
      will propagate at submission time. Exceptions raised inside the tasks are stored on their futures
      and are not raised by this generator (because it does not call future.result()).
    - If `items` is empty, the generator yields nothing.
    - The ordering of completion is non-deterministic and depends on task runtime.
    """
    futures = {executor.submit(func, x): x for x in items[:limit]}
    next_idx = limit
    for future in as_completed(futures):
        yield None
        if next_idx < len(items):
            new_future = executor.submit(func, items[next_idx])
            futures[new_future] = items[next_idx]
            next_idx += 1
        del futures[future]

def start_unthrottled_pbar(ex, func, desc: str, items: list, **func_kwargs):
    return list(no_leave_pbar(unthrottled_map(ex, partial(func, **func_kwargs), items), total=len(items), desc=desc))

def start_throttled_pbar(ex, func, desc: str, items: list, limit: int, **func_kwargs):
    return list(no_leave_pbar(throttled_map(ex, partial(func, **func_kwargs), items, limit), total=len(items), desc=desc))

def _convert_process_count(n: int) -> int:
    mem_per_process = n / 127.696 # Calibrated using 128 GB RAM machines....
    if (mem_per_process * os.cpu_count()) > psutil.virtual_memory().total / (1024 ** 3):
        return os.cpu_count()
    
    return int(psutil.virtual_memory().total / (1024 ** 3) / mem_per_process)

def buffer_dem(dem: str, 
               output_dir: str,
               dems: list[str], 
               dem_type: str, 
               buffer_distance: float, 
               valid_tiles: list[list[int]] = None, 
               as_vrt: bool = True,
               overwrite: bool = False) -> str:
    """
    Create a buffered version of a single DEM tile by assembling all DEMs that intersect
    the tile's buffered bounding box and writing either a VRT or a warped raster.

    Parameters
    ----------
    dem : str
        Path to the input DEM tile to buffer.
    output_dir : str
        Base output directory where the buffered DEM (or VRT) will be written. The final
        file is placed under "<output_dir>/<dem_to_dir(dem)>/dems/" with the same
        basename as `dem`. If `as_vrt` is True the resulting filename will end with ".vrt".
    dems : list[str]
        List of available DEM paths to search for neighboring tiles that intersect the
        buffered extent.
    dem_type : str
        A string used by get_dem_in_extent to select/parse DEMs from `dems` (passed through
        to helper functions; semantics depend on the calling code).
    buffer_distance : float
        Buffer distance to apply to the DEM tile's bounding box. NOTE: the function
        assumes the DEM is in geographic coordinates (degrees). The buffer distance is
        interpreted in the same linear units as the geotransform; because the function
        explicitly rejects non-degree data (by checking gt[5]), this value should be
        given in degrees (decimal degrees).
    valid_tiles : list[list[int]] | None, optional
        Optional collection of tile identifiers / coordinate pairs representing an area of
        interest. If provided, the function will skip processing and return an empty
        string for tiles that are not in the valid set (checked via
        is_tile_in_valid_tiles). Default is None (no valid-tile checking).
    as_vrt : bool, optional
        If True (default) the function builds a VRT of the buffered extent (faster,
        lightweight). If False, a GDAL warp is performed and a raster file (same basename
        as `dem`) is written.
    overwrite : bool, optional
        If True, forces re-creation of the buffered DEM even if the output file already exists.

    Returns
    -------
    str
        Path to the created buffered DEM file (VRT path if as_vrt is True, or the warped
        raster path if as_vrt is False). If `valid_tiles` is provided and this tile is
        outside the area of interest, returns an empty string. If the target output file
        already exists and is readable (opens_right returns True), the existing path is
        returned immediately.
    """
    out_dir = dem_to_dir(dem, output_dir)
    output_dem = os.path.join(out_dir, 'dems', os.path.basename(dem))
    if as_vrt:
        output_dem = output_dem.replace('.tif', '.vrt')
    
    if opens_right(output_dem) and not overwrite:
        return output_dem

    width, height, gt, _ = get_dataset_info(dem)
    minx, miny, maxx, maxy = convert_gt_to_bbox(gt, width, height)

    # Check if tile is in the area of interest
    if valid_tiles and not is_tile_in_valid_tiles(minx, miny, valid_tiles):
        return ''
    
    os.makedirs(os.path.dirname(output_dem), exist_ok=True)
    if abs(gt[5]) > 0.01:
        raise ValueError("I think this is not in degrees, so buffering is not supported.")
    
    minx -= buffer_distance
    maxx += buffer_distance
    miny -= buffer_distance
    maxy += buffer_distance

    surrounding_dems = get_dem_in_extent(minx, miny, maxx, maxy, dems, dem_type)
    if dem not in surrounding_dems:
        raise ValueError("The original DEM is not in the candidates list.")
    
    if as_vrt:
        vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest',
                                                outputBounds=(minx, miny, maxx, maxy))
        gdal.BuildVRT(output_dem, surrounding_dems, options=vrt_options)
    else:
        warp_options = gdal.WarpOptions(resampleAlg='nearest',
                                        outputBounds=(minx, miny, maxx, maxy))
        gdal.Warp(output_dem, surrounding_dems, options=warp_options)
    
    return output_dem

def rasterize_streams(dem: str, 
                    dem_type: str, 
                    bounds: dict[str, tuple[float, float, float, float]], 
                    min_stream_order: int = 1,
                    overwrite: bool = False):
    """
    Rasterize vector stream layers into a GIS-aligned raster (streams.tif) matching a given DEM.
    This function creates a single-band GeoTIFF named "streams.tif" under the directory
    derived from the provided DEM and dem_type (os.path.join(_dir(dem, 2), f'inputs={dem_type}', 'streams.tif')).
    The output raster is created with the same size, geotransform and projection as the input DEM
    and uses gdal.GDT_Int32 with ZSTD compression (PREDICTOR=2). Each selected vector stream file
    is rasterized into band 1 using the integer attribute "LINKNO".

    Parameters
    ----------
    dem : str
        Path to the reference DEM raster. Used to derive output path and to obtain
        width, height, geotransform and projection via get_dataset_info().
    dem_type : str
        Short identifier used to build the output path (placed under inputs={dem_type}).
    bounds : dict[str, tuple[float, float, float, float]]
        Mapping of vector stream file paths to their bounding boxes.
        Each bbox must be a tuple (minx, miny, maxx, maxy) in the same coordinate system
        as the DEM. Only vector files whose bbox intersects the DEM extent are considered.
    min_stream_order : int, optional
        Minimum stream order to include (default 1). When >1, an attribute filter
        "CAST(strmOrder AS INTEGER) >= {min_stream_order}" is applied to each OGR layer
        before rasterization.
    overwrite : bool, optional
        If True, forces re-creation of the streams.tif file even if it already exists.
    """
    stream_file = os.path.join(_dir(dem, 2), f'inputs={dem_type}', 'streams.tif')

    if opens_right(stream_file) and not overwrite:
        return
    
    width, height, gt, proj = get_dataset_info(dem)
    minx, miny, maxx, maxy = convert_gt_to_bbox(gt, width, height)

    stream_files = []
    for f, bbox in bounds.items():
        if bbox[0] <= maxx and bbox[1] <= maxy and bbox[2] >= minx and bbox[3] >= miny:
            stream_files.append(f)

    if not stream_files:
        return
    
    os.makedirs(_dir(stream_file), exist_ok=True)
    stream_ds: gdal.Dataset = gdal.GetDriverByName('GTiff').Create(stream_file, width, height, 1, gdal.GDT_Int32, options=['COMPRESS=ZSTD', 'PREDICTOR=2'])
    stream_ds.SetGeoTransform(gt)
    stream_ds.SetProjection(proj)

    # Rasterize the streams
    for streams_file in stream_files:
        temp: gdal.Dataset = ogr.Open(streams_file)
        layer: ogr.Layer = temp.GetLayer()
        if min_stream_order > 1:
            layer.SetAttributeFilter(f"CAST(strmOrder AS INTEGER) >= {min_stream_order}")

        gdal.RasterizeLayer(stream_ds, 
                            [1], 
                            layer, 
                            options=[f"ATTRIBUTE=LINKNO"],)

        stream_ds.FlushCache()
        temp = None
        layer = None

    stream_ds = None
    clean_stream_raster(stream_file)

def warp_land_use(dem: str, 
                  dem_type: str, 
                  landcover_directory: str, 
                  save_vrt: bool = True,
                  overwrite: bool = False):
    """
    Generate or warp a land-use raster to match a target DEM's extent, resolution, and projection.
    This function constructs an output path for a land-use raster (land_use.tif) next to the
    provided DEM, determines which ESA WorldCover tiles intersect the DEM footprint, and
    either builds a VRT referencing those tiles or warps them into a single GeoTIFF that
    matches the DEM geometry. If no source tiles are available, a placeholder raster filled
    with the value 10 (tree cover) is created to ensure downstream processing can proceed.
    
    Parameters
    ----------
    dem (str):
        Filesystem path to the reference DEM raster used to determine output geometry,
        resolution, extent, and projection.
    dem_type (str):
        A short identifier used when constructing the output directory path (used in
        inputs=<dem_type>/land_use.tif).
    landcover_directory (str):
        Local directory containing pre-downloaded ESA WorldCover tiles named
        "<tile>.tif". If a tile is not present locally, the function falls back to a
        canonical S3 /vsis3 path for ESA WorldCover tiles.
    save_vrt (bool, optional):
        If True (default) build and save a VRT that references the source tiles using
        gdal.BuildVRT. If False, perform a GDAL warp to produce a single compressed
        GeoTIFF matching the DEM geometry.
    overwrite (bool, optional):
        If True, forces re-creation of the land_use.tif file even if it already exists.

    Notes:
        - Resampling algorithm is 'mode' to preserve categorical landcover classes.
        - Output raster band type is gdal.GDT_Byte when warping to GeoTIFF.
        - Caller must ensure any auxiliary helpers and constants (opens_right, get_dataset_info,
          convert_gt_to_bbox, ESA_TILES_FILE, _dir) are defined in the module scope.
    """
    lu_file = os.path.join(_dir(dem, 2), f'inputs={dem_type}', 'land_use.tif')

    if opens_right(lu_file) and not overwrite:
        return
    
    width, height, gt, proj = get_dataset_info(dem)
    bbox = convert_gt_to_bbox(gt, width, height)
    tiles = set(gpd.read_file(ESA_TILES_FILE, bbox=bbox, ignore_geometry=True, use_arrow=True)['ll_tile'])

    landcover_files = []
    for tile in tiles:
        if os.path.exists(os.path.join(landcover_directory, f"{tile}.tif")):
            landcover_files.append(os.path.join(landcover_directory, f"{tile}.tif"))
        else:
            landcover_files.append(f"/vsis3/esa-worldcover/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")

    if not landcover_files:
        # Let's make a fake landcover file for arc to use. 
        # Fill it with 10, since the areas that don't have it tend to be tropical (10 is trees)
        ds: gdal.Dataset = gdal.GetDriverByName('GTiff').Create(lu_file, width, height, 1, gdal.GDT_Byte, {'COMPRESS': 'ZSTD', 'PREDICTOR': '2'})
        ds.SetGeoTransform(gt)
        ds.SetProjection(proj)
        ds.GetRasterBand(1).Fill(10)
        return

    if save_vrt:
        xres = gt[1]
        yres = abs(gt[5])
        options = gdal.BuildVRTOptions(outputBounds=bbox,
                            outputSRS=proj,
                            xRes=xres,
                            yRes=yres,
                            resampleAlg='mode')
        gdal.BuildVRT(lu_file, landcover_files, options=options)
    else:
        options = gdal.WarpOptions(format='GTiff',
                            outputType=gdal.GDT_Byte,
                            creationOptions=["COMPRESS=ZSTD", "PREDICTOR=2"],
                            outputBounds=bbox,
                            width=width,
                            height=height,
                            outputBoundsSRS=proj,
                            resampleAlg='mode',
                            dstSRS=proj)
        gdal.Warp(lu_file, landcover_files, options=options)

def download_flows(stream_file: str, 
                   rps: list[int] = None,
                   forecast_date: str = None):
    """
    Download and prepare flow and forecast CSV files for a given stream network file.
    This function ensures the presence of several CSV files under a "flow_files"
    directory adjacent to the provided stream_file. It reads hydrologic datasets
    (provided by helper functions _get_fdc, _get_rp, and _get_forecast) and writes
    CSV outputs for:
    - bmf.csv: base/minimum flow and a padded maximum flow estimate (used as a
        baseline/metadata file).
    - baseflow.csv: two-column CSV with river_id and baseflow.
    - flows_<rps>.csv: return-period specific flow values (when rps is provided).
    - <forecast_date>.csv: forecasted maximum flows for the specified forecast date
        (when forecast_date is provided).
    Files are only created/overwritten when they do not exist or when their first
    line does not start with 'river_id' (a simple validity check).

    Parameters
    ----------
    stream_file : str
            Path to the stream network file (used to derive the folder in which
            "flow_files" will be created and to extract river/link IDs via get_linknos).
    rps : list[int], optional
            List of return periods to export into a flows CSV. When provided, a file
            named flows_<comma-separated-rps>.csv will be written. If a requested return
            period is not present in the rp dataset, a message is printed and the
            function returns early.
    forecast_date : str, optional
            Forecast identifier (typically a date string) used to fetch forecast data via
            _get_forecast. When provided, a CSV named <forecast_date>.csv will be written
            containing the maximum forecast discharge per river/link.

    Behavior / Side effects
    ----------------------
    - Creates a directory named "flow_files" next to the stream_file (using _dir).
    - Reads FDC and return-period datasets using helper functions _get_fdc() and
        _get_rp(). For forecasts, uses _get_forecast(forecast_date).
    - Filters river/link IDs returned by get_linknos(stream_file) to those present
        in the datasets before querying the datasets.
    - Writes CSVs with rounded numeric values (three decimal places).
    - For bmf.csv:
            - Computes a per-river "baseflow" as the minimum hourly_monthly p_exceed=0
                value across months, enforcing a minimum of 0.1.
            - Computes a "max" column from the max_simulated values and applies padding
                with the formula (max + 50) * 1.5.
            - Drops certain columns (p_exceed, gumbel, logpearson3) from the output.
    - For flows_<rps>.csv:
            - Exports return-period values from the rp dataset; fills missing
                logpearson3 values with gumbel and writes an unstacked table by
                return_period.
    - For forecast CSV:
            - Selects the maximum Qout across time and ensemble for the requested
                forecast_date and writes it as forecast_max.
    - If a requested return period is missing from the rp dataset, the function
        prints an informative message listing available return periods and returns
        without writing the flows file.

    Return
    ------
    None
            This function performs filesystem I/O and writes CSV files; it does not
            return a value.
    """
    flow_file_dir = os.path.join(_dir(stream_file, 2), 'flow_files')
    bmf = os.path.join(flow_file_dir, 'bmf.csv')
    bf_file = os.path.join(flow_file_dir, 'baseflow.csv')
    linknos = None

    if not os.path.exists(bmf) or not open(bmf).readline().startswith('river_id')  or \
        not os.path.exists(bf_file) or not open(bf_file).readline().startswith('river_id'):
        os.makedirs(flow_file_dir, exist_ok=True)
        fdc_ds = _get_fdc()
        rp_ds = _get_rp()

        linknos = get_linknos(stream_file)
        existing = set(fdc_ds['river_id'].values)
        linknos = list(set(linknos) & existing)

        rp_df = rp_ds.sel(river_id=linknos, return_period=2).to_dataframe().drop(columns='return_period')
        df = (
            fdc_ds.sel(river_id=linknos, p_exceed=0)
            ['hourly_monthly']
            .min(dim='month')
            .to_dataframe()
            .join(rp_df, on='river_id')
            .reset_index()
            .drop(columns=['p_exceed', 'gumbel', 'logpearson3'])
            .rename(columns={'hourly_monthly': 'baseflow', 'max_simulated': 'max'})   
            .round(3)
        )
        df['max'] = (df['max'] + 50) * 1.5 # Add some padding 
        df['baseflow'] = np.maximum(df['baseflow'], 0.1)
        df.to_csv(bmf, index=False)
        df[['river_id', 'baseflow']].to_csv(bf_file, index=False)

    flow_file = os.path.join(flow_file_dir, f"flows_{','.join(map(str, rps))}.csv")
    if rps and (not os.path.exists(flow_file) or not open(flow_file).readline().startswith('river_id')):
        if linknos is None:
            rp_ds = _get_rp()
            fdc_ds = _get_fdc()
            linknos = get_linknos(stream_file)
            existing = set(fdc_ds['river_id'].values)
            linknos = list(set(linknos) & existing)
        try:
            df = rp_ds.sel(river_id=linknos, return_period=rps).to_dataframe()
        except KeyError:
            print(f"Return period {rps} not found in the dataset. Available return periods are {', '.join(rp_ds['return_period'].values.astype(str))}")
            return
        
        if not df.empty:
            df['logpearson3'] = df['logpearson3'].fillna(df['gumbel'])
            df['logpearson3'].unstack(level='return_period').round().to_csv(flow_file)

    forecast_file = os.path.join(flow_file_dir, f'{forecast_date}.csv')
    if forecast_date and (not os.path.exists(forecast_file) or not open(forecast_file).readline().startswith('river_id')):
        if linknos is None:
            fc_zarr = _get_forecast(forecast_date)
            linknos = get_linknos(stream_file)
            existing = set(fc_zarr['rivid'].values)
            linknos = list(set(linknos) & existing)
        fc_df = (
            fc_zarr
            .sel(rivid=linknos)
            ['Qout']
            .max(dim=['time', 'ensemble'])
            .to_dataframe()
            .reset_index()
            .rename(columns={'Qout': 'forecast_max'})
            .round(3)
        )
        fc_df.to_csv(forecast_file, index=False)

def prepare_water_mask(dem: str, dem_type: str, water_value: int = 80):
    """
    Prepare a binary water mask GeoTIFF from a land use raster.
    The function locates an "inputs={dem_type}" subdirectory relative to the directory
    returned by _dir(dem, 2), reads the land_use.tif raster located there, and
    creates a water_mask.tif raster in the same inputs directory. Pixels in the
    output are 1 where the land use value equals `water_value` and 0 otherwise.

    Parameters
    ----------
    dem : str
        Path to a DEM file used to derive the output directory (passed to _dir).
    dem_type : str
        Identifier used to construct the inputs subdirectory name: "inputs={dem_type}".
    water_value : int, optional
        Integer code in the land use raster that denotes water (default: 80).

    Returns
    -------
    None
        The function writes the water mask to disk as a single-band GTiff and
        returns None. If opens_right(water_mask, True) indicates the target is
        already present/accessible, the function returns early without rewriting.

    Side effects
    -----------
    - Reads inputs/<...>/land_use.tif using GDAL.
    - Writes inputs/<...>/water_mask.tif with:
      - 1 band, GDT_Byte (uint8)
      - Compression options: COMPRESS=ZSTD, PREDICTOR=2
      - GeoTransform and Projection copied from the source land use raster
      - NoData value set to 0
    - Pixel values: 1 for water (land_use == water_value), 0 for non-water.

    Notes
    -----
    - The function assumes the land_use raster is a single-band, integer-coded raster.
    - The output array is explicitly cast to uint8, so values will be limited to 0 and 1.
    - ZSTD compression requires GDAL to be built with ZSTD support.
    """
    out_dir = _dir(dem, 2)

    inputs_dir = os.path.join(out_dir, f'inputs={dem_type}')
    land_use = os.path.join(inputs_dir, 'land_use.tif')
    water_mask = os.path.join(inputs_dir, 'water_mask.tif')
    if opens_right(water_mask, True):
        return
    
    ds: gdal.Dataset = gdal.Open(land_use)
    array: np.ndarray = ds.ReadAsArray()
    array = np.where(array == water_value, 1, 0).astype(np.uint8)
    out_ds: gdal.Dataset = gdal.GetDriverByName('GTiff').Create(water_mask, array.shape[1], array.shape[0], 1, gdal.GDT_Byte, options=['COMPRESS=ZSTD', 'PREDICTOR=2'])
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.GetRasterBand(1).SetNoDataValue(0)
    out_ds.WriteArray(array)
    out_ds.FlushCache()

def prepare_inputs(dem: str, 
                   dem_type: str, 
                   mannings_table: str, 
                   rps: list[int] = [],
                   forecast_date: str = "",
                   arc_args: dict = {}, 
                   c2f_bathymetry_args: dict = {}, 
                   c2f_floodmap_args: dict = {}):
    """
    Prepare input files and directory structure required by ARC and Curve2Flood workflows.
    This function inspects provided flow metadata, computes empirical geometry limits,
    creates required directories, and writes multiple "main input" text files consumed
    by ARC and Curve2Flood utilities.

    Parameters
    ----------
    dem : str
        Path to the source DEM raster file used as the base digital elevation model.
    dem_type : str
        A short identifier for the DEM variant (used to name subdirectories and files,
        e.g. 'srtm', 'copernicus', etc.).
    mannings_table : str
        Path to a Manning's n lookup table (raster or csv) referenced in the ARC inputs.
    rps : list[int], optional
        List of return-period identifiers (integers). When provided, the function will
        create floodmap input specifications for flow files named using these return
        periods (e.g. flows_10, flows_50). Default is [] (no return-period outputs).
    forecast_date : str, optional
        Optional forecast date string. If provided, a floodmap input configuration
        for that forecast will be created (a flow file named by the date is expected).
        Default is "" (no forecast-specific outputs).
    arc_args : dict, optional
        Optional mapping of ARC-related parameter overrides. 
        Any missing keys fall back to the listed defaults.
    c2f_bathymetry_args : dict, optional
        Optional overrides used when writing bathymetry input. Currently the function
        reads the 'TW_MultFact' key (default 1.5) to populate the bathymetry input file.
    c2f_floodmap_args : dict, optional
        Optional overrides used when writing floodmap input files. The function uses
        'TW_MultFact' (default 1.5) when present.

    Returns
    -------
    None

    Side effects
    ------------
    - Reads the flow metadata file at out_dir/flow_files/bmf.csv to compute:
        - max_q: maximum value in the 'max' column
        - x_sect_dist: an empirically-derived cross-section distance based on max_q
        - max_tw: an empirically-derived plausible top-width limit based on max_q
    - Writes multiple text input files containing configuration parameters for ARC
        and Curve2Flood, inserting computed values (x_sect_dist, max_tw) and defaults
        or provided overrides from arc_args / c2f_* dictionaries.
    - Creates several directories used by the pipeline if they do not already exist.
    - Does not overwrite certain outputs if helper function opens_right(...) indicates
        the destination file already "opens right" (i.e., exists & is valid according
        to that helper).
    
    Notes
    -----
    - The function expects the module-level helper functions _dir(path, n) and
        opens_right(path) to be available and to behave as used in the implementing module.
    - Mutable default arguments are used in the signature (rps=[], arc_args={}, ...).
        This is a potential pitfall: callers should prefer passing explicit lists/dicts
        or the implementation should be refactored to use None and create defaults
        inside the body.
    - The exact empirical formulas used to compute x_sect_dist and max_tw are applied
        as part of the input creation logic; these are intended to produce conservative
        geometry limits for subsequent processing.
    """
    out_dir = _dir(dem, 2)
    stream_file = os.path.join(out_dir, f'inputs={dem_type}', 'streams.tif')
    
    bmf = os.path.join(_dir(dem, 2), 'flow_files', 'bmf.csv')

    inputs_dir = os.path.join(out_dir, f'inputs={dem_type}')
    vdt_dir = os.path.join(out_dir, 'vdts')
    bathy_dir = os.path.join(out_dir, 'bathymetry')
    os.makedirs(bathy_dir, exist_ok=True)
    os.makedirs(vdt_dir, exist_ok=True)

    vdt = os.path.join(vdt_dir, f'vdt={dem_type}.parquet')
    bathy = os.path.join(bathy_dir, f'bathy={dem_type}.tif')
    land_use = os.path.join(inputs_dir, 'land_use.tif')
    bmf = os.path.join(out_dir, 'flow_files', 'bmf.csv')
    main_input_file = os.path.join(inputs_dir, f'inputs=arc.txt')

    # Use empirical equation to determine x-section distance
    max_q = pd.read_csv(bmf, usecols=['max'], na_filter=False).values.max() # This is the fastest way to get the maximum value
    x_sect_dist = int(min(7500, (5e7 / max_q) + 4000 + (0.0001 * max_q)) / 2) # Divided by two, because this eq is based on top width, and x_sect_dist = 0.5 * tw

    with open(main_input_file, 'w') as f:
        f.write("# Input files - Required\n")
        f.write(f"DEM_File\t{dem}\n")
        f.write(f"Stream_File\t{stream_file}\n")
        f.write(f"LU_Raster_SameRes\t{land_use}\n")
        f.write(f"LU_Manning_n\t{mannings_table}\n")
        f.write(f"Flow_File\t{bmf}\n")
        f.write(f"Degree_Manip\t{arc_args.get('Degree_Manip', 6.5)}\n")
        f.write(f"Degree_Interval\t{arc_args.get('Degree_Interval', 1)}\n")
        f.write(f"Low_Spot_Range\t{arc_args.get('Low_Spot_Range', 2 if dem_type == 'fabdem' else 4)}\n")
        f.write(f"Gen_Slope_Dist\t{arc_args.get('Gen_Slope_Dist', 10 if dem_type == "fabdem" else 20)}\n")
        f.write(f"Gen_Dir_Dist\t{arc_args.get('Gen_Dir_Dist', 10 if dem_type == "fabdem" else 20)}\n")
        f.write(f"X_Section_Dist\t{x_sect_dist}\n")

        f.write("\n# Output files - Required\n")
        f.write(f"VDT_Database_NumIterations\t{arc_args.get('VDT_Database_NumIterations', 15)}\n")
        f.write(f"Print_VDT_Database\t{vdt}\n")

        f.write("\n# Parameters - Required\n")
        f.write(f"Flow_File_ID\triver_id\n")
        f.write(f"Flow_File_BF\tbaseflow\n")
        f.write(f"Flow_File_QMax\tmax\n")
        f.write(f"Spatial_Units\tdeg\n")

        f.write("\n# ARC Bathymetry\n")
        f.write(f"FindBanksBasedOnLandCover\t{arc_args.get('FindBanksBasedOnLandCover', True)}\n")
        f.write(f"AROutBATHY\t{bathy}\n")
        f.write(f"BATHY_Out_File\t{bathy}\n")
    
    bathy_dem_dir = os.path.join(out_dir, 'burned_dems')
    os.makedirs(bathy_dem_dir, exist_ok=True)
    floodmaps_dir = os.path.join(out_dir, 'floodmaps', f'dem={dem_type}')
    os.makedirs(floodmaps_dir, exist_ok=True)

    # Use an empirical equation to get max plausible top width
    max_tw = round(max(2000 * (max_q ** 0.15), 1000), 1)

    burned_dem = os.path.join(bathy_dem_dir, f'dem_burned={dem_type}.tif')
    main_input_file = os.path.join(inputs_dir, f'inputs=burned.txt')
    floodmap = os.path.join(floodmaps_dir, f'bankfull.tif')
    bf_file = os.path.join(out_dir, 'flow_files', 'baseflow.csv')
    water_mask = os.path.join(inputs_dir, 'water_mask.tif')

    if not opens_right(burned_dem):
        with open(main_input_file, 'w') as f:
            f.write("# Main input file for ARC and Curve2Flood\n\n")

            f.write("\n# Input files - Required\n")
            f.write(f"DEM_File\t{dem}\n")
            f.write(f"Stream_File\t{stream_file}\n")
            f.write(f"LU_Raster_SameRes\t{land_use}\n")
            f.write(f"COMID_Flow_File\t{bf_file}\n")
            f.write(f"AROutBATHY\t{bathy}\n")
            f.write(f"BATHY_Out_File\t{bathy}\n")
            f.write(f"Print_VDT_Database\t{vdt}\n")

            f.write(f"BathyWaterMask\t{water_mask}\n")

            f.write("\n# Output files - Optional\n")
            f.write(f"FSOutBATHY\t{burned_dem}\n")
            f.write(f"OutFLD\t{floodmap}\n")

            f.write(f"TopWidthPlausibleLimit\t{max_tw}\n")
            f.write(f"TW_MultFact\t{c2f_bathymetry_args.get('TW_MultFact', 1.5)}\n")
            f.write(f"Flood_WaterLC_and_STRM_Cells\tTrue\n")
    
    # Floodmaps for realz
    names = []
    if rps:
        names.append(f'flows_{",".join(map(str, rps))}')
    if forecast_date:
        names.append(forecast_date)

    for name in names:
        flow_file = os.path.join(out_dir, 'flow_files', f'{name}.csv')
        floodmap = os.path.join(floodmaps_dir, f"{name}.tif")
        main_input_file = os.path.join(inputs_dir, f"inputs={name}.txt")

        if not opens_right(floodmap):
            with open(main_input_file, 'w') as f:
                f.write("# Main input file for ARC and Curve2Flood\n\n")
                f.write("\n# Input files - Required\n")
                f.write(f"DEM_File\t{burned_dem}\n")
                f.write(f"Stream_File\t{stream_file}\n")
                f.write(f"LU_Raster_SameRes\t{land_use}\n")
                f.write(f"COMID_Flow_File\t{flow_file}\n")
                f.write(f"Print_VDT_Database\t{vdt}\n")

                f.write("\n# Output files - Optional\n")
                f.write(f"OutFLD\t{floodmap}\n")

                f.write("\n# Parameters - Optional\n")
                f.write(f"TW_MultFact\t{c2f_floodmap_args.get('TW_MultFact', 1.5)}\n")
                f.write(f"TopWidthPlausibleLimit\t{max_tw}\n")

def run_arc(input_file: str, dem_type: str, overwrite: bool = False):
    """
    Run the ARC model.

    Parameters
    ----------
    input_file : str
        Path to the ARC input text file.
    dem_type : str
        DEM type identifier used to construct output paths.
    overwrite : bool, optional  
        If True, forces re-execution even if output files already exist.
    """
    out_dir = _dir(input_file, 2)

    vdt = os.path.join(out_dir, 'vdts', f'vdt={dem_type}.parquet')

    if opens_right(vdt) and not overwrite:
        return
    
    try:
        Arc(input_file, quiet=True).run()
    except:
        print(input_file)
        raise

def run_c2f_bathymetry(input_file: str, dem_type: str, overwrite: bool = False):
    """
    Run the Curve2Flood bathymetry generation.
    Parameters
    ----------
    input_file : str
        Path to the Curve2Flood bathymetry input text file.
    dem_type : str
        DEM type identifier used to construct output paths.
    overwrite : bool, optional  
        If True, forces re-execution even if output files already exist.
    """
    out_dir = _dir(input_file, 2)

    burned_dem = os.path.join(out_dir, 'burned_dems', f'dem_burned={dem_type}.tif')
    vdt = os.path.join(out_dir, 'vdts', f'vdt={dem_type}.parquet')

    if (not opens_right(burned_dem) or overwrite) and os.path.exists(vdt):
        try:
            Curve2Flood_MainFunction(input_file, quiet=True)
        except Exception as e:
            print(input_file)
            raise e

def run_c2f_floodmaps(input_file: str, dem_type: str, overwrite: bool = False):
    """
    Run the Curve2Flood floodmap generation.
    Parameters
    ----------
    input_file : str
        Path to the Curve2Flood floodmap input text file.
    dem_type : str
        DEM type identifier used to construct output paths.
    overwrite : bool, optional  
        If True, forces re-execution even if output files already exist.
    """
    out_dir = _dir(input_file, 2)

    vdt = os.path.join(out_dir, 'vdts', f'vdt={dem_type}.parquet')
    floodmap = os.path.join(out_dir, 'floodmaps', f'dem={dem_type}', os.path.basename(input_file).replace('inputs=', '').replace('.txt', '.tif'))

    if (not opens_right(floodmap) or overwrite) and os.path.exists(vdt) :
        try:
            Curve2Flood_MainFunction(input_file, quiet=True, flood_vdt_cells=False)
        except Exception as e:
            print(input_file)
            raise e


def get_oceans_raster(oceans_pq: str, bbox: tuple[float], width, height, gt, proj) -> np.ndarray:
    """
    Rasterize ocean geometries from a Parquet vector file into a binary in-memory raster.
    Reads ocean polygons from a Parquet file (using geopandas) restricted to the provided
    bounding box, rasterizes those geometries into an in-memory GDAL raster, and returns
    the resulting numpy array mask where ocean pixels are 1 and non-ocean pixels are 0.

    Parameters
    ----------
    oceans_pq : str
        Filesystem path to a Parquet file containing vector geometries (must include a
        "geometry" column). The file is read with geopandas.read_parquet using the
        `bbox` argument to reduce IO.
    bbox : tuple[float, float, float, float]
        Bounding box (minx, miny, maxx, maxy) used to filter geometries prior to
        rasterization. Only geometries that intersect this box will be considered.
    width : int
        Width (number of columns) of the output raster in pixels.
    height : int
        Height (number of rows) of the output raster in pixels.
    gt : sequence of 6 floats
        GDAL geotransform for the output raster (origin x, pixel width, x rotation,
        origin y, y rotation, pixel height). Passed to GDAL Dataset.SetGeoTransform.
    proj : str
        Spatial reference for the output raster in a form accepted by GDAL's
        SetProjection (commonly a WKT string).

    Returns
    -------
    numpy.ndarray | None
        2D numpy array of shape (height, width) with dtype compatible with GDAL.GDT_Byte
        (typically uint8). Ocean pixels are set to 1 (burn value) and other pixels are 0.
        If no geometries intersect the provided bbox or the GeoDataFrame is empty, the
        function returns None.
    """
    gdf = gpd.read_parquet(oceans_pq, columns=['geometry'], bbox=bbox)
    if gdf.empty:
        return
    
    gdf = gdf[gdf.intersects(box(*bbox))]
    if gdf.empty:
        return
    
    # Step 1: Convert GeoDataFrame to OGR Layer (in memory)
    vector_ds: gdal.Dataset = ogr.GetDriverByName('Memory').CreateDataSource('temp')
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

def unbuffer_remove(floodmaps: list[str], dem_type: str, buffer_distance: float, oceans_pq: str):
    """
    Remove buffering from one or more floodmap raster files and mask out ocean areas, writing the
    modified rasters back to their original paths as Cloud Optimized GeoTIFFs (COGs).
    This function expects that all provided floodmaps cover the same geographic extent. It opens
    each floodmap with GDAL, checks whether the floodmap already matches the expected unbuffered
    dimensions for the given DEM type, and if not, translates (resamples / crops) the floodmap to
    the unbuffered extent. After translation, it masks out ocean pixels using get_oceans_raster and
    overwrites the original floodmap with a COG created from an in-memory raster.
    Behavior by dem_type:
    - "fabdem": The canonical unbuffered size is 3600x3600. If a floodmap already has these
        dimensions the file is left unchanged. Otherwise the floodmap is reprojected / cropped to
        a 3600x3600 window equal to the input geotransform minus/plus buffer_distance.
    - "alos" or "tilezen": The function discovers the associated DEM VRT in a sibling "dems"
        directory. If the floodmap's pixel dimensions already match the DEM dimensions, the file is
        left unchanged. Otherwise the floodmap is cropped to the DEM-derived unbuffered width and
        height, where the new size is computed by subtracting twice the buffer distance (in geounits)
        from the DEM dimensions.
    - Any other dem_type will raise ValueError.
    The first call to get_oceans_raster establishes an oceans mask for the target window; that mask
    is re-used for subsequent floodmaps in the same call to avoid repeated ocean lookups.

    Parameters
    ----------
    floodmaps : list[str]
            Paths to floodmap raster files to process. Each file will be opened with GDAL and potentially
            overwritten by a COG with the unbuffered and ocean-masked data.
    dem_type : str
            One of "fabdem", "alos", or "tilezen". Controls how the unbuffered target size is computed.
    buffer_distance : float
            Distance (in the same spatial units as the raster geotransform) to remove from each side of the
            floodmap (i.e., the size of the buffer to strip). Used to compute the unbuffered bounding box.
    oceans_pq : str
            Source identifier (path or pq dataset) passed to get_oceans_raster to obtain an oceans mask
            for the target bounding box.

    Returns
    -------
    None
            The function writes modified floodmap files in place and does not return a Python value.
    Side effects and notes
    - Overwrites the input floodmap files by creating COGs with compression options
        ['COMPRESS=ZSTD', 'PREDICTOR=2'].
    - Uses GDAL in-memory ("MEM") datasets for intermediate processing.
    - Masks ocean pixels by setting them to the floodmap nodata value (0).
    - Assumes floodmaps cover the same area and that get_oceans_raster accepts the parameters
        (oceans_pq, (minx, miny, maxx, maxy), width, height, gt, proj) and returns a 2D numpy array
        where ocean pixels have value 1.
    - Buffer distance arithmetic assumes gt[1] (pixel width) and gt[5] (pixel height) are non-zero.
    """
    if not floodmaps:
        return
    
    oceans_array = None

    # We assume these floodmaps cover the same area
    for floodmap in floodmaps:
        ds: gdal.Dataset = gdal.Open(floodmap)
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        minx = gt[0]
        maxx = gt[0] + ds.RasterXSize * gt[1]
        miny = gt[3] + ds.RasterYSize * gt[5]
        maxy = gt[3]
        width = ds.RasterXSize
        height = ds.RasterYSize
        ds = None

        # Check if this has already been unbuffered
        if dem_type in {'fabdem'}:
            if width == 3600 and height == 3600:
                continue
            minx += buffer_distance
            maxx -= buffer_distance
            miny += buffer_distance
            maxy -= buffer_distance
            width = 3600
            height = 3600

            options = gdal.TranslateOptions(format='MEM',
                                            projWin=(minx, maxy, maxx, miny),
                                            width=width,
                                            height=height,
                                            noData=0
                                            )
            u_ds: gdal.Dataset = gdal.Translate('', floodmap, options=options)

            gt = u_ds.GetGeoTransform()
            proj = u_ds.GetProjection()
            correct_size: np.ndarray = u_ds.ReadAsArray()
        elif dem_type in {'alos', 'tilezen'}:
            # Need to compare to the dem
            dem_path = glob.glob(os.path.join(_dir(floodmap, 3), 'dems', f'*{dem_type}*.vrt'))[0]
            dem_ds: gdal.Dataset = gdal.Open(dem_path)
            dem_width = dem_ds.RasterXSize
            dem_height = dem_ds.RasterYSize
            dem_ds = None
            if  width != dem_width and height != dem_height:
                continue
            minx += buffer_distance
            maxx -= buffer_distance
            miny += buffer_distance
            maxy -= buffer_distance
            # Compute new width and height
            width = dem_width - int(2 * buffer_distance / abs(gt[1]))
            height = dem_height - int(2 * buffer_distance / abs(gt[5]))

            options = gdal.TranslateOptions(format='MEM',
                                            projWin=(minx, maxy, maxx, miny),
                                            width=width,
                                            height=height,
                                            noData=0
                                            )
            u_ds: gdal.Dataset = gdal.Translate('', floodmap, options=options)
            gt = u_ds.GetGeoTransform()
            proj = u_ds.GetProjection()
            correct_size: np.ndarray = u_ds.ReadAsArray()
                 
        elif dem_type not in {'fabdem'}:
            raise ValueError(f"dem_type {dem_type} not recognized")

        # Now, clip out oceans
        if oceans_array is None:
            oceans_array = get_oceans_raster(oceans_pq, (minx, miny, maxx, maxy), width, height, gt, proj)

        correct_size[oceans_array == 1] = 0

        # Create a new MEM dataset to hold the modified array
        mem_ds: gdal.Dataset = gdal.GetDriverByName('MEM').Create('', width, height, 1, gdal.GDT_Byte)
        mem_ds.SetGeoTransform(u_ds.GetGeoTransform())
        mem_ds.SetProjection(u_ds.GetProjection())
        mem_ds.GetRasterBand(1).WriteArray(correct_size)
        mem_ds.GetRasterBand(1).SetNoDataValue(0)

        # Now safely write to COG
        gdal.GetDriverByName("COG").CreateCopy(floodmap, mem_ds, options=['COMPRESS=ZSTD', 'PREDICTOR=2'])

        # Cleanup
        mem_ds = None
        u_ds = None

# def majority_vote_multiple_bands(floodmap_files: list[str], output_file: str):
#     if opens_right(output_file):
#         return

#     arrays = []
#     gt = None
#     proj = None
#     width = None
#     height = None
#     for floodmap in sorted(floodmap_files, reverse=True):
#         # Tilezen will be first, then fab, then alos
#         if gt is None:
#             ds: gdal.Dataset = gdal.Open(floodmap)
#             gt = ds.GetGeoTransform()
#             proj = ds.GetProjection()
#             width = ds.RasterXSize
#             height = ds.RasterYSize
#         else:
#             options = gdal.WarpOptions(format='MEM',
#                                         outputBounds=(gt[0], gt[3] + height * gt[5], gt[0] + width * gt[1], gt[3]),
#                                         outputBoundsSRS=proj,
#                                         dstSRS=proj,
#                                         width=width,
#                                         height=height,
#                                         resampleAlg='mode')
#             ds: gdal.Dataset = gdal.Warp('', floodmap, options=options)

#         arrays.append(ds.ReadAsArray() > 0)
#         ds = None

#     stacked_data = np.stack(arrays, axis=0)
#     majority_data = np.where(np.sum(stacked_data, axis=0) >= (len(arrays) / 2), 1, 0).astype(np.uint8)

#     out_ds: gdal.Dataset = gdal.GetDriverByName('GTiff').Create(output_file, majority_data.shape[1], majority_data.shape[0], len(arrays), gdal.GDT_Byte, options=['COMPRESS=ZSTD', 'PREDICTOR=2'])
#     out_ds.SetGeoTransform(ds.GetGeoTransform())
#     out_ds.SetProjection(ds.GetProjection())
#     out_ds.GetRasterBand(1).SetNoDataValue(0)
#     out_ds.WriteArray(majority_data)
#     out_ds.FlushCache()

def majority_vote_all_return_periods(floodmap_files: list[str], overwrite: bool = False):
    """
    Compute a majority-vote composite flood map from multiple return-period raster files and save
    the result as a Cloud Optimized GeoTIFF named "majority_vote_all_return_periods.tif" in the
    directory two levels above the first input file.
    The function:
    - Skips processing if the target output already "opens right" (as determined by opens_right()).
    - If only one input file is provided, copies it to the output location using the COG driver.
    - Otherwise, aligns all input rasters to the spatial grid of the first file (using gdal.Warp with
        an in-memory target), stacks their arrays, and computes a per-pixel majority vote across a
        set of discrete class thresholds. The computed majority class values are written to a single
        uint8 raster and saved as a COG with ZSTD compression and predictor 2.
    
    Parameters:
    ----------
    floodmap_files (list[str]):
        Ordered list of filepaths to raster return-period flood maps to be combined.
    overwrite (bool, optional):
        If True, forces re-computation even if the output file already exists and opens right.
    """
    output_file = os.path.join(_dir(floodmap_files[0], 2), 'majority_vote_all_return_periods.tif')
    if opens_right(output_file) and not overwrite:
        return
    
    if len(floodmap_files) == 1:
        # Just copy the file
        gdal.GetDriverByName("COG").CreateCopy(output_file, gdal.Open(floodmap_files[0]), options=['COMPRESS=ZSTD', 'PREDICTOR=2'])
        return

    size_to_use = 'alos' if any('alos' in f for f in floodmap_files) else ('tilezen' if any('tilezen' in f for f in floodmap_files) else 'fabdem')
    floodmap_files.sort(key=lambda x: 0 if size_to_use in x else 1)

    stacked_array = []
    rps = floodmap_files[0]
    width, height, gt, proj = get_dataset_info(rps)
    stacked_array.append(gdal.Open(rps).ReadAsArray())


    for rps in floodmap_files[1:]:
            options = gdal.WarpOptions(format='MEM',
                               width=width,
                                 height=height,
                                 dstSRS=proj,)
    
            rp_ds: gdal.Dataset = gdal.Warp('', rps, options=options)
            stacked_array.append(rp_ds.ReadAsArray())

    stacked_data = np.stack(stacked_array, axis=0)
    output_array = np.zeros((height, width), dtype=np.uint8)
    for _class in np.linspace(0, 100, 7, dtype=np.uint8)[1:]:
        class_mask = (stacked_data >= _class)
        votes = np.sum(class_mask, axis=0)
        majority_mask = votes > (len(stacked_array) / 2)
        output_array[majority_mask] = _class

    ds: gdal.Dataset = gdal.GetDriverByName('MEM').Create('', width, height, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    ds.GetRasterBand(1).SetNoDataValue(0)
    ds.WriteArray(output_array)
    ds.FlushCache()

    gdal.GetDriverByName('COG').CreateCopy(output_file, ds, options=['COMPRESS=ZSTD', 'PREDICTOR=2'])



