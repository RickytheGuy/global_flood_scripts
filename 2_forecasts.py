import glob, os, logging
from collections import defaultdict
from multiprocessing import Pool

import s3fs
import tqdm
import json
import pandas as pd
import geopandas as gpd
import xarray as xr
from osgeo import gdal
import numpy as np
from osgeo import ogr, osr
from shapely.geometry import box
from curve2flood import Curve2Flood_MainFunction

gdal.UseExceptions()

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

from global_floodmaps.utility_functions import _dir
from global_floodmaps.parallel_functions import bar_map, bar_starmap


def map_river_id_to_tiles(stream_file: str) -> dict[int, list[list[int]]]:
    """Process a single stream file and return river_id -> tiles mapping."""
    local_map = defaultdict(list)

    streams_gdf = gpd.read_parquet(stream_file).to_crs(4326)

    for river_id, geom in streams_gdf[["LINKNO", "geometry"]].itertuples(index=False):
        minx, miny, maxx, maxy = geom.bounds

        lon_start = int(minx)
        lon_end = int(maxx) + 1
        lat_start = int(miny)
        lat_end = int(maxy) + 1

        for lon in range(lon_start, lon_end):
            for lat in range(lat_start, lat_end):
                local_map[river_id].append([lon, lat])

    return local_map

def create_id_to_tiles_from_geometry(stream_files: list[str], output_json: str, processes: int = None):
    id_to_tile = defaultdict(list)

    if processes is None:
        processes = os.cpu_count()

    with Pool(processes) as pool:
        for local_map in bar_map(
            pool, 
            map_river_id_to_tiles, 
            stream_files,
            total=len(stream_files),
            desc="Mapping river IDs to tiles based on geometry"
        ):
            for river_id, tiles in local_map.items():
                id_to_tile[river_id].extend(tiles)

    with open(output_json, "w") as f:
        json.dump(id_to_tile, f)

def invert_dict(input_dict: dict[int, list[list[int]]]) -> dict[tuple[int, int], set[int]]:
    inverted_dict = defaultdict(set)
    for key, value_list in input_dict.items():
        for item in value_list:
            inverted_dict[tuple(item)].add(key)
    return inverted_dict

def get_streams_above_rp5(date: str) -> pd.DataFrame:
    ds = xr.open_zarr(f's3://geoglows-v2-forecasts/{date}.zarr', storage_options={'anon': True})
    rp_ds = xr.open_zarr('s3://geoglows-v2/retrospective/return-periods.zarr', storage_options={'anon': True})
    max_median_flow = ds['Qout'].isel(time=0, rivid=slice(4_707_000, 4_708_000)).median(dim='ensemble').to_dataframe().drop(columns='time')
    
    
    rp5 = rp_ds['logpearson3'].sel(river_id=max_median_flow.index, return_period=5).to_dataframe().drop(columns='return_period')
    max_median_flow = max_median_flow.join(rp5, how='inner')
    max_median_flow.index.name = 'river_id'
    return max_median_flow[max_median_flow['Qout'] >= max_median_flow['logpearson3']].drop(columns='logpearson3')


def get_tile_dict(rivers_above_rp5_df: pd.DataFrame, riverid_to_tiles_json: str):
    with open(riverid_to_tiles_json, 'r') as f:
        riverid_to_tiles = json.load(f)

    rivers_above_rp5 = set(rivers_above_rp5_df.index)

    id_tiles = {int(key): value for key, value in riverid_to_tiles.items() if int(key) in rivers_above_rp5}
    tiles_to_riverid = invert_dict(id_tiles)
    return tiles_to_riverid


def make_floodmap(flow_file: str, dem: str) -> tuple[str, str] | None:
    floodmap = flow_file.replace('flow_files', f'floodmaps{os.sep}dem={dem}').replace('.csv', '.tif')
    source_dir = flow_file.split('flow_files', 1)[0].rstrip(os.sep)

    dem_file = os.path.join(source_dir, 'burned_dems', f'dem_burned={dem}.tif')
    if not os.path.exists(dem_file):
        return None

    vdt = os.path.join(source_dir, 'vdts', f'vdt={dem}.parquet')
    if not os.path.exists(vdt):
        return None
    
    if os.path.exists(floodmap):
        return floodmap
    
    os.makedirs(os.path.dirname(floodmap), exist_ok=True)
    
    max_q = pd.read_csv(flow_file, usecols=['1'], na_filter=False).values.max()
    max_tw = round(max(2000 * (max_q ** 0.15), 1000), 1)

    Curve2Flood_MainFunction(args={
        'DEM_File': dem_file,
        'Print_VDT_Database': vdt,
        'COMID_Flow_File': flow_file,
        'OutFLD': floodmap,
        'TopWidthPlausibleLimit': max_tw,
    },
    flood_vdt_cells=False,
    quiet=True
    )

    return floodmap

def run_c2f(flow_files: list[str], processes: int = None) -> list[str]:
    args_list = []
    for flow_file in flow_files:
        for dem in ['fabdem', 'alos', 'tilezen']:
            args_list.append((flow_file, dem))

    if processes is None:
        processes = os.cpu_count()

    with Pool(processes=processes) as pool:
        results = bar_starmap(
            pool,
            make_floodmap,
            args_list,
            total=len(args_list),
            desc="Running Curve2Flood"
        )

    results = [res for res in results if res is not None]
    return results

def get_dataset_info(path: str):
    ds: gdal.Dataset = gdal.Open(path)
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    projection = ds.GetProjection()
    return width, height, gt, projection

def get_oceans_raster(oceans_pq: str, bbox: tuple[float], width, height, gt, proj) -> np.ndarray:
    gdal.AllRegister()
    ogr.RegisterAll()

    gdf = gpd.read_parquet(oceans_pq, columns=['geometry'], bbox=bbox)
    if gdf.empty:
        return
    
    gdf = gdf[gdf.intersects(box(*bbox))]
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


def unbuffer_remove(floodmap: str, dem_type: str, buffer_distance: float, oceans_pq: str):
    # if not os.path.exists(floodmap):
    #     return 
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
            return
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
        try:
            dem_path = glob.glob(os.path.join(_dir(floodmap, 3), 'dems', f'*{dem_type}*.vrt'))[0]
        except IndexError:
            return
        
        dem_ds: gdal.Dataset = gdal.Open(dem_path)
        dem_width = dem_ds.RasterXSize
        dem_height = dem_ds.RasterYSize
        dem_ds = None
        if width != dem_width and height != dem_height:
            return
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
    oceans_array = get_oceans_raster(oceans_pq, (minx, miny, maxx, maxy), width, height, gt, proj)

    land_use = os.path.join(_dir(floodmap, 3), f'inputs={dem_type}', 'land_use.tif')
    if os.path.exists(land_use):
        lu_ds: gdal.Dataset = gdal.Translate('', land_use, options=gdal.TranslateOptions(format='MEM', projWin=(minx, maxy, maxx, miny), width=width, height=height))
        land_use_array: np.ndarray = lu_ds.ReadAsArray()
        lu_ds = None
        correct_size[land_use_array == 80] = 100 

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

def majority_vote(floodmap_files: list[str], 
                  output_dir: str = '') -> str:
    output_file = os.path.join(_dir(floodmap_files[0], 2), f'majority_vote_{os.path.basename(floodmap_files[0])}')
        
    # if os.path.exists(output_file):
    #     return output_file
    
    os.makedirs(_dir(output_file), exist_ok=True)

    size_to_use = 'alos' if any('alos' in f for f in floodmap_files) else ('tilezen' if any('tilezen' in f for f in floodmap_files) else 'fabdem')
    floodmap_files.sort(key=lambda x: 0 if size_to_use in x else 1)

    stacked_array = []
    first_floodmap = floodmap_files[0]
    width, height, gt, proj = get_dataset_info(first_floodmap)
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + width * gt[1]
    miny = maxy + height * gt[5]
    stacked_array.append(gdal.Open(first_floodmap).ReadAsArray())


    for first_floodmap in floodmap_files[1:]:
        if not os.path.exists(first_floodmap):
            continue
        options = gdal.WarpOptions(format='MEM', width=width, height=height, dstSRS=proj, resampleAlg='mode')
        rp_ds: gdal.Dataset = gdal.Warp('', first_floodmap, options=options)
        stacked_array.append(rp_ds.ReadAsArray())

    stacked_data = np.stack(stacked_array, axis=0)
    output_array = np.zeros((height, width), dtype=np.uint8)
    for _class in np.unique(stacked_data)[1:]:
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
    return output_file

def mp_majority_vote(floodmap_groups: list[list[str]]) -> list[str]:
    output_files = []
    with Pool(processes=os.cpu_count()) as pool:
        for output_file in tqdm.tqdm(
            pool.imap_unordered(majority_vote, floodmap_groups),
            total=len(floodmap_groups),
            desc="Majority voting floodmaps"
        ):
            if output_file is not None:
                output_files.append(output_file)
    return output_files

def unbuffer_remove_helper(args):
    return unbuffer_remove(*args)

def mp_unbuffer_remove(floodmap_dirs: list[list[str]], buffer_distance: float, oceans_pq: str):
    floodmaps = [f for sublist in floodmap_dirs for f in sublist]

    with Pool(processes=min(os.cpu_count()-2, len(floodmap_dirs))) as pool:
        for dem_type in ['fabdem', 'alos', 'tilezen']:
            args_list = [(fmap, dem_type, buffer_distance, oceans_pq
                         ) for fmap in floodmaps]
            list(tqdm.tqdm(
                pool.imap_unordered(unbuffer_remove_helper, args_list),
                total=len(args_list),
                desc=f"Unbuffering and removing oceans from floodmaps {dem_type}"
            ))

def get_floodmap_groups(floodmaps: list[str]) -> list[list[str]]:
    floodmap_dirs = defaultdict(list)
    for floodmap in floodmaps:
        floodmap_dir = _dir(floodmap, 2)
        
        if os.path.exists(floodmap):
            floodmap_dirs[floodmap_dir].append(floodmap)
    floodmap_dirs = list(floodmap_dirs.values())
    floodmap_dirs = [f for f in floodmap_dirs if isinstance(f, list)]
    return floodmap_dirs

def get_maptable_df(date: str, cache_file: str = None) -> pd.DataFrame:
    if cache_file and os.path.exists(cache_file):
        return pd.read_parquet(cache_file)
    
    s3 = s3fs.S3FileSystem(anon=True)
    maptables = s3.glob(f's3://geoglows-v2-forecast-products/map-tables/{date}/mapstyletable_*_{date}.parquet')
    df = pd.concat([pd.read_parquet(f"s3://{path}", storage_options={'anon': True}) for path in maptables], ignore_index=True)
    if cache_file:
        df.to_parquet(cache_file)
    return df

def filter_flows(df: pd.DataFrame, min_rp: int) -> pd.DataFrame:
    """
    sometimes, a return period of 100 has streamflow value of .1, which is not something we should model
    So let us filter out those whose mean is < 1
    Also, let's filter to only those with return period >= min_rp
    """
    return df[(df['mean'] >= 1)  & (df['ret_per'] >= min_rp)]

def get_forecast_peak_flows(date: str, min_rp: int, cache_file: str = None) -> pd.DataFrame:
    cache_file_2 = cache_file.replace('.parquet', '_filtered.parquet') if cache_file else None
    if cache_file_2 and os.path.exists(cache_file_2):
        return pd.read_parquet(cache_file_2, index_col='comid')
    df = get_maptable_df(date, cache_file=cache_file)

    df = filter_flows(df, min_rp)
    df_max = df.groupby('comid')['ret_per'].max()
    df_max = df_max[df_max.values >= min_rp]
    df = df[df['comid'].isin(df_max.index)]
    df = df.groupby(['comid']).max()[['mean', 'ret_per']]
    if cache_file_2:
        df.to_parquet(cache_file_2)

    return df

def get_forecast_peak_ensemble_flows(date: str, min_rp: int, cache_file: str) -> pd.DataFrame:
    cache_file = cache_file.replace('.parquet', '_ensemble.parquet')
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)
    
    rp_ds = xr.open_zarr('s3://geoglows-v2/retrospective/return-periods.zarr', storage_options={'anon': True})
    rp_threshold = (
        rp_ds["gumbel"]
        .sel(return_period=rp_ds.return_period[rp_ds.return_period >= min_rp].min())
        .rename({"river_id": "rivid"})
    )

    ds = xr.open_zarr(f's3://geoglows-v2-forecasts/{date}.zarr', storage_options={'anon': True}, chunks='auto')
    forecast_max = ds['Qout'].max(dim='time')
    above_rp = forecast_max >= rp_threshold
    rivid_mask = above_rp.any(dim="ensemble")
    filtered = forecast_max.sel(rivid=rivid_mask)
    df = (
        filtered
        .to_dataframe()
        .reset_index()
        .pivot(index="rivid", columns="ensemble", values="Qout")
    )
    df.to_parquet(cache_file)
    return df

def get_tiles(df: pd.DataFrame, tile_path: str):
    with open(tile_path, 'r') as f:
        tile_dict: dict[str, list[list[int]]] = json.load(f)

    tile_id_dict: dict[tuple[int, int], list[int]] = defaultdict(list)
    for comid in df.index:
        for tile in tile_dict[str(comid)]:
            tile_key = (tile[0], tile[1])
            tile_id_dict[tile_key].append(comid)

    return tile_id_dict

def create_flow_files_for_single_flow(df: pd.DataFrame, tiles: dict[tuple[int, int], list[int]], date: str, output_folder: str):
    flow_files = []
    for tile_key, comid_list in tiles.items():
        tile_filename = os.path.join(output_folder, f'lon={tile_key[0]}', f'lat={tile_key[1]}', 'flow_files', f'{date}.csv')
        flow_files.append(tile_filename)
        os.makedirs(os.path.dirname(tile_filename), exist_ok=True)
        tile_df = df.loc[comid_list]
        tile_df[['mean']].to_csv(tile_filename)

    return flow_files

def create_ensemble_flow_files(df: pd.DataFrame, tiles: dict[tuple[int, int], list[int]], date: str, output_folder: str):
    flow_files = []
    for tile_key, comid_list in tiles.items():
        tile_filename = os.path.join(output_folder, f'lon={tile_key[0]}', f'lat={tile_key[1]}', 'flow_files', f'{date}_ensemble.csv')
        flow_files.append(tile_filename)
        os.makedirs(os.path.dirname(tile_filename), exist_ok=True)
        tile_df = df.loc[comid_list]
        tile_df.to_csv(tile_filename)

    return flow_files


if __name__ == "__main__":
    date = '2025121200'
    forecast_data_dir = '/Users/rickyrosas/floodmap_forecast_data'
    dems = sorted(glob.glob('/Users/rickyrosas/tests/lat=*/Bathymetry/GEOGLOWS_lat=*_dem_burned=fabdem_buffered_FS_Bathy.tif'))
    cache_file = os.path.join(forecast_data_dir, f'comdid_flow_{date}.parquet')
    id_to_tile_json = os.path.join(forecast_data_dir, 'riverid_to_tiles.json')
    seas_parquet = os.path.join(forecast_data_dir, 'seas_buffered.parquet')

    if not os.path.exists(id_to_tile_json):
        stream_files = glob.glob('/Users/rickyrosas/streamlines/streams_*.parquet')
        create_id_to_tiles_from_geometry(stream_files, id_to_tile_json)

    logging.info(f"Starting forecast for date: {date}")
    # df = get_forecast_peak_ensemble_flows(date, 10, cache_file)
    df = get_forecast_peak_flows(date, 10, cache_file)
    logging.info("Retrieved forecast peak flows")
    tiles = get_tiles(df, id_to_tile_json)

    # One of these:
    flow_files = create_flow_files_for_single_flow(df, tiles, date, output_folder) # 15 mins
    # flow_files = create_ensemble_flow_files(df, tiles, date, output_folder) # 24 hours

    floodmaps = run_c2f(flow_files)
    floodmaps_dirs = get_floodmap_groups(floodmaps)
    mp_unbuffer_remove(floodmaps_dirs, 0.1, seas_parquet)
    majority_floodmaps = mp_majority_vote(floodmaps_dirs)
    logging.info("Floodmaps created successfully")

