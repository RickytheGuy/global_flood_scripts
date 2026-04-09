import glob
import os
import multiprocessing as mp
from functools import partial

os.environ["KMP_WARNINGS"] = "0"

from nencarta import process_watershed, set_log_level
from global_floodmaps.utility_functions import get_streamlines_in_dem_extent, _dir, get_return_period_flows_in_dem_extent, buffer_dem
from global_floodmaps.parallel_functions import bar_map, bar_starmap

set_log_level('ERROR')

if __name__ == "__main__":
    dems = sorted(glob.glob('/Users/Shared/flood_map_tiles/lon=-111/lat=*/burned_dems/dem_burned=fabdem.tif'))
    streamlines = glob.glob('/Users/Shared/streamlines/streams_*.parquet')
    output_dir = '/Users/rickyrosas/tests'
    buffered_dem_dir = '/Users/rickyrosas/buffered_dems'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(buffered_dem_dir, exist_ok=True)

    with mp.Pool() as pool:
        streamlines_for_each_dem = bar_map(
            pool, 
            partial(get_streamlines_in_dem_extent, streamlines=streamlines), 
            dems,
            total=len(dems), desc="Finding streamlines in DEM extents")

        dems_with_streams = []
        buffered_dems = []
        streamlines_to_use = []
        for dem, streamlines in zip(dems, streamlines_for_each_dem):
            if streamlines is not None and len(streamlines) > 0:
                dems_with_streams.append(dem)
                streamlines_to_use.append(streamlines[0])

                buffered_dem = os.path.join(buffered_dem_dir, os.path.basename(_dir(dem, 2)) + '_' + os.path.basename(dem).replace('.tif', '_buffered.vrt'))
                buffered_dems.append(buffered_dem)
            else:
                print(f"No streamlines found in extent of {dem}, skipping.")

        to_do = []
        for dem, buffered_dem in zip(dems_with_streams, buffered_dems):
            if not os.path.isfile(buffered_dem):
                to_do.append((dem, buffered_dem))

        if to_do:
            bar_starmap(
                pool, 
                partial(buffer_dem, all_dems=dems, as_vrt=True, buffer_distance=0.01),
                to_do,
                total=len(to_do), desc="Buffering DEMs"
            )

        rp_flow_files = []
        names = []
        for dem in buffered_dems:
            lat = os.path.basename(dem)[:6]
            names.append(lat)

            flow_file = os.path.join(output_dir, lat, 'return_period_flows.csv')
            os.makedirs(os.path.dirname(flow_file), exist_ok=True)
            rp_flow_files.append(flow_file)
        
        bar_starmap(
            pool,
            get_return_period_flows_in_dem_extent, 
            zip(buffered_dems, streamlines_to_use, [[10, 25, 50, 100]]*len(streamlines_to_use), rp_flow_files),
            total=len(buffered_dems),
            desc="Calculating return period flows"
        )

        kwargs = {
            # "name": 'test',
            # "flowline": streamlines_to_use[0],
            # "dem_dir": os.path.dirname(dems_with_streams[0]),
            "output_dir": output_dir,
            "bathy_use_banks": False,
            "flood_waterlc_and_strm_cells": False,
            "land_watervalue": 80,
            "clean_dem": False,
            "mapper": "Curve2Flood",
            "process_stream_network": True,
            "use_specified_depth_for_bathy_mask": False,
            "find_banks_based_on_landcover": True,
            "specify_depths_for_bathy_mask": False,
            "create_reach_average_curve_file": False,
            "use_warning_flags_to_download_dem": False,
            "specified_bathyflow_field": "p_exceed_50",
            "specified_highflow_field": "rp100_premium",
            "streamflow_source": "GEOGLOWS",
            "overwrite_floodmaps": False,
            "make_fist_inputs": False,
            # "dem_filter": "*fabdem*.vrt",
            "floodmap_mode": "user",
            # "user_flow_files": rp_flow_files[0],
            "make_curvefile": False,
            "make_ap_database": False,
            "vdt_file_extension": "parquet",
            "make_depth_maps": False,
            "make_velocity_maps": False,
            "make_wse_maps": False,
            "floodmap_identifier": "",
            "bathy_args": {
                "X_Section_Dist": 5000, 
                "Degree_Manip": 6.1,
                "Degree_Interval": 1.5, 
                "Low_Spot_Range": 2,
                "Str_Limit_Val": None,
                "Gen_Dir_Dist": 10,
                "Gen_Slope_Dist": 10,
                "Stream_Slope_Method": "local_average_corrected"
            },
            "move_stream_network_to_new_locations": False,
            "quiet": True,
            "land_use_cache_dir": "/Users/Shared/esa_landcover"
        }
        
        args = []
        for dem, stream, flow_file, name in zip(buffered_dems, streamlines_to_use, rp_flow_files, names):
            args.append(kwargs | {
                "name": name,
                "flowline": stream,
                "dem_dir": os.path.dirname(dem),
                "user_flow_files": flow_file,
                "dem_filter": os.path.basename(dem)
            })

        bar_map(pool, process_watershed, args, total=len(args), desc="Processing watersheds")

    
# First naive time: 15:15
# With LU cache: 14:38
