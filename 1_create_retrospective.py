import glob
import os

from nencarta import process_watershed, set_log_level


if __name__ == "__main__":
    dems = glob.glob('/Users/Shared/flood_map_tiles/lon=-111/lat=*/burned_dems/dem_burned=fabdem.tif')
    set_log_level('INFO')

    process_watershed({
        "name": 'test',
        "flowline": os.path.normpath(input_dict["flowline"]),
        "dem_dir": os.path.normpath(input_dict["dem_dir"]),
        "output_dir": os.path.normpath(input_dict["output_dir"]),
        "bathy_use_banks": input_dict.get("bathy_use_banks", False),
        "flood_waterlc_and_strm_cells": input_dict.get("flood_waterlc_and_strm_cells", False),
        "land_watervalue": input_dict.get("land_watervalue", 80),
        "clean_dem": clean_dem,
        "mapper": input_dict.get("mapper", "FloodSpreader"),
        "process_stream_network": input_dict.get("process_stream_network", False),
        "use_specified_depth_for_bathy_mask": use_specified_depth_for_bathy_mask,
        "age_of_forecast_days": input_dict.get("age_of_forecast_days", 7),
        "find_banks_based_on_landcover": input_dict.get("find_banks_based_on_landcover", True),
        "specify_depths_for_bathy_mask": specify_depths_for_bathy_mask,
        "create_reach_average_curve_file": input_dict.get("create_reach_average_curve_file", False),
        "use_warning_flags_to_download_dem": input_dict.get("use_warning_flags_to_download_dem", False),
        "geoglows_vpu": input_dict.get("geoglows_vpu"),
        "forensic_forecast_date": validate_forecast_date(input_dict.get("forensic_forecast_date"), streamflow_source),
        "forensic_forecast_hour": forensic_forecast_hour,
        "specified_bathyflow_field":input_dict.get("specified_bathyflow_field", "p_exceed_50"),
        "specified_highflow_field":input_dict.get("specified_highflow_field", "rp100_premium"),
        "StrmOrder_Field": input_dict.get("StrmOrder_Field"),
        "StrmOrder_Lower": input_dict.get("StrmOrder_Lower"),
        "StrmOrder_Upper": input_dict.get("StrmOrder_Upper"),
        "q_baseflow_threshold": float_or_none(input_dict.get("q_baseflow_threshold")),
        "lake_filter_json": norm_or_none(input_dict.get("lake_filter_json")),
        "estimate_consequences": input_dict.get("estimate_consequences", False),
        "streamflow_source": streamflow_source,
        "nwm_api_key": nwm_api_key,
        "overwrite_floodmaps": input_dict.get("overwrite_floodmaps", True),
        "remove_old_forecast_files": input_dict.get("remove_old_forecast_files", False),
        "make_fist_inputs": input_dict.get("make_fist_inputs", True),
        "dem_filter": dem_filter,
        "floodmap_mode": floodmap_mode,
        "user_flow_files": user_flow_files,
        "make_curvefile": input_dict.get("make_curvefile", True),
        "make_ap_database": input_dict.get("make_ap_database", True),
        "vdt_file_extension": input_dict.get("vdt_file_extension", 'txt'),
        "mannings_text_file": norm_or_none(input_dict.get("mannings_text_file")),
        "bathy_args": input_dict.get("bathy_args", {}),
        "floodmap_args": input_dict.get("floodmap_args", {}),
        "make_depth_maps": input_dict.get("make_depth_maps", True),
        "make_velocity_maps": input_dict.get("make_velocity_maps", True),
        "make_wse_maps": input_dict.get("make_wse_maps", True),
        "floodmap_identifier": input_dict.get("floodmap_identifier", ""),
        "move_stream_network_to_new_locations": input_dict.get("move_stream_network_to_new_locations", False),
        "new_strm_threshold_km2": float_or_none(input_dict.get("new_strm_threshold_km2")),
        "min_match_score": float_or_none(input_dict.get("min_match_score")),
        "quiet": input_dict.get("quiet", False),
    })