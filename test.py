from global_floodmaps import LocalDomain, ModelManager

domain = LocalDomain(directory=r"C:\Users\lrr43\Downloads\test_domain")

import glob
source_dems = glob.glob(r"E:\ricky\fab_dems\*.tif")

(
    domain.assign_source_dem(source_dems)
    .assign_dem(buffer=True, bbox=(-98.5, 29.4, -98.3, 29.6), overwrite=False, vrt=True)
    .generate_stream_raster_from_RFS(glob.glob(r'C:\Users\lrr43\Documents\streamlines\*.parquet'), overwrite=False)
    .generate_land_cover(glob.glob(r'C:\Users\lrr43\Documents\lu\*.tif'), overwrite=False, vrt=True)
    .generate_bathy_water_mask()
    .generate_base_max_flows()
    .define_arc_configs()
    .generate_flood_flow_file_from_base_max_file('rp100')
)

model_manager = ModelManager(domains=domain)
model_manager.run(overwrite=False, processes=1, pbar=True)