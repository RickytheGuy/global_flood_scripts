from global_floodmaps import LocalDomain, ModelManager

domain = LocalDomain(directory=r"C:\Users\lrr43\Downloads\test_domain")

import glob

@profile
def main():
    source_dems = glob.glob(r"E:\ricky\fab_dems\*.tif")

    (
        domain.assign_source_dem(source_dems)
        .assign_dem(buffer=True, bbox=(-99, 29, -98, 30), overwrite=True, vrt=True)
        .generate_stream_raster_from_RFS(glob.glob(r'C:\Users\lrr43\Documents\streamlines\*.parquet'), overwrite=True)
        .generate_land_cover(glob.glob(r'C:\Users\lrr43\Documents\lu\*.tif'), overwrite=True, vrt=True)
        .generate_bathy_water_mask(overwrite=True)
        .generate_base_max_flows(overwrite=True)
        .define_arc_configs(overwrite=True)
        .generate_flood_flow_file_from_base_max_file('rp100', overwrite=True)
    )

    model_manager = ModelManager(domains=domain)
    model_manager.run(overwrite=True, processes=1, pbar=True)

if __name__ == "__main__":
    main()