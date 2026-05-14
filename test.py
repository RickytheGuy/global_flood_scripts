import os, glob
import multiprocessing as mp

import tqdm

from global_floodmaps import LocalDomain, ModelManager, load_default_tiles

def process_tile(args):
    x, y = args
    # print(f"Processing tile at x={x}, y={y}")
    domain = LocalDomain(os.path.join(OUTPUT_DIR, f"lon={x}", f"lat={y}"))
    (
        domain.assign_source_dem(FABDEMS)
        .assign_dem(buffer=True, bbox=(x, y, x+1, y+1), overwrite=False, vrt=True)
        .generate_stream_raster_from_RFS(STREAMLINES, overwrite=False, raise_error_if_no_streams=False)
        .generate_land_cover(LAND_COVER, overwrite=False, vrt=True)
        .generate_bathy_water_mask()
        .generate_base_max_flows(parquet=False)
        .define_arc_configs()
    )

    return domain

OUTPUT_DIR = "/Users/Shared/datasets/flood_map_tiles_2026/"
FABDEMS = glob.glob("/Users/Shared/datasets/fabdem/*.tif")
STREAMLINES = glob.glob("/Users/Shared/datasets/streamlines/*.parquet")
LAND_COVER = glob.glob("/Users/Shared/datasets/esa_landcover/*.tif")

if __name__ == "__main__":
    gdf = load_default_tiles()
    args = []
    for x, y in gdf[['x', 'y']].itertuples(index=False):
        args.append((x, y))

    domains = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # for arg in tqdm.tqdm(args):
        #     domains.append(process_tile(arg))
        for domain in tqdm.tqdm(pool.imap_unordered(process_tile, args), total=len(args)):
            domains.append(domain)

    model_manager = ModelManager(domains=domains)
    model_manager.run(overwrite=False, processes=mp.cpu_count(), pbar=True)