import json
from global_floodmaps import FloodManager


if __name__ == "__main__":
    bbox = [-110, 37, -109, 38]

    dem_dirs = [
        r"C:\Users\lrr43\Documents\fabdems",
        r"C:\Users\lrr43\Documents\alos_dems",
        r"C:\Users\lrr43\Documents\tilezen"
    ]
    dem_names = ['fabdem', 'alos', 'tilezen']
    output_dir = r"C:\Users\lrr43\Documents\global_flood_maps"
    landcover_directory = r"C:\Users\lrr43\Documents\lu"
    oceans_pq = r"C:\Users\lrr43\Documents\worldmaps\seas_buffered.parquet"
    streamlines_directory = r"C:\Users\lrr43\Documents\worldmaps\streamlines"

    (
        FloodManager(
        dem_dirs=dem_dirs,
        dem_names=dem_names,
        output_dir=output_dir,
        landcover_directory=landcover_directory,
        streamlines_directory=streamlines_directory,
        oceans_pq=oceans_pq,
        # bbox=bbox,
        rps=[2, 5, 10, 25, 50, 100],
        number_of_tiles=2700,
        offset=0,
        # overwrite_majority_maps=True,
        # overwrite_floodmaps=True,
        # overwrite_burned_dems=True,
        # overwrite_vdts=True,
        # overwrite_buffered_dems=True,
        # overwrite_landuse=True,
        # overwrite_streams=True,
        )
        .download_tilezen(r"C:\Users\lrr43\Documents\tilezen")
        .download_alos(r"C:\Users\lrr43\Documents\alos_dems")
        .download_fabdem(r"C:\Users\lrr43\Documents\fabdems")
        .download_landcover()
        .download_streamlines()
        .download_oceans_pq(oceans_pq)
        .run_all()
    )