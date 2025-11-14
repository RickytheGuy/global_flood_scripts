import json
from global_floodmaps import FloodManager


if __name__ == "__main__":
    bbox = [128, 0, 129, 1]

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
    s3_dir = 's3://global-floodmaps/tiles/'

    tries = 0
    while tries < 20:
        try:
            (
                FloodManager(
                dem_dirs=dem_dirs,
                dem_names=dem_names,
                output_dir=output_dir,
                landcover_directory=landcover_directory,
                streamlines_directory=streamlines_directory,
                oceans_pq=oceans_pq,
                s3_dir=s3_dir,
                # bbox=bbox,
                rps=[2, 5, 10, 25, 50, 100],
                number_of_tiles=40,
                offset=2640,
                # overwrite_majority_maps=True,
                # overwrite_floodmaps=True,
                # overwrite_burned_dems=True,
                # overwrite_vdts=True,
                # overwrite_buffered_dems=True,
                # overwrite_landuse=True,
                # overwrite_streams=True,
                )
                .set_log_level(10)  # DEBUG level
                .download_from_s3(r"C:\Users\lrr43\Documents\tilezen", "tilezen", no_download=True)
                .download_from_s3(r"C:\Users\lrr43\Documents\alos_dems", "alos", no_download=True)
                .download_from_s3(r"C:\Users\lrr43\Documents\fabdems", "fabdem", no_download=True)
                # .download_landcover()
                # .download_streamlines()
                # .download_oceans_pq(oceans_pq)
                .run_all()
            )
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            # print(f"An error occurred: {e}")
            tries += 1
            print(f"Retrying... Attempt {tries}/20")