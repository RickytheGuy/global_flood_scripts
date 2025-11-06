import json
from global_floodmaps import FloodManager


if __name__ == "__main__":
    bbox = [-121, 39, -120, 40]

    dem_dirs = [
        r"C:\Users\lrr43\Documents\fabdems",
        r"C:\Users\lrr43\Documents\alos_dems",
        r"C:\Users\lrr43\Documents\tilezen"
    ]
    dem_names = ['fabdem', 'alos', 'tilezen']
    output_dir = r"C:\Users\lrr43\Downloads\testing"
    landcover_directory = r"C:\Users\lrr43\Documents\lu"
    oceans_pq = r"C:\Users\lrr43\Documents\worldmaps\seas_buffered.parquet"
    tiles_file = r"C:\Users\lrr43\Documents\tiles_in_geoglowsv2.parquet"
    with open(r"C:\Users\lrr43\Documents\worldmaps\stream_bounds.json", 'r') as f:
        stream_bounds = json.load(f)

    (
        FloodManager(
        dem_dirs=dem_dirs,
        dem_names=dem_names,
        output_dir=output_dir,
        landcover_directory=landcover_directory,
        stream_bounds=stream_bounds,
        oceans_pq=oceans_pq,
        bbox=bbox,
        rps=[2, 5, 10, 25, 50, 100],
        valid_tiles_file=tiles_file,
        # overwrite_majority_maps=True
        overwrite_floodmaps=True,
        # overwrite_burned_dems=True,
        # overwrite_vdts=True,
        # overwrite_buffered_dems=True
        )
        # .download_tilezen(r"C:\Users\lrr43\Documents\tilezen", overwrite=True)
        .run_all()
    )