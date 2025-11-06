import os
import glob
from concurrent.futures import ProcessPoolExecutor

import tqdm
import natsort
import pandas as pd
from osgeo import gdal

from .parallel_functions import (
    buffer_dem, run_arc, rasterize_streams, warp_land_use, _convert_process_count,
    start_unthrottled_pbar, download_flows, prepare_water_mask, prepare_inputs, 
    start_throttled_pbar, run_c2f_bathymetry, run_c2f_floodmaps, unbuffer_remove,
    majority_vote_all_return_periods
)

from .utility_functions import (
    filter_files_in_extent_by_lat_lon_dirs, get_dem_in_extent
)


gdal.UseExceptions()

class FloodManager:
    def __init__(self,
                 dem_dirs: list[str],
                 dem_names: list[str],
                 output_dir: str,
                 landcover_directory: str,
                 stream_bounds: dict[str, tuple[float, float, float, float]],
                 oceans_pq: str,
                 bbox: tuple[float, float, float, float] = None,
                 mannings_table: str = None,
                 rps: list[int] = None,
                 forecast_date: str = None,
                 valid_tiles_file: str = None,
                 buffer_distance: float = 0.1,
                 buffer_dems_as_vrt: bool = True, 
                 arc_args: dict = {},
                 c2f_bathymetry_args: dict = {},
                 c2f_floodmap_args: dict = {},
                 overwrite_floodmaps: bool = False,
                 overwrite_burned_dems: bool = False,
                 overwrite_vdts: bool = False,
                 overwrite_streams: bool = False,
                 overwrite_landuse: bool = False,
                 overwrite_buffered_dems: bool = False):
        """
        Initialize a FloodManager instance.

        Parameters
        ----------
        dem_dirs : list[str]
            List of filesystem directories containing digital elevation model (DEM) tiles.
        dem_names : list[str]
            Parallel list of names/identifiers corresponding to each entry in `dem_dirs`.
        output_dir : str
            Directory path where outputs (intermediate and final) will be written.
        landcover_directory : str
            Directory path containing landcover data used in processing.
        stream_bounds : dict[str, tuple[float, float, float, float]]
            Mapping of stream parquets to bounding boxes (minx, miny, maxx, maxy) used to constrain
            processing for particular stream segments.
        oceans_pq : str
            Path to a parquet file or dataset describing ocean/water extent information.
        bbox : tuple[float, float, float, float], optional
            Global bounding box (minx, miny, maxx, maxy) used to filter tiles and operations. If None,
            defaults to (-180.0, -90.0, 180.0, 90.0).
        mannings_table : str, optional
            Path to a Manning's n lookup table. If not provided, a default table bundled with the
            package (data/default_mannings_table.txt next to this module) will be used.
        rps : list[int], optional
            List of return periods (e.g., flood recurrence intervals) to simulate or produce floodmaps for.
        forecast_date : str, optional
            Forecast date identifier (e.g., 'YYYYMMDD') used to label outputs or drive time-dependent logic.
        valid_tiles_file : str, optional
            Path to a parquet file containing tile coordinates (columns 'x', 'y'). If provided, the file is
            read and tiles are filtered to those intersecting `bbox`; otherwise `valid_tiles` will be set to
            None and tile selection will be handled elsewhere.
        buffer_distance : float, default 0.1
            Distance (in the same units as tile coordinates, typically degrees) used to buffer tile extents
            when assembling inputs.
        buffer_dems_as_vrt : bool, default True
            If True, DEMs will be buffered/combined into a virtual raster (VRT) rather than copied or
            reprojected tile-by-tile.
        arc_args : dict, optional
            Arbitrary keyword arguments forwarded to ARC.
        c2f_bathymetry_args : dict, optional
            Arbitrary keyword arguments forwarded to curve2flood.
        c2f_floodmap_args : dict, optional
            Arbitrary keyword arguments forwarded to curve2flood.
        """
        
        if len(dem_dirs) != len(dem_names):
            raise ValueError("dem_dirs and dem_names must have the same length.")
        
        self.dem_dirs = dem_dirs
        self.dem_names = dem_names
        self.output_dir = output_dir
        self.landcover_directory = landcover_directory
        self.stream_bounds = stream_bounds
        self.oceans_pq = oceans_pq

        if not bbox:
            self.bbox = (-180.0, -90.0, 180.0, 90.0)
        else:
            self.bbox = bbox
        if mannings_table:
            self.mannings_table = mannings_table
        else:
            self.mannings_table = os.path.join(os.path.dirname(__file__), 'data', 'default_mannings_table.txt')
        self.rps = rps
        self.forecast_date = forecast_date
        if valid_tiles_file:
            self.valid_tiles = pd.read_parquet(valid_tiles_file, columns=['x', 'y']).to_numpy().tolist()
            self.valid_tiles = [tile for tile in self.valid_tiles if
                                (tile[0] + 1 > self.bbox[0] and tile[0] < self.bbox[2] and
                                 tile[1] + 1 > self.bbox[1] and tile[1] < self.bbox[3])]
        else:
            self.valid_tiles = None
        self.buffer_distance = buffer_distance
        self.buffer_dems_as_vrt = buffer_dems_as_vrt
        self.arc_args = arc_args
        self.c2f_bathymetry_args = c2f_bathymetry_args
        self.c2f_floodmap_args = c2f_floodmap_args

        self.overwrite_floodmaps = overwrite_floodmaps
        self.overwrite_burned_dems = overwrite_burned_dems
        self.overwrite_vdts = overwrite_vdts
        self.overwrite_streams = overwrite_streams
        self.overwrite_landuse = overwrite_landuse
        self.overwrite_buffered_dems = overwrite_buffered_dems

    def run_one_dem_type(self, ex: ProcessPoolExecutor, dem_type: str):
        original_dems = glob.glob(os.path.join(self.dem_dirs[self.dem_names.index(dem_type)], '*.tif'), recursive=True)
        og_dems_filtered = get_dem_in_extent(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], original_dems, dem_type)
        buffered_dems = []
        for buffered_dem in start_unthrottled_pbar(ex, 
                                                   buffer_dem,
                                                   f"Buffering DEMs for {dem_type}",
                                                   og_dems_filtered,
                                                   dems=original_dems,
                                                   output_dir=self.output_dir,
                                                   dem_type=dem_type,
                                                   buffer_distance=self.buffer_distance,
                                                   valid_tiles=self.valid_tiles,
                                                   as_vrt=self.buffer_dems_as_vrt,
                                                   overwrite=self.overwrite_buffered_dems):
            if buffered_dem:
                buffered_dems.append(buffered_dem)

        limit = _convert_process_count({'fabdem': 31, 'alos': 20, 'tilezen': 21}.get(dem_type, os.cpu_count()))
        start_throttled_pbar(ex, rasterize_streams, f"Rasterizing streams for {dem_type}", 
                             buffered_dems, limit, dem_type=dem_type, 
                             bounds=self.stream_bounds, overwrite=self.overwrite_streams)

        start_unthrottled_pbar(ex, warp_land_use, f"Warping land use for {dem_type}", 
                               buffered_dems, dem_type=dem_type, 
                               landcover_directory=self.landcover_directory, overwrite=self.overwrite_landuse)

        stream_files = glob.glob(os.path.join(self.output_dir, '*', '*', f'inputs={dem_type}', 'streams.tif'))
        stream_files = filter_files_in_extent_by_lat_lon_dirs(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], stream_files)
        start_unthrottled_pbar(ex, download_flows, f"Downloading flows for {dem_type}", 
                               stream_files, rps=self.rps, 
                               forecast_date=self.forecast_date)
        start_unthrottled_pbar(ex, prepare_water_mask, f"Preparing water masks for {dem_type}", 
                               buffered_dems, dem_type=dem_type)

        start_unthrottled_pbar(ex, prepare_inputs, f"Preparing input files for {dem_type}", 
                               buffered_dems, dem_type=dem_type, mannings_table=self.mannings_table, 
                               rps=self.rps, forecast_date=self.forecast_date, arc_args=self.arc_args, 
                               c2f_bathymetry_args=self.c2f_bathymetry_args, 
                               c2f_floodmap_args=self.c2f_floodmap_args)

        inputs = glob.glob(os.path.join(self.output_dir, '*', '*', f'inputs={dem_type}', 'inputs=arc.txt'))
        inputs = filter_files_in_extent_by_lat_lon_dirs(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], inputs)
        start_unthrottled_pbar(ex, run_arc, f"Running ARC for {dem_type}", inputs, dem_type=dem_type, overwrite=self.overwrite_vdts)

        inputs = glob.glob(os.path.join(self.output_dir, '*', '*', f'inputs={dem_type}', 'inputs=burned.txt'))
        inputs = filter_files_in_extent_by_lat_lon_dirs(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], inputs)
        limit = _convert_process_count({'fabdem': 32, 'alos': 17, 'tilezen': 25}.get(dem_type, os.cpu_count()))
        start_throttled_pbar(ex, run_c2f_bathymetry, f"Preparing burned DEMs for {dem_type}", 
                               inputs, limit, dem_type=dem_type, overwrite=self.overwrite_burned_dems)

        names = []
        if self.rps:
            names.append(f'flows_{",".join(map(str, self.rps))}')
        if self.forecast_date:
            names.append(self.forecast_date)
        inputs = []
        for name in names:
            inputs.extend(glob.glob(os.path.join(self.output_dir, '*', '*', f'inputs={dem_type}', f'inputs={name}.txt'))) 
        inputs = filter_files_in_extent_by_lat_lon_dirs(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], inputs)
        limit = _convert_process_count({'fabdem': 32, 'alos': 23, 'tilezen': 27}.get(dem_type, os.cpu_count()))
        start_throttled_pbar(ex, run_c2f_floodmaps, f"Creating floodmaps for {dem_type}", 
                               inputs, limit, dem_type=dem_type, overwrite=self.overwrite_floodmaps)
        
        floodmap_dirs = glob.glob(os.path.join(self.output_dir, '*', '*', f'floodmaps={dem_type}'))
        floodmap_dirs = filter_files_in_extent_by_lat_lon_dirs(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], floodmap_dirs)
        floodmap_groups = [glob.glob(os.path.join(floodmap_dir, '*.tif')) for floodmap_dir in floodmap_dirs]
        start_unthrottled_pbar(ex, unbuffer_remove, f"Unbuffering floodmaps for {dem_type}", 
                               floodmap_groups, dem_type=dem_type, buffer_distance=self.buffer_distance, oceans_pq=self.oceans_pq)

    def run_all(self):
        run_all_rps_majority = set(self.rps) == {2, 5, 10, 25, 50, 100}
        with ProcessPoolExecutor(os.cpu_count()) as ex, tqdm.tqdm(total=len(self.dem_names)+int(run_all_rps_majority)) as pbar:
            for dem_type in self.dem_names:
                pbar.set_description(f"Processing DEM type: {dem_type}")
                self.run_one_dem_type(ex, dem_type)
                pbar.update(1)

            if run_all_rps_majority:
                fmaps = []
                for dir in glob.glob(os.path.join(self.output_dir, '*', '*', 'floodmaps')):
                    cluster = []
                    for dem_type in self.dem_names:
                        rp_tif = os.path.join(dir, f"dem={dem_type}", 'flows_2,5,10,25,50,100.tif')
                        if  os.path.exists(rp_tif):
                            cluster.append(rp_tif)
                    fmaps.append(cluster)

                start_unthrottled_pbar(ex, majority_vote_all_return_periods, "Majority voting floodmaps across DEM types", fmaps, overwrite=self.overwrite_floodmaps)
                pbar.update(1)
            



