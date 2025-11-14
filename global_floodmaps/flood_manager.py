import json
import os
import glob
import logging
import subprocess
from itertools import chain
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import s3fs
import tqdm
import pandas as pd
from osgeo import gdal

from .parallel_functions import (
    buffer_dem, run_arc, rasterize_streams, warp_land_use, _get_num_processes,
    start_unthrottled_pbar, download_flows, prepare_water_mask, prepare_inputs, 
    start_throttled_pbar, run_c2f_bathymetry, run_c2f_floodmaps, unbuffer_remove,
    majority_vote, download_tilezen_in_area, download_alos_in_area, 
    download_fabdem_tile, download_alos_tile, download_tilezen_tile
)

from .utility_functions import (
    filter_files_in_extent_by_lat_lon_dirs, get_dem_in_extent, generate_bounding_args,
    extract_base_path
)

from ._constants import DEFAULT_TILES_FILE, STREAM_BOUNDS_FILE
from .logger import LOG, add_file_handler


gdal.UseExceptions()

class FloodManager:
    def __init__(self,
                 dem_dirs: list[str],
                 dem_names: list[str],
                 output_dir: str,
                 landcover_directory: str,
                 streamlines_directory: str,
                 oceans_pq: str,
                 s3_dir: str = None,
                 bbox: tuple[float, float, float, float] = None,
                 number_of_tiles: int = None,
                 offset: int = 0,
                 mannings_table: str = None,
                 rps: list[int] = None,
                 forecast_date: str = None,
                 valid_tiles_file: str = DEFAULT_TILES_FILE,
                 buffer_distance: float = 0.1,
                 buffer_dems_as_vrt: bool = True, 
                 arc_args: dict = {},
                 c2f_bathymetry_args: dict = {},
                 c2f_floodmap_args: dict = {},
                 overwrite_majority_maps: bool = False,
                 overwrite_floodmaps: bool = False,
                 overwrite_burned_dems: bool = False,
                 overwrite_vdts: bool = False,
                 overwrite_streams: bool = False,
                 overwrite_landuse: bool = False,
                 overwrite_buffered_dems: bool = False,):
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
        assert len(self.dem_dirs) == len(self.dem_names), "dem_dirs and dem_names must have the same length."
        for dem_dir, dem_name in zip(self.dem_dirs, self.dem_names):
            if dem_name.lower() not in dem_dir.lower():
                LOG.warning(f"DEM name '{dem_name}' does not appear to match directory '{dem_dir}'.")

        self.og_dem_dict = {}
        self.output_dir = output_dir
        self.landcover_directory = landcover_directory
        self.streamlines_directory = streamlines_directory
        with open(STREAM_BOUNDS_FILE, 'r') as f:
            self.stream_bounds = json.load(f)
        self.stream_bounds = {os.path.join(self.streamlines_directory, key): value for key, value in self.stream_bounds.items()}
        self.oceans_pq = oceans_pq
        self.s3_dir = s3_dir
        self.s3_cache = set()
        if self.s3_dir:
            if self.s3_dir.endswith('/'):
                self.s3_dir = self.s3_dir[:-1]
            s3 = s3fs.S3FileSystem(anon=True)
            LOG.info(f"Building S3 cache for {self.s3_dir}...")
            self.s3_cache = set(s3.glob(f"{self.s3_dir}/**"))
            self.s3_cache = [f"/vsis3/{f}" for f in self.s3_cache]

        if not bbox:
            self.bbox = (-180, -90, 180, 90)
        else:
            self.bbox = bbox

        self.number_of_tiles = number_of_tiles 
        self.offset = offset

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

        self.overwrite_majority_maps = overwrite_majority_maps
        self.overwrite_floodmaps = overwrite_floodmaps
        self.overwrite_burned_dems = overwrite_burned_dems
        self.overwrite_vdts = overwrite_vdts
        self.overwrite_streams = overwrite_streams
        self.overwrite_landuse = overwrite_landuse
        self.overwrite_buffered_dems = overwrite_buffered_dems

    def _run_one_dem_type(self, ex: ProcessPoolExecutor, dem_type: str):
        if dem_type in self.og_dem_dict:
            original_dems = self.og_dem_dict[dem_type]
        else:
            original_dems = glob.glob(os.path.join(self.dem_dirs[self.dem_names.index(dem_type)], '*.tif'), recursive=True)

        og_dems_filtered = get_dem_in_extent(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], original_dems, dem_type)
        buffered_dems = []
        for buffered_dem in start_unthrottled_pbar(ex, 
                                                   buffer_dem,
                                                   f"Buffering DEMs for {dem_type}",
                                                   og_dems_filtered,
                                                   dems=original_dems,
                                                   output_dir=self.output_dir,
                                                   s3_dir=self.s3_dir,
                                                   s3_cache=self.s3_cache,
                                                   dem_type=dem_type,
                                                   buffer_distance=self.buffer_distance,
                                                   valid_tiles=self.valid_tiles,
                                                   as_vrt=self.buffer_dems_as_vrt,
                                                   overwrite=self.overwrite_buffered_dems):
            if buffered_dem:
                buffered_dems.append(buffered_dem)

        limit = _get_num_processes({'fabdem': 8, 'alos': 9, 'tilezen': 8}.get(dem_type, 9))
        stream_files = start_throttled_pbar(ex, rasterize_streams, f"Rasterizing streams for {dem_type} ({limit})", 
                             buffered_dems, limit, dem_type=dem_type, s3_dir=self.s3_dir, s3_cache=self.s3_cache,
                             bounds=self.stream_bounds, overwrite=self.overwrite_streams)
        stream_files = [f for f in stream_files if f]
        
        limit = _get_num_processes({'fabdem': 3.84, 'alos': 3.91, 'tilezen': 3.57}.get(dem_type, 4))
        start_throttled_pbar(ex, warp_land_use, f"Warping land use for {dem_type} ({limit})", 
                               buffered_dems, limit, dem_type=dem_type, s3_dir=self.s3_dir, s3_cache=self.s3_cache,
                               landcover_directory=self.landcover_directory, overwrite=self.overwrite_landuse)

        start_throttled_pbar(ex, download_flows, f"Downloading flows for {dem_type} ({limit})", 
                               stream_files, limit, rps=self.rps, s3_dir=self.s3_dir, s3_cache=self.s3_cache,
                               forecast_date=self.forecast_date)
        
        mask_limit = _get_num_processes({'fabdem': 3.84, 'alos': 5, 'tilezen': 3.57}.get(dem_type, 4))
        start_throttled_pbar(ex, prepare_water_mask, f"Preparing water masks for {dem_type} ({mask_limit})", 
                               buffered_dems, mask_limit, dem_type=dem_type, s3_dir=self.s3_dir, s3_cache=self.s3_cache)

        outputs = start_throttled_pbar(ex, prepare_inputs, f"Preparing input files for {dem_type} ({limit})", 
                               buffered_dems, limit, dem_type=dem_type, mannings_table=self.mannings_table, 
                               rps=self.rps, forecast_date=self.forecast_date, overwrite_arc=self.overwrite_vdts,
                               overwrite_c2f_bathymetry=self.overwrite_burned_dems,
                               overwrite_c2f_floodmap=self.overwrite_floodmaps, arc_args=self.arc_args, 
                               c2f_bathymetry_args=self.c2f_bathymetry_args, 
                               c2f_floodmap_args=self.c2f_floodmap_args,
                               s3_dir=self.s3_dir, s3_cache=self.s3_cache)
        arc_inputs = list(chain.from_iterable([i[0] for i in outputs if i[0]]))
        burned_inputs = list(chain.from_iterable([i[1] for i in outputs if i[1]]))
        floodmap_inputs = list(chain.from_iterable([i[2] for i in outputs if i[2]]))

        limit = _get_num_processes({'fabdem': 1.7, 'alos': 3.5, 'tilezen': 3.5}.get(dem_type, 4))
        start_throttled_pbar(ex, run_arc, f"Running ARC for {dem_type} ({limit})", arc_inputs, limit, 
                             s3_dir=self.s3_dir, s3_cache=self.s3_cache, dem_type=dem_type, overwrite=self.overwrite_vdts)

        limit = _get_num_processes({'fabdem': 4.85, 'alos': 6.2, 'tilezen': 5.07}.get(dem_type, 6))
        start_throttled_pbar(ex, run_c2f_bathymetry, f"Preparing burned DEMs for {dem_type} ({limit})", 
                               burned_inputs, limit, s3_dir=self.s3_dir, s3_cache=self.s3_cache, dem_type=dem_type, overwrite=self.overwrite_burned_dems)

        limit = _get_num_processes({'fabdem': 4.76, 'alos': 6, 'tilezen': 4}.get(dem_type, 6))
        floodmaps = start_throttled_pbar(ex, run_c2f_floodmaps, f"Creating floodmaps for {dem_type} ({limit})", 
                               floodmap_inputs, limit, s3_dir=self.s3_dir, s3_cache=self.s3_cache, dem_type=dem_type, overwrite=self.overwrite_floodmaps)
        
        floodmap_dirs = defaultdict(list)
        for floodmap in floodmaps:
            floodmap_dir = os.path.dirname(floodmap)
            floodmap_dirs[floodmap_dir].append(floodmap)
        floodmap_dirs = list(floodmap_dirs.values())

        limit = _get_num_processes({'fabdem': 4.71, 'alos': 4.68, 'tilezen': 3.18}.get(dem_type, 5))
        start_throttled_pbar(ex, unbuffer_remove, f"Unbuffering floodmaps for {dem_type} ({limit})", 
                             floodmap_dirs, limit, dem_type=dem_type, buffer_distance=self.buffer_distance, 
                             oceans_pq=self.oceans_pq)
        
        return floodmaps

    def run_all(self) -> 'FloodManager':
        run_majority_rps = bool(self.rps)
        with ProcessPoolExecutor(os.cpu_count()) as ex, tqdm.tqdm(total=len(self.dem_names)+int(run_majority_rps)+1) as pbar:
            floodmaps = []
            for dem_type in self.dem_names:
                pbar.set_description(f"Processing DEM type: {dem_type}")
                floodmaps.extend(self._run_one_dem_type(ex, dem_type))
                pbar.update(1)

            if run_majority_rps:
                floodmap_dirs = defaultdict(list)
                for floodmap in floodmaps:
                    floodmap_dir = os.path.dirname(floodmap)
                    if os.path.basename(floodmap) == f'flows_{",".join(map(str, self.rps))}.tif':
                        floodmap_dirs[floodmap_dir].append(floodmap)
                floodmap_dirs = list(floodmap_dirs.values())

                start_unthrottled_pbar(ex, majority_vote, "Majority voting floodmaps across DEM types", floodmap_dirs,
                                       s3_cache=self.s3_cache, s3_dir=self.s3_dir, overwrite=self.overwrite_majority_maps)
                pbar.update(1)

            floodmap_dirs = defaultdict(list)
            for floodmap in floodmaps:
                floodmap_dir = os.path.dirname(floodmap)
                if os.path.basename(floodmap) == 'bankfull.tif':
                    floodmap_dirs[floodmap_dir].append(floodmap)
            start_unthrottled_pbar(ex, majority_vote, "Majority voting bankfull floodmaps across DEM types", floodmap_dirs, overwrite=self.overwrite_majority_maps)
            pbar.update(1)

        return self

    def download_tilezen(self, output_dir: str, z_level: int = 12, overwrite: bool = False) -> 'FloodManager':
        minx, miny, maxx, maxy = self.bbox
        os.makedirs(output_dir, exist_ok=True)

        args = generate_bounding_args(minx, miny, maxx, maxy, self.valid_tiles, self.number_of_tiles, self.offset)

        if not args:
            LOG.warning("No Tilezen tiles to download in the specified bounding box.")
            return self

        with ProcessPoolExecutor(min((os.cpu_count() * 2) - 4, len(args))) as ex:
            dems = start_unthrottled_pbar(ex, download_tilezen_in_area, f"Downloading Tilezen DEMs", args, output_dir=output_dir,
                                   overwrite=overwrite, z=z_level)
        self.og_dem_dict['tilezen'] = dems

        return self
    
    def download_alos(self, output_dir: str, overwrite: bool = False) -> 'FloodManager':
        minx, miny, maxx, maxy = self.bbox
        os.makedirs(output_dir, exist_ok=True)

        args = generate_bounding_args(minx, miny, maxx, maxy, self.valid_tiles, self.number_of_tiles, self.offset)

        if not args:
            LOG.warning("No ALOS tiles to download in the specified bounding box.")
            return self
        
        home = os.path.expanduser("~")
        netrc_path = os.path.join(home, '.netrc')
        if not os.path.exists(netrc_path):
            raise FileNotFoundError(f".netrc file not found in {home}. Please create one with your ASF credentials. It should look like:\n\nmachine urs.earthdata.nasa.gov\nlogin YOUR_USERNAME\npassword YOUR_PASSWORD\n")

        with ProcessPoolExecutor(min((os.cpu_count() * 2) - 4, len(args))) as ex:
            dems = start_unthrottled_pbar(ex, download_alos_in_area, f"Downloading ALOS DEMs", args, output_dir=output_dir,
                                   overwrite=overwrite)

        self.og_dem_dict['alos'] = [d for d in dems if d]

        return self

    def download_from_s3(self, output_dir: str, type: str, overwrite: bool = False, no_download: bool = False) -> 'FloodManager':
        minx, miny, maxx, maxy = self.bbox
        os.makedirs(output_dir, exist_ok=True)

        args = generate_bounding_args(minx, miny, maxx, maxy, self.valid_tiles, self.number_of_tiles, self.offset)

        if not args:
            LOG.warning("No FABDEM tiles to download in the specified bounding box.")
            return self

        assert type in ['fabdem', 'alos', 'tilezen'], f"DEM type {type} not recognized."
        download_func = {
            'fabdem': download_fabdem_tile,
            'alos': download_alos_tile,
            'tilezen': download_tilezen_tile
        }[type]

        with ProcessPoolExecutor(min((os.cpu_count() * 2) - 4, len(args))) as ex:
            dems = start_unthrottled_pbar(ex, download_func, f"Downloading {type} DEMs", args, output_dir=output_dir,
                                          download=not no_download, overwrite=overwrite, s3_cache=self.s3_cache)

        self.og_dem_dict[type] = [d for d in dems if d]

        return self

    def define_memory_usage(self, dem_type: str) -> 'FloodManager':
        """Estimate and define memory usage for processing based on DEM types and system resources."""
        from memory_profiler import memory_usage

        assert dem_type in self.dem_names, f"DEM type {dem_type} not recognized."

        original_dems = glob.glob(os.path.join(self.dem_dirs[self.dem_names.index(dem_type)], '*.tif'), recursive=True)
        og_dems_filtered = get_dem_in_extent(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], original_dems, dem_type)
        og_dems_filtered = og_dems_filtered[-1]  # Test with a single DEM to estimate memory

        mem_usage, buffered_dem = memory_usage((buffer_dem, (og_dems_filtered,),
                                  {'dems': original_dems,
                                   'output_dir': self.output_dir,
                                   'dem_type': dem_type,
                                   'buffer_distance': self.buffer_distance,
                                   'valid_tiles': self.valid_tiles,
                                   'as_vrt': self.buffer_dems_as_vrt,
                                   'overwrite': self.overwrite_buffered_dems}),
                                 max_usage=True, retval=True)
        LOG.info(f"Estimated memory usage for buffering DEMs of type {dem_type}: {(mem_usage/1024):.2f} GB")

        mem_usage, _ = memory_usage((rasterize_streams, (buffered_dem,),
                                  {'dem_type': dem_type,
                                   'bounds': self.stream_bounds,
                                   'overwrite': self.overwrite_streams}),
                                 max_usage=True, retval=True)
        LOG.info(f"Estimated memory usage for rasterizing streams of type {dem_type}: {(mem_usage/1024):.2f} GB")
        
        mem_usage, _ = memory_usage((warp_land_use, (buffered_dem,),
                                  {'dem_type': dem_type,
                                   'landcover_directory': self.landcover_directory,
                                   'overwrite': self.overwrite_landuse}),
                                 max_usage=True, retval=True)
        LOG.info(f"Estimated memory usage for warping land use of type {dem_type}: {(mem_usage/1024):.2f} GB")

        mem_usage, _ = memory_usage((prepare_water_mask, (buffered_dem,),
                                  {'dem_type': dem_type}),
                                 max_usage=True, retval=True)
        LOG.info(f"Estimated memory usage for preparing water mask of type {dem_type}: {(mem_usage/1024):.2f} GB")

        mem_usage, _ = memory_usage((prepare_inputs, (buffered_dem,),
                                  {'dem_type': dem_type,
                                   'mannings_table': self.mannings_table,
                                   'rps': self.rps,
                                   'forecast_date': self.forecast_date,
                                   'overwrite_arc': self.overwrite_vdts,
                                   'overwrite_c2f_bathymetry': self.overwrite_burned_dems,
                                   'overwrite_c2f_floodmap': self.overwrite_floodmaps,
                                   'arc_args': self.arc_args,
                                   'c2f_bathymetry_args': self.c2f_bathymetry_args,
                                   'c2f_floodmap_args': self.c2f_floodmap_args}),
                                 max_usage=True, retval=True)
        LOG.info(f"Estimated memory usage for preparing inputs of type {dem_type}: {(mem_usage/1024):.2f} GB")

        stream_files = glob.glob(os.path.join(self.output_dir, '*', '*', f'inputs={dem_type}', 'streams.tif'))
        stream_files = filter_files_in_extent_by_lat_lon_dirs(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], stream_files)
        stream_file = stream_files[0]  # Test with a single stream file to estimate memory
        mem_usage, _ = memory_usage((download_flows, (stream_file,),
                                  {'rps': self.rps,
                                   'forecast_date': self.forecast_date}),
                                 max_usage=True, retval=True)
        LOG.info(f"Estimated memory usage for downloading flows of type {dem_type}: {(mem_usage/1024):.2f} GB")

        inputs = glob.glob(os.path.join(self.output_dir, '*', '*', f'inputs={dem_type}', 'inputs=arc.txt'))
        inputs = filter_files_in_extent_by_lat_lon_dirs(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], inputs)
        inputs = inputs[0]  # Test with a single input to estimate memory
        mem_usage, _ = memory_usage((run_arc, (inputs,),
                                  {'dem_type': dem_type, 'overwrite': self.overwrite_vdts}),
                                 max_usage=True, retval=True)
        LOG.info(f"Estimated memory usage for running ARC of type {dem_type}: {(mem_usage/1024):.2f} GB")
        inputs = glob.glob(os.path.join(self.output_dir, '*', '*', f'inputs={dem_type}', 'inputs=burned.txt'))
        inputs = filter_files_in_extent_by_lat_lon_dirs(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], inputs)
        inputs = inputs[0]  # Test with a single input to estimate memory
        mem_usage, _ = memory_usage((run_c2f_bathymetry, (inputs,),
                                  {'dem_type': dem_type, 'overwrite': self.overwrite_burned_dems}),
                                 max_usage=True, retval=True)
        LOG.info(f"Estimated memory usage for preparing burned DEMs of type {dem_type}: {(mem_usage/1024):.2f} GB")

        names = []
        if self.rps:
            names.append(f'flows_{",".join(map(str, self.rps))}')
        if self.forecast_date:
            names.append(self.forecast_date)
        inputs = []
        for name in names:
            inputs.extend(glob.glob(os.path.join(self.output_dir, '*', '*', f'inputs={dem_type}', f'inputs={name}.txt'))) 
        inputs = filter_files_in_extent_by_lat_lon_dirs(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], inputs)
        inputs = inputs[0]  # Test with a single input to estimate memory
        mem_usage, _ = memory_usage((run_c2f_floodmaps, (inputs,),
                                  {'dem_type': dem_type, 'overwrite': self.overwrite_floodmaps}),
                                 max_usage=True, retval=True)
        LOG.info(f"Estimated memory usage for creating floodmaps of type {dem_type}: {(mem_usage/1024):.2f} GB")

        floodmap_dirs = glob.glob(os.path.join(self.output_dir, '*', '*', f'floodmaps', f'dem={dem_type}'))
        floodmap_dirs = filter_files_in_extent_by_lat_lon_dirs(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], floodmap_dirs)
        floodmap_groups = [glob.glob(os.path.join(floodmap_dir, '*.tif')) for floodmap_dir in floodmap_dirs]
        mem_usage, _ = memory_usage((unbuffer_remove, (floodmap_groups[0],),
                                  {'dem_type': dem_type,
                                   'buffer_distance': self.buffer_distance,
                                   'oceans_pq': self.oceans_pq}),
                                 max_usage=True, retval=True)
        LOG.info(f"Estimated memory usage for unbuffering floodmaps of type {dem_type}: {(mem_usage/1024):.2f} GB")
        return self
    
    def download_landcover(self, overwrite: bool = False) -> 'FloodManager':
        command = ["s5cmd", "--no-sign-request", "cp" if overwrite else "sync"]
        if not overwrite:
            command.append("--size-only")
        command.extend(["s3://global-floodmaps/landcover/*", self.landcover_directory])

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            LOG.error("Error downloading landcover data:")
            LOG.error(result.stderr)

        return self

    def download_streamlines(self, overwrite: bool = False) -> 'FloodManager':
        command = ["s5cmd", "--no-sign-request", "cp" if overwrite else "sync"]
        if not overwrite:
            command.append("--size-only")
        command.extend(["s3://global-floodmaps/streamlines/*", self.streamlines_directory])

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            LOG.error("Error downloading streamlines data:")
            LOG.error(result.stderr) 

        return self
    
    def download_oceans_pq(self, output_path: str, overwrite: bool = False) -> 'FloodManager':
        if not overwrite and os.path.exists(output_path):
            return self 
        
        result = subprocess.run([
            "s5cmd", "--no-sign-request", "cp",
            "s3://global-floodmaps/configs/seas_buffered.parquet",
            output_path
        ], capture_output=True, text=True)

        if result.returncode != 0:
            LOG.error("Error downloading oceans parquet data:")
            LOG.error(result.stderr) 

        return self
    
    def add_logging_file_handler(self, log_file: str, level: int = logging.WARNING) -> 'FloodManager':
        add_file_handler(LOG, log_file, level)
        return self
    
    def set_log_level(self, level: int) -> 'FloodManager':
        for handler in LOG.handlers:
            handler.setLevel(level)
        return self