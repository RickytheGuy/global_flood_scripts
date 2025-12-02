import json
import os
import glob
import logging
import subprocess
import tempfile
from itertools import chain
import multiprocessing as mp
from collections import defaultdict

import s3fs
import pandas as pd
from osgeo import gdal

from .parallel_functions import (
    buffer_dem, run_arc, rasterize_streams, warp_land_use, _get_num_processes,
    start_unthrottled_pbar, download_flows, prepare_water_mask, prepare_inputs, 
    start_throttled_pbar, run_c2f_bathymetry, run_c2f_floodmaps, unbuffer_remove,
    majority_vote, download_tilezen_in_area, download_alos_in_area, 
    download_fabdem_tile, download_alos_tile, download_tilezen_tile, init_s3,
    _init_s3_cache, _dir
)

from .utility_functions import (
    filter_files_in_extent_by_lat_lon_dirs, get_dem_in_extent, generate_bounding_args,
    extract_base_path, opens_right
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
                 s3_dems_dir: str = None,
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
            s3_cache = set(s3.glob(f"{self.s3_dir}/**"))
            if s3_dems_dir:
                if s3_dems_dir.endswith('/'):
                    s3_dems_dir = s3_dems_dir[:-1]
                add = set(s3.glob(f"{s3_dems_dir}/**"))
                s3_cache.update(add)
            s3_cache = [f"/vsis3/{f}{os.linesep}" for f in s3_cache]
            self._s3_temp_cache_file = tempfile.NamedTemporaryFile(delete=False)
            with open(self._s3_temp_cache_file.name, 'w') as f:
                f.writelines(s3_cache)

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

        self.blacklist_file = os.path.join(os.path.dirname(__file__), 'data', '.blacklisted')
        if os.path.exists(self.blacklist_file):
            with open(self.blacklist_file, 'r') as f:
                self.blacklisted_files = set(line.strip() for line in f if line.strip())
        else:
            self.blacklisted_files = set()


    def _run_one_dem_type(self, pool, dem_type: str) -> list[str]:
        if dem_type in self.og_dem_dict:
            original_dems = self.og_dem_dict[dem_type]
        else:
            original_dems = glob.glob(os.path.join(self.dem_dirs[self.dem_names.index(dem_type)], '*.tif'), recursive=True)

        og_dems_filtered = get_dem_in_extent(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], original_dems, dem_type)
        buffered_dems = []
        for buffered_dem in start_unthrottled_pbar(pool, 
                                                   buffer_dem,
                                                   f"Buffering DEMs for {dem_type}",
                                                   og_dems_filtered,
                                                   dems=original_dems,
                                                   output_dir=self.output_dir,
                                                   s3_dir=self.s3_dir,
                                                   dem_type=dem_type,
                                                   buffer_distance=self.buffer_distance,
                                                   valid_tiles=self.valid_tiles,
                                                   as_vrt=self.buffer_dems_as_vrt,
                                                   overwrite=self.overwrite_buffered_dems):
            if buffered_dem:
                buffered_dems.append(buffered_dem)

        limit = _get_num_processes({'fabdem': 8, 'alos': 9, 'tilezen': 4}.get(dem_type, 9))
        stream_files = start_throttled_pbar(pool, rasterize_streams, f"Rasterizing streams for {dem_type} ({limit})", 
                             buffered_dems, limit, dem_type=dem_type, s3_dir=self.s3_dir,
                             bounds=self.stream_bounds, overwrite=self.overwrite_streams)
        stream_files = [f for f in stream_files if f]
        
        limit = _get_num_processes({'fabdem': 3.84, 'alos': 3.91, 'tilezen': 3.57}.get(dem_type, 4))
        start_throttled_pbar(pool, warp_land_use, f"Warping land use for {dem_type} ({limit})", 
                               buffered_dems, limit, dem_type=dem_type, s3_dir=self.s3_dir,
                               landcover_directory=self.landcover_directory, overwrite=self.overwrite_landuse)

        start_throttled_pbar(pool, download_flows, f"Downloading flows for {dem_type} ({limit})", 
                               stream_files, limit, rps=self.rps, s3_dir=self.s3_dir,
                               forecast_date=self.forecast_date)
        
        mask_limit = _get_num_processes({'fabdem': 3.84, 'alos': 5, 'tilezen': 3.57}.get(dem_type, 4))
        start_throttled_pbar(pool, prepare_water_mask, f"Preparing water masks for {dem_type} ({mask_limit})", 
                               buffered_dems, mask_limit, dem_type=dem_type, s3_dir=self.s3_dir)

        outputs = start_throttled_pbar(pool, prepare_inputs, f"Preparing input files for {dem_type} ({limit})", 
                               buffered_dems, limit, dem_type=dem_type, mannings_table=self.mannings_table, 
                               rps=self.rps, forecast_date=self.forecast_date, overwrite_arc=self.overwrite_vdts,
                               overwrite_c2f_bathymetry=self.overwrite_burned_dems,
                               overwrite_c2f_floodmap=self.overwrite_floodmaps, arc_args=self.arc_args, 
                               c2f_bathymetry_args=self.c2f_bathymetry_args, 
                               c2f_floodmap_args=self.c2f_floodmap_args, s3_dir=self.s3_dir)
        arc_inputs = list(chain.from_iterable([i[0] for i in outputs if i[0]]))
        arc_inputs = [i for i in arc_inputs if i not in self.blacklisted_files]
        burned_inputs = list(chain.from_iterable([i[1] for i in outputs if i[1]]))
        floodmap_inputs = list(chain.from_iterable([i[2] for i in outputs if i[2]]))


        limit = _get_num_processes({'fabdem': 1.7, 'alos': 3.5, 'tilezen': 3.5}.get(dem_type, 4))
        files_to_blacklist = start_throttled_pbar(pool, run_arc, f"Running ARC for {dem_type} ({limit})", arc_inputs, limit, 
                             s3_dir=self.s3_dir, dem_type=dem_type, overwrite=self.overwrite_vdts)
        
        if files_to_blacklist:
            with open(self.blacklist_file, 'a') as f:
                for file in files_to_blacklist:
                    if file and file not in self.blacklisted_files:
                        f.write(f"{file}\n")
                        self.blacklisted_files.add(file)
        

        limit = _get_num_processes({'fabdem': 4.85, 'alos': 6.4, 'tilezen': 5.07}.get(dem_type, 6))
        bankfull_floodmaps = start_throttled_pbar(pool, run_c2f_bathymetry, f"Preparing burned DEMs for {dem_type} ({limit})", 
                               burned_inputs, limit, s3_dir=self.s3_dir, dem_type=dem_type, overwrite=self.overwrite_burned_dems)
        bankfull_floodmaps = [f for f in bankfull_floodmaps if f]

        limit = _get_num_processes({'fabdem': 4.76, 'alos': 3, 'tilezen': 4}.get(dem_type, 6))
        floodmaps = start_throttled_pbar(pool, run_c2f_floodmaps, f"Creating floodmaps for {dem_type} ({limit})", 
                               floodmap_inputs, limit, s3_dir=self.s3_dir, dem_type=dem_type, overwrite=self.overwrite_floodmaps)
        floodmaps = [f for f in floodmaps if f]
        floodmaps.extend(bankfull_floodmaps)

        floodmap_dirs = defaultdict(list)
        for floodmap in floodmaps:
            floodmap_dir = os.path.dirname(floodmap)
            floodmap_dirs[floodmap_dir].append(floodmap)
        floodmap_dirs = list(floodmap_dirs.values())

        limit = _get_num_processes({'fabdem': 4.71, 'alos': 4.68, 'tilezen': 3.18}.get(dem_type, 5))
        start_throttled_pbar(pool, unbuffer_remove, f"Unbuffering floodmaps for {dem_type} ({limit})", 
                             floodmap_dirs, limit, dem_type=dem_type, buffer_distance=self.buffer_distance, 
                             oceans_pq=self.oceans_pq)
        
        return floodmaps

    def run_all(self) -> 'FloodManager':
        run_majority_rps = bool(self.rps)
        floodmaps = []
        with mp.Pool(os.cpu_count(), initializer=_init_s3_cache, initargs=(self._s3_temp_cache_file.name,)) as pool:
            for dem_type in self.dem_names:
                floodmaps.extend(self._run_one_dem_type(pool, dem_type))

            if run_majority_rps:
                floodmap_dirs = defaultdict(list)
                for floodmap in floodmaps:
                    floodmap_dir = _dir(floodmap, 2)
                    
                    if os.path.basename(floodmap) == f'flows_{",".join(map(str, self.rps))}.tif':
                        floodmap_dirs[floodmap_dir].append(floodmap)
                floodmap_dirs = list(floodmap_dirs.values())

                start_unthrottled_pbar(pool, majority_vote, "Majority voting floodmaps across DEM types", floodmap_dirs,
                                    s3_dir=self.s3_dir, output_dir=self.output_dir, overwrite=self.overwrite_majority_maps)

            floodmap_dirs = defaultdict(list)
            for floodmap in floodmaps:
                floodmap_dir = _dir(floodmap, 2)
                if os.path.basename(floodmap) == 'bankfull.tif':
                    floodmap_dirs[floodmap_dir].append(floodmap)

            start_unthrottled_pbar(pool, majority_vote, "Majority voting bankfull floodmaps across DEM types", floodmap_dirs,
                                s3_dir=self.s3_dir, output_dir=self.output_dir, overwrite=self.overwrite_majority_maps)

        return self

    def download_tilezen(self, output_dir: str, z_level: int = 12, overwrite: bool = False) -> 'FloodManager':
        minx, miny, maxx, maxy = self.bbox
        os.makedirs(output_dir, exist_ok=True)

        args = generate_bounding_args(minx, miny, maxx, maxy, self.valid_tiles, self.number_of_tiles, self.offset)

        if not args:
            LOG.warning("No Tilezen tiles to download in the specified bounding box.")
            return self

        with mp.Pool(min((os.cpu_count() * 2) - 4, len(args))) as pool:
            dems = start_unthrottled_pbar(pool, download_tilezen_in_area, f"Downloading Tilezen DEMs", args, output_dir=output_dir,
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

        with mp.Pool(min((os.cpu_count() * 2) - 4, len(args))) as pool:
            dems = start_unthrottled_pbar(pool, download_alos_in_area, f"Downloading ALOS DEMs", args, output_dir=output_dir,
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

        with mp.Pool(min(int(os.cpu_count() * 1.5), len(args)), initializer=init_s3, initargs=(self._s3_temp_cache_file.name,)) as pool:
            dems = start_unthrottled_pbar(pool, download_func, f"Downloading {type} DEMs", args, output_dir=output_dir,
                                   download=not no_download, overwrite=overwrite)

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