"""Helpers for generating retrospective flood maps from buffered DEM tiles.

This module turns the one-off logic from ``1_create_retrospective.py`` into a
small API that is easier to document, test, and reuse.
"""

from __future__ import annotations

import glob
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from nencarta import process_watershed

from .parallel_functions import bar_map, bar_starmap
from .utility_functions import (
    _dir,
    buffer_dem,
    clip_streamlines_to_dem,
    get_return_period_flows_in_dem_extent,
    get_streamlines_in_dem_extent,
    unbuffer_and_mask_oceans,
)


@dataclass(frozen=True)
class RetrospectivePaths:
    """Filesystem locations required to run the retrospective workflow.

    Attributes
    ----------
    dem_glob
        Glob pattern pointing at the burned DEM tiles to process.
    streamlines_glob
        Glob pattern for the source stream parquet files.
    output_dir
        Root folder where nencarta writes retrospective products.
    buffered_dem_dir
        Workspace for buffered DEMs and clipped streamline parquet files.
    oceans_path
        GeoParquet used to remove ocean cells from final flood maps.
    land_use_cache_dir
        Land-cover cache directory passed through to nencarta.
    """

    dem_glob: str
    streamlines_glob: str
    output_dir: str
    buffered_dem_dir: str
    oceans_path: str
    land_use_cache_dir: str


@dataclass(frozen=True)
class RetrospectiveTilePlan:
    """All file paths needed to process a single DEM tile retrospectively."""

    tile_name: str
    source_dem: str
    streamline_paths: tuple[str, ...]
    buffered_dem: str
    clipped_streamlines: str
    flow_file: str


@dataclass
class RetrospectiveConfig:
    """Runtime options for the retrospective flood-map workflow.

    The defaults mirror the original `1_create_retrospective.py` behavior:
    10, 25, 50, and 100 year return periods; a 0.05 degree DEM buffer; and VRT
    outputs for buffered DEM mosaics.
    """

    paths: RetrospectivePaths
    rps: tuple[int, ...] = (10, 25, 50, 100)
    buffer_distance: float = 0.05
    as_vrt: bool = True
    processes: int | None = None
    nencarta_kwargs: dict[str, Any] = field(default_factory=dict)


def default_nencarta_kwargs(output_dir: str, land_use_cache_dir: str) -> dict[str, Any]:
    """Return the nencarta settings used by the original retrospective script."""

    return {
        "output_dir": output_dir,
        "bathy_use_banks": False,
        "flood_waterlc_and_strm_cells": False,
        "land_watervalue": 80,
        "clean_dem": False,
        "mapper": "Curve2Flood",
        "process_stream_network": True,
        "use_specified_depth_for_bathy_mask": False,
        "find_banks_based_on_landcover": True,
        "specify_depths_for_bathy_mask": False,
        "create_reach_average_curve_file": False,
        "use_warning_flags_to_download_dem": False,
        "specified_bathyflow_field": "p_exceed_50",
        "specified_highflow_field": "rp100_premium",
        "streamflow_source": "GEOGLOWS",
        "overwrite_floodmaps": False,
        "make_fist_inputs": False,
        "floodmap_mode": "user",
        "make_curvefile": False,
        "make_ap_database": False,
        "vdt_file_extension": "parquet",
        "make_depth_maps": False,
        "make_velocity_maps": False,
        "make_wse_maps": False,
        "floodmap_identifier": "",
        "bathy_args": {
            "X_Section_Dist": 5000,
            "Degree_Manip": 6.1,
            "Degree_Interval": 1.5,
            "Low_Spot_Range": 2,
            "Str_Limit_Val": None,
            "Gen_Dir_Dist": 10,
            "Gen_Slope_Dist": 10,
            "Stream_Slope_Method": "local_average_corrected",
        },
        "move_stream_network_to_new_locations": False,
        "quiet": True,
        "land_use_cache_dir": land_use_cache_dir,
    }


def discover_retrospective_inputs(paths: RetrospectivePaths) -> tuple[list[str], list[str]]:
    """Resolve and validate the DEM and streamline inputs for a run."""

    dems = sorted(glob.glob(paths.dem_glob))
    streamlines = sorted(glob.glob(paths.streamlines_glob))

    if not dems:
        raise FileNotFoundError(f"No DEMs matched {paths.dem_glob!r}")
    if not streamlines:
        raise FileNotFoundError(f"No streamlines matched {paths.streamlines_glob!r}")

    os.makedirs(paths.output_dir, exist_ok=True)
    os.makedirs(paths.buffered_dem_dir, exist_ok=True)
    return dems, streamlines


def _tile_name_from_dem(dem: str) -> str:
    """Return the ``lat=...`` tile identifier used in the output folder layout."""

    return os.path.basename(_dir(dem, 2))


def _buffered_dem_name(dem: str, tile_name: str, as_vrt: bool) -> str:
    stem = os.path.splitext(os.path.basename(dem))[0]
    suffix = ".vrt" if as_vrt else ".tif"
    return f"{tile_name}_{stem}_buffered{suffix}"


def collect_retrospective_tile_plans(
    dems: list[str],
    streamlines: list[str],
    paths: RetrospectivePaths,
    *,
    as_vrt: bool = True,
    processes: int | None = None,
) -> list[RetrospectiveTilePlan]:
    """Build per-tile processing plans for DEMs that intersect streamlines.

    Only DEMs with at least one overlapping stream parquet are kept. Each plan
    contains the source DEM path plus the derived output paths used later in the
    workflow.
    """

    with mp.Pool(processes=processes) as pool:
        streamlines_for_each_dem = bar_map(
            pool,
            partial(get_streamlines_in_dem_extent, streamlines=streamlines),
            dems,
            total=len(dems),
            desc="Finding streamlines in DEM extents",
        )

    plans: list[RetrospectiveTilePlan] = []
    for dem, matched_streamlines in zip(dems, streamlines_for_each_dem):
        if not matched_streamlines:
            continue

        tile_name = _tile_name_from_dem(dem)
        stem = os.path.splitext(os.path.basename(dem))[0]
        buffered_dem = os.path.join(
            paths.buffered_dem_dir,
            _buffered_dem_name(dem, tile_name, as_vrt),
        )
        clipped_streamlines = os.path.join(
            paths.buffered_dem_dir,
            f"{tile_name}_{stem}_streamlines.parquet",
        )
        flow_file = os.path.join(paths.output_dir, tile_name, "return_period_flows.csv")
        os.makedirs(os.path.dirname(flow_file), exist_ok=True)

        plans.append(
            RetrospectiveTilePlan(
                tile_name=tile_name,
                source_dem=dem,
                streamline_paths=tuple(matched_streamlines),
                buffered_dem=buffered_dem,
                clipped_streamlines=clipped_streamlines,
                flow_file=flow_file,
            )
        )

    return plans


def _build_nencarta_arguments(
    plans: list[RetrospectiveTilePlan],
    nencarta_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert tile plans into ``process_watershed`` keyword dictionaries."""

    return [
        nencarta_kwargs
        | {
            "name": plan.tile_name,
            "flowline": plan.clipped_streamlines,
            "dem_dir": os.path.dirname(plan.buffered_dem),
            "user_flow_files": plan.flow_file,
            "dem_filter": os.path.basename(plan.buffered_dem),
        }
        for plan in plans
    ]


def _build_unbuffer_arguments(
    plans: list[RetrospectiveTilePlan],
    output_dir: str,
    oceans_path: str,
) -> list[tuple[str, str, str, str]]:
    """Find finished flood maps and pair them with the files needed to crop them."""

    unbuffer_args: list[tuple[str, str, str, str]] = []
    for plan in plans:
        floodmaps = glob.glob(
            os.path.join(
                output_dir,
                plan.tile_name,
                "FloodMap",
                f"GEOGLOWS_{plan.tile_name}*_ARC_Flood_return_period_flows.tif",
            )
        )
        if not floodmaps:
            continue

        land_use_files = glob.glob(
            os.path.join(
                output_dir,
                plan.tile_name,
                "LAND",
                f"{plan.tile_name}*_LAND_Raster.tif",
            )
        )
        if not land_use_files:
            continue

        unbuffer_args.append(
            (plan.source_dem, floodmaps[0], land_use_files[0], oceans_path)
        )

    return unbuffer_args


def run_retrospective_workflow(config: RetrospectiveConfig) -> list[RetrospectiveTilePlan]:
    """Run the full retrospective flood-map workflow.

    Returns
    -------
    list[RetrospectiveTilePlan]
        The tile plans that were processed. This makes it easy to inspect which
        DEMs, clipped stream files, and return-period CSV files were used.
    """

    dems, streamlines = discover_retrospective_inputs(config.paths)
    plans = collect_retrospective_tile_plans(
        dems,
        streamlines,
        config.paths,
        as_vrt=config.as_vrt,
        processes=config.processes,
    )
    if not plans:
        return []

    nencarta_kwargs = default_nencarta_kwargs(
        output_dir=config.paths.output_dir,
        land_use_cache_dir=config.paths.land_use_cache_dir,
    )
    nencarta_kwargs.update(config.nencarta_kwargs)

    buffer_args = [
        (plan.source_dem, plan.buffered_dem)
        for plan in plans
        if not os.path.isfile(plan.buffered_dem)
    ]
    clip_args = [
        (
            plan.buffered_dem,
            list(plan.streamline_paths),
            plan.clipped_streamlines,
        )
        for plan in plans
    ]
    flow_args = [
        (
            plan.buffered_dem,
            plan.clipped_streamlines,
            list(config.rps),
            plan.flow_file,
        )
        for plan in plans
    ]
    nencarta_args = _build_nencarta_arguments(plans, nencarta_kwargs)

    with mp.Pool(processes=config.processes) as pool:
        if buffer_args:
            bar_starmap(
                pool,
                partial(
                    buffer_dem,
                    all_dems=dems,
                    as_vrt=config.as_vrt,
                    buffer_distance=config.buffer_distance,
                ),
                buffer_args,
                total=len(buffer_args),
                desc="Buffering DEMs",
            )

        bar_starmap(
            pool,
            clip_streamlines_to_dem,
            clip_args,
            total=len(clip_args),
            desc="Clipping streamlines to buffered DEM extents",
        )
        bar_starmap(
            pool,
            get_return_period_flows_in_dem_extent,
            flow_args,
            total=len(flow_args),
            desc="Calculating return period flows",
        )
        bar_map(
            pool,
            process_watershed,
            nencarta_args,
            total=len(nencarta_args),
            desc="Processing watersheds",
        )

        unbuffer_args = _build_unbuffer_arguments(
            plans,
            output_dir=config.paths.output_dir,
            oceans_path=config.paths.oceans_path,
        )
        if unbuffer_args:
            bar_starmap(
                pool,
                unbuffer_and_mask_oceans,
                unbuffer_args,
                total=len(unbuffer_args),
                desc="Unbuffering DEMs and masking oceans",
            )

    return plans
