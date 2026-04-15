# Retrospective Flood Maps

`1_create_retrospective.py` is now a thin configuration layer around
`global_floodmaps.retrospective`. The package module is the best place to look if
you want to understand, reuse, or extend the retrospective flood-map process.

## What the script does

The script generates retrospective flood maps for every burned DEM tile that has
matching streamlines:

1. Discover input DEMs and streamline parquet files.
2. Build a processing plan per tile.
3. Buffer each DEM into a VRT or GeoTIFF mosaic.
4. Clip streamlines to the buffered DEM footprint.
5. Export GEOGLOWS return-period flows for the stream IDs inside that tile.
6. Run `nencarta.process_watershed(...)`.
7. Crop the buffered flood map back to the original DEM extent and remove ocean cells.

## Files and directories expected by the workflow

The workflow is configured through `RetrospectivePaths`:

```python
from global_floodmaps.retrospective import RetrospectivePaths

paths = RetrospectivePaths(
    dem_glob="/path/to/lon=*/lat=*/burned_dems/dem_burned=fabdem.tif",
    streamlines_glob="/path/to/streamlines/streams_*.parquet",
    output_dir="/path/to/retrospective_outputs",
    buffered_dem_dir="/path/to/buffered_dems",
    oceans_path="/path/to/seas_buffered.parquet",
    land_use_cache_dir="/path/to/esa_landcover",
)
```

Meaning of each path:

- `dem_glob`: burned DEM tiles to process.
- `streamlines_glob`: vector reaches used to identify streams present in each DEM tile.
- `output_dir`: nencarta output root.
- `buffered_dem_dir`: workspace for buffered DEMs and clipped streamline parquet files.
- `oceans_path`: ocean polygons used to zero out marine flood pixels.
- `land_use_cache_dir`: land-cover cache passed through to nencarta.

## Minimal Python API

You can run the retrospective workflow directly from Python:

```python
from global_floodmaps.retrospective import (
    RetrospectiveConfig,
    RetrospectivePaths,
    default_nencarta_kwargs,
    run_retrospective_workflow,
)

paths = RetrospectivePaths(
    dem_glob="/path/to/lon=-111/lat=*/burned_dems/dem_burned=fabdem.tif",
    streamlines_glob="/path/to/streamlines/streams_*.parquet",
    output_dir="/path/to/tests",
    buffered_dem_dir="/path/to/buffered_dems",
    oceans_path="/path/to/seas_buffered.parquet",
    land_use_cache_dir="/path/to/esa_landcover",
)

config = RetrospectiveConfig(
    paths=paths,
    rps=(10, 25, 50, 100),
    buffer_distance=0.05,
    as_vrt=True,
    nencarta_kwargs=default_nencarta_kwargs(
        output_dir=paths.output_dir,
        land_use_cache_dir=paths.land_use_cache_dir,
    ),
)

plans = run_retrospective_workflow(config)
print(f"Processed {len(plans)} tiles")
```

## Step-by-step explanation

### 1. Discover inputs

`discover_retrospective_inputs(...)` expands the DEM and streamline glob
patterns and creates the output folders if needed. The workflow stops early if
either glob matches nothing.

### 2. Build per-tile plans

`collect_retrospective_tile_plans(...)` intersects each DEM with the streamline
parquet bounds using `get_streamlines_in_dem_extent(...)`.

For every DEM tile with at least one matching streamline file, the plan records:

- the source burned DEM,
- the buffered DEM path,
- the clipped streamline parquet path,
- the output return-period CSV path,
- and the tile name used by nencarta output folders.

### 3. Buffer the DEM

`buffer_dem(...)` expands the DEM footprint by `buffer_distance` degrees and
mosaics neighboring DEMs into either:

- a VRT when `as_vrt=True`, or
- a raster when `as_vrt=False`.

This prevents tile-edge artifacts during the hydraulic run.

### 4. Clip streamlines to the buffered DEM

`clip_streamlines_to_dem(...)` reads only the stream geometries that intersect
the buffered DEM extent and writes them into a single tile-specific parquet.

### 5. Export return-period flows

`get_return_period_flows_in_dem_extent(...)` collects all `LINKNO` values from
the clipped streams and writes a CSV of GEOGLOWS return-period flows for the
requested recurrence intervals.

By default the workflow exports:

- 10-year,
- 25-year,
- 50-year,
- 100-year flows.

### 6. Run nencarta

`run_retrospective_workflow(...)` builds a keyword-argument dictionary for each
tile and calls `nencarta.process_watershed(...)`.

The defaults come from `default_nencarta_kwargs(...)`, which captures the
settings that were previously embedded inside the original script.

The most important tile-specific fields are:

- `name`: tile folder name such as `lat=34`.
- `flowline`: clipped stream parquet.
- `dem_dir`: directory containing the buffered DEM.
- `user_flow_files`: tile return-period CSV.
- `dem_filter`: basename of the buffered DEM to use.

### 7. Unbuffer and mask oceans

After nencarta finishes, `unbuffer_and_mask_oceans(...)`:

- resamples the flood map back to the original unbuffered DEM shape,
- restores open-water pixels from the land-use raster,
- masks ocean polygons to zero.

This produces tile-aligned final flood maps instead of buffered edge products.

## Customizing the run

Useful knobs in `RetrospectiveConfig`:

- `rps`: change the return periods written to the flow CSV.
- `buffer_distance`: expand or shrink the DEM buffer.
- `as_vrt`: choose VRTs for lightweight buffering or rasters for materialized outputs.
- `processes`: control multiprocessing pool size.
- `nencarta_kwargs`: override any default nencarta setting.

Example override:

```python
config = RetrospectiveConfig(
    paths=paths,
    nencarta_kwargs=default_nencarta_kwargs(
        output_dir=paths.output_dir,
        land_use_cache_dir=paths.land_use_cache_dir,
    ) | {
        "overwrite_floodmaps": True,
        "quiet": False,
    },
)
```

## Output layout

The retrospective workflow writes:

- buffered DEMs and clipped streams under `buffered_dem_dir`,
- return-period CSV files under `output_dir/<tile_name>/return_period_flows.csv`,
- nencarta flood-map products under `output_dir/<tile_name>/...`.

The exact subfolders inside each tile directory are created by nencarta, but the
workflow expects at least:

- `FloodMap/GEOGLOWS_<tile>..._ARC_Flood_return_period_flows.tif`
- `LAND/<tile>..._LAND_Raster.tif`

## Practical notes

- The workflow assumes DEMs are in geographic coordinates because buffering is
  specified in degrees.
- Stream parquet files work best after running `0_add_covering_bbox.py`.
- The code only processes DEMs that actually intersect at least one streamline file.
- The current script example is configured for a single longitude strip, but the
  API is generic and works for any glob pattern.
