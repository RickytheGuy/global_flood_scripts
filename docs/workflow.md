# Workflow Overview

The current flood-map pipeline is easiest to understand as three stages.

## 1. Prepare streamline parquet files

Run `0_add_covering_bbox.py` once for a streamline dataset when the parquet files
do not yet contain covering-bbox metadata. This makes bbox reads much faster
when `geopandas.read_parquet(..., bbox=...)` is used later in the workflow.

High-level effect:

- Reads every `streams_*.parquet` file.
- Rewrites the file with `write_covering_bbox=True`.
- Improves the clipping performance used by retrospective and forecast runs.

## 2. Create retrospective flood maps

Run `1_create_retrospective.py` to build flood maps from a set of burned DEMs and
GEOGLOWS return-period flows.

High-level steps:

1. Find DEM tiles and matching streamline parquet files.
2. Buffer each DEM so hydraulics are not cut off at tile edges.
3. Clip streamlines to each buffered DEM extent.
4. Export return-period flow CSV files for the stream IDs inside each tile.
5. Run `nencarta.process_watershed(...)`.
6. Crop the resulting flood maps back to the original DEM footprint and mask oceans.

The reusable implementation for this stage now lives in
`global_floodmaps.retrospective`.

## 3. Create forecast flood maps

Run `2_forecasts.py` when you already have burned DEMs and VDT products and want
to convert forecast peak flows into flood maps.

That script:

- Maps river IDs to tiles.
- Pulls forecast peak flows from GEOGLOWS.
- Writes tile-specific flow CSV files.
- Runs Curve2Flood for each DEM type.
- Unbuffers the outputs and computes majority-vote maps.
