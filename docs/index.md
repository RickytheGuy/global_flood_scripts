# Global Flood Scripts

This repository packages the flood-map workflow you have been prototyping in the
top-level scripts:

- `0_add_covering_bbox.py` adds covering-bbox metadata to streamline parquet files.
- `1_create_retrospective.py` generates retrospective flood maps from burned DEMs.
- `2_forecasts.py` turns forecast peak flows into flood maps using precomputed inputs.

The documentation currently centers on the retrospective path because it is the
clearest end-to-end recipe for generating flood maps from DEM tiles and GEOGLOWS
return-period flows.

Start here:

- [Workflow overview](workflow.md) for the three-script pipeline.
- [Retrospective flood maps](retrospective.md) for the detailed `1_create_retrospective.py` process.
