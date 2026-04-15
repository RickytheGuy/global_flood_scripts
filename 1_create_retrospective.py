"""Example entrypoint for retrospective flood-map generation.

Update the filesystem paths below to match your environment, then run:

```bash
python 1_create_retrospective.py
```

The actual workflow lives in :mod:`global_floodmaps.retrospective`.
"""

from __future__ import annotations

import os

from nencarta import set_log_level

from global_floodmaps.retrospective import (
    RetrospectiveConfig,
    RetrospectivePaths,
    default_nencarta_kwargs,
    run_retrospective_workflow,
)

os.environ["KMP_WARNINGS"] = "0"
set_log_level("ERROR")


if __name__ == "__main__":
    paths = RetrospectivePaths(
        dem_glob="/Users/Shared/flood_map_tiles/lon=-111/lat=*/burned_dems/dem_burned=fabdem.tif",
        streamlines_glob="/Users/Shared/streamlines/streams_*.parquet",
        output_dir="/Users/rickyrosas/tests",
        buffered_dem_dir="/Users/rickyrosas/buffered_dems",
        oceans_path="/Users/rickyrosas/seas_buffered.parquet",
        land_use_cache_dir="/Users/Shared/esa_landcover",
    )

    config = RetrospectiveConfig(
        paths=paths,
        nencarta_kwargs=default_nencarta_kwargs(
            output_dir=paths.output_dir,
            land_use_cache_dir=paths.land_use_cache_dir,
        ),
    )

    run_retrospective_workflow(config)
