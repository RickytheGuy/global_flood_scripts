from __future__ import annotations

import os
import json
import pickle
import threading
from pathlib import Path
from typing import Any

import lmdb

import pyogrio
import geopandas as gpd
from shapely.geometry import box


_CACHE_LOCK = threading.RLock()

_CACHE_DIR = Path.home() / ".vector_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

LMDB_PATH = _CACHE_DIR / "vector_metadata.lmdb"
_LMDB_ENV = lmdb.open(
    str(LMDB_PATH),
    map_size=1024**3,  # 1 GB
    subdir=False,
    lock=True,
    sync=False,
    metasync=False,
    readahead=False,
    writemap=True,
    max_readers=512,
)

def _cache_key(filepath: str) -> bytes:
    return os.path.abspath(filepath).encode()

def _get_file_timestamp(filepath: str) -> float:
    return os.path.getmtime(filepath)

def _load_cached_metadata(
    filepath: str,
    timestamp: float,
) -> dict[str, Any] | None:

    key = _cache_key(filepath)

    with _LMDB_ENV.begin(write=False) as txn:
        value = txn.get(key)

    if value is None:
        return None

    try:
        entry = pickle.loads(value)
    except Exception:
        return None

    if entry["timestamp"] != timestamp:
        return None

    return entry["metadata"]

def _save_cached_metadata(
    filepath: str,
    timestamp: float,
    metadata: dict[str, Any],
) -> None:

    key = _cache_key(filepath)

    value = pickle.dumps(
        {
            "timestamp": timestamp,
            "metadata": metadata,
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    with _LMDB_ENV.begin(write=True) as txn:
        txn.put(key, value)

class Vector:
    def __init__(self, filepath: str):
        self.filepath = os.path.abspath(filepath)

        self._metadata = self._load_or_compute_metadata()

    @property
    def bbox(self) -> tuple:
        return tuple(self._metadata["bbox"])

    @property
    def epsg_4326_bbox(self) -> tuple:
        return tuple(self._metadata["epsg_4326_bbox"])
    
    @property
    def projection(self) -> str:
        return self._metadata["projection"]

    def _load_or_compute_metadata(self) -> dict[str, Any]:
        timestamp = _get_file_timestamp(self.filepath)

        # Fast path
        metadata = _load_cached_metadata(
            self.filepath,
            timestamp,
        )

        if metadata is not None:
            return metadata

        # Prevent duplicate computation within process threads
        with _CACHE_LOCK:
            metadata = _load_cached_metadata(
                self.filepath,
                timestamp,
            )

            if metadata is not None:
                return metadata

            metadata = self._compute_metadata()

            _save_cached_metadata(
                self.filepath,
                timestamp,
                metadata,
            )

        return metadata
    
    def _compute_metadata(self) -> dict[str, Any]:
        info = pyogrio.read_info(self.filepath, force_total_bounds=True)
        projection = info['crs']
        if projection is None:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(self.filepath)

            # Retrieve the file metadata
            metadata = parquet_file.metadata
            geo_metadata = json.loads(metadata.metadata[b'geo'].decode('utf-8'))
            projection = ":".join(map(str,geo_metadata['columns']['geometry']['crs']['id'].values()))

        bbox = info['total_bounds']
        if projection is not None and projection != 'EPSG:4326':
            minx, miny, maxx, maxy = bbox
            gdf_bbox = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs=projection).to_crs(4326)
            epsg_4326_bbox = gdf_bbox.total_bounds
        else:
            epsg_4326_bbox = bbox

        return {
            "projection": projection,
            "bbox": bbox,
            "epsg_4326_bbox": epsg_4326_bbox,
        }
