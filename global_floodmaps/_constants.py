import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ESA_TILES_FILE = os.path.join(THIS_DIR, 'data', 'esa_tiles.gpkg')
DEFAULT_TILES_FILE = os.path.join(THIS_DIR, 'data', 'tiles_in_geoglowsv2.parquet')
STREAM_BOUNDS_FILE = os.path.join(THIS_DIR, 'data', 'stream_bounds.json')
STORAGE_OPTIONS = {"anon": True, 'config_kwargs': {'response_checksum_validation':'when_required'}}

