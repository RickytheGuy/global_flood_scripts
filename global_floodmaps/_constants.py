import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ESA_TILES_FILE = os.path.join(THIS_DIR, 'data', 'esa_tiles.gpkg')
STORAGE_OPTIONS = {"anon": True, 'config_kwargs': {'response_checksum_validation':'when_required'}}

