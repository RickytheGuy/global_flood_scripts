import multiprocessing as mp
import glob

from global_floodmaps.utility_functions import rewrite_file_as_parquet_with_covering_bbox
from global_floodmaps.parallel_functions import bar_map

if __name__ == "__main__":
    streamlines = glob.glob('/Users/Shared/streamlines/streams_*.parquet')

    with mp.Pool() as pool:
        bar_map(
            pool, 
            rewrite_file_as_parquet_with_covering_bbox, 
            streamlines,
            total=len(streamlines), desc="Adding covering bbox to streamlines parquet files")