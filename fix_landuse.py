import glob
import tqdm
import os
import multiprocessing as mp

os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

from osgeo import gdal

gdal.UseExceptions()

def handle_tile(land_use_tile: str):
    # Open each file as text, since these are vrts, replace incorrect path, read with gdal, and rewrite
    with open(land_use_tile, 'r') as f:
        try:
            content = f.read()
            if not content.startswith('<VRTDataset'):
                return
        except UnicodeDecodeError:
            return
        except Exception as e:
            raise e
        
    corrected_content = content.replace('C:\\Users\\lrr43\\Documents\\lu\\', '/vsis3/global-floodmaps/landcover/').replace(
        '>lu/', '>/vsis3/global-floodmaps/landcover/'
    )
    with open(land_use_tile, 'w') as f:
        f.write(corrected_content)

    driver: gdal.Driver = gdal.GetDriverByName('MEM')
    try:
        ds: gdal.Dataset = driver.CreateCopy('', gdal.Open(land_use_tile), strict=0)
    except Exception as e:
        print(f"Error processing {land_use_tile}: {e}")
        return
    ds.FlushCache()
    out_driver: gdal.Driver = gdal.GetDriverByName('GTiff')
    out_ds: gdal.Dataset = out_driver.CreateCopy(land_use_tile, ds, strict=0, options=['COMPRESS=ZSTD', 'PREDICTOR=2'])
    out_ds.FlushCache()
    out_ds = None
    ds = None

def main():
    land_use_tiles = glob.glob('/Users/Shared/flood_map_tiles/lon=*/lat=*/inputs=*/land_use.tif')

    with mp.Pool(processes=mp.cpu_count()*4) as pool:
        list(tqdm.tqdm(
            pool.imap_unordered(handle_tile, land_use_tiles),
            total=len(land_use_tiles),
            desc="Fixing land use tiles"
        ))

if __name__ == "__main__":
     main()