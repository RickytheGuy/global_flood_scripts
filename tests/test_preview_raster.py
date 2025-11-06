from global_floodmaps import convert_area_to_single_vrt
import glob, os

files = glob.glob(r"C:\Users\lrr43\Documents\global_flood_maps\global_flood_maps\lon=*\lat=*\floodmaps\dem=fabdem\rp=100.tif")
bbox = [-180, -90, 180, 90]

convert_area_to_single_vrt(bbox, files, os.path.join(r"C:\Users\lrr43\Downloads", "all_flows_fabdem.vrt"))