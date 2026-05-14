from propcache import  cached_property

import geopandas as gpd
from shapely.geometry import box
from osgeo import gdal, osr

gdal.UseExceptions()


class Raster:
    def __init__(self, filepath):
        self.filepath = filepath

    @cached_property
    def ds(self) -> gdal.Dataset:
        return gdal.Open(self.filepath)

    @cached_property
    def geotransform(self) -> tuple:
        return self.ds.GetGeoTransform()

    @cached_property
    def projection(self) -> str:
        return self.ds.GetProjection()

    @cached_property
    def shape(self) -> tuple:
        return self.ds.RasterXSize, self.ds.RasterYSize
    
    @cached_property
    def resolution(self) -> tuple:
        gt = self.geotransform
        return abs(gt[1]), abs(gt[5])

    @cached_property
    def bbox(self) -> tuple:
        gt = self.geotransform
        width, height = self.shape

        minx = gt[0]
        maxx = gt[0] + width * gt[1]
        miny = gt[3] + height * gt[5]
        maxy = gt[3]

        if miny > maxy:
            miny, maxy = maxy, miny

        return minx, miny, maxx, maxy

    @cached_property
    def epsg_4326_bbox(self) -> tuple:
        bbox = self.bbox
        projection = self.projection
        srs = osr.SpatialReference(projection)
        srs.AutoIdentifyEPSG()
        if srs.IsGeographic() and srs.GetAuthorityCode(None) == '4326':
            return bbox
        
        return tuple(
            gpd.GeoSeries(
                [box(*self.bbox)],
                crs=self.projection
            ).to_crs("EPSG:4326").total_bounds
        )