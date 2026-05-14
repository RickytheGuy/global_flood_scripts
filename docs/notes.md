# Global DEM Generation
To generate one global DEM file from a text file of individual DEM tiles, follow the steps below.

```{bash}
gdalbuildvrt -input_file_list files.txt global_dem.vrt
```

If downloading from a public S3 bucket, set the `AWS_NO_SIGN_REQUEST` environment variable to `YES` to avoid authentication issues, or provide them if necessary.
```{bash}
gdalbuildvrt --config AWS_NO_SIGN_REQUEST YES -input_file_list files.txt global_dem.vrt
```

Then, convert the VRT to a GeoTIFF:
```{bash}
gdal_translate  --config AWS_NO_SIGN_REQUEST YES --config GDAL_CACHEMAX 50% --config GDAL_MAX_DATASET_POOL_SIZE 1000 --config GDAL_MAX_DATASET_POOL_RAM_USAGE 50GB fabdem_global.vrt fabdem_v1-2.tif -of COG -co COMPRESS=DEFLATE -co BLOCKSIZE=2048 -co BIGTIFF=YES -co NUM_THREADS=ALL_CPUS
```