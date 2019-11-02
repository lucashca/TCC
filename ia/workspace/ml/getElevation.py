import rasterio


coords = ((-13.028610576500245, -38.11269908126678))

elevation = 'srtm/SC-24-Z-C.tif'

with rasterio.open(elevation) as src:
    vals = src.sample(coords)
    for val in vals:
        print(val[0]) 