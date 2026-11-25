import os
import io
import math
import time
import hashlib
import numpy as np
import requests
from PIL import Image
import geopandas as gpd
from rasterio.transform import from_bounds
from rasterio.features import shapes, sieve
from shapely.geometry import shape

CACHE_DIR = "cache"
CLASSES = {1: "building", 2: "road"}


def bbox_from_center(lat, lon, width_m=300.0, height_m=None):
    """A box around a point, in degrees. 300 m over 1024 px is about 0.3 m/pixel, like LoveDA."""
    height_m = height_m or width_m
    dlat = (height_m / 2) / 111320.0
    dlon = (width_m / 2) / (111320.0 * math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)   # west, south, east, north


def fetch_wms(base_url, layer, bbox, size=(1024, 1024), srs="EPSG:4326",
              version="1.1.1", fmt="image/png", timeout=60):
    """Ask a WMS for one picture. Saves it in cache/ so the demo works offline later."""
    west, south, east, north = bbox
    if version == "1.3.0" and srs == "EPSG:4326":
        box = f"{south},{west},{north},{east}"        # 1.3.0 wants latitude first
    else:
        box = f"{west},{south},{east},{north}"
    params = {"service": "WMS", "version": version, "request": "GetMap",
              "layers": layer, "styles": "", "bbox": box,
              "width": size[0], "height": size[1], "format": fmt}
    params["crs" if version == "1.3.0" else "srs"] = srs
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.md5(repr((base_url, sorted(params.items()))).encode()).hexdigest()
    path = os.path.join(CACHE_DIR, key + ".png")
    if os.path.exists(path):                          # already downloaded: no internet needed
        return np.array(Image.open(path).convert("RGB"))
    r = requests.get(base_url, params=params, timeout=timeout)
    if not r.headers.get("Content-Type", "").startswith("image"):
        raise RuntimeError("WMS did not return an image: " + r.text[:300])
    with open(path, "wb") as f:
        f.write(r.content)
    return np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))


def mask_to_gdf(mask, bbox, min_pixels=40, crs="EPSG:4326"):
    """Pixel mask -> polygons with real-world coordinates."""
    h, w = mask.shape
    west, south, east, north = bbox
    transform = from_bounds(west, south, east, north, w, h)   # pixel (row, col) <-> coordinates
    names, geoms = [], []
    for value, name in CLASSES.items():
        binary = (mask == value).astype(np.uint8)
        if binary.sum() == 0:
            continue
        binary = sieve(binary, size=min_pixels, connectivity=8)   # drop specks under 40 pixels
        for geom, _ in shapes(binary, mask=binary.astype(bool),
                              transform=transform, connectivity=8):
            names.append(name)
            geoms.append(shape(geom))                 # GeoJSON-style dict -> shapely polygon
    return gpd.GeoDataFrame({"class": names}, geometry=geoms, crs=crs)


def compute_stats(gdf):
    stats = {"buildings": int((gdf["class"] == "building").sum()),
             "building_area_m2": 0.0, "road_area_m2": 0.0}
    if len(gdf) == 0:
        return stats
    try:
        g = gdf.to_crs(4326)
        proj = g.to_crs(g.estimate_utm_crs())         # metres, so areas make sense
        area = proj.area
        stats["building_area_m2"] = float(area[g["class"] == "building"].sum())
        stats["road_area_m2"] = float(area[g["class"] == "road"].sum())
    except Exception:
        pass
    return stats


def overlay(rgb, mask, alpha=0.5):
    out = rgb.astype(np.float32).copy()
    for value, colour in {1: (255, 60, 60), 2: (255, 220, 0)}.items():
        m = mask == value
        out[m] = (1 - alpha) * out[m] + alpha * np.array(colour, dtype=np.float32)
    return out.astype(np.uint8)


def process(rgb, bbox, segmenter, min_pixels=40):
    start = time.time()
    mask = segmenter.predict(rgb)
    gdf = mask_to_gdf(mask, bbox, min_pixels=min_pixels)
    return {"mask": mask, "gdf": gdf, "geojson": gdf.to_json(),
            "stats": compute_stats(gdf), "seconds": time.time() - start}