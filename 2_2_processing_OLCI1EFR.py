# ===============================================================
# Processing of Sentinel-3 OLCI L1 EFR imagery for chlorophyll-a retrieval
# SNAP C2RCC processor — SIMILE protocol implementation
# RAW / LITE / STRICT masking strategies
# CIPAIS report — Lake Lugano
# Version of 16.12.2025, created by Yara Leone
# ===============================================================


import sys
sys.path.append(r'C:\\Users\\pocket\\.snap\\snap-python')

import os, zipfile
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_origin

import snappy
from snappy import ProductIO, GPF
jpy = snappy.jpy

import pandas as pd

print("SNAP + snappy loaded.")

# -----------------------------
# Global parameters
# -----------------------------
BASE_DIR   = r"outputs\sentinel3b_l1efr"
LAKE_SHP   = r"inputs\SIMILE LAGO LUGANO SHP\SIMILE_laghi_UTM32.shp"

# ----------------------------------
# Load temperature table (Observed)
# ----------------------------------

excel_path = r"inputs\\temp_observed_estimated_2024.xlsx"

df_temp = None
try:
    raw = pd.read_excel(excel_path, header=None)

    df_temp = raw.iloc[:, 26:29].copy()  # columns AA–AC
    df_temp.columns = ["Month", "Estimated", "Observed"]

    df_temp = df_temp.dropna(subset=["Month", "Observed"], how="any")

    df_temp["Month"] = df_temp["Month"].astype(str).str.strip().str.lower()

    print("Observed temperature table loaded correctly.")

except Exception as e:
    print("ERROR reading Excel → temperatures unavailable.")
    print(e)

# --------------------------------------
# Function to get temperature by month
# --------------------------------------
def get_temperature_for_product(prod_root):
    if df_temp is None:
        return 20.0

    base = os.path.basename(prod_root)

    try:
        parts = base.split("_")
        date_block = None
        for p in parts:
            if len(p) >= 8 and p[:8].isdigit():
                date_block = p[:8]
                break

        if date_block is None:
            raise ValueError("Date not found in product name.")

        month_num = int(date_block[4:6])

        month_name = [
            "january","february","march","april","may","june",
            "july","august","september","october","november","december"
        ][month_num - 1]

        row = df_temp[df_temp["Month"] == month_name]

        if row.empty:
            raise ValueError("Month not found in Observed table.")

        temp = float(row["Observed"].values[0])
        return temp

    except Exception as e:
        print("Unable to extract temperature → default value 20°C applied.")
        print(e)
        return 20.0


# Masking modes:
MASK_MODE = "LITE"


# -----------------------------
# ZIP extraction
# -----------------------------
def extract_zip(zip_path, extract_dir):
    if os.path.isdir(zip_path):
        return zip_path
    if zip_path.lower().endswith(".zip"):
        out_path = os.path.splitext(zip_path)[0]
        if not os.path.exists(out_path):
            try:
                print(f"Extracting {os.path.basename(zip_path)} ...")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_dir)
            except:
                print("Invalid ZIP:", zip_path)
                return None
        return out_path
    return None

def find_product_root(path):
    if path is None:
        return None
    for root, dirs, files in os.walk(path):
        if "xfdumanifest.xml" in files:
            return root
    return None

# -----------------------------
# Lake geometry → WKT
# -----------------------------
def load_lake_bbox_wkt(shp):
    gdf = gpd.read_file(shp)

    gdf4326 = gdf.to_crs(4326)
    geom = gdf4326.geometry.union_all().buffer(0.0001)
    minx, miny, maxx, maxy = geom.bounds

    wkt = (f"POLYGON (({minx} {miny}, {maxx} {miny}, "
           f"{maxx} {maxy}, {minx} {maxy}, {minx} {miny}))")

    print("\n=== Lake BBOX WKT ===")
    print(wkt)
    return gdf, wkt

# -----------------------------
# Robust subset (WGS84)
# -----------------------------
def subset_bbox(product, bbox_wkt):
    HashMap = jpy.get_type('java.util.HashMap')
    params = HashMap()
    params.put('copyMetadata', True)
    params.put('geoRegion', bbox_wkt)
    params.put('crs', 'EPSG:4326')

    subset = GPF.createProduct("Subset", params, product)

    print("\n=== Subset size ===")
    print("width  =", subset.getSceneRasterWidth())
    print("height =", subset.getSceneRasterHeight())
    return subset

# -----------------------------
# C2RCC parameters (SIMILE)
# -----------------------------
def c2rcc_config():
    HashMap = jpy.get_type('java.util.HashMap')
    p = HashMap()
    p.put("CHLexp", 0.65)
    p.put("CHLfac", 19.8)
    p.put("salinity", 0.2) # Lake Lugano surface average 1988-2024 
    p.put("temperature", float(TEMPERATURE_C))
    p.put("outputToaReflectances", False)
    p.put("outputOutOfScopeValues", True)
    p.put("useNNValues", True)
    p.put("outputInWater", True)
    p.put("targetResolution", 300)
    return p

# -----------------------------
# Reproject
# -----------------------------
def reproject_to_utm32_snap(product):
    HashMap = jpy.get_type('java.util.HashMap')
    params = HashMap()
    params.put("crs", "EPSG:32632")
    params.put("resampling", "Nearest")
    params.put("pixelSizeX", 300.0)
    params.put("pixelSizeY", 300.0)
    return GPF.createProduct("Reproject", params, product)

# -----------------------------
# Read band
# -----------------------------
def read_band(product, name):
    b = product.getBand(name)
    w = product.getSceneRasterWidth()
    h = product.getSceneRasterHeight()
    arr = jpy.array('float', w * h)
    b.readPixels(0, 0, w, h, arr)
    return np.array(arr, dtype=np.float32).reshape(h, w)

# -----------------------------
# SIMILE flags
# -----------------------------
def get_simile_masks(product):
    for b in product.getBandNames():
        if "flag" in b.lower():
            flag_band = product.getBand(b)
            fc = flag_band.getFlagCoding()
            if fc is None:
                continue

            masks = {}
            wanted = ["Cloud_risk", "Rtosa_OOS", "Rtosa_OOR", "Rhow_OOR"]
            for w in wanted:
                try:
                    masks[w] = fc.getFlagMask(w)
                except:
                    pass

            if masks:
                print("\n=== SIMILE flags ===")
                print(list(masks.keys()))
                return flag_band, masks
    return None, {}

# -----------------------------
# Apply mask
# -----------------------------
def apply_mask(product, chl_raw):
    flag_band, masks = get_simile_masks(product)

    if MASK_MODE.upper() == "RAW":
        print("\nMASK_MODE = RAW → no masking performed.")
        return chl_raw

    w = product.getSceneRasterWidth()
    h = product.getSceneRasterHeight()

    jflags = jpy.array('int', w*h)
    flag_band.readPixels(0, 0, w, h, jflags)
    flags = np.array(jflags, dtype=np.int32).reshape(h, w)

    mask = np.zeros((h,w), dtype=bool)

    if MASK_MODE.upper() == "STRICT":
        for k, m in masks.items():
            mask |= (flags & int(m)) != 0

    if MASK_MODE.upper() == "LITE":
        if "Cloud_risk" in masks:
            mask |= (flags & int(masks["Cloud_risk"])) != 0

    print("\n=== Mask stats ===")
    print("Pixels masked:", np.sum(mask))
    print("Total pixels :", mask.size)

    chl = chl_raw.copy()
    chl[mask] = np.nan
    return chl

# -----------------------------
# Export GeoTIFF
# -----------------------------
def export_tif(product, arr, path, crs_epsg):
    w = product.getSceneRasterWidth()
    h = product.getSceneRasterHeight()

    temp_base = path.replace(".tif", "_snap_tmp")
    ProductIO.writeProduct(product, temp_base, "GeoTIFF")

    snap_tif = temp_base + ".tif"

    with rasterio.open(snap_tif) as src:
        meta = src.meta.copy()

    meta.update({
        "count": 1,
        "dtype": "float32",
        "nodata": -9999.0
    })

    with rasterio.open(path, "w", **meta) as dst:
        data = np.where(np.isfinite(arr), arr, -9999).astype("float32")
        dst.write(data, 1)

    try:
        os.remove(snap_tif)
    except:
        pass

# -----------------------------
# Clip raster
# -----------------------------
def clip_with_shapefile(src_tif, gdf, out_tif):
    with rasterio.open(src_tif) as src:
        if gdf.crs.to_string() != src.crs.to_string():
            gdf = gdf.to_crs(src.crs)
        clipped, transform = mask(src, gdf.geometry, crop=True)
        meta = src.meta.copy()
        meta.update({
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "transform": transform
        })
        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(clipped)

# -----------------------------
# Process a product
# -----------------------------
def process_one(prod_root):
    print("\n=== Processing:", prod_root)

    prod_path = os.path.join(prod_root, "xfdumanifest.xml")
    product = ProductIO.readProduct(prod_path)
    if product is None:
        print("ERROR reading product.")
        return

    gdf, bbox_wkt = load_lake_bbox_wkt(LAKE_SHP)
    subset = subset_bbox(product, bbox_wkt)

    if subset.getSceneRasterWidth() < 5 or subset.getSceneRasterHeight() < 5:
        print("Subset too small → probably outside swath.")
        return

    global TEMPERATURE_C
    TEMPERATURE_C = get_temperature_for_product(prod_root)
    print(f"Temperature applied for this product: {TEMPERATURE_C} °C")

    c2 = GPF.createProduct("C2RCC.OLCI", c2rcc_config(), subset)
    c2_utm = reproject_to_utm32_snap(c2)

    chl_band = None
    for b in c2_utm.getBandNames():
        if b.lower() == "conc_chl":
            chl_band = b
            break
    if chl_band is None:
        print("No conc_chl found.")
        return

    chl_raw = read_band(c2_utm, chl_band)
    chl_raw[chl_raw <= 0] = np.nan

    print("\n=== Raw CHL stats ===")
    print("Finite pixels:", np.sum(np.isfinite(chl_raw)))
    print("NaN pixels   :", np.sum(~np.isfinite(chl_raw)))

    chl_masked = apply_mask(c2_utm, chl_raw)

    # === Statistical trimming: remove extreme values outside 2.5–97.5 percentile ===
    try:
        p_low  = np.nanpercentile(chl_masked, 2.5)
        p_high = np.nanpercentile(chl_masked, 97.5)

        print(f"\nStatistical trimming thresholds → low: {p_low:.3f}, high: {p_high:.3f}")

        trimmed = chl_masked.copy()
        trimmed[(trimmed < p_low) | (trimmed > p_high)] = np.nan

        chl_masked = trimmed

        print("Trimming applied: extreme values removed.")
    except Exception as e:
        print("Trimming failed → using non-trimmed data.")
        print(e)

    finite = chl_masked[np.isfinite(chl_masked)]

    print("\n=== Final CHL stats ===")
    print("Finite:", finite.size)

    if finite.size == 0:
        print("No valid data after masking.")
        return

    print("min =", np.nanmin(finite))
    print("max =", np.nanmax(finite))
    print("med =", np.nanmedian(finite))

    tmp_tif = prod_root + "_tmp.tif"
    clip_tif = prod_root + "_clip.tif"
    out_tif = prod_root + f"_Chl_{MASK_MODE}.tif"

    export_tif(c2_utm, chl_masked, tmp_tif, "EPSG:32632")
    clip_with_shapefile(tmp_tif, gdf, clip_tif)

    os.replace(clip_tif, out_tif)
    os.remove(tmp_tif)

    print("Output:", out_tif)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    product_dirs = []
    print("\nScanning directory:", BASE_DIR)

    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            if f.lower().endswith(".zip"):
                prod_path = extract_zip(os.path.join(root, f), BASE_DIR)
                prod_root = find_product_root(prod_path)
                if prod_root and prod_root not in product_dirs:
                    product_dirs.append(prod_root)

    for root, dirs, files in os.walk(BASE_DIR):
        if "xfdumanifest.xml" in files:
            if root not in product_dirs:
                product_dirs.append(root)

    print("\nDetected products:")
    for d in product_dirs:
        print(" •", os.path.basename(d))

    for prod in product_dirs:
        try:
            process_one(prod)
        except Exception as e:
            print("ERROR:", e)

    print("\nProcessing complete.")
