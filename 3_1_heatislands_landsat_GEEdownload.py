# ========================================================================
# Landsat 8–9 Land Surface Temperature (LST) download
# USGS STAC (Collection 2 Level-2 inventory) + Google Earth Engine
# Heat island analysis — Canton of Ticino
# Version of 16.12.2025, created by Yara Leone
# ========================================================================



import os, sys, time, requests
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta, timezone
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

import ee, geemap

# ========================================================================
# 1. USER SETTINGS
# ========================================================================
AOI_SHP = r"C:\Users\pocket\Desktop\Rendu supsi\CIPAIS_automation\inputs\TICINO SHP\Tessin_AOI.shp"
OUT_DIR = r"C:\Users\pocket\Desktop\Rendu supsi\CIPAIS_automation\outputs\landsat_lst_ticino"
os.makedirs(OUT_DIR, exist_ok=True)

START = "2025-06-01"
END   = "2025-09-30"

MIN_COVERAGE = 50.0  # minimum AOI coverage (%) to download a scene

LST_BANDS = ["ST_B10", "ST_EMIS", "ST_DRAD", "QA_PIXEL"]

STAC_URL = "https://landsatlook.usgs.gov/stac-server/search"

# ========================================================================
# 2. LOAD AOI SHAPEFILE
# ========================================================================
print("Loading AOI shapefile...")

if not os.path.exists(AOI_SHP):
    print(f"AOI shapefile not found: {AOI_SHP}")
    sys.exit(1)

gdf = gpd.read_file(AOI_SHP)
if gdf.empty:
    raise ValueError("AOI shapefile is empty or invalid.")

if gdf.crs is None or gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(4326)

aoi_geom = unary_union(list(gdf.geometry))
aoi_area_km2 = gdf.to_crs("EPSG:32632").area.sum() / 1e6

minx, miny, maxx, maxy = aoi_geom.bounds
bbox = [float(minx), float(miny), float(maxx), float(maxy)]

print(f"AOI loaded. Area = {aoi_area_km2:.2f} km²")

# ========================================================================
# 3. QUERY USGS STAC (OFFICIAL INVENTORY)
# ========================================================================
print("\nQuerying USGS STAC server (official inventory)...")

body = {
    "collections": ["landsat-c2l2-sr", "landsat-c2l2-st"],
    "bbox": bbox,
    "datetime": f"{START}T00:00:00Z/{END}T23:59:59Z",
    "limit": 1000
}

r = requests.post(STAC_URL, json=body, timeout=120)
r.raise_for_status()
results = r.json()
features = results.get("features", [])

print(f"STAC returned {len(features)} scenes.")

if not features:
    print("No scenes found for this AOI and date range.")
    sys.exit(0)

# ========================================================================
# 4. PARSE STAC RESULTS + AOI COVERAGE + WRS PATH/ROW
# ========================================================================
def parse_rfc3339(dt_str):
    if dt_str.endswith("Z"):
        dt_str = dt_str.replace("Z", "+00:00")
    return datetime.fromisoformat(dt_str).astimezone(timezone.utc)
summary = []
seen_keys = set()  # to avoid duplicates from SR + ST items

print("\nComputing AOI coverage and extracting WRS path/row...")

for feat in features:
    props = feat.get("properties", {})
    geom  = feat.get("geometry")
    scene_id = feat.get("id", "unknown_id")

    # Datetime (UTC)
    dt = parse_rfc3339(props["datetime"])
    dt_label = dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    acq_date = dt.date().isoformat()  # YYYY-MM-DD

    # Cloud cover
    cloud = props.get("eo:cloud_cover", float("nan"))

    # Platform → satellite code
    platform = props.get("platform", "").lower()
    if "8" in platform:
        sat_code = "LC08"
    elif "9" in platform:
        sat_code = "LC09"
    else:
        sat_code = "UNKNOWN"

    # WRS path/row from STAC
    wrs_path = props.get("landsat:wrs_path", None)
    wrs_row  = props.get("landsat:wrs_row", None)

    if wrs_path is None or wrs_row is None:
        continue  # skip if we cannot match in EE

    # AOI coverage = intersection(footprint, AOI) / AOI area
    if geom:
        g = shape(geom)
        inter = g.intersection(aoi_geom)
        if inter.is_empty:
            cov = 0.0
        else:
            inter_area_km2 = (
                gpd.GeoSeries([inter], crs="EPSG:4326")
                .to_crs("EPSG:32632")
                .area.iloc[0] / 1e6
            )
            cov = min((inter_area_km2 / aoi_area_km2) * 100.0, 100.0)
    else:
        cov = 0.0

    # Key to avoid duplicates from SR/ST duplicates:
    # (satellite, wrs_path, wrs_row, acquisition_date)
    key = (sat_code, int(wrs_path), int(wrs_row), acq_date)
    if key in seen_keys:
        continue
    seen_keys.add(key)

    summary.append({
        "scene_id": scene_id,         # STAC scene id (for reference only)
        "satellite": sat_code,        # LC08 or LC09
        "wrs_path": int(wrs_path),
        "wrs_row": int(wrs_row),
        "acq_date": acq_date,         # YYYY-MM-DD
        "datetime": dt_label,         # full timestamp
        "cloud": cloud,
        "coverage": cov
    })

if not summary:
    print("No valid scenes with WRS path/row found.")
    sys.exit(0)

# ========================================================================
# 5. SAVE INVENTORY TO EXCEL + PRINT TABLE
# ========================================================================
df = pd.DataFrame(summary)

# Keep only required columns in the desired order
df = df[["datetime", "satellite", "cloud", "coverage"]]

# Rename columns for clarity in Excel and console
df = df.rename(columns={
    "datetime": "Datetime (UTC)",
    "satellite": "Satellite name",
    "cloud": "Cloud cover (%)",
    "coverage": "Canton coverage (%)"
})

# Sort by datetime
df = df.sort_values("Datetime (UTC)")

excel_path = os.path.join(OUT_DIR, "landsat_lst_usgs_inventory.xlsx")
df.to_excel(excel_path, index=False)

print(f"\nInventory exported to Excel:\n  {excel_path}")

print("\n" + "="*95)
print("USGS STAC INVENTORY — TICINO")
print("="*95)
print(f"{'Datetime (UTC)':<28} {'Satellite name':<16} {'Cloud cover (%)':>18} {'Canton coverage (%)':>20}")
print("-"*95)

for _, s in df.iterrows():
    cloud_val = f"{s['Cloud cover (%)']:.1f}" if isinstance(s['Cloud cover (%)'], (float, int)) else "n/a"
    cov_val   = f"{s['Canton coverage (%)']:.2f}"
    print(f"{s['Datetime (UTC)']:<28} {s['Satellite name']:<16} {cloud_val:>18} {cov_val:>20}")

print("="*95)
print(f"Total scenes listed: {len(df)}")
print("="*95)

# ========================================================================
# 6. INITIALIZE EARTH ENGINE
# ========================================================================
print("\nInitializing Google Earth Engine...")

try:
    ee.Initialize()
    print("Earth Engine initialized.\n")
except Exception:
    ee.Authenticate()
    ee.Initialize()
    print("Earth Engine initialized after authentication.\n")

# Convert AOI to Earth Engine geometry
aoi_ee = ee.Geometry(mapping(aoi_geom))
region = aoi_ee.buffer(2000)  # small buffer around AOI, for export region

# ========================================================================
# 7. FUNCTION TO FIND CORRESPONDING EE IMAGE
# ========================================================================
def find_ee_image(sat_code, wrs_path, wrs_row, acq_date_str):

    if sat_code == "LC08":
        col_id = "LANDSAT/LC08/C02/T1_L2"
    elif sat_code == "LC09":
        col_id = "LANDSAT/LC09/C02/T1_L2"
    else:
        return None

    start = datetime.fromisoformat(acq_date_str)
    end   = start + timedelta(days=1)

    col = (ee.ImageCollection(col_id)
           .filterBounds(aoi_ee)
           .filterDate(acq_date_str, end.date().isoformat())
           .filter(ee.Filter.eq("WRS_PATH", wrs_path))
           .filter(ee.Filter.eq("WRS_ROW", wrs_row))
           .filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
           )

    if col.size().getInfo() == 0:
        return None

    return col.first()

# ========================================================================
# 8. SELECT SCENES ≥ MIN_COVERAGE
# ========================================================================
selected = [s for s in summary if s["coverage"] >= MIN_COVERAGE]

print(f"\n{len(selected)} scenes meet the ≥ {MIN_COVERAGE:.1f}% coverage threshold.")

if not selected:
    print("No scenes meet the minimum coverage threshold. Nothing to download.")
    sys.exit(0)

choice = input("\nDownload LST bands (ST_B10, ST_EMIS, ST_DRAD, QA_PIXEL) for these scenes? (y/n): ").strip().lower()
if choice not in ["y", "yes"]:
    print("Download canceled by user.")
    sys.exit(0)

# ========================================================================
# 9. DOWNLOAD LST BANDS VIA EARTH ENGINE
# ========================================================================
download_count = 0  # will count scenes with at least one band exported
download_log = []   # will store info for the second Excel summary

for s in selected:

    sat_code = s["satellite"]
    wrs_path = s["wrs_path"]
    wrs_row  = s["wrs_row"]
    acq_date = s["acq_date"]
    scene_id = s["scene_id"]

    print(f"\nMatching EE image for scene:")
    print(f"  STAC id   : {scene_id}")
    print(f"  Satellite : {sat_code}, Path {wrs_path}, Row {wrs_row}, Date {acq_date}")

    img = find_ee_image(sat_code, wrs_path, wrs_row, acq_date)
    if img is None:
        print("  ! No matching EE image found, skipping.")
        continue

    # Check that required LST bands exist
    try:
        bands = img.bandNames().getInfo()
    except Exception as e:
        print(f"  ! Failed to read band names from EE image: {e}")
        continue

    missing = [b for b in LST_BANDS if b not in bands]
    if missing:
        print(f"  ! Missing LST bands {missing}, skipping scene.")
        continue

    # Create output folder for this scene
    scene_dir = os.path.join(OUT_DIR, f"{sat_code}_{wrs_path:03d}{wrs_row:03d}_{acq_date}")
    os.makedirs(scene_dir, exist_ok=True)

    # Export LST bands
    downloaded_bands = []     # track which bands were downloaded
    downloaded_any_band = False

    for b in LST_BANDS:

        out_tif = os.path.join(scene_dir, f"{sat_code}_{wrs_path:03d}{wrs_row:03d}_{acq_date}_{b}.tif")

        if os.path.exists(out_tif):
            print(f"  - {b}: already exists, skipping.")
            continue

        print(f"  - Exporting {b} → {out_tif}")
        try:
            geemap.ee_export_image(
                ee_object=img.select(b).clip(aoi_ee),
                filename=out_tif,
                scale=30,
                region=region,
                crs="EPSG:4326"
            )
            downloaded_bands.append(b)
            downloaded_any_band = True

        except Exception as e:
            print(f"    ! Export failed for {b}: {e}")

    # Record scene information if it downloaded at least one band
    if downloaded_any_band:
        download_count += 1

        download_log.append({
            "Datetime (UTC)": s["datetime"],
            "Satellite name": s["satellite"],
            "Cloud cover (%)": s["cloud"],
            "Canton coverage (%)": s["coverage"],
            "Bands downloaded": ", ".join(downloaded_bands)
        })

    time.sleep(2)

print(f"\nDownload completed. Total scenes downloaded: {download_count}")
print("All files saved under:")
print(f"  {OUT_DIR}")

# ========================================================================
# 10. SAVE DOWNLOAD SUMMARY EXCEL
# ========================================================================
if download_log:
    dl_df = pd.DataFrame(download_log)
    dl_path = os.path.join(OUT_DIR, "landsat_lst_GEE_downloaded.xlsx")
    dl_df.to_excel(dl_path, index=False)
    print(f"\nDownload summary exported to:\n  {dl_path}")
else:
    print("\nNo scenes were downloaded. No summary generated.")

sys.exit(0)

