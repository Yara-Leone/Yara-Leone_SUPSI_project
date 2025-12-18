# ========================================================================
# Sentinel-3 SLSTR Land Surface Temperature (LST) download
# Copernicus Data Space Ecosystem (STAC – sentinel-3-sl-2-lst-ntc)
# Heat island analysis — Canton of Ticino
# Version of 16.12.2025, created by Yara Leone
# ========================================================================


import geopandas as gpd
import os
import requests
import time
import pandas as pd
from datetime import datetime, timezone
from shapely.geometry import shape
import sys

# -----------------------------------------------------
# 1) USER SETTINGS
# -----------------------------------------------------
shapefile_path = r"C:\Users\pocket\Desktop\Rendu supsi\CIPAIS_automation\inputs\TICINO SHP\Tessin_AOI.shp"

output_dir = r"C:\Users\pocket\Desktop\Rendu supsi\CIPAIS_automation\outputs\sentinel3_lst_ticino"
os.makedirs(output_dir, exist_ok=True)

COPERNICUS_CLIENT_ID = "your_cdse_client_id"
COPERNICUS_CLIENT_SECRET = "your_cdse_client_secret"

MIN_COVERAGE_FILTER = 20.0
DAY_START = 8
DAY_END = 18

START = "2025-06-01"
END   = "2025-09-30"

# -----------------------------------------------------
# 2) TOKEN AUTHENTICATION
# -----------------------------------------------------
def get_copernicus_token(client_id, client_secret):
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials"
    }
    headers = {"content-type": "application/x-www-form-urlencoded"}
    r = requests.post(url, headers=headers, data=payload)
    r.raise_for_status()
    token_data = r.json()
    token_data["expires_at"] = time.time() + token_data["expires_in"]
    return token_data

print("Connecting to Copernicus Data Space…")
token_info = get_copernicus_token(COPERNICUS_CLIENT_ID, COPERNICUS_CLIENT_SECRET)
ACCESS_TOKEN = token_info["access_token"]
print("Token obtained.\n")

# -----------------------------------------------------
# 3) LOAD AOI
# -----------------------------------------------------
print("Loading AOI shapefile…")
aoi_gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
aoi_geom = aoi_gdf.unary_union
minx, miny, maxx, maxy = aoi_geom.bounds
bbox_expanded = [minx - 0.05, miny - 0.05, maxx + 0.05, maxy + 0.05]
aoi_area_km2 = aoi_gdf.to_crs("EPSG:32632").area.sum() / 1e6

print(f"AOI area: {aoi_area_km2:.2f} km²\n")

# -----------------------------------------------------
# 4) STAC SEARCH
# -----------------------------------------------------
print(f"Searching Sentinel-3 SLSTR LST products from {START} to {END}…\n")

search_url = "https://stac.dataspace.copernicus.eu/v1/search"
headers_stac = {"Content-Type": "application/json", "Accept": "application/geo+json"}

search_body = {
    "collections": ["sentinel-3-sl-2-lst-ntc"],
    "bbox": bbox_expanded,
    "datetime": f"{START}T00:00:00Z/{END}T23:59:59Z",
    "limit": 10000
}

try:
    response = requests.post(search_url, headers=headers_stac, json=search_body, timeout=120)
    response.raise_for_status()
    results = response.json()
    all_features = results.get("features", [])
    print(f"STAC returned {len(all_features)} products.\n")
except Exception as e:
    print("STAC query failed:", e)
    sys.exit(1)

if not all_features:
    print("No products found.")
    sys.exit(0)

# -----------------------------------------------------
# 5) PARSE FEATURES + TRUE COVERAGE
# -----------------------------------------------------
summary = []

def parse_rfc3339(dt_str):
    if dt_str is None:
        return None
    if dt_str.endswith("Z"):
        dt_str = dt_str.replace("Z", "+00:00")
    return datetime.fromisoformat(dt_str).astimezone(timezone.utc)

print("Computing intersection coverage with Tessin AOI…\n")

for feat in all_features:
    props = feat.get("properties", {})
    assets = feat.get("assets", {})
    scene_id = feat.get("id", "unknown_id")

    dt = parse_rfc3339(props.get("datetime"))
    dt_label = dt.strftime("%Y-%m-%dT%H:%M:%S%z") if dt else "unknown"

    hour = dt.hour

    platform = props.get("platform", "UNK").upper()
    geom = feat.get("geometry")

    if geom:
        g = shape(geom)
        inter = g.intersection(aoi_geom)

        if inter.is_empty:
            coverage = 0.0
        else:
            inter_area_km2 = (
                gpd.GeoSeries([inter], crs="EPSG:4326")
                .to_crs("EPSG:32632")
                .area.iloc[0] / 1e6
            )
            coverage = min((inter_area_km2 / aoi_area_km2) * 100, 100)
    else:
        coverage = 0.0

    summary.append({
        "Datetime (UTC)": dt,
        "HourUTC": hour,
        "Satellite": platform,
        "Ticino coverage (%)": coverage,
        "id": scene_id,
        "assets": assets
    })

df = pd.DataFrame(summary)
print(summary[0])

# (IMPORTANT) the sys.exit() here was removed so your script continues

# -----------------------------------------------------
# 6) FILTER : COVERAGE + DAYTIME
# -----------------------------------------------------
print("Coverage stats BEFORE filtering:")
print(" - min coverage :", df["Ticino coverage (%)"].min())
print(" - max coverage :", df["Ticino coverage (%)"].max(), "\n")

before = len(df)

df = df[df["HourUTC"].between(DAY_START, DAY_END)]
df = df[df["Ticino coverage (%)"] >= MIN_COVERAGE_FILTER].reset_index(drop=True)

after = len(df)

print(f"Applying filters ≥ {MIN_COVERAGE_FILTER}% & DAYTIME (08–18 UTC)…")
print(f"Products before filtering : {before}")
print(f"Products after filtering  : {after}\n")

if after == 0:
    print("No products meet the filtering criteria.")
    sys.exit(0)

# -----------------------------------------------------
# 7) SAVE INVENTORY TO EXCEL + PRINT TABLE 
# -----------------------------------------------------
df_table = df[[
    "Datetime (UTC)",
    "Satellite",
    "Ticino coverage (%)"
]]

df_table = df_table.rename(columns={
    "Satellite": "Satellite name",
    "Ticino coverage (%)": "Canton coverage (%)"
})

# Trier avant de formater
df_table = df_table.sort_values("Datetime (UTC)")

# >>> ICI LE FIX IMPORTANT <<< 
# On convertit les datetimes en TEXTE UTC, pour éviter tout décalage
df_table["Datetime (UTC)"] = df_table["Datetime (UTC)"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# Export Excel
excel_path = os.path.join(output_dir, "sentinel3_lst_copernicus_inventory.xlsx")
df_table.to_excel(excel_path, index=False)

print(f"\nInventory exported to Excel:\n  {excel_path}")

print("\n" + "="*95)
print("SENTINEL-3 SLSTR LST INVENTORY — TICINO")
print("="*95)
print(f"{'Datetime (UTC)':<28} {'Satellite name':<16} {'Canton coverage (%)':>20}")
print("-"*95)

for _, s in df_table.iterrows():
    cov_val = f"{s['Canton coverage (%)']:.2f}"
    print(f"{s['Datetime (UTC)']:<28} {s['Satellite name']:<16} {cov_val:>20}")

print("="*95)
print(f"Total scenes listed: {len(df_table)}")
print("="*95)


# ASK FOR DOWNLOAD
choice = input("\nDownload LST files now? (y/n): ").strip().lower()
if choice not in ["y", "yes", "o", "oui"]:
    print("Canceled by user.")
    sys.exit(0)

# -----------------------------------------------------
# 9) DOWNLOAD FULL SAFE PRODUCT
# -----------------------------------------------------
def s3_to_https(url):
    return url.replace(
        "s3://eodata/",
        "https://eodata.dataspace.copernicus.eu/"
    )

def download_file(url, out_path):
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

print("\nStarting SAFE product downloads…\n")

download_log = []

for _, row in df.iterrows():
    scene_id = row["id"]
    assets = row["assets"]

    if "product" not in assets:
        print(f"[SKIP] No product asset for {scene_id}")
        continue

    url = s3_to_https(assets["product"]["href"])
    out_path = os.path.join(output_dir, f"{scene_id}.zip")

    print(f"[DOWNLOAD] {scene_id}.zip")

    try:
        download_file(url, out_path)
        download_log.append({
            "Datetime (UTC)": row["Datetime (UTC)"],
            "Satellite": row["Satellite"],
            "Ticino coverage (%)": row["Ticino coverage (%)"],
            "Downloaded files": f"{scene_id}.zip"
        })
    except Exception as e:
        print("  ! Failed:", e)


# -----------------------------------------------------
# 10) DOWNLOAD SUMMARY
# -----------------------------------------------------
if download_log:
    dl_df = pd.DataFrame(download_log)
    dl_file = os.path.join(output_dir, "sentinel3_lst_downloaded.xlsx")
    dl_df.to_excel(dl_file, index=False)
    print(f"\nDownload summary written to:\n  {dl_file}")
else:
    print("\nNo files downloaded.")

print("\nDone.")
sys.exit(0)
