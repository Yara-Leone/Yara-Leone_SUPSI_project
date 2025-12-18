# ===============================================================
# Sentinel-3 OLCI L1 EFR download
# Copernicus Data Space Ecosystem (STAC)
# CIPAIS report — Lake Lugano
# Version of 16.12.2025, created by Yara Leone
# ===============================================================

import os
import time
import requests
import geopandas as gpd

# -----------------------------
# GLOBAL PARAMETERS
# -----------------------------
LAKE_SHP  = r"inputs\lake_lugano_shp_epsg4326\lac_lugano.shp"  #epsg:4623 projection needed for CSDE
os.makedirs("outputs", exist_ok=True) 
OUT_DIR   = r"outputs\sentinel3b_l1efr"
os.makedirs(OUT_DIR, exist_ok=True)

# Analysis period 
START_DATE = "2024-01-01"
END_DATE   = "2024-12-31"

BUFFER_DEGREES = 0.1
PAUSE_S = 3

# Copernicus Data Space credentials
COPERNICUS_CLIENT_ID = "your_cdse_client_id"
COPERNICUS_CLIENT_SECRET = "your_cdse_client_secret"

# Sentinel-3 OLCI L1 EFR (NTC) collection
COLLECTION = "sentinel-3-olci-1-efr-ntc"

# -----------------------------
# CDSE Authentication
# -----------------------------
def get_copernicus_token(client_id, client_secret):
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials"
    }
    headers = {"content-type": "application/x-www-form-urlencoded"}
    r = requests.post(url, headers=headers, data=payload, timeout=60)
    r.raise_for_status()
    token_data = r.json()
    token_data["expires_at"] = time.time() + token_data.get("expires_in", 0)
    return token_data

print("Connecting to Copernicus Data Space...")
token_info = get_copernicus_token(COPERNICUS_CLIENT_ID, COPERNICUS_CLIENT_SECRET)
ACCESS_TOKEN = token_info["access_token"]
print("Copernicus token successfully obtained.\n")

# -----------------------------
# Load AOI
# -----------------------------
print("Loading shapefile...")
gdf = gpd.read_file(LAKE_SHP).to_crs("EPSG:4326")
gdf["geometry"] = gdf.buffer(BUFFER_DEGREES)
geom = gdf.unary_union
bbox = geom.bounds
print(f"BBOX: {bbox}\n")

# -----------------------------
# STAC Search
# -----------------------------
print(f"Searching Sentinel-3 OLCI L1 EFR images ({START_DATE} → {END_DATE})...")

search_url = "https://stac.dataspace.copernicus.eu/v1/search"
headers = {"Content-Type": "application/json", "Accept": "application/geo+json"}

body = {
    "collections": [COLLECTION],
    "bbox": bbox,
    "datetime": f"{START_DATE}T00:00:00Z/{END_DATE}T23:59:59Z",
    "limit": 1000
}

r = requests.post(search_url, headers=headers, json=body, timeout=120)
r.raise_for_status()
features = r.json().get("features", [])
print(f"{len(features)} products found.\n")

if not features:
    raise SystemExit("No products found for this period.")

# -----------------------------
# Summary table
# -----------------------------
import pandas as pd

records = []
for feat in features:
    props = feat.get("properties", {})
    datetime_full = props.get("datetime", "N/A")
    # Date + UTC time (ex: 2024-05-16 10:31:52)
    try:
        dt_str = datetime_full.replace("T", " ").replace("Z", "")
    except Exception:
        dt_str = datetime_full
    records.append({
        "Date & Time (UTC)": dt_str,
        "Satellite": props.get("platform", "N/A"),
        "ProductType": props.get("s3:productType", "OL_1_EFR"),
    })

df = pd.DataFrame(records)
print("Products found:\n")
print(df.to_string(index=True))

df = pd.DataFrame(records)
print("Products found:\n")
print(df.to_string(index=True))

# -----------------------------
# Save summary Excel in OUT_DIR
# -----------------------------
excel_out = os.path.join(OUT_DIR, "sentinel3_olci_luganoforCIPAIS_inventory.xlsx")
df.to_excel(excel_out, index=False)
print(f"\nExcel summary saved to:\n{excel_out}\n")

# -------------------------------------
# Robust download (fixes '$value' bug)
# -------------------------------------
AUTH_HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

# Confirmation before download
choice = input("\nDo you want to download these products? (y/n): ").strip().lower()
if choice not in ["y", "yes"]:
    print("Download cancelled.")
    exit()

def get_zip_href(assets):
    # Returns the direct link to the .zip file (often .../$value)
    if "product" in assets and "href" in assets["product"]:
        return assets["product"]["href"]
    for a in assets.values():
        href = a.get("href", "")
        if ".zip" in href or "$value" in href:
            return href
    return None

for i, feat in enumerate(features):
    print("\n" + "=" * 70)
    print(f"[{i+1}/{len(features)}] Downloading product...")
    print("=" * 70)

    assets = feat.get("assets", {})
    href = get_zip_href(assets)
    if not href:
        print("No ZIP link found, skipping.")
        continue

    # Convert s3:// → https://
    if href.startswith("s3://eodata/"):
        href = href.replace("s3://eodata/", "https://eodata.dataspace.copernicus.eu/")

    # Clean filename
    base_name = os.path.basename(href)
    if base_name == "$value" or base_name == "":
        # Extract product name from STAC ID
        pid = feat.get("id", "sentinel3_product").replace("/", "_")
        base_name = f"{pid}.zip"
    elif base_name.endswith("$value"):
        base_name = base_name.replace("$value", "zip")

    out_zip = os.path.join(OUT_DIR, base_name)

    if os.path.exists(out_zip):
        print(f"File already exists: {out_zip}")
        continue

    print(f"Downloading: {base_name}")
    try:
        with requests.get(href, headers=AUTH_HEADERS, stream=True, timeout=600) as r:
            r.raise_for_status()
            # Check that the response is a ZIP file
            content_type = r.headers.get("Content-Type", "").lower()
            if "application/zip" not in content_type and not href.lower().endswith(".zip"):
                print(f"Non-ZIP response ({content_type}) — probably an error JSON.")
                text_preview = r.text[:500]
                print(f"--- Server response ---\n{text_preview}\n--- End ---")
                continue
            with open(out_zip, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
        print(f"Saved: {out_zip}")
    except Exception as e:
        print(f"Error during download: {e}")

    time.sleep(PAUSE_S)

print("\nSentinel-3 OLCI L1 EFR download completed successfully!")
print(f"Output directory: {OUT_DIR}")
