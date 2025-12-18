# ===============================================================
# Sentinel-3 Chl-a product validation and comparison
# Processed Sentinel-3 imagery vs reference chlorophyll-a maps
# CIPAIS report — Lake Lugano
# Version of 16.12.2025, created by Yara Leone
# ===============================================================


import os
import re
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Input folders
# ---------------------------------------------------------

reference_dir = r"C:\Users\pocket\Desktop\Rendu supsi\CIPAIS_automation\inputs\chla_reference_images_2024"
s3_dir        = r"C:\Users\pocket\Desktop\Rendu supsi\CIPAIS_automation\outputs\sentinel3b_l1efr"

# Output folder = same as S3 rasters
output_root = s3_dir


# ---------------------------------------------------------
# 2. Italian month abbreviations → "MM"
# ---------------------------------------------------------

month_map_it = {
    "gen": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "mag": "05",
    "giu": "06",
    "lug": "07",
    "ago": "08",
    "set": "09",
    "ott": "10",
    "nov": "11",
    "dic": "12",
}

# Reverse lookup: "03" → "mar"
month_map_rev = {v: k for k, v in month_map_it.items()}


# ---------------------------------------------------------
# 3. Raster loading function
# ---------------------------------------------------------

def load_raster(path):
    """Load a raster and convert nodata to NaN."""
    src = rasterio.open(path)
    data = src.read(1).astype(float)
    nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    return src, data


# ---------------------------------------------------------
# 4. Load all reference maps into dictionary
# ---------------------------------------------------------

reference_dict = {}

print("\n=== Scanning reference maps ===")

for f in os.listdir(reference_dir):

    if not f.lower().endswith(".tif"):
        continue

    f_low = f.lower()
    for abb in month_map_it.keys():
        if f_low.startswith(f"chla_{abb}"):
            reference_dict[abb] = f
            break

for abb, fname in reference_dict.items():
    print(f"  {abb} → {fname}")

if not reference_dict:
    raise SystemExit("No valid .tif reference maps found !")


# ---------------------------------------------------------
# 5. Process all Sentinel-3 rasters
# ---------------------------------------------------------

s3_files = [f for f in os.listdir(s3_dir) if f.lower().endswith(".tif")]

print("\n=== Beginning comparison ===")

for s3_file in s3_files:

    # Extract year + month from pattern YYYYMMDD
    match = re.search(r"(202\d)(\d{2})\d{2}", s3_file)
    if not match:
        print("Skipping (no date found):", s3_file)
        continue

    yyyy = match.group(1)
    mm   = match.group(2)

    abb = month_map_rev.get(mm)

    if not abb:
        print("Skipping (unknown month):", s3_file)
        continue

    print(f"\n=== Processing month: {abb} {yyyy} ===")

    # Find reference raster
    ref_file = reference_dict.get(abb)
    if ref_file is None:
        print("No reference map for:", abb)
        continue

    ref_path = os.path.join(reference_dir, ref_file)
    s3_path  = os.path.join(s3_dir, s3_file)

    print("→ Reference:", ref_file)
    print("→ S3 raster:", s3_file)

    # Tag used in filenames
    tag = f"__REF_{ref_file.replace('.tif','')}__S3_{s3_file.replace('.tif','')}"

    # Load both rasters
    srcA, A = load_raster(ref_path)
    srcB, B = load_raster(s3_path)

    # ---------------------------------------------------------
    # Align if needed
    # ---------------------------------------------------------

    if srcA.crs != srcB.crs or srcA.transform != srcB.transform:
        print("  Aligning S3 raster to reference grid...")
        B_aligned = np.empty_like(A)

        reproject(
            source=B,
            destination=B_aligned,
            src_transform=srcB.transform,
            src_crs=srcB.crs,
            dst_transform=srcA.transform,
            dst_crs=srcA.crs,
            resampling=Resampling.bilinear
        )
    else:
        B_aligned = B

    # ---------------------------------------------------------
    # Mask valid pixels
    # ---------------------------------------------------------

    mask = ~np.isnan(A) & ~np.isnan(B_aligned)

    if not np.any(mask):
        print("No overlapping valid pixels!")
        continue

    A_m = A[mask]
    B_m = B_aligned[mask]

    diff = A - B_aligned
    rel_diff = (A - B_aligned) / B_aligned * 100


    # ---------------------------------------------------------
    # Output directory per month + year
    # ---------------------------------------------------------

    out_dir = os.path.join(output_root, f"{abb}_{yyyy}_comparison")
    os.makedirs(out_dir, exist_ok=True)


    # ---------------------------------------------
    # Generate figures
    # ---------------------------------------------
    
    # 1. Side-by-side images  
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(A, cmap="viridis")
    plt.title(f"Reference ({abb} {yyyy})")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(B_aligned, cmap="viridis")
    plt.title(f"Sentinel-3 ({s3_file[:16]})")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"1_side_by_side{tag}.png"))
    plt.close()


    # 2. Difference map
    plt.figure(figsize=(8, 6))
    vmax = np.nanmax(np.abs(diff))
    plt.imshow(diff, cmap="RdBu", vmin=-vmax, vmax=vmax)
    plt.title("Difference (Reference − S3)")
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, f"2_difference_map{tag}.png"))
    plt.close()

    # 3. Relative difference (%)
    plt.figure(figsize=(8, 6))
    plt.imshow(rel_diff, cmap="coolwarm")
    plt.title("Relative difference (%)")
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, f"3_relative_difference{tag}.png"))
    plt.close()

    # 4. Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(A_m, B_m, s=3, alpha=0.3)
    plt.xlabel("Reference")
    plt.ylabel("Sentinel-3")
    plt.title("Pixel-to-pixel scatter")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"4_scatter{tag}.png"))
    plt.close()

    # 5. Histogram comparison
    plt.figure(figsize=(8, 6))
    plt.hist(A_m, bins=80, alpha=0.5, label="Reference")
    plt.hist(B_m, bins=80, alpha=0.5, label="Sentinel-3")
    plt.legend()
    plt.title("Histogram comparison")
    plt.savefig(os.path.join(out_dir, f"5_histograms{tag}.png"))
    plt.close()

    print(f"  → Figures saved in: {out_dir}")

print("\n=== COMPARISON COMPLETED SUCCESSFULLY ===\n")
