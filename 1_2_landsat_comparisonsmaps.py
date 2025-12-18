# ===============================================================
# Landsat Land Surface Temperature (LST) validation and comparison
# Processed Landsat 8–9 LST vs reference temperature datasets
# CIPAIS report — Lake Lugano
# Version of 16.12.2025, created by Yara Leone
# ===============================================================


import os
import re
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

# ---------------------------------------------
# 1. Input folders
# ---------------------------------------------

reference_dir = r"inputs\temp_reference_images_2024"
os.makedirs("outputs", exist_ok=True) 
clean_dir = r"outputs\landsat8-9_L2SP"

# Outputs will be created inside the clean directory
output_root = clean_dir


# ---------------------------------------------
# 2. Month number → month name mapping
# ---------------------------------------------

month_map = {
    "01": "january",
    "02": "february",
    "03": "march",
    "04": "april",
    "05": "may",
    "06": "june",
    "07": "july",
    "08": "august",
    "09": "september",
    "10": "october",
    "11": "november",
    "12": "december"
}


# ---------------------------------------------
# 3. Raster loading function
# ---------------------------------------------

def load_raster(path):
    src = rasterio.open(path)
    data = src.read(1).astype(float)

    nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan

    return src, data


# ---------------------------------------------
# 4. Process all CLEAN rasters
# ---------------------------------------------

clean_files = [f for f in os.listdir(clean_dir) if f.endswith(".tif")]

for clean_file in clean_files:

    # Extract month (MM) & year (YYYY) from pattern "YYYYMMDD"
    match = re.search(r"(202\d)(\d{2})\d{2}", clean_file)
    if not match:
        print("Skipping (no date found):", clean_file)
        continue

    yyyy = match.group(1)
    mm = match.group(2)

    month_name = month_map.get(mm)

    if not month_name:
        print("Skipping (unknown month):", clean_file)
        continue

    print(f"\n=== Processing: {month_name} {yyyy} ===")

    # Find reference raster starting with the month name
    reference_file = None
    for f in os.listdir(reference_dir):
        if f.lower().startswith(month_name.lower()):
            reference_file = f
            break

    if reference_file is None:
        print("No reference file found for:", month_name)
        continue

    # Build absolute paths
    ref_path = os.path.join(reference_dir, reference_file)
    clean_path = os.path.join(clean_dir, clean_file)

    print("→ Reference:", reference_file)
    print("→ Clean raster:", clean_file)

    # Tag added to all exported plots
    tag = f"__REF_{reference_file.replace('.tif','')}__CLEAN_{clean_file.replace('.tif','')}"

    # Load both rasters
    srcA, A = load_raster(ref_path)
    srcB, B = load_raster(clean_path)

    # Align if needed (different CRS / transform)
    if srcA.crs != srcB.crs or srcA.transform != srcB.transform:
        print("  Aligning clean raster to reference grid...")
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

    # Mask valid pixels
    mask = ~np.isnan(A) & ~np.isnan(B_aligned)
    A_m = A[mask]
    B_m = B_aligned[mask]

    # Differences
    diff = A - B_aligned
    rel_diff = (A - B_aligned) / B_aligned * 100

    # Output folder: year + month → e.g. april_2024
    out_dir = os.path.join(output_root, f"{month_name}_{yyyy}")
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------
    # Generate figures
    # ---------------------------------------------

    # 1. Side-by-side display
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(A, cmap="viridis")
    plt.title(f"Reference ({month_name} {yyyy})")
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(B_aligned, cmap="viridis")
    plt.title("Clean Landsat")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"1_side_by_side{tag}.png"))
    plt.close()

    # 2. Difference map
    plt.figure(figsize=(8,6))
    vmax = np.nanmax(abs(diff))
    plt.imshow(diff, cmap="RdBu", vmin=-vmax, vmax=vmax)
    plt.title("Difference (Reference - Clean)")
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, f"2_difference_map{tag}.png"))
    plt.close()

    # 3. Relative difference (%)
    plt.figure(figsize=(8,6))
    plt.imshow(rel_diff, cmap="coolwarm")
    plt.title("Relative difference (%)")
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, f"3_relative_difference{tag}.png"))
    plt.close()

    # 4. Scatter plot
    plt.figure(figsize=(6,6))
    plt.scatter(A_m, B_m, s=2, alpha=0.2)
    plt.xlabel("Reference")
    plt.ylabel("Clean")
    plt.title("Pixel-to-pixel scatter plot")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"4_scatter{tag}.png"))
    plt.close()

    # 5. Histogram comparison
    plt.figure(figsize=(8,6))
    plt.hist(A_m, bins=100, alpha=0.5, label="Reference")
    plt.hist(B_m, bins=100, alpha=0.5, label="Clean")
    plt.legend()
    plt.title("Histogram comparison")
    plt.savefig(os.path.join(out_dir, f"5_histograms{tag}.png"))
    plt.close()

    print(f"  → Figures saved to: {out_dir}")

print("\n=== BATCH PROCESS COMPLETED ===")
