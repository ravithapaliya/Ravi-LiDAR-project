import os
import glob
import time
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
import gc
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. USER SETTINGS
# =========================================================
YEARS = [2002, 2007, 2012, 2017, 2023] 

qol_path = r"C:\Users\ravit\Documents\tree research work new\QOL_NPA_2020_final_projected\QOL_NPA_2020_final_projected.shp"

# The base directory where your Vertical_Structure folders are located
base_input_drive = "F:\\"

# The folder where the final CSV summaries will be saved
output_folder = r"F:\MYPHDRESEARCHOUTPUTS\Neighborhood_3D_Summaries"
os.makedirs(output_folder, exist_ok=True)

# =========================================================
# 2. LOAD QOL POLYGONS
# =========================================================
print("Loading QOL polygons...")
qol = gpd.read_file(qol_path)
qol = qol[qol.geometry.notnull() & ~qol.geometry.is_empty].copy()
qol['NPA_ID'] = qol['NPA_ID'].astype(int)

# =========================================================
# 3. HIGH-SPEED AGGREGATION LOOP
# =========================================================
for year in YEARS:
    print(f"\n==============================================")
    print(f"Aggregating 3D metrics for {year}...")
    print(f"==============================================")
    
    start_time = time.time()
    tiles_folder = os.path.join(base_input_drive, f"Vertical_Structure_{year}")
    output_csv = os.path.join(output_folder, f"NPA_3D_Structure_Summary_{year}.csv")
    
    if not os.path.exists(tiles_folder):
        print(f"Folder not found for {year}: {tiles_folder}. Skipping...")
        continue

    # Find all 3D structure tiles for the current year
    tif_files = glob.glob(os.path.join(tiles_folder, "*_3D_Structure*.tif"))
    print(f"Found {len(tif_files)} 3D Structure tiles.")
    
    if len(tif_files) == 0:
        continue

    # Initialize a dictionary to accumulate pixel sums across all tiles
    # This prevents us from having to merge massive rasters together
    npa_stats = {npa: {
        "fhd_sum": 0.0, "vci_sum": 0.0,
        "under_sum": 0.0, "mid_sum": 0.0, "over_sum": 0.0,
        "veg_pixels": 0, "total_pixels": 0
    } for npa in qol['NPA_ID']}

    # Process TILE by TILE (opens each file exactly once -> EXTREMELY FAST)
    for i, tif in enumerate(tif_files, 1):
        try:
            with rasterio.open(tif) as src:
                tile_bbox = box(*src.bounds)
                
                # Find which NPAs touch this specific tile
                intersecting_qol = qol[qol.intersects(tile_bbox)]
                
                if intersecting_qol.empty:
                    continue
                    
                # Create an integer grid where each pixel value is the NPA_ID
                shapes = ((geom, npa_id) for geom, npa_id in zip(intersecting_qol.geometry, intersecting_qol.NPA_ID))
                npa_grid = rasterize(
                    shapes,
                    out_shape=(src.height, src.width),
                    transform=src.transform,
                    fill=0,
                    dtype='uint32'
                )

                # Read the 5 structural bands
                fhd = src.read(1)
                vci = src.read(2)
                under = src.read(3)
                mid = src.read(4)
                over = src.read(5)

            # Pre-calculate boolean masks to speed up NumPy math
            valid_mask = (fhd != -9999.0) & np.isfinite(fhd)
            veg_mask = valid_mask & ((under + mid + over) > 0)
            
            # Find the unique Neighborhood IDs that actually exist in this tile's pixels
            present_npas = np.unique(npa_grid[valid_mask])
            
            # Instantly sum up the data for each neighborhood using the masks
            for npa in present_npas:
                if npa == 0: 
                    continue # 0 is the background (outside of any NPA)
                    
                npa_mask = (npa_grid == npa)
                npa_valid = npa_mask & valid_mask
                npa_veg = npa_mask & veg_mask
                
                if np.any(npa_valid):
                    npa_stats[npa]["under_sum"] += float(np.sum(under[npa_valid]))
                    npa_stats[npa]["mid_sum"] += float(np.sum(mid[npa_valid]))
                    npa_stats[npa]["over_sum"] += float(np.sum(over[npa_valid]))
                    npa_stats[npa]["total_pixels"] += int(np.count_nonzero(npa_valid))
                    
                if np.any(npa_veg):
                    npa_stats[npa]["fhd_sum"] += float(np.sum(fhd[npa_veg]))
                    npa_stats[npa]["vci_sum"] += float(np.sum(vci[npa_veg]))
                    npa_stats[npa]["veg_pixels"] += int(np.count_nonzero(npa_veg))
                    
        except Exception as e:
            print(f"Error reading tile {os.path.basename(tif)}: {e}")
            
        finally:
            # Drop massive arrays from RAM instantly without NameErrors
            fhd = vci = under = mid = over = npa_grid = valid_mask = veg_mask = None
            gc.collect()
            
        # Print progress every 50 tiles so you know it's working
        if i % 50 == 0:
            print(f"  Processed {i}/{len(tif_files)} tiles...")

    # =========================================================
    # 4. CALCULATE FINAL MEANS & EXPORT
    # =========================================================
    year_summary = []
    for npa, s in npa_stats.items():
        if s["total_pixels"] > 0:
            mean_fhd = s["fhd_sum"] / s["veg_pixels"] if s["veg_pixels"] > 0 else 0.0
            mean_vci = s["vci_sum"] / s["veg_pixels"] if s["veg_pixels"] > 0 else 0.0
            
            year_summary.append({
                "NPA_ID": npa,
                "Year": year,
                "Mean_FHD": mean_fhd,
                "Mean_VCI": mean_vci,
                "Total_Understory_Density": s["under_sum"],
                "Total_Midstory_Density": s["mid_sum"],
                "Total_Overstory_Density": s["over_sum"],
                "Total_Vegetated_Pixels": s["veg_pixels"],
                "Total_Analyzed_Pixels": s["total_pixels"]
            })

    if year_summary:
        df_out = pd.DataFrame(year_summary)
        df_out.to_csv(output_csv, index=False)
        mins = (time.time() - start_time) / 60.0
        print(f"-> DONE! Saved {year} aggregation to {os.path.basename(output_csv)} in {mins:.2f} minutes.")

print("\nAll years aggregated successfully!")