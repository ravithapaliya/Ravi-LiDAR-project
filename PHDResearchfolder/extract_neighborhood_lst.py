import os
import glob
import time
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. USER SETTINGS & PATHS
# =========================================================
lst_input_folder = r"E:\Landsat_LST_Charlotte_RQ3\landsat_lst_downscaled_1m"
qol_path = r"C:\Users\ravit\Documents\tree research work new\QOL_NPA_2020_final_projected\QOL_NPA_2020_final_projected.shp"

# The script will automatically create this folder if it doesn't exist yet!
output_folder = r"F:\MYPHDRESEARCHOUTPUTS\others\LST_Summaries"
os.makedirs(output_folder, exist_ok=True)

# =========================================================
# 2. LOAD QOL POLYGONS
# =========================================================
print("Loading QOL neighborhood polygons...")
qol = gpd.read_file(qol_path)
qol = qol[qol.geometry.notnull() & ~qol.geometry.is_empty].copy()
qol['NPA_ID'] = qol['NPA_ID'].astype(int)

# =========================================================
# 3. HIGH-SPEED LST EXTRACTION LOOP
# =========================================================
tif_files = glob.glob(os.path.join(lst_input_folder, "*.tif"))
print(f"Found {len(tif_files)} LST raster files to process.\n")

for i, tif in enumerate(tif_files, 1):
    start_time = time.time()
    filename = os.path.basename(tif)
    filename_no_ext = os.path.splitext(filename)[0]
    
    output_csv = os.path.join(output_folder, f"QOL_NPA_LST_Summary_{filename_no_ext}.csv")
    
    if os.path.exists(output_csv):
        print(f"[{i}/{len(tif_files)}] Skipping (already processed): {filename_no_ext}")
        continue
        
    print(f"[{i}/{len(tif_files)}] Processing LST: {filename_no_ext}")
    
    try:
        with rasterio.open(tif) as src:
            # Check CRS and reproject QOL polygons on the fly if they don't match
            if qol.crs != src.crs:
                qol_proj = qol.to_crs(src.crs)
            else:
                qol_proj = qol
                
            # Create an integer grid where each pixel value is the NPA_ID
            shapes = ((geom, npa_id) for geom, npa_id in zip(qol_proj.geometry, qol_proj.NPA_ID))
            npa_grid = rasterize(
                shapes,
                out_shape=(src.height, src.width),
                transform=src.transform,
                fill=0,
                dtype='uint32'
            )
            
            # Read the LST band (Band 1)
            lst_data = src.read(1)
            nodata_val = src.nodata if src.nodata is not None else -9999.0
            
        # Create a mask for valid LST pixels (ignore nodata and background)
        valid_mask = (lst_data != nodata_val) & np.isfinite(lst_data) & (lst_data > -50) & (npa_grid > 0)
        
        # Flatten the arrays to 1D to use highly optimized NumPy grouping
        valid_npa_ids = npa_grid[valid_mask]
        valid_lst_vals = lst_data[valid_mask]
        
        if len(valid_lst_vals) == 0:
            print(f"  -> Warning: No valid LST data found intersecting neighborhoods.")
            continue
            
        # Group by NPA_ID and calculate the mean LST
        # bincount calculates the sum of LST and the count of pixels simultaneously
        lst_sum = np.bincount(valid_npa_ids, weights=valid_lst_vals)
        pixel_count = np.bincount(valid_npa_ids)
        
        # Find which NPA IDs actually have data
        active_npas = np.nonzero(pixel_count)[0]
        
        mean_lst = lst_sum[active_npas] / pixel_count[active_npas]
        
        # Format results into a DataFrame
        results_df = pd.DataFrame({
            'NPA_ID': active_npas,
            'mean_lst': mean_lst,
            'valid_lst_pixels': pixel_count[active_npas]
        })
        
        results_df.to_csv(output_csv, index=False)
        mins = (time.time() - start_time) / 60.0
        print(f"  -> Success: Exported {len(results_df)} neighborhoods in {mins:.2f} mins")
        
    except Exception as e:
        print(f"  -> Error processing {filename}: {e}")

print("\nAll LST extraction complete!")