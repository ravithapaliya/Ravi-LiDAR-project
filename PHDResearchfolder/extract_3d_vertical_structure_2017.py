import os
import time
import math
import gc
import numpy as np
import laspy
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from scipy.stats import binned_statistic_2d
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. USER SETTINGS
# =========================================================
# Update these paths to match your 2002, 2007, 2012, 2017, or 2023 folders
las_folder = r"F:\MecklenburgCountyGIS\LiDAR2017\Masspoints"
output_folder = r"C:\Users\ravit\Documents\Phd_Novel_3D_Results\Vertical_Structure_2017"

target_crs = CRS.from_epsg(2264)
resolution = 3.28084  # 1 meter in US Survey Feet

# Height Bins (in feet) for Vertical Stratification
# e.g., Understory (3-15ft), Midstory (15-50ft), Overstory (>50ft)
HEIGHT_BINS = [3.0, 15.0, 50.0, 150.0] 

# Classes
ground_classes = [2]
veg_classes = [1, 3, 4, 5] # Update based on the specific LiDAR year (e.g., 2007 is 1, 2017 is 3,4,5)

skip_completed = True

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================
def calculate_fhd_vci(z_veg, bins):
    """
    Calculates Foliage Height Diversity (FHD) and Vertical Complexity Index (VCI)
    for a 1D array of vegetation heights inside a single pixel.
    """
    if len(z_veg) == 0:
        return 0.0, 0.0, [0]*len(bins)
    
    # Calculate point counts in each vertical bin
    counts, _ = np.histogram(z_veg, bins=[0.0] + bins)
    
    # Calculate proportions (p_i) for each bin
    total_pts = len(z_veg)
    proportions = counts / total_pts
    
    # FHD = -Sum(p_i * ln(p_i))
    fhd = 0.0
    for p in proportions:
        if p > 0:
            fhd -= p * np.log(p)
            
    # VCI = FHD / ln(number of height bins)
    vci = fhd / np.log(len(bins)) if len(bins) > 1 else 0.0
    
    return fhd, vci, counts

def write_multiband_raster(path, arrays, transform, crs, band_names):
    meta = {
        "driver": "GTiff",
        "height": arrays[0].shape[0],
        "width": arrays[0].shape[1],
        "count": len(arrays),
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
        "nodata": -9999.0
    }
    with rasterio.open(path, "w", **meta) as dst:
        for i, arr in enumerate(arrays, start=1):
            dst.write(arr.astype("float32"), i)
            dst.set_band_description(i, band_names[i-1])

# =========================================================
# 3. MAIN PROCESSING LOOP
# =========================================================
os.makedirs(output_folder, exist_ok=True)
las_files = sorted([f for f in os.listdir(las_folder) if f.lower().endswith(".las")])
print(f"Found {len(las_files)} LAS files for 3D Vertical Analysis.")

for filename in las_files:
    tile_id = os.path.splitext(filename)[0]
    out_tif = os.path.join(output_folder, f"{tile_id}_3D_Structure.tif")
    
    if skip_completed and os.path.exists(out_tif):
        print(f"Skipping {tile_id}...")
        continue
        
    print(f"\nProcessing 3D Structure for: {tile_id}")
    start_t = time.time()
    
    try:
        # 1. Read LAS
        las_path = os.path.join(las_folder, filename)
        las = laspy.read(las_path)
        
        x = np.asarray(las.x, dtype=np.float32)
        y = np.asarray(las.y, dtype=np.float32)
        z = np.asarray(las.z, dtype=np.float32)
        cls = np.asarray(las.classification, dtype=np.int16)
        
        # 2. Extract Ground and Vegetation
        ground_m = np.isin(cls, ground_classes)
        veg_m = np.isin(cls, veg_classes)
        
        if np.sum(ground_m) < 10 or np.sum(veg_m) < 10:
            print("  Skipping: Insufficient ground or vegetation points.")
            continue
            
        # 3. Fast DTM Generation (using 2D binning min-Z)
        x_min, x_max = math.floor(np.min(x)), math.ceil(np.max(x))
        y_min, y_max = math.floor(np.min(y)), math.ceil(np.max(y))
        
        ncols = int(math.ceil((x_max - x_min) / resolution))
        nrows = int(math.ceil((y_max - y_min) / resolution))
        
        x_edges = np.linspace(x_min, x_max, ncols + 1)
        y_edges = np.linspace(y_max, y_min, nrows + 1) # y decreases top to bottom
        transform = from_origin(x_min, y_max, resolution, resolution)
        
        # Compute ground elevation per pixel
        dtm_stats, _, _, _ = binned_statistic_2d(
            y[ground_m], x[ground_m], z[ground_m], 
            statistic='min', bins=[y_edges[::-1], x_edges]
        )
        dtm = np.flipud(dtm_stats) # Align to raster axes
        
        # Fill missing DTM gaps with local median (simple gap fill)
        from scipy.ndimage import generic_filter
        dtm[np.isnan(dtm)] = np.nanmedian(dtm)
        if np.isnan(dtm).any():
            dtm = np.nan_to_num(dtm, nan=np.nanmean(dtm))

        # 4. Normalize Vegetation Heights
        # Map each veg point to its raster cell to subtract ground Z
        col_indices = np.digitize(x[veg_m], x_edges) - 1
        row_indices = np.digitize(y[veg_m], y_edges[::-1]) - 1
        row_indices = (nrows - 1) - row_indices # Flip for array indexing
        
        # Filter out-of-bounds indices
        valid_idx = (col_indices >= 0) & (col_indices < ncols) & (row_indices >= 0) & (row_indices < nrows)
        
        x_v = x[veg_m][valid_idx]
        y_v = y[veg_m][valid_idx]
        z_v_raw = z[veg_m][valid_idx]
        
        r_idx = row_indices[valid_idx]
        c_idx = col_indices[valid_idx]
        
        # Normalized veg height
        z_v_norm = z_v_raw - dtm[r_idx, c_idx]
        
        # Keep only valid heights
        valid_h = (z_v_norm >= HEIGHT_BINS[0]) & (z_v_norm <= HEIGHT_BINS[-1])
        r_idx = r_idx[valid_h]
        c_idx = c_idx[valid_h]
        z_v_norm = z_v_norm[valid_h]
        
        # 5. Calculate 3D Metrics per Pixel
        # Initialize output arrays
        fhd_arr = np.full((nrows, ncols), -9999.0, dtype=np.float32)
        vci_arr = np.full((nrows, ncols), -9999.0, dtype=np.float32)
        understory_arr = np.full((nrows, ncols), -9999.0, dtype=np.float32)
        midstory_arr = np.full((nrows, ncols), -9999.0, dtype=np.float32)
        overstory_arr = np.full((nrows, ncols), -9999.0, dtype=np.float32)
        
        # Group points by pixel
        # Flatten 2D indices to 1D for faster grouping
        flat_idx = r_idx * ncols + c_idx
        sort_args = np.argsort(flat_idx)
        
        flat_idx_sorted = flat_idx[sort_args]
        z_sorted = z_v_norm[sort_args]
        
        # Find boundaries where the pixel changes
        split_pts = np.unique(flat_idx_sorted, return_index=True)[1][1:]
        z_grouped = np.split(z_sorted, split_pts)
        idx_unique = np.unique(flat_idx_sorted)
        
        for pixel_flat_idx, z_pixel in zip(idx_unique, z_grouped):
            r = pixel_flat_idx // ncols
            c = pixel_flat_idx % ncols
            
            fhd, vci, counts = calculate_fhd_vci(z_pixel, HEIGHT_BINS)
            
            fhd_arr[r, c] = fhd
            vci_arr[r, c] = vci
            understory_arr[r, c] = counts[1] # Bin 1: 3-15ft
            midstory_arr[r, c] = counts[2]   # Bin 2: 15-50ft
            overstory_arr[r, c] = counts[3]  # Bin 3: 50-150ft
            
        # 6. Write Outputs
        band_names = ["Foliage_Height_Diversity", "Vertical_Complexity_Index", 
                      "Understory_Density", "Midstory_Density", "Overstory_Density"]
        
        arrays = [fhd_arr, vci_arr, understory_arr, midstory_arr, overstory_arr]
        write_multiband_raster(out_tif, arrays, transform, target_crs, band_names)
        
        print(f"  Success. 3D Metrics generated in {((time.time() - start_t) / 60.0):.2f} mins")
        
    except Exception as e:
        print(f"  Error processing {filename}: {e}")
        
    finally:
        gc.collect()

print("\nAll 3D Structure processing completed.")
