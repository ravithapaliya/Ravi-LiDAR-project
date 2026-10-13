import os
import time
import math
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
import laspy
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.features import rasterize
from shapely.geometry import box
from scipy.stats import binned_statistic_2d
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. USER SETTINGS
# =========================================================
bare_earth_folder = r"F:\MecklenburgCountyGIS\LiDAR2023\BareEarth_las"
classified_las_folder = r"F:\MecklenburgCountyGIS\LiDAR2023\Classified_las"
output_folder = r"C:\Users\ravit\Documents\Phd_Novel_3D_Results\Vertical_Structure_2023"
building_shapefile = r"F:\MYPHDRESEARCHDATA\Mecklenburg_2023_BuildingFootprints\Mecklenburg_2023_BuildingFootprints.shp"

target_crs = CRS.from_epsg(2264)
resolution = 3.28084  # 1 meter in US Survey Feet

# Height Bins (in feet) for Vertical Stratification
HEIGHT_BINS = [3.0, 15.0, 50.0, 150.0] 

# Excluded classes to isolate vegetation
excluded_classes = np.array([2, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=np.int16)

skip_completed = True

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================
def find_matching_bare_earth_file(filename, bare_folder):
    candidates = [
        os.path.join(bare_folder, filename),
        os.path.join(bare_folder, filename.replace(".las", ".LAS")),
        os.path.join(bare_folder, filename.lower()),
        os.path.join(bare_folder, filename.upper()),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    base = os.path.splitext(filename)[0].lower()
    all_bare = [f for f in os.listdir(bare_folder) if f.lower().endswith(".las")]
    for f in all_bare:
        if os.path.splitext(f)[0].lower() == base:
            return os.path.join(bare_folder, f)
    return None

def calculate_fhd_vci(z_veg, bins):
    if len(z_veg) == 0:
        return 0.0, 0.0, [0]*len(bins)
    
    counts, _ = np.histogram(z_veg, bins=[0.0] + bins)
    total_pts = len(z_veg)
    proportions = counts / total_pts
    
    fhd = 0.0
    for p in proportions:
        if p > 0:
            fhd -= p * np.log(p)
            
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

def rasterize_buildings_for_tile(buildings_gdf, sindex, tile_bbox, out_shape, transform):
    idx = list(sindex.intersection(tile_bbox.bounds))
    if len(idx) == 0:
        return np.zeros(out_shape, dtype=np.uint8)

    subset = buildings_gdf.iloc[idx]
    subset = subset[subset.intersects(tile_bbox)]

    if subset.empty:
        return np.zeros(out_shape, dtype=np.uint8)

    mask = rasterize(
        [(geom, 1) for geom in subset.geometry if geom.is_valid],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    )
    return mask

# =========================================================
# 3. CORE TILE PROCESSING FUNCTION (Memory Isolated)
# =========================================================
def process_tile(filename, buildings_gdf, bldg_sindex):
    tile_id = os.path.splitext(filename)[0]
    out_tif = os.path.join(output_folder, f"{tile_id}_3D_Structure_2023.tif")
    
    if skip_completed and os.path.exists(out_tif):
        print(f"Skipping {tile_id}...")
        return
        
    print(f"\nProcessing 3D Structure for: {tile_id}")
    start_t = time.time()
    
    classified_path = os.path.join(classified_las_folder, filename)
    bare_path = find_matching_bare_earth_file(filename, bare_earth_folder)
    
    if bare_path is None:
        print(f"  Skipping. No matching bare earth LAS found for {filename}")
        return
        
    # --- MEMORY OPTIMIZATION STEP 1: Process Classified LAS ---
    with laspy.open(classified_path) as f_cls:
        las_cls = f_cls.read()
        
        # Build mask without converting coordinates yet
        veg_m = ~np.isin(las_cls.classification, excluded_classes)
        
        if np.sum(veg_m) < 10:
            print("  Skipping: Insufficient candidate vegetation points.")
            return
            
        # Extract ONLY vegetation points directly to float32 to prevent float64 RAM crash
        x_v = np.float32(las_cls.x[veg_m])
        y_v = np.float32(las_cls.y[veg_m])
        z_v_raw = np.float32(las_cls.z[veg_m])
        
    # The 'with' block closes the file. We clear massive arrays from RAM immediately.
    del las_cls, veg_m
    gc.collect() 
    
    # --- MEMORY OPTIMIZATION STEP 2: Process Bare Earth LAS ---
    with laspy.open(bare_path) as f_be:
        las_be = f_be.read()
        gnd_m = np.isin(las_be.classification, [2])
        
        if np.sum(gnd_m) < 10:
            print("  Skipping: Insufficient ground points.")
            return
            
        x_g = np.float32(las_be.x[gnd_m])
        y_g = np.float32(las_be.y[gnd_m])
        z_g = np.float32(las_be.z[gnd_m])

    del las_be, gnd_m
    gc.collect()

    # --- Step 3: Establish Tile Extent ---
    x_min = min(math.floor(np.min(x_v)), math.floor(np.min(x_g)))
    x_max = max(math.ceil(np.max(x_v)), math.ceil(np.max(x_g)))
    y_min = min(math.floor(np.min(y_v)), math.floor(np.min(y_g)))
    y_max = max(math.ceil(np.max(y_v)), math.ceil(np.max(y_g)))
    
    ncols = int(math.ceil((x_max - x_min) / resolution))
    nrows = int(math.ceil((y_max - y_min) / resolution))
    
    x_edges = np.linspace(x_min, x_max, ncols + 1)
    y_edges = np.linspace(y_max, y_min, nrows + 1)
    transform = from_origin(x_min, y_max, resolution, resolution)
    
    # --- Step 4: Generate DTM (Digital Terrain Model) ---
    dtm_stats, _, _, _ = binned_statistic_2d(
        y_g, x_g, z_g, 
        statistic='min', bins=[y_edges[::-1], x_edges]
    )
    dtm = np.flipud(dtm_stats)
    
    dtm[np.isnan(dtm)] = np.nanmedian(dtm)
    if np.isnan(dtm).any():
        dtm = np.nan_to_num(dtm, nan=np.nanmean(dtm))

    del x_g, y_g, z_g
    gc.collect()

    # --- Step 5: Normalize Vegetation Heights ---
    col_indices = np.digitize(x_v, x_edges) - 1
    row_indices = np.digitize(y_v, y_edges[::-1]) - 1
    row_indices = (nrows - 1) - row_indices 
    
    valid_idx = (col_indices >= 0) & (col_indices < ncols) & (row_indices >= 0) & (row_indices < nrows)
    
    z_v_raw = z_v_raw[valid_idx]
    r_idx = row_indices[valid_idx]
    c_idx = col_indices[valid_idx]
    
    z_v_norm = z_v_raw - dtm[r_idx, c_idx]
    
    valid_h = (z_v_norm >= HEIGHT_BINS[0]) & (z_v_norm <= HEIGHT_BINS[-1])
    r_idx = r_idx[valid_h]
    c_idx = c_idx[valid_h]
    z_v_norm = z_v_norm[valid_h]
    
    del x_v, y_v, z_v_raw, col_indices, row_indices, valid_idx
    gc.collect()
    
    if len(z_v_norm) == 0:
        print("  Skipping: No vegetation points found between 3ft and 150ft.")
        return

    # --- Step 6: Calculate 3D Metrics per Pixel ---
    fhd_arr = np.full((nrows, ncols), -9999.0, dtype=np.float32)
    vci_arr = np.full((nrows, ncols), -9999.0, dtype=np.float32)
    understory_arr = np.full((nrows, ncols), -9999.0, dtype=np.float32)
    midstory_arr = np.full((nrows, ncols), -9999.0, dtype=np.float32)
    overstory_arr = np.full((nrows, ncols), -9999.0, dtype=np.float32)
    
    flat_idx = r_idx * ncols + c_idx
    sort_args = np.argsort(flat_idx)
    
    flat_idx_sorted = flat_idx[sort_args]
    z_sorted = z_v_norm[sort_args]
    
    split_pts = np.unique(flat_idx_sorted, return_index=True)[1][1:]
    z_grouped = np.split(z_sorted, split_pts)
    idx_unique = np.unique(flat_idx_sorted)
    
    for pixel_flat_idx, z_pixel in zip(idx_unique, z_grouped):
        r = pixel_flat_idx // ncols
        c = pixel_flat_idx % ncols
        
        fhd, vci, counts = calculate_fhd_vci(z_pixel, HEIGHT_BINS)
        
        fhd_arr[r, c] = fhd
        vci_arr[r, c] = vci
        understory_arr[r, c] = counts[1] 
        midstory_arr[r, c] = counts[2]   
        overstory_arr[r, c] = counts[3]  
        
    # --- Step 7: Mask out Buildings ---
    tile_bbox = box(x_min, y_min, x_max, y_max)
    building_mask = rasterize_buildings_for_tile(
        buildings_gdf, bldg_sindex, tile_bbox, fhd_arr.shape, transform
    )
    
    fhd_arr[building_mask == 1] = -9999.0
    vci_arr[building_mask == 1] = -9999.0
    understory_arr[building_mask == 1] = -9999.0
    midstory_arr[building_mask == 1] = -9999.0
    overstory_arr[building_mask == 1] = -9999.0

    # --- Step 8: Export the 5-Band GeoTIFF ---
    band_names = ["Foliage_Height_Diversity", "Vertical_Complexity_Index", 
                  "Understory_Density", "Midstory_Density", "Overstory_Density"]
    
    arrays = [fhd_arr, vci_arr, understory_arr, midstory_arr, overstory_arr]
    write_multiband_raster(out_tif, arrays, transform, target_crs, band_names)
    
    print(f"  Success. 3D Metrics generated in {((time.time() - start_t) / 60.0):.2f} mins")

# =========================================================
# 4. EXECUTION LOOP
# =========================================================
if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)
    
    print("Loading 2023 building footprints...")
    buildings = gpd.read_file(building_shapefile)
    if buildings.crs != target_crs:
        buildings = buildings.to_crs(target_crs)
    buildings = buildings[buildings.geometry.notnull() & ~buildings.geometry.is_empty].copy()
    bldg_sindex = buildings.sindex
    
    classified_files = sorted([f for f in os.listdir(classified_las_folder) if f.lower().endswith(".las")])
    print(f"Found {len(classified_files)} Classified LAS files for 2023 3D Vertical Analysis.")

    for filename in classified_files:
        try:
            # Process tile executes completely inside its own memory scope
            process_tile(filename, buildings, bldg_sindex)
        except Exception as e:
            print(f"  Critical Error processing {filename}: {e}")
        finally:
            # Enforce RAM cleanup after each tile regardless of success/fail
            gc.collect()

    print("\nAll 2023 3D Structure processing completed.")