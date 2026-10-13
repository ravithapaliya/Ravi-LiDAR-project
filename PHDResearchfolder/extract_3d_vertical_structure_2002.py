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
bare_earth_folder = r"F:\MecklenburgCountyGIS\LiDAR2002\Bare Earth\LAS"
reflective_folder = r"F:\MecklenburgCountyGIS\LiDAR2002\Reflective Surface\New folder"
output_folder = r"C:\Users\ravit\Documents\Phd_Novel_3D_Results\Vertical_Structure_2002"
building_shapefile = r"F:\MYPHDRESEARCHDATA\Mecklenburg_Buildings_2002\Mecklenburg_Buildings_2002.shp"

target_crs = CRS.from_epsg(2264)
resolution = 3.28084  # 1 meter in US Survey Feet

# Height Bins (in feet) for Vertical Stratification
# Bin 1: Understory (3-15ft), Bin 2: Midstory (15-50ft), Bin 3: Overstory (>50ft)
HEIGHT_BINS = [3.0, 15.0, 50.0, 150.0] 

# Classes based on 2002 Metadata
ground_classes = [2]
building_classes = [6]
noise_classes = [7]
water_classes = [9]
ignored_classes = [20, 21]

skip_completed = True

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================
def find_matching_bare_earth_file(filename, bare_folder):
    """Finds the corresponding bare earth LAS tile for a reflective tile."""
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
    """Calculates Foliage Height Diversity (FHD) and Vertical Complexity Index (VCI)."""
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
    """Writes the 3D metric arrays to a multi-band GeoTIFF."""
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
    """Creates a binary mask of building footprints for the tile to exclude building points."""
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
# 3. LOAD BUILDING FOOTPRINTS
# =========================================================
os.makedirs(output_folder, exist_ok=True)

print("Loading 2002 building footprints...")
buildings = gpd.read_file(building_shapefile)

if buildings.crs is None:
    raise ValueError("Building shapefile has no CRS defined.")

if buildings.crs != target_crs:
    buildings = buildings.to_crs(target_crs)

buildings = buildings[buildings.geometry.notnull() & ~buildings.geometry.is_empty].copy()
buildings_sindex = buildings.sindex

# =========================================================
# 4. MAIN PROCESSING LOOP
# =========================================================
reflective_files = sorted([f for f in os.listdir(reflective_folder) if f.lower().endswith(".las")])
print(f"Found {len(reflective_files)} Reflective LAS files for 2002 3D Vertical Analysis.")

for filename in reflective_files:
    tile_id = os.path.splitext(filename)[0]
    out_tif = os.path.join(output_folder, f"{tile_id}_3D_Structure_2002.tif")
    
    if skip_completed and os.path.exists(out_tif):
        print(f"Skipping {tile_id}...")
        continue
        
    print(f"\nProcessing 3D Structure for: {tile_id}")
    start_t = time.time()
    
    try:
        # 1. Locate and read the dual LAS files
        reflective_path = os.path.join(reflective_folder, filename)
        bare_path = find_matching_bare_earth_file(filename, bare_earth_folder)
        
        if bare_path is None:
            print(f"  Skipping. No matching bare earth LAS found for {filename}")
            continue
            
        las_ref = laspy.read(reflective_path)
        las_be = laspy.read(bare_path)
        
        x_ref = np.asarray(las_ref.x, dtype=np.float32)
        y_ref = np.asarray(las_ref.y, dtype=np.float32)
        z_ref = np.asarray(las_ref.z, dtype=np.float32)
        cls_ref = np.asarray(las_ref.classification, dtype=np.int16)
        
        x_be = np.asarray(las_be.x, dtype=np.float32)
        y_be = np.asarray(las_be.y, dtype=np.float32)
        z_be = np.asarray(las_be.z, dtype=np.float32)
        
        # 2. Extract Vegetation Candidates
        excluded_classes = set(ground_classes + building_classes + noise_classes + water_classes + ignored_classes)
        veg_m = ~np.isin(cls_ref, list(excluded_classes))
        
        if len(x_be) < 10 or np.sum(veg_m) < 10:
            print("  Skipping: Insufficient ground or vegetation candidate points.")
            continue
            
        # 3. Establish Tile Extent based on Reflective Points
        x_min, x_max = math.floor(np.min(x_ref)), math.ceil(np.max(x_ref))
        y_min, y_max = math.floor(np.min(y_ref)), math.ceil(np.max(y_ref))
        
        ncols = int(math.ceil((x_max - x_min) / resolution))
        nrows = int(math.ceil((y_max - y_min) / resolution))
        
        x_edges = np.linspace(x_min, x_max, ncols + 1)
        y_edges = np.linspace(y_max, y_min, nrows + 1)
        transform = from_origin(x_min, y_max, resolution, resolution)
        
        # 4. Generate DTM (Digital Terrain Model) from Bare Earth Data
        dtm_stats, _, _, _ = binned_statistic_2d(
            y_be, x_be, z_be, 
            statistic='min', bins=[y_edges[::-1], x_edges]
        )
        dtm = np.flipud(dtm_stats)
        
        # Fill missing DTM gaps (if any) with local median
        dtm[np.isnan(dtm)] = np.nanmedian(dtm)
        if np.isnan(dtm).any():
            dtm = np.nan_to_num(dtm, nan=np.nanmean(dtm))

        # 5. Normalize Vegetation Heights
        col_indices = np.digitize(x_ref[veg_m], x_edges) - 1
        row_indices = np.digitize(y_ref[veg_m], y_edges[::-1]) - 1
        row_indices = (nrows - 1) - row_indices 
        
        valid_idx = (col_indices >= 0) & (col_indices < ncols) & (row_indices >= 0) & (row_indices < nrows)
        
        z_v_raw = z_ref[veg_m][valid_idx]
        r_idx = row_indices[valid_idx]
        c_idx = col_indices[valid_idx]
        
        # Normalized height = Raw Z - DTM Z
        z_v_norm = z_v_raw - dtm[r_idx, c_idx]
        
        # Filter strictly for vegetation (above 3ft and under 150ft)
        valid_h = (z_v_norm >= HEIGHT_BINS[0]) & (z_v_norm <= HEIGHT_BINS[-1])
        r_idx = r_idx[valid_h]
        c_idx = c_idx[valid_h]
        z_v_norm = z_v_norm[valid_h]
        
        # 6. Calculate 3D Metrics per Pixel
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
            understory_arr[r, c] = counts[1] # Bin 1: 3-15ft
            midstory_arr[r, c] = counts[2]   # Bin 2: 15-50ft
            overstory_arr[r, c] = counts[3]  # Bin 3: 50-150ft
            
        # 7. Mask out Buildings to ensure no roof points are counted as trees
        tile_bbox = box(x_min, y_min, x_max, y_max)
        building_mask = rasterize_buildings_for_tile(
            buildings, buildings_sindex, tile_bbox, fhd_arr.shape, transform
        )
        
        fhd_arr[building_mask == 1] = -9999.0
        vci_arr[building_mask == 1] = -9999.0
        understory_arr[building_mask == 1] = -9999.0
        midstory_arr[building_mask == 1] = -9999.0
        overstory_arr[building_mask == 1] = -9999.0
            
        # 8. Export the 5-Band GeoTIFF
        band_names = ["Foliage_Height_Diversity", "Vertical_Complexity_Index", 
                      "Understory_Density", "Midstory_Density", "Overstory_Density"]
        
        arrays = [fhd_arr, vci_arr, understory_arr, midstory_arr, overstory_arr]
        write_multiband_raster(out_tif, arrays, transform, target_crs, band_names)
        
        print(f"  Success. 3D Metrics generated in {((time.time() - start_t) / 60.0):.2f} mins")
        
    except Exception as e:
        print(f"  Error processing {filename}: {e}")
        
    finally:
        # Clear variables and run garbage collection
        # del las_ref, las_be, x_ref, y_ref, z_ref, cls_ref, x_be, y_be, z_be
        # del dtm, z_v_raw, z_v_norm, fhd_arr, vci_arr, understory_arr, midstory_arr, overstory_arr
        gc.collect()

print("\nAll 2002 3D Structure processing completed.")