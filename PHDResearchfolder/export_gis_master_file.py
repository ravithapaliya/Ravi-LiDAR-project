import os
import pandas as pd
import numpy as np
import geopandas as gpd
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. PATHS
# =========================================================
vert_csv_folder = r"F:\MYPHDRESEARCHOUTPUTS\others\Neighborhood_3D_Summaries"
qol_shp_path = r"C:\Users\ravit\Documents\tree research work new\QOL_NPA_2020_final_projected\QOL_NPA_2020_final_projected.shp"

gis_output_folder = r"F:\MYPHDRESEARCHOUTPUTS\others\GIS_Spatial_Results"
os.makedirs(gis_output_folder, exist_ok=True)

def get_bhm_csv_path(year):
    return fr"F:\MYPHDRESEARCHOUTPUTS\Countywide_Combined_Results_{year}\QOL_NPA_BuildingHeight_Min10ft_Max900ft_Outputs\QOL_NPA_Building_Height_Summary_Min10ft_Max900ft_{year}.csv"

def get_chm_csv_path(year):
    return fr"F:\MYPHDRESEARCHOUTPUTS\Countywide_Combined_Results_{year}\QOL_NPA_Min2m_Outputs\QOL_NPA_Canopy_Summary_Min2m_{year}.csv"

def find_height_column(df, is_chm=False):
    candidates = ['mean_canopy_height_m', 'mean_canopy_height_ft', 'mean_chm_height_m', 'mean_chm_height_ft'] if is_chm else \
                 ['mean_building_height_m', 'mean_building_height_ft', 'mean_bhm_height_m', 'mean_bhm_height_ft']
    for c in candidates:
        if c in df.columns: return c
    for c in df.columns:
        if 'mean' in c.lower() and 'height' in c.lower(): return c
    return None

# =========================================================
# 2. LOAD 2002 (START) AND 2023 (END) DATA
# =========================================================
print("Loading 2002 and 2023 structural data for GIS and CSV export...")
years = [2002, 2023]
data_dict = {}

for y in years:
    df_v = pd.read_csv(os.path.join(vert_csv_folder, f"NPA_3D_Structure_Summary_{y}.csv"))
    df_b = pd.read_csv(get_bhm_csv_path(y))
    df_c = pd.read_csv(get_chm_csv_path(y))

    bhm_col = find_height_column(df_b, is_chm=False)
    chm_col = find_height_column(df_c, is_chm=True)

    merged = df_v.merge(df_b[['NPA_ID', bhm_col, 'building_cover']], on='NPA_ID', how='inner')
    merged.rename(columns={bhm_col: 'mean_bldg_ht'}, inplace=True)
    
    chm_cols = ['NPA_ID', 'canopy_cover']
    if chm_col: chm_cols.append(chm_col)
    merged = merged.merge(df_c[chm_cols], on='NPA_ID', how='inner')
    if chm_col: merged.rename(columns={chm_col: 'mean_canopy_ht'}, inplace=True)
    else: merged['mean_canopy_ht'] = 0.0

    # Calculate normalized fractions
    total_veg = merged['Total_Understory_Density'] + merged['Total_Midstory_Density'] + merged['Total_Overstory_Density']
    total_veg = total_veg.replace(0, np.nan)
    merged['Under_Frac'] = (merged['Total_Understory_Density'] / total_veg).fillna(0)
    merged['Over_Frac'] = (merged['Total_Overstory_Density'] / total_veg).fillna(0)
    
    data_dict[y] = merged

# =========================================================
# 3. CALCULATE DELTAS (2002 to 2023)
# =========================================================
df_start = data_dict[2002].copy().add_suffix('_02')
df_start.rename(columns={'NPA_ID_02': 'NPA_ID'}, inplace=True)

df_end = data_dict[2023].copy().add_suffix('_23')
df_end.rename(columns={'NPA_ID_23': 'NPA_ID'}, inplace=True)

df_deltas = df_start.merge(df_end, on='NPA_ID', how='inner')

# Calculate exact changes
df_deltas['Chg_Under'] = df_deltas['Under_Frac_23'] - df_deltas['Under_Frac_02']
df_deltas['Chg_Over'] = df_deltas['Over_Frac_23'] - df_deltas['Over_Frac_02']
df_deltas['Chg_2D_Cov'] = df_deltas['canopy_cover_23'] - df_deltas['canopy_cover_02']
df_deltas['Chg_CHM_Ht'] = df_deltas['mean_canopy_ht_23'] - df_deltas['mean_canopy_ht_02']
df_deltas['Chg_BHM_Ht'] = df_deltas['mean_bldg_ht_23'] - df_deltas['mean_bldg_ht_02']
df_deltas['Chg_Bldg_Cov'] = df_deltas['building_cover_23'] - df_deltas['building_cover_02']

# Define Neighborhood Zones based on baseline density
df_deltas['Zone_Num'] = pd.qcut(df_deltas['building_cover_02'], q=3, labels=[1, 2, 3]).astype(int)
df_deltas['Zone_Desc'] = df_deltas['Zone_Num'].map({1: 'Low Density', 2: 'Medium Density', 3: 'Urban Core'})

# Keep only the columns necessary for GIS and review
export_cols = [
    'NPA_ID', 'Zone_Desc', 'Zone_Num',
    'Chg_Under', 'Chg_Over', 'Chg_2D_Cov', 'Chg_CHM_Ht', 
    'Chg_BHM_Ht', 'Chg_Bldg_Cov',
    'Under_Frac_02', 'Under_Frac_23', 'canopy_cover_02', 'canopy_cover_23',
    'building_cover_02', 'building_cover_23'
]
df_final = df_deltas[export_cols].copy()

# =========================================================
# 4. EXPORT TO CSV AND GEOPACKAGE
# =========================================================
# Export 1: Standard CSV for viewing in Excel
output_csv = os.path.join(gis_output_folder, "Mecklenburg_Structural_Drivers_2002_2023.csv")
df_final.to_csv(output_csv, index=False)
print(f"Saved CSV for data review: {output_csv}")

# Export 2: GeoPackage for spatial mapping
print("Merging with QOL polygons for GIS export...")
try:
    qol = gpd.read_file(qol_shp_path)
    qol['NPA_ID'] = qol['NPA_ID'].astype(int)

    gdf_export = qol.merge(df_final, on='NPA_ID', how='inner')

    output_gpkg = os.path.join(gis_output_folder, "Mecklenburg_Structural_Drivers_2002_2023.gpkg")
    gdf_export.to_file(output_gpkg, driver="GPKG")
    print(f"Saved Map-ready GeoPackage: {output_gpkg}")
except Exception as e:
    print(f"Warning: Could not create GeoPackage. Ensure your QOL shapefile path is correct. Error: {e}")

print("\nExport complete!")