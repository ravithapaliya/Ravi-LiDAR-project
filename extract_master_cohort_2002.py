import os
import glob
import pandas as pd
import geopandas as gpd

input_folder = r"F:\MYPHDRESEARCHOUTPUTS\Countywide_Combined_Results_2002\QOL_NPA_Min2m_Outputs"
output_file = r"F:\MYPHDRESEARCHOUTPUTS\Master_Tree_ID_Cohort_2002.shp"

search_pattern = os.path.join(input_folder, "NPA_*", "*.shp")
shp_files = glob.glob(search_pattern)

gdfs = []
for shp in shp_files:
    try:
        gdf = gpd.read_file(shp)
        if not gdf.empty:
            gdfs.append(gdf)
    except Exception as e:
        pass

master_gdf = pd.concat(gdfs, ignore_index=True)
master_gdf = gpd.GeoDataFrame(master_gdf, geometry='geometry', crs=gdfs[0].crs)

if 'height_m' not in master_gdf.columns and 'height_ft' in master_gdf.columns:
    master_gdf['height_m'] = master_gdf['height_ft'] * 0.3048

primary_cohort = master_gdf[(master_gdf['height_m'] >= 2.0) & (master_gdf['height_m'] <= 6.0)]

if len(primary_cohort) < 1000:
    final_cohort = master_gdf[(master_gdf['height_m'] >= 2.0) & (master_gdf['height_m'] <= 12.0)]
else:
    final_cohort = primary_cohort

final_cohort = final_cohort.copy()
final_cohort['Tree_ID'] = range(1, len(final_cohort) + 1)
final_cohort['geometry'] = final_cohort.geometry.centroid

final_cohort.to_file(output_file)