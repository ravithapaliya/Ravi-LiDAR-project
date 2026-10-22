import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. PATHS TO ALL FOUR DATASETS
# =========================================================
vert_csv_folder = r"F:\MYPHDRESEARCHOUTPUTS\others\Neighborhood_3D_Summaries"
lst_csv_folder  = r"F:\MYPHDRESEARCHOUTPUTS\others\LST_Summaries"
plot_folder     = r"F:\MYPHDRESEARCHOUTPUTS\others\Final_RQ_Figures"
os.makedirs(plot_folder, exist_ok=True)

def get_bhm_csv_path(year):
    return fr"F:\MYPHDRESEARCHOUTPUTS\Countywide_Combined_Results_{year}\QOL_NPA_BuildingHeight_Min10ft_Max900ft_Outputs\QOL_NPA_Building_Height_Summary_Min10ft_Max900ft_{year}.csv"

def get_chm_csv_path(year):
    return fr"F:\MYPHDRESEARCHOUTPUTS\Countywide_Combined_Results_{year}\QOL_NPA_Min2m_Outputs\QOL_NPA_Canopy_Summary_Min2m_{year}.csv"

lst_year_map = {
    2002: "2000", 
    2007: "2005", 
    2012: "2010", 
    2017: "2015", 
    2023: "2020"
}

def get_lst_csv_path(structural_year):
    lst_yr = lst_year_map.get(structural_year)
    search_pattern = os.path.join(lst_csv_folder, f"*_{lst_yr}_summer*.csv")
    matches = glob.glob(search_pattern)
    if matches:
        return matches[0] 
    return None

def find_height_column(df, is_chm=False):
    """Dynamically finds the mean height column whether it uses _m or _ft"""
    candidates = ['mean_canopy_height_m', 'mean_canopy_height_ft', 'mean_chm_height_m', 'mean_chm_height_ft'] if is_chm else \
                 ['mean_building_height_m', 'mean_building_height_ft', 'mean_bhm_height_m', 'mean_bhm_height_ft']
    
    for c in candidates:
        if c in df.columns:
            return c
            
    # Fallback if specific names aren't found
    for c in df.columns:
        if 'mean' in c.lower() and 'height' in c.lower():
            return c
    return None

# =========================================================
# 2. LOAD & MERGE ALL DATASETS
# =========================================================
print("Loading and merging 3D Voxel, CHM, BHM, and LST data...")
years = [2002, 2007, 2012, 2017, 2023]
all_data = []

for y in years:
    try:
        df_v = pd.read_csv(os.path.join(vert_csv_folder, f"NPA_3D_Structure_Summary_{y}.csv"))
        df_b = pd.read_csv(get_bhm_csv_path(y))
        df_c = pd.read_csv(get_chm_csv_path(y))
        
        lst_path = get_lst_csv_path(y)
        if not lst_path:
            print(f"Warning: No summer LST file found for structural year {y}. Skipping year.")
            continue
        df_l = pd.read_csv(lst_path)

        bhm_height_col = find_height_column(df_b, is_chm=False)
        chm_height_col = find_height_column(df_c, is_chm=True)

        if not chm_height_col:
            print(f"Warning: Could not find CHM height column in {y}. Target bar will be blank.")

        # Merge BHM
        merged = df_v.merge(df_b[['NPA_ID', bhm_height_col, 'building_cover']], on='NPA_ID', how='inner')
        merged.rename(columns={bhm_height_col: 'mean_building_height'}, inplace=True)
        
        # Merge CHM
        chm_cols = ['NPA_ID', 'canopy_cover']
        if chm_height_col:
            chm_cols.append(chm_height_col)
            
        merged = merged.merge(df_c[chm_cols], on='NPA_ID', how='inner')
        if chm_height_col:
            merged.rename(columns={chm_height_col: 'mean_canopy_height'}, inplace=True)
        else:
            merged['mean_canopy_height'] = 0.0
            
        # Merge LST
        merged = merged.merge(df_l[['NPA_ID', 'mean_lst']], on='NPA_ID', how='inner')
        merged['Year'] = y
        all_data.append(merged)
        
    except Exception as e:
        print(f"Skipping {y} due to error: {e}")

df_master = pd.concat(all_data, ignore_index=True)

df_2002 = df_master[df_master['Year'] == 2002].copy()
df_2002['Zone'] = pd.qcut(df_2002['building_cover'], q=3, labels=['Low Density (Suburbs)', 'Medium Density', 'High Density (Urban Core)'])
zone_map = dict(zip(df_2002['NPA_ID'], df_2002['Zone']))
df_master['Neighborhood_Zone'] = df_master['NPA_ID'].map(zone_map)
df_master = df_master.dropna(subset=['Neighborhood_Zone'])

# =========================================================
# FIGURE 1: TEMPORAL TRAJECTORIES (STRUCTURE VS HEAT)
# =========================================================
print("Generating Figure 1: Structural & Thermal Trajectories...")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics = [
    ('canopy_cover', 'Horizontal Canopy Cover (CHM 2D)', axes[0]),
    ('Total_Understory_Density', 'Vertical Understory Density (3D)', axes[1]),
    ('mean_lst', 'Mean Land Surface Temp (LST °C)', axes[2])
]

for col, title, ax in metrics:
    sns.lineplot(data=df_master, x='Year', y=col, hue='Neighborhood_Zone', marker='o', ax=ax, errorbar=None, palette="Dark2")
    ax.set_title(f"{title}")
    ax.set_xticks([2002, 2007, 2012, 2017, 2023])
    if ax != axes[0]: ax.get_legend().remove()

plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "Fig1_Trajectories_Structure_and_Heat.png"), dpi=300)
plt.close()

# =========================================================
# 3. CALCULATE INTERVAL DELTAS
# =========================================================
intervals = [(2002, 2023)]
interval_dfs = []

for y_start, y_end in intervals:
    df_s = df_master[df_master['Year'] == y_start].copy().add_suffix('_start')
    df_s.rename(columns={'NPA_ID_start': 'NPA_ID', 'Neighborhood_Zone_start': 'Zone'}, inplace=True)
    df_e = df_master[df_master['Year'] == y_end].copy().add_suffix('_end')
    df_e.rename(columns={'NPA_ID_end': 'NPA_ID'}, inplace=True)
    
    df_int = df_s.merge(df_e, on='NPA_ID', how='inner')
    
    df_int['Delta_Understory'] = df_int['Total_Understory_Density_end'] - df_int['Total_Understory_Density_start']
    df_int['Delta_Overstory'] = df_int['Total_Overstory_Density_end'] - df_int['Total_Overstory_Density_start']
    df_int['Delta_FHD'] = df_int['Mean_FHD_end'] - df_int['Mean_FHD_start']
    df_int['Delta_Horizontal'] = df_int['canopy_cover_end'] - df_int['canopy_cover_start']
    df_int['Delta_CHM_Height'] = df_int['mean_canopy_height_end'] - df_int['mean_canopy_height_start']
    df_int['Delta_BHM_Height'] = df_int['mean_building_height_end'] - df_int['mean_building_height_start']
    df_int['Delta_Bldg_Cover'] = df_int['building_cover_end'] - df_int['building_cover_start']
    df_int['Delta_LST'] = df_int['mean_lst_end'] - df_int['mean_lst_start']
    
    interval_dfs.append(df_int)

df_deltas = pd.concat(interval_dfs, ignore_index=True)

# =========================================================
# FIGURE 2: DENSIFICATION VS UNDERSTORY VS HEAT (SCATTER)
# =========================================================
print("Generating Figure 2: Sprawl, Understory, and Heat Scatter...")
plt.figure(figsize=(10, 6))

scatter = sns.scatterplot(
    data=df_deltas, 
    x='Delta_Bldg_Cover', 
    y='Delta_Understory', 
    hue='Delta_LST', 
    palette='coolwarm',
    size='Delta_LST',
    sizes=(20, 150),
    alpha=0.8
)

plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.title("The Densification Trade-off (2002-2023): Sprawl, Understory Loss, and Heat")
plt.xlabel("Increase in Building Cover Fraction (Sprawl)")
plt.ylabel("Change in Understory Density (3-15ft)")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Increase in\nTemp (°C)")
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "Fig2_Densification_Understory_Heat_Scatter.png"), dpi=300)
plt.close()

# =========================================================
# FIGURE 3: XGBOOST FEATURE IMPORTANCE 
# =========================================================
print("Generating Figure 3: XGBoost Horizontal vs Vertical Heat Drivers...")
features = ['Delta_Horizontal', 'Delta_Understory', 'Delta_Overstory', 'Delta_FHD', 'Delta_CHM_Height']
feature_names = [
    '2D Horizontal Canopy Loss', 
    '3D Understory Density Loss', 
    '3D Overstory Density Loss', 
    'Foliage Height Diversity Loss',
    'Overall Canopy Height Loss (CHM)'
]

df_model = df_deltas.dropna(subset=features + ['Delta_LST'])
X = df_model[features]
X_inverted = X * -1 
y = df_model['Delta_LST']

model = xgb.XGBRegressor(n_estimators=150, max_depth=3, random_state=42)
model.fit(X_inverted, y)

importance_scores = model.feature_importances_

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=importance_scores, y=feature_names, palette="magma", ax=ax)
ax.set_title("Which Structural Losses Drive the Highest Heat Penalties? (XGBoost)")
ax.set_xlabel("Relative Importance in Predicting LST Increase")

plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "Fig3_Horizontal_vs_Vertical_Heat_Drivers.png"), dpi=300)
plt.close()

print(f"\nSuccess! 3 customized, publication-ready figures for your RQ saved to {plot_folder}")