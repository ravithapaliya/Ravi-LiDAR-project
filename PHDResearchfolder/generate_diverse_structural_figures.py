import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. PATHS & HELPER FUNCTIONS
# =========================================================
vert_csv_folder = r"F:\MYPHDRESEARCHOUTPUTS\others\Neighborhood_3D_Summaries"
plot_folder = r"F:\MYPHDRESEARCHOUTPUTS\others\Diverse_RQ_Figures"
os.makedirs(plot_folder, exist_ok=True)

def get_bhm_csv_path(year):
    return fr"F:\MYPHDRESEARCHOUTPUTS\Countywide_Combined_Results_{year}\QOL_NPA_BuildingHeight_Min10ft_Max900ft_Outputs\QOL_NPA_Building_Height_Summary_Min10ft_Max900ft_{year}.csv"

def get_chm_csv_path(year):
    return fr"F:\MYPHDRESEARCHOUTPUTS\Countywide_Combined_Results_{year}\QOL_NPA_Min2m_Outputs\QOL_NPA_Canopy_Summary_Min2m_{year}.csv"

def find_height_column(df, is_chm=False):
    """Dynamically finds the mean height column whether it uses _m or _ft"""
    candidates = ['mean_canopy_height_m', 'mean_canopy_height_ft', 'mean_chm_height_m', 'mean_chm_height_ft'] if is_chm else \
                 ['mean_building_height_m', 'mean_building_height_ft', 'mean_bhm_height_m', 'mean_bhm_height_ft']
    for c in candidates:
        if c in df.columns: return c
    for c in df.columns:
        if 'mean' in c.lower() and 'height' in c.lower(): return c
    return None

# =========================================================
# 2. LOAD, MERGE, & NORMALIZE ALL YEARS
# =========================================================
print("Loading BHM, CHM, and 3D data...")
years = [2002, 2007, 2012, 2017, 2023]
all_data = []

for y in years:
    try:
        df_v = pd.read_csv(os.path.join(vert_csv_folder, f"NPA_3D_Structure_Summary_{y}.csv"))
        df_b = pd.read_csv(get_bhm_csv_path(y))
        df_c = pd.read_csv(get_chm_csv_path(y))

        bhm_col = find_height_column(df_b, is_chm=False)
        chm_col = find_height_column(df_c, is_chm=True)

        merged = df_v.merge(df_b[['NPA_ID', bhm_col, 'building_cover']], on='NPA_ID', how='inner')
        merged.rename(columns={bhm_col: 'mean_building_height'}, inplace=True)
        
        chm_cols = ['NPA_ID', 'canopy_cover']
        if chm_col: chm_cols.append(chm_col)
        merged = merged.merge(df_c[chm_cols], on='NPA_ID', how='inner')
        if chm_col: merged.rename(columns={chm_col: 'mean_canopy_height'}, inplace=True)
        else: merged['mean_canopy_height'] = 0.0

        # CRITICAL FIX: Normalize raw LiDAR point densities into Structural Fractions
        total_veg = merged['Total_Understory_Density'] + merged['Total_Midstory_Density'] + merged['Total_Overstory_Density']
        # Prevent division by zero in barren neighborhoods
        total_veg = total_veg.replace(0, np.nan) 
        
        merged['Understory_Fraction'] = (merged['Total_Understory_Density'] / total_veg).fillna(0)
        merged['Overstory_Fraction'] = (merged['Total_Overstory_Density'] / total_veg).fillna(0)
        
        merged['Year'] = y
        all_data.append(merged)
        
    except Exception as e:
        print(f"Skipping {y}: {e}")

df_master = pd.concat(all_data, ignore_index=True)

# Define Neighborhood Zones based on 2002 Baseline Density
df_2002 = df_master[df_master['Year'] == 2002].copy()
df_2002['Zone'] = pd.qcut(df_2002['building_cover'], q=3, labels=['Low Density (Suburban)', 'Medium Density', 'High Density (Urban Core)'])
zone_map = dict(zip(df_2002['NPA_ID'], df_2002['Zone']))
df_master['Neighborhood_Zone'] = df_master['NPA_ID'].map(zone_map)
df_master = df_master.dropna(subset=['Neighborhood_Zone'])

# =========================================================
# FIGURE 1: TEMPORAL TRENDS (USING NORMALIZED FRACTIONS)
# =========================================================
print("Generating Temporal Trendlines...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.set_theme(style="whitegrid")

metrics = [
    ('canopy_cover', 'Horizontal Canopy Cover (2D)', axes[0]),
    ('Understory_Fraction', 'Understory Structural Fraction (3-15ft)', axes[1]),
    ('Overstory_Fraction', 'Overstory Structural Fraction (>50ft)', axes[2])
]

for col, title, ax in metrics:
    sns.lineplot(data=df_master, x='Year', y=col, hue='Neighborhood_Zone', marker='o', ax=ax, errorbar=None, palette="Dark2")
    ax.set_title(f"Temporal Trend: {title}")
    ax.set_xticks(years)
    if ax != axes[0]: ax.get_legend().remove()

plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "Fig1_Temporal_Trends_By_Zone.png"), dpi=300)
plt.close()

# =========================================================
# 3. CALCULATE INTERVAL DELTAS FOR MODELING
# =========================================================
intervals = [(2002, 2007), (2007, 2012), (2012, 2017), (2017, 2023)]
interval_dfs = []

for y_start, y_end in intervals:
    df_s = df_master[df_master['Year'] == y_start].copy().add_suffix('_start')
    df_s.rename(columns={'NPA_ID_start': 'NPA_ID', 'Neighborhood_Zone_start': 'Zone'}, inplace=True)
    df_e = df_master[df_master['Year'] == y_end].copy().add_suffix('_end')
    df_e.rename(columns={'NPA_ID_end': 'NPA_ID'}, inplace=True)
    
    df_int = df_s.merge(df_e, on='NPA_ID', how='inner')
    
    # Calculate deltas using the robust fractions
    df_int['Delta_Understory_Frac'] = df_int['Understory_Fraction_end'] - df_int['Understory_Fraction_start']
    df_int['Delta_Overstory_Frac'] = df_int['Overstory_Fraction_end'] - df_int['Overstory_Fraction_start']
    df_int['Delta_Horizontal'] = df_int['canopy_cover_end'] - df_int['canopy_cover_start']
    df_int['Delta_CHM_Height'] = df_int['mean_canopy_height_end'] - df_int['mean_canopy_height_start']
    df_int['Delta_BHM_Height'] = df_int['mean_building_height_end'] - df_int['mean_building_height_start']
    df_int['Delta_Bldg_Cover'] = df_int['building_cover_end'] - df_int['building_cover_start']
    
    interval_dfs.append(df_int)

df_deltas = pd.concat(interval_dfs, ignore_index=True)

# =========================================================
# FIGURE 2: XGBOOST FEATURE IMPORTANCE
# =========================================================
print("Training XGBoost to extract Feature Importance...")
features = ['mean_building_height_start', 'building_cover_start', 'Delta_BHM_Height', 'Delta_Bldg_Cover']
feature_names = ['Baseline Bldg Height', 'Baseline Bldg Cover', 'Change in Bldg Height', 'Change in Bldg Cover']

targets = ['Delta_Horizontal', 'Delta_Understory_Frac', 'Delta_Overstory_Frac', 'Delta_CHM_Height']
importance_data = {}

for target in targets:
    df_model = df_deltas.dropna(subset=features + [target])
    X, y = df_model[features], df_model[target]
    
    # Absolute values for targets so we model the magnitude of disruption
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X, np.abs(y))
    importance_data[target] = model.feature_importances_

fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.2
x = np.arange(len(features))

ax.bar(x - bar_width*1.5, importance_data['Delta_Horizontal'], bar_width, label='Drives 2D Canopy Change', color='skyblue')
ax.bar(x - bar_width*0.5, importance_data['Delta_CHM_Height'], bar_width, label='Drives CHM Height Change', color='steelblue')
ax.bar(x + bar_width*0.5, importance_data['Delta_Understory_Frac'], bar_width, label='Drives 3D Understory Change', color='lightgreen')
ax.bar(x + bar_width*1.5, importance_data['Delta_Overstory_Frac'], bar_width, label='Drives 3D Overstory Change', color='forestgreen')

ax.set_xticks(x)
ax.set_xticklabels(feature_names, rotation=15, ha='right')
ax.set_title("Which Urban Factors Drive Different Types of Canopy Loss? (XGBoost)")
ax.set_ylabel("Relative Importance")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "Fig2_Driver_Importance_BarChart.png"), dpi=300)
plt.close()

# =========================================================
# FIGURE 3: BIVARIATE SCATTER PLOTS
# =========================================================
print("Generating Bivariate Scatter Plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.scatterplot(data=df_deltas, x='Delta_Bldg_Cover', y='Delta_Understory_Frac', hue='Zone', alpha=0.7, ax=axes[0], palette="Dark2")
axes[0].set_title("Sprawl (Bldg Cover) vs Understory Fraction")
axes[0].set_xlabel("Increase in Building Cover Fraction")
axes[0].set_ylabel("Change in Understory Fraction")

sns.scatterplot(data=df_deltas, x='Delta_BHM_Height', y='Delta_Overstory_Frac', hue='Zone', alpha=0.7, ax=axes[1], palette="Dark2")
axes[1].set_title("Vertical Densification (BHM) vs Overstory Fraction")
axes[1].set_xlabel("Increase in Mean Building Height")
axes[1].set_ylabel("Change in Overstory Fraction")

plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "Fig3_BHM_vs_Structure_Scatters.png"), dpi=300)
plt.close()

print(f"\nSuccess! 3 diverse, normalized, publication-ready figures saved to {plot_folder}")