import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. PATHS TO STRUCTURAL DATA & YOUR LST DATA
# =========================================================
vert_csv_folder = r"F:\MYPHDRESEARCHOUTPUTS\others\Neighborhood_3D_Summaries"

# IMPORTANT: Update this to point to your actual Landsat LST neighborhood summaries
lst_csv_folder = r"F:\MYPHDRESEARCHOUTPUTS\others\LST_Summaries" 

def get_bhm_csv_path(year):
    return fr"F:\MYPHDRESEARCHOUTPUTS\Countywide_Combined_Results_{year}\QOL_NPA_BuildingHeight_Min10ft_Max900ft_Outputs\QOL_NPA_Building_Height_Summary_Min10ft_Max900ft_{year}.csv"

def get_chm_csv_path(year):
    return fr"F:\MYPHDRESEARCHOUTPUTS\Countywide_Combined_Results_{year}\QOL_NPA_Min2m_Outputs\QOL_NPA_Canopy_Summary_Min2m_{year}.csv"

# =========================================================
# 2. LOAD, MERGE, AND CREATE INTERVALS
# =========================================================
years = [2002, 2007, 2012, 2017, 2023]
all_data = {}

for y in years:
    try:
        df_v = pd.read_csv(os.path.join(vert_csv_folder, f"NPA_3D_Structure_Summary_{y}.csv"))
        df_b = pd.read_csv(get_bhm_csv_path(y))
        df_c = pd.read_csv(get_chm_csv_path(y))
        
        # Load your Landsat LST data (assuming columns: NPA_ID, mean_lst)
        # df_lst = pd.read_csv(os.path.join(lst_csv_folder, f"QOL_NPA_LST_Summary_{y}.csv"))

        merged = df_v.merge(df_b[['NPA_ID', 'mean_building_height_ft', 'building_cover']], on='NPA_ID', how='inner')
        merged = merged.merge(df_c[['NPA_ID', 'canopy_cover']], on='NPA_ID', how='inner')
        
        # MOCK LST DATA FOR SCRIPT TESTING (Replace with actual merge above)
        np.random.seed(y)
        merged['mean_lst'] = np.random.uniform(30, 45, size=len(merged))
        
        all_data[y] = merged
    except Exception as e:
        pass

intervals = [(2002, 2007), (2007, 2012), (2012, 2017), (2017, 2023)]
interval_dfs = []

for y_start, y_end in intervals:
    if y_start in all_data and y_end in all_data:
        df_start = all_data[y_start].copy().add_suffix('_start')
        df_start.rename(columns={'NPA_ID_start': 'NPA_ID'}, inplace=True)
        df_end = all_data[y_end].copy().add_suffix('_end')
        df_end.rename(columns={'NPA_ID_end': 'NPA_ID'}, inplace=True)
        
        df_int = df_start.merge(df_end, on='NPA_ID', how='inner')
        
        # Predictor Deltas (Structural Changes)
        df_int['Delta_Understory'] = df_int['Total_Understory_Density_end'] - df_int['Total_Understory_Density_start']
        df_int['Delta_Overstory'] = df_int['Total_Overstory_Density_end'] - df_int['Total_Overstory_Density_start']
        df_int['Delta_Horizontal_Canopy'] = df_int['canopy_cover_end'] - df_int['canopy_cover_start']
        df_int['Delta_Building_Cover'] = df_int['building_cover_end'] - df_int['building_cover_start']
        
        # TARGET: Change in Land Surface Temperature
        df_int['Delta_LST'] = df_int['mean_lst_end'] - df_int['mean_lst_start']
        
        interval_dfs.append(df_int)

df_master = pd.concat(interval_dfs, ignore_index=True)
df_master['Neighborhood_Zone_Code'] = pd.qcut(df_master['building_cover_start'], q=3, labels=[0, 1, 2]).astype(int)

# =========================================================
# 3. XGBOOST: HOW STRUCTURE IMPACTS HEAT
# =========================================================
features = [
    'Delta_Understory',
    'Delta_Overstory',
    'Delta_Horizontal_Canopy',
    'Delta_Building_Cover',
    'Neighborhood_Zone_Code'
]

df_model = df_master.dropna(subset=features + ['Delta_LST']).copy()
X = df_model[features]
y = df_model['Delta_LST']

print("\n--- Modeling Impacts of 3D/2D Canopy Change on LST ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train, y_train)

explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X)

plot_folder = r"F:\MYPHDRESEARCHOUTPUTS\others\XGBoost_SHAP_Plots"

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.title("Drivers of Land Surface Temperature Change (LST)")
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "SHAP_Summary_LST_Impacts.png"), dpi=300)
plt.close()

# Evaluate Understory vs Heat
shap.dependence_plot("Delta_Understory", shap_values.values, X, interaction_index="Delta_Building_Cover", show=False, cmap=plt.get_cmap("coolwarm"))
plt.title("Understory Loss vs LST Change")
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "SHAP_Dependence_Understory_vs_LST.png"), dpi=300)
plt.close()

print(f"Thermal models completed. Saved to {plot_folder}")