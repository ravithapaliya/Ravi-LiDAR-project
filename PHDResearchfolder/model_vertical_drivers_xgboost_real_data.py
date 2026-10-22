import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. DEFINE PATHS TO YOUR ACTUAL GENERATED CSVS
# =========================================================
# 3D Vertical Structure CSVs
vert_csv_folder = r"F:\MYPHDRESEARCHOUTPUTS\others\Neighborhood_3D_Summaries"

# Function to generate your specific BHM and CHM paths
def get_bhm_csv_path(year):
    return fr"F:\MYPHDRESEARCHOUTPUTS\Countywide_Combined_Results_{year}\QOL_NPA_BuildingHeight_Min10ft_Max900ft_Outputs\QOL_NPA_Building_Height_Summary_Min10ft_Max900ft_{year}.csv"

def get_chm_csv_path(year):
    return fr"F:\MYPHDRESEARCHOUTPUTS\Countywide_Combined_Results_{year}\QOL_NPA_Min2m_Outputs\QOL_NPA_Canopy_Summary_Min2m_{year}.csv"

# =========================================================
# 2. LOAD AND STACK ALL 5 YEARS INTO INTERVALS
# =========================================================
print("Loading all datasets for 2002, 2007, 2012, 2017, and 2023...")
years = [2002, 2007, 2012, 2017, 2023]
all_data = {}

# Load and merge the 3 CSVs (Vertical, BHM, CHM) for every individual year
for y in years:
    try:
        # Load Vertical Structure
        df_v = pd.read_csv(os.path.join(vert_csv_folder, f"NPA_3D_Structure_Summary_{y}.csv"))
        # Load Building Height Models (BHM)
        df_b = pd.read_csv(get_bhm_csv_path(y))
        # Load Canopy Height Models (CHM)
        df_c = pd.read_csv(get_chm_csv_path(y))

        # Merge them together for this specific year
        merged = df_v.merge(df_b[['NPA_ID', 'mean_building_height_ft', 'building_cover']], on='NPA_ID', how='inner')
        merged = merged.merge(df_c[['NPA_ID', 'canopy_cover']], on='NPA_ID', how='inner')
        
        all_data[y] = merged
    except Exception as e:
        print(f"Warning: Could not load data for {y}. Error: {e}")

# Create Stacked Intervals (02-07, 07-12, 12-17, 17-23)
print("Calculating interval changes (Deltas) across all epochs...")
intervals = [(2002, 2007), (2007, 2012), (2012, 2017), (2017, 2023)]
interval_dfs = []

for y_start, y_end in intervals:
    if y_start in all_data and y_end in all_data:
        df_start = all_data[y_start].copy().add_suffix('_start')
        df_start.rename(columns={'NPA_ID_start': 'NPA_ID'}, inplace=True)
        
        df_end = all_data[y_end].copy().add_suffix('_end')
        df_end.rename(columns={'NPA_ID_end': 'NPA_ID'}, inplace=True)
        
        df_int = df_start.merge(df_end, on='NPA_ID', how='inner')
        df_int['Time_Interval'] = f"{y_start}-{y_end}"
        
        # Calculate the Change Variables (Deltas) for this specific interval
        df_int['Delta_FHD'] = df_int['Mean_FHD_end'] - df_int['Mean_FHD_start']
        df_int['Delta_Understory'] = df_int['Total_Understory_Density_end'] - df_int['Total_Understory_Density_start']
        df_int['Delta_Overstory'] = df_int['Total_Overstory_Density_end'] - df_int['Total_Overstory_Density_start']
        
        df_int['Delta_Building_Height_ft'] = df_int['mean_building_height_ft_end'] - df_int['mean_building_height_ft_start']
        df_int['Delta_Building_Cover'] = df_int['building_cover_end'] - df_int['building_cover_start']
        
        interval_dfs.append(df_int)

# Combine all intervals into a massive master dataset for XGBoost
df_master = pd.concat(interval_dfs, ignore_index=True)

# =========================================================
# 3. CREATE NEIGHBORHOOD ZONES (CRASH-PROOF METHOD)
# =========================================================
print("Classifying neighborhoods into Urban Zones...")
# We use pandas qcut to safely divide neighborhoods into 3 equal bins based on their baseline building density
# 0 = Low Density (Suburbs), 1 = Medium Density, 2 = High Density (Urban Core)
df_master['Neighborhood_Zone_Code'] = pd.qcut(df_master['building_cover_start'], q=3, labels=[0, 1, 2]).astype(int)

# Define our features (X)
features = [
    'Mean_FHD_start', 
    'Total_Understory_Density_start', 
    'Total_Overstory_Density_start',
    'building_cover_start',
    'Delta_Building_Height_ft', 
    'Delta_Building_Cover', 
    'Neighborhood_Zone_Code'
]

# Drop missing data
df_model = df_master.dropna(subset=features + ['Delta_Understory', 'Delta_Overstory', 'Delta_FHD']).copy()

# =========================================================
# 4. MULTI-TARGET XGBOOST & SHAP ANALYSIS
# =========================================================
targets = {
    "Understory_Change": "Delta_Understory",
    "Overstory_Change": "Delta_Overstory",
    "Vertical_Complexity_Change": "Delta_FHD"
}

plot_folder = r"F:\MYPHDRESEARCHOUTPUTS\XGBoost_SHAP_Plots"
os.makedirs(plot_folder, exist_ok=True)

for target_name, target_col in targets.items():
    print(f"\n--- Modeling {target_name} ---")
    
    X = df_model[features]
    y = df_model[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=4, 
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    print(f"R-squared: {r2_score(y_test, y_pred):.3f} | RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

    # SHAP Explanation
    explainer = shap.Explainer(xgb_model, X_train)
    shap_values = explainer(X)

    # 1. Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"Drivers of {target_name} (Multi-Decade Intervals)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"SHAP_Summary_{target_name}.png"), dpi=300)
    plt.close()

    # 2. Dependence Plot: Building Densification vs Change (Colored by Zone)
    shap.dependence_plot(
        "Delta_Building_Cover", 
        shap_values.values, 
        X, 
        interaction_index="Neighborhood_Zone_Code", 
        show=False,
        cmap=plt.get_cmap("viridis")
    )
    plt.title(f"Tipping Point: Building Densification vs {target_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"SHAP_Dependence_BldgCover_vs_{target_name}.png"), dpi=300)
    plt.close()

print(f"\nAll models completed! SHAP plots saved to {plot_folder}")