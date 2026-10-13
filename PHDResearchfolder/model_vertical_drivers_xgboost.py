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
# 1. LOAD AND MERGE THE 5 CSVS
# =========================================================
csv_folder = r"F:\MYPHDRESEARCHOUTPUTS\Neighborhood_3D_Summaries"
years = [2002, 2007, 2012, 2017, 2023]

# Load all years into a single dataframe
dfs = []
for y in years:
    file_path = os.path.join(csv_folder, f"NPA_3D_Structure_Summary_{y}.csv")
    if os.path.exists(file_path):
        df_year = pd.read_csv(file_path)
        dfs.append(df_year)

df_all = pd.concat(dfs, ignore_index=True)

# Pivot the data to a "wide" format so each NPA is one row with columns for every year
df_wide = df_all.pivot(index='NPA_ID', columns='Year')
df_wide.columns = [f"{col[0]}_{col[1]}" for col in df_wide.columns]
df_wide.reset_index(inplace=True)

# =========================================================
# 2. CALCULATE THE DELTAS (TARGET VARIABLES)
# =========================================================
# We will look at the long-term structural change from 2002 to 2023
df_wide['Delta_FHD'] = df_wide['Mean_FHD_2023'] - df_wide['Mean_FHD_2002']
df_wide['Delta_Understory'] = df_wide['Total_Understory_Density_2023'] - df_wide['Total_Understory_Density_2002']
df_wide['Delta_Overstory'] = df_wide['Total_Overstory_Density_2023'] - df_wide['Total_Overstory_Density_2002']

# =========================================================
# 3. MOCK INTEGRATION OF URBAN PREDICTORS (YOUR "X" FACTORS)
# =========================================================
# In your actual research, you will merge your Building Height (BHM) and urban form CSVs here.
# For demonstration, we will use the baseline 2002 forest states as internal predictors, 
# but you MUST replace the mock columns below with your real BHM/urbanization data.

# MOCK DATA: df_urban = pd.read_csv("Mecklenburg_BHM_Neighborhood_Summaries.csv")
# MOCK MERGE: df_model = pd.merge(df_wide, df_urban, on="NPA_ID")

df_model = df_wide.copy()
# Simulating urban driver variables you have generated in previous steps:
np.random.seed(42)
df_model['Delta_Building_Height_ft'] = np.random.uniform(0, 40, size=len(df_model))
df_model['Delta_Building_Density'] = np.random.uniform(0, 0.3, size=len(df_model))
df_model['Mean_SVF_2023'] = np.random.uniform(0.2, 0.9, size=len(df_model))

# Define our features (X) and target (y)
# Let's predict what caused the UNDERSTORY to change (you can run this again for Overstory or FHD)
features = [
    'Mean_FHD_2002',              # Baseline complexity
    'Total_Understory_Density_2002', # Baseline understory
    'Delta_Building_Height_ft',   # Urbanization pressure 1
    'Delta_Building_Density',     # Urbanization pressure 2
    'Mean_SVF_2023'               # Built environment proxy
]

# Drop any rows with missing data
df_model = df_model.dropna(subset=features + ['Delta_Understory'])

X = df_model[features]
y = df_model['Delta_Understory']

# =========================================================
# 4. XGBOOST MODEL TRAINING
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining XGBoost Model to predict changes in Understory Density...")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
print(f"Model R-squared: {r2_score(y_test, y_pred):.3f}")
print(f"Model RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

# =========================================================
# 5. SHAP ANALYSIS FOR NOVEL INSIGHTS
# =========================================================
print("\nGenerating SHAP values to explain specific neighborhood drivers...")
explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X)

# Create output folder for plots
plot_folder = r"F:\MYPHDRESEARCHOUTPUTS\XGBoost_SHAP_Plots"
os.makedirs(plot_folder, exist_ok=True)

# Generate Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.title("Drivers of Understory Canopy Change (2002-2023)")
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "SHAP_Summary_Understory.png"), dpi=300)
plt.close()

# Generate Dependence Plot for Building Height
shap.dependence_plot("Delta_Building_Height_ft", shap_values.values, X, show=False)
plt.title("Tipping Point: Building Height vs Understory Loss")
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "SHAP_Dependence_BHM_vs_Understory.png"), dpi=300)
plt.close()

print(f"SHAP plots saved to {plot_folder}")