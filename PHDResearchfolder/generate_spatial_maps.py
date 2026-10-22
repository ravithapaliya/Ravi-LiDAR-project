import os
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. PATHS
# =========================================================
gis_output_folder = r"F:\MYPHDRESEARCHOUTPUTS\others\GIS_Spatial_Results"
gpkg_path = os.path.join(gis_output_folder, "Mecklenburg_Structural_Drivers_2002_2023.gpkg")
plot_folder = r"F:\MYPHDRESEARCHOUTPUTS\others\Final_RQ_Figures"

os.makedirs(plot_folder, exist_ok=True)

# =========================================================
# 2. LOAD SPATIAL DATA
# =========================================================
print("Loading spatial GeoPackage...")
try:
    gdf = gpd.read_file(gpkg_path)
except Exception as e:
    print(f"Error loading GeoPackage. Ensure you ran the export script first. {e}")
    exit()

# =========================================================
# 3. GENERATE 3-PANEL CHOROPLETH MAP
# =========================================================
print("Generating spatial visualizations...")
fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='white')

# Set a shared color range for the canopy maps so they are directly comparable
# Canopy loss will be brown/red, canopy gain will be green
vmin_canopy = min(gdf['Chg_Under'].min(), gdf['Chg_2D_Cov'].min())
vmax_canopy = max(gdf['Chg_Under'].max(), gdf['Chg_2D_Cov'].max())

# Map 1: Sprawl (Increase in Building Cover)
gdf.plot(
    column='Chg_Bldg_Cov', 
    ax=axes[0], 
    cmap='Reds', 
    legend=True,
    legend_kwds={'label': "Increase in Building Cover Fraction\n(2002 - 2023)", 'orientation': "horizontal"}
)
axes[0].set_title("Urban Sprawl & Densification", fontsize=16, fontweight='bold')
axes[0].axis('off')

# Map 2: 2D Horizontal Canopy Loss
gdf.plot(
    column='Chg_2D_Cov', 
    ax=axes[1], 
    cmap='BrBG',  # Brown to Blue-Green
    vmin=-0.3, # Standardize the scale for visual contrast
    vmax=0.3,
    legend=True,
    legend_kwds={'label': "Change in 2D Canopy Fraction\n(Negative = Loss)", 'orientation': "horizontal"}
)
axes[1].set_title("Horizontal Canopy Change (2D CHM)", fontsize=16, fontweight='bold')
axes[1].axis('off')

# Map 3: 3D Vertical Understory Loss
gdf.plot(
    column='Chg_Under', 
    ax=axes[2], 
    cmap='BrBG', 
    vmin=-0.3, 
    vmax=0.3,
    legend=True,
    legend_kwds={'label': "Change in 3D Understory Fraction\n(Negative = Loss)", 'orientation': "horizontal"}
)
axes[2].set_title("Vertical Understory Change (3D LiDAR)", fontsize=16, fontweight='bold')
axes[2].axis('off')

# Add a main title
fig.suptitle("Spatial Disconnect: Densification Impacts on 2D vs 3D Forest Structure (Mecklenburg County)", fontsize=20, fontweight='bold', y=0.95)

plt.tight_layout(rect=[0, 0, 1, 0.93])
output_map = os.path.join(plot_folder, "Fig4_Spatial_Choropleth_Comparison.png")
plt.savefig(output_map, dpi=300, bbox_inches='tight')
plt.close()

print(f"Success! High-resolution spatial map saved to: {output_map}")