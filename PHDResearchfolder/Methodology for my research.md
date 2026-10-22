# **3.1. Study Area and Spatial Framework** 

The spatial framework for this longitudinal study was established using the Quality of Life (QOL) Neighborhood Profile Areas (NPAs) across Mecklenburg County, North Carolina. To track localized structural and thermal shifts, all subsequent high-resolution raster and point cloud datasets were aggregated to these neighborhood delineations. Neighborhoods were stratified into three baseline typologies—Low Density (Suburbs), Medium Density, and High Density (Urban Core), based on their initial building cover fractions in 2002 to assess heterogeneous responses to urbanization. 

# **3.2. Extraction and Normalization of 3D Forest Structure** 

To quantify internal forest architecture, raw LiDAR point clouds from five temporal epochs (2002, 2007, 2012, 2017, and 2023) were processed. A highly optimized voxelization workflow was utilized to extract point densities across specific vertical strata for each neighborhood. The vertical profile was partitioned into an Understory layer (3 to 15 feet), a Midstory layer, and an Overstory layer (>50 feet). 

A critical methodological challenge in longitudinal LiDAR analysis is the variation in sensor resolution and pulse density across flight years (e.g., higher point densities in 2017 and 2023 compared to 2002). To neutralize hardware discrepancies and prevent artificial spikes in structural metrics, raw vertical point counts were normalized. Total vegetation volume was calculated for each neighborhood, and the individual strata were converted into **Structural Fractions** (e.g., Understory Fraction = Understory Point Density / Total Vegetation Density). This standardization ensured that temporal trajectories isolated genuine structural degradation rather than sensor artifacts. 

Simultaneously, standard 2D horizontal Canopy Height Models (CHM) and Building Height Models (BHM) were processed to extract mean canopy height, canopy cover percentage, building cover fraction (sprawl), and mean building height (vertical densification). 

# **3.3. Thermal Data Integration (LST)** 

To assess the microclimate impacts of these structural changes, 1-meter downscaled Landsat Land Surface Temperature (LST) raster datasets were integrated. The thermal extraction workflow dynamically matched the structural LiDAR epochs to the closest corresponding summer Landsat composites (e.g., linking the 2002 LiDAR to the 2000 summer LST baseline). An array-based rasterization technique masked non-valid pixels and extracted the mean LST for every active QOL neighborhood, ensuring exact spatial alignment between the canopy metrics and the local thermal environment. 

# **3.4. Machine Learning Modeling of Urban Drivers** 

To determine the specific urbanization factors driving canopy loss, Extreme Gradient Boosting (XGBoost) regression models were deployed. The analysis was bifurcated to explicitly contrast the drivers of horizontal versus vertical structural loss: 

1. **Horizontal Driver Modeling:** Evaluated how changes in building cover and baseline building heights predicted the loss of traditional 2D CHM canopy cover. 

2. **Vertical Driver Modeling:** Evaluated how the same urbanization metrics predicted the collapse of the 3D Understory and Overstory Structural Fractions, as well as the loss of overall Foliage Height Diversity (FHD). 

By calculating the interval deltas (the exact mathematical change in each variable between the start and end years), the XGBoost algorithms identified non-linear tipping points in the urban ecosystem. Feature importance scores were extracted to rank whether horizontal sprawl or vertical densification was the dominant destructive force for each specific forest stratum. 

# **3.5. Modeling Thermal Impacts** 

A final XGBoost model was constructed to isolate the structural drivers of neighborhood heat penalties. In this model, the target variable was the change in mean summer LST ( _Δ_ LST). The predictor variables included the structural losses measured in both 2D (horizontal canopy cover loss, overall CHM height loss) and 3D (understory fraction loss, overstory fraction loss). SHAP (SHapley Additive exPlanations) dependence plots were generated to visualize how the localized destruction of specific canopy strata—particularly the 3-to-15-foot understory—correlated with localized temperature spikes. 

# **3.6. Spatial Visualization and Cartography** 

To transition the statistical findings into a geographic context, the 21-year interval changes (2002– 2023) for all structural, building, and thermal metrics were synthesized. These comprehensive deltas were joined back to the QOL neighborhood shapefile and exported as a master spatial database (GeoPackage). 

Using these spatial datasets, multi-panel bivariate choropleth maps were generated to visually contrast the spatial distribution of 2D canopy loss against 3D understory collapse. This cartographic mapping served as the final validation step, proving geographically that dense clusters of understory degradation frequently occurred in densifying neighborhoods where the overhead 2D canopy appeared deceptively stable. 

