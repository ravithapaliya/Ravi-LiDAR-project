# **4. Results** 

# **4.1. The Temporal Divergence of 2D and 3D Forest Structures** 

A primary objective of this research was to determine if standard horizontal canopy metrics accurately reflect the internal ecological state of the urban forest over time. The longitudinal analysis (2002–2023) of Mecklenburg County reveals a severe temporal divergence between twodimensional (2D) canopy cover and three-dimensional (3D) vertical structural fractions. 

As illustrated in the temporal trend models (Figure 1), the countywide 2D horizontal canopy cover followed a relatively stable, arc-like trajectory across the two-decade study period. Across all neighborhood typologies, 2D canopy peaked around 2012 before experiencing a moderate decline through 2023. However, the normalized 3D structural metrics exposed a concealed ecological collapse. The understory structural fraction (representing vegetation between 3 and 15 feet) did not mirror the stability of the overhead canopy. Instead, understory fractions experienced extreme volatility, culminating in a precipitous crash between 2017 and 2023. This divergence proves that relying exclusively on 2D Canopy Height Models (CHM) creates a "hidden" loss phenomenon, masking the rapid hollowing-out of the internal forest volume occurring beneath a deceptively stable canopy ceiling. 

# **4.2. Distinct Urban Morphologies Drive Stratified Structural Loss** 

The XGBoost feature importance models were deployed to determine which specific urbanization factors drove the destruction of the canopy. The results quantitatively prove that different modes of urban densification destroy distinctly different vertical strata of the forest ecosystem. 

The models (Figure 2, Feature Importance) demonstrate that horizontal sprawl, quantified as the increase in building cover fraction ( _Δ_ Building Cover), is the primary driver of 3D understory loss. As neighborhood lot lines maximize and building footprints expand, the lower and intermediate vegetation layers are systematically eradicated. Conversely, vertical densification, measured by increases in baseline building height ( _Δ_ Building Height), was identified as the dominant predictor for the loss of standard 2D canopy cover and overall canopy height. This indicates that mid-rise and high-rise developments inherently require the removal of the mature, overarching canopy, whereas lower-density sprawl targets the understory brush and mid-level trees. 

# **4.3. Threshold Dynamics: The 5% Densification Tipping Point** 

A critical finding of this study is that urban forest degradation is non-linear. The bivariate scatter plots and SHAP dependence plots measuring structural changes against building densification reveal a stark "tipping point" in the ecosystem. 

The models show that forest structures do not slowly or proportionally degrade as urban concrete is introduced. Instead, the data reveals that an increase of just 2% to 5% in a neighborhood's building cover fraction triggers an abrupt and severe collapse in both understory density and overall Foliage Height Diversity (FHD). Prior to this 5% threshold, canopy structures demonstrate a degree of resilience; however, once this specific densification threshold is breached, the internal vertical complexity of the forest undergoes a rapid, non-linear phase transition toward total structural simplification. 

# **4.4. Spatial Heterogeneity: Suburban Vulnerability vs. Urban Core Resilience** 

The structural response to urban densification is highly dependent on the historical baseline of the neighborhood zone. By stratifying the data into Low-Density (Suburbs), Medium Density, and HighDensity (Urban Core) typologies, the analysis revealed significant spatial heterogeneity in canopy resilience. 

The SHAP dependence models highlight that the most extreme and chaotic reductions in vertical complexity occurred almost exclusively in low-density suburban zones (represented by the purple/dark markers in the scatter distributions). Because these older suburbs historically maintained thick, multi-layered forest volumes, they possessed the most ecological complexity to lose during recent waves of infill development. In contrast, the High-Density Urban Core zones exhibited a relatively flat structural response to new development between 2002 and 2023. This is not indicative of ecological resilience, but rather a reflection of a pre-simplified baseline; the complex natural tree layers in the urban core had already been cleared out prior to the 2002 onset of this study. 

# **4.5. Microclimate Implications: 3D Complexity as a Thermal Determinant** 

The final phase of the analysis modeled how these distinct structural losses drove localized heat penalties, utilizing the downscaled Landsat Land Surface Temperature (LST) deltas. The thermal XGBoost models challenge the traditional urban planning paradigm that relies exclusively on 2D shade for microclimate cooling. 

The results establish that the destruction of internal 3D structural complexity is a superior determinant of neighborhood warming. The loss of Foliage Height Diversity (FHD) and 3D Overstory Density were the strongest predictors of severe localized LST increases, significantly outpacing the heat penalties associated with standard 2D canopy loss. Neighborhoods that experienced the acute collapse of their 3-to-15-foot understory strata consistently exhibited the highest spikes in localized warming. 

# **4.6. Spatial Disconnect Across Mecklenburg County** 

The geographic synthesis of these metrics (Figure 4, Spatial Choropleth Comparison) visually validates the statistical models across Mecklenburg County. The multi-panel mapping explicitly shows the spatial disconnect: dense geographic clusters in the rapidly developing suburban peripheries display severe 3D understory depletion, even in tracts where the 2D canopy map registers as stable. This spatial evidence confirms that urbanization is actively hollowing out the ecological volume of Charlotte's suburban canopy, leaving behind structurally simplified, hotter neighborhoods. 

