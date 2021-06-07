# RUSLE
Computing the Mean (annual) soil loss (in ton/ha/year) with the RUSLE model.

:warning: **CONDA ENVIRONMENT**  
**Be sure to have your ArcgisPro conda environment installed.

## Inputs

**Aoi** : 
AOI path.

**Location** : 
Location of the AOI.
Can be :
- "Europe" : AOI located in Europe
- Or "Global" : AOI located outside Europe

**Fcover_method** :
Calculation method of the Fraction of green Vegetation Coverportal (Fcover).
Can be :
- "Already provided" : When the Fcover raster already exists. You will have to load it in the **"FCover"** parameter.
- Or "To be calculated" : When the Fcover raster does not exist. It will be calculated with the following formula :

_(NDVI - NDVI_s)/(NDVI_v - NDVI_s)_
With :
_NDVI_ : Normalized Difference Vegetation Index
; _NDVI_s_ : NDVI Min
; _NDVI_v_ : NDVI Max

To produce it you will have to load a **NIR infrared** and a **Red band** that cover the AOI.


![Arcgis  pro toolbox](static/Arcgis_pro_Toolbox.PNG)



