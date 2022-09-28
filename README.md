# RUSLE
Computing the Mean (annual) soil loss (in ton/ha/year) with the RUSLE model.

:warning: **CONDA ENVIRONMENT**  
**Be sure to have your ArcgisPro conda environment installed.
See [here](https://git.unistra.fr/sertit/arcgis-pro/sertit-eo-conda-environment) for more information.**


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

To produce it you will have to load a **NIR infrared band** and a **Red band** that cover the AOI.

**Landcover** : 
Name of the Landcover that will be used.
Can be :
- "Corine Land Cover - 2018 (100m)"
- "Global Land Cover - Copernicus 2020 (100m)"
- "GlobCover - ESA 2005 (300m)"
- "GlobeLand30 - China 2020 (30m)"
- "P03" : P03 produce. The **"P03 raster"** need to be load

**Delineation** : Wildfire delineation path

**LS_method** : Calculation method of the Slope angle and length (LS factor).
Can be :
- "Already provided" : When the LS raster already exists. You will have to load it in the **"LS raster"** parameter.
- Or "To be calculated" : When the LS raster does not exist. To produce it you will have to **chose or load the DEM** that will be use to produce it with the "DEM" parameter.

**DEM** (if LS method == "To be calculated") : Name of the DEM that will be used.
Can be :
- "EUDEM 25m"
- "SRTM 30m"
- "MERIT 5 deg"
- "Other" : A DEM other than those listed above. Need to be load in the **"DEM raster"** parameter

**Output resolution** : Resolution of the output raster

**Coordinate system** : Coordinate system of the output raster

**Output folder** : Output folder

![Arcgis  pro toolbox](static/Arcgis_pro_Toolbox.PNG)



