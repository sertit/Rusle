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
- "P03" : P03 produce. The **"P03 raster"** need to be load . Should have the same values as CLC (need to be recode if not)

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

![Arcgis  pro toolbox](Arcgis_pro_Toolbox.PNG)


## CLI
This tool is also usable by command line:
```shell
rusle_core.py [-h] -aoi AOI_PATH -loc {Europe,Global} -fc {Already provided,To be calculated} [-fcp FCOVER_PATH] [-nir NIR_PATH] [-red RED_PATH] -lulc {Corine Land Cover - 2018 100m),Global Land Cover - Copernicus 2020 (100m,P03}
                [-p03 P03_PATH] [-del DEL_PATH] -ls {Already provided,To be calculated} [-lsp LS_PATH] [-dem {EUDEM 25m,SRTM 30m,MERIT 5 deg,Other}] [-demp OTHER_DEM_PATH] -res OUTPUT_RESOLUTION -epsg EPSG_CODE -o OUTPUT

optional arguments:
  -h, --help            show this help message and exit
  -aoi AOI_PATH, --aoi_path AOI_PATH
                        AOI path as a shapefile.
  -loc {Europe,Global}, --location {Europe,Global}
                        Location
  -fc {Already provided,To be calculated}, --fcover_method {Already provided,To be calculated}
                        Fcover Method
  -fcp FCOVER_PATH, --fcover_path FCOVER_PATH
                        Framework path if Already provided
  -nir NIR_PATH, --nir_path NIR_PATH
                        NIR band path if Fcover is To be calculated
  -red RED_PATH, --red_path RED_PATH
                        RED band path if Fcover is To be calculated
  -lulc {Corine Land Cover - 2018 (100m),Global Land Cover - Copernicus 2020 (100m),P03}, --landcover_name {Corine Land Cover - 2018 (100m),Global Land Cover - Copernicus 2020 (100m),P03}
                        Land Cover Name
  -p03 P03_PATH, --p03_path P03_PATH
                        P03 Path if lulc = P03. Should have the same nomenclature as CLC
  -del DEL_PATH, --del_path DEL_PATH
                        Fire delineation path
  -ls {Already provided,To be calculated}, --ls_method {Already provided,To be calculated}
                        LS Method
  -lsp LS_PATH, --ls_path LS_PATH
                        LS Path if ls Already provided
  -dem {EUDEM 25m,SRTM 30m,MERIT 5 deg,Other}, --dem_name {EUDEM 25m,SRTM 30m,MERIT 5 deg,Other}
                        DEM Name if ls To be calculated
  -demp OTHER_DEM_PATH, --other_dem_path OTHER_DEM_PATH
                        DEM path if ls To be calculated and dem = Other
  -res OUTPUT_RESOLUTION, --output_resolution OUTPUT_RESOLUTION
                        Output resolution
  -epsg EPSG_CODE, --epsg_code EPSG_CODE
                        EPSG code
  -o OUTPUT, --output OUTPUT
                        Output directory.
```

Example:
```shell
cd D:\RUSLE\
conda activate arcgispro-eo
python .\rusle.py -aoi D:\TLedauphin\02_Temp_traitement\test_rusle\emsn073_aoi_32631.shp -loc "Europe" -fc "To be calculated" -nir D:\TLedauphin\02_Temp_traitement\test_rusle\T31TDH_20200805T104031_B08_10m.jp2 -red D:\TLedauphin\02_Temp_traitement\test_rusle\T31TDH_20200805T104031_B04_10m.jp2 -lulc "Corine Land Cover - 2018 (100m)" -del D:\TLedauphin\02_Temp_traitement\test_rusle\emsn073_del_32631.shp -ls "To be calculated" -dem "EUDEM 25m" -res 10 -epsg 32631 -o D:\TLedauphin\02_Temp_traitement\test_rusle\output
```
