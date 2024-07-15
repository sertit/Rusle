# RUSLE
Computing the Mean (annual) soil loss (in ton/ha/year) with the [RUSLE 2015 model](https://web.jrc.ec.europa.eu/policy-model-inventory/explore/models/model-rusle2015/).

:warning: **CONDA ENVIRONMENT**  
**Be sure to have your ArcgisPro conda environment installed.
See [here](https://git.unistra.fr/sertit/arcgis-pro/sertit-eo-conda-environment) for more information.**


## Arcgis pro inputs

### Basic inputs

**Aoi** (mandatory)

AOI path (shapefile, geojson, kml) or WKT strings.

**Location** (mandatory)

Location of the AOI.
Can be :
- "Europe" : AOI located in Europe
- Or "Global" : AOI located outside Europe

**Nir infrared band** and **Red band**

Path to the nir infrared and red bands to compute the Fraction of green Vegetation Coverportal (Fcover). 
You can provide your own Fcover raster in the advanced section.

**Output folder**

Path to the output folder.

### Advanced

Everything is optional in this section.

**Fcover**

You can provide an existing Fraction of green Vegetation Coverportal (Fcover) raster file.
If not provided, the Fcover is calculated  with the following formula :

```math
(NDVI - NDVI_s)/(NDVI_v - NDVI_s)
```

With :
- `NDVI` : Normalized Difference Vegetation Index
- `NDVI_s` : NDVI Min
- `NDVI_v` : NDVI Max

To produce it you will have to load a **NIR infrared band** and a **Red band** that cover the AOI.

**Landcover** (optional)

Name of the Landcover that will be used.
Can be :
- "Corine Land Cover - 2018 (100m)"
- "Global Land Cover - Copernicus 2019 (100m)"
- "P03" : P03 produce. The **"P03 raster"** need to be load . Should have the same values as CLC (need to be recode if not)

**Fire delineation** (optional)

Wildfire delineation path

Default: None

**LS Raster** (optional)

Optional path to the Slope angle and length (LS factor) raster. If not provided, it is calculated thanks to the DEM.

Default: None

**DEM** (optional)

The DEM is used if the LS raster is not provided.

Can be :
- "EUDEM 25m"
- "COPDEM 30m"
- "SRTM 30m"
- "MERIT 5 deg"
- "Other" : A DEM other than those listed above. Need to be load in the **"DEM raster"** parameter

Default: "COPDEM 30m"

**Output resolution** (Optional)

Resolution of the output raster in the unit of the output coordinate system. 

Default : 10 meters

![Arcgis  pro toolbox](Arcgis_pro_Toolbox.PNG)


## CLI

This tool is also usable by command line:
```shell
Usage: rusle [OPTIONS]

  Import osm charter with the CLI. Returns:

Options:
  -aoi, --aoi_path TEXT           Path to the AOI (shp, geojson) or WKT string
                                  [required]
  -loc, --location [Europe|Global]
                                  Location of the AOI  [required]
  -nir, --nir_path PATH           NIR band path needed if no fcover raster is
                                  provided.
  -red, --red_path PATH           RED band path needed if no fcover raster is
                                  provided.
  --satellite_product, --sat_product PATH
                                  Alternative to red and nir options. Path to
                                  a satellite product with at least the Nir
                                  and Red bands
  -lulc, --landcover_name [Corine Land Cover - 2018 (100m)|Global Land Cover - Copernicus 2019 (100m)|P03]
                                  Land Cover Name  [default: Global Land Cover
                                  - Copernicus 2019 (100m)]
  -fcp, --fcover_path PATH        Path to a Fraction of green Vegetation
                                  Coverportal (Fcover) raster file. If not
                                  provided, it will be calculated from nir and
                                  red bands or satellite products
  -p03, --p03_path PATH           P03 Path if lulc =  P03. Should have the
                                  same nomenclature as CLC
  -p03, -lsp, --ls_path PATH      Optional path to the Slope angle and length
                                  (LS factor) raster. If not provided, it is
                                  calculated thanks to the DEM.
  -del, --del_path PATH           Fire delineation path
  -dem, --dem_name [COPDEM 30m|EUDEM 25m|SRTM 30m|MERIT 5 deg|Other]
                                  DEM Name needed if ls_path option is not
                                  provided.  [default: COPDEM 30m]
  -demp, --other_dem_path PATH    DEM path if dem = Other
  -res, --output_resolution INTEGER RANGE
                                  Output resolution  [1<=x<=1000]
  -epsg, --epsg_code INTEGER RANGE
                                  EPSG code, 4326 is not accepted. By default,
                                  it is the EPSG code of the AOI UTM zone.
                                  [1024<=x<=32767]
  -o, --output DIRECTORY          Output directory.   [required]
  --ftep BOOLEAN                  Set this flag if the command line is run on
                                  the ftep platform.
  --help                          Show this message and exit.
```

Example:
```shell
conda activate arcgispro-eo
python rusle.py -aoi emsn073_aoi_32631.shp -loc "Europe" -nir T31TDH_20200805T104031_B08_10m.jp2 -red T31TDH_20200805T104031_B04_10m.jp2 -del emsn073_del_32631.shp -o output
```
