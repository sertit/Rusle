"""
This file is part of RUSLE.

RUSLE is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

RUSLE is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with RUSLE.
If not, see <https://www.gnu.org/licenses/>.
"""

import gc
import logging
import os
import sqlite3
import warnings
from enum import unique

import cloudpathlib
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import whitebox_workflows as wbw
import xarray
from eoreader.bands import NIR, RED
from eoreader.reader import Reader
from odc.geo import xr  # noqa
from pysheds.grid import Grid
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterstats import zonal_stats
from sertit import AnyPath, files, misc, rasters, rasters_rio, strings, vectors
from sertit.misc import ListEnum
from sertit.unistra import get_geodatastore

np.seterr(divide="ignore", invalid="ignore")

DEBUG = False
LOGGING_FORMAT = "%(asctime)s - [%(levelname)s] - %(message)s"
LOGGER = logging.getLogger("RUSLE")


def geodatastore(ftep=False):
    """
    This function returns the root path to the geo data store (DEM, Soil database...).
    Args:
        ftep: If True, the path to the s3 bucket for the ftep platform is returned. Else, get_geodatastore from sertit utils module is called.

    Returns:
        The path to the geo data store.

    """
    if ftep:
        return AnyPath("s3://eo4sdg-data-sertit")
    else:
        return get_geodatastore()


class DataPath:
    GLOBAL_DIR = None

    @classmethod
    def load_paths(cls, ftep=False):
        cls.GLOBAL_DIR = geodatastore(ftep) / "GLOBAL"
        cls.WORLD_COUNTRIES_PATH = (
            cls.GLOBAL_DIR / "World_countries_poly" / "world_countries_poly.shp"
        )
        cls.EUROPE_COUNTRIES_PATH = (
            cls.GLOBAL_DIR / "World_countries_poly" / "europe_countries_poly.shp"
        )

        cls.HWSD_PATH = (
            cls.GLOBAL_DIR / "FAO_Harmonized_World_Soil_Database" / "hwsd.tif"
        )
        cls.DBFILE_PATH = (
            cls.GLOBAL_DIR / "FAO_Harmonized_World_Soil_Database" / "HWSD.mdb"
        )
        cls.DBFILE_PATH_SQL = (
            cls.GLOBAL_DIR / "FAO_Harmonized_World_Soil_Database" / "HWSD.db"
        )

        cls.R_EURO_PATH = (
            cls.GLOBAL_DIR / "European_Soil_Database_v2" / "Rfactor" / "Rf_gp1.tif"
        )

        cls.K_EURO_PATH = (
            cls.GLOBAL_DIR / "European_Soil_Database_v2" / "Kfactor" / "K_new_crop.tif"
        )
        cls.K_GLOBAL_PATH = (
            cls.GLOBAL_DIR
            / "European_Soil_Database_v2"
            / "Kfactor"
            / "Global"
            / "K_GloSEM_factor.tif"
        )
        cls.LS_EURO_PATH = (
            cls.GLOBAL_DIR
            / "European_Soil_Database_v2"
            / "LS_100m"
            / "EU_LS_Mosaic_100m.tif"
        )
        cls.P_EURO_PATH = (
            cls.GLOBAL_DIR
            / "European_Soil_Database_v2"
            / "Pfactor"
            / "EU_PFactor_V2.tif"
        )

        cls.CLC_PATH = (
            cls.GLOBAL_DIR
            / "Corine_Land_Cover"
            / "CLC_2018"
            / "clc2018_clc2018_v2018_20_raster100m"
            / "CLC2018_CLC2018_V2018_20.tif"
        )
        cls.R_GLOBAL_PATH = (
            cls.GLOBAL_DIR / "Global_Rainfall_Erosivity" / "GlobalR_NoPol.tif"
        )

        cls.GLC_PATH = (
            cls.GLOBAL_DIR
            / "Global_Land_Cover"
            / "2019"
            / "PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif"
        )

        cls.WC_PATH = (
            cls.GLOBAL_DIR / "ESA_WorldCover" / "2021" / "ESA_WorldCover_10m_2021.vrt"
        )

        cls.GC_PATH = (
            cls.GLOBAL_DIR
            / "Globcover_2009"
            / "Globcover2009_V2.3_Global_"
            / "GLOBCOVER_L4_200901_200912_V2.3.tif"
        )
        cls.GL_PATH = ""  # To add

        cls.EUDEM_PATH = cls.GLOBAL_DIR / "EUDEM_v2" / "eudem_dem_3035_europe.tif"
        cls.SRTM30_PATH = cls.GLOBAL_DIR / "SRTM_30m_v4" / "index.vrt"
        cls.MERIT_PATH = (
            cls.GLOBAL_DIR
            / "MERIT_Hydrologically_Adjusted_Elevations"
            / "MERIT_DEM.vrt"
        )
        cls.COPDEM30_PATH = cls.GLOBAL_DIR / "COPDEM_30m" / "COPDEM_30m.vrt"
        cls.GADM_PATH = cls.GLOBAL_DIR / "GADM" / "gadm_410.gdb"

    # Buffer apply to the AOI
    AOI_BUFFER = 5000


@unique
class InputParameters(ListEnum):
    """
    List of the input parameters
    """

    AOI_PATH = "aoi_path"
    LOCATION = "location"
    FCOVER_PATH = "fcover_path"
    NIR_PATH = "nir_path"
    RED_PATH = "red_path"
    SATELLITE_PRODUCT_PATH = "satellite_product_path"
    LANDCOVER_NAME = "landcover_name"
    P03_PATH = "p03_path"
    DEL_PATH = "del_path"
    LS_PATH = "ls_path"
    DEM_NAME = "dem_name"
    OTHER_DEM_PATH = "other_dem_path"
    OUTPUT_RESOLUTION = "output_resolution"
    REF_EPSG = "ref_epsg"
    OUTPUT_DIR = "output_dir"
    TMP_DIR = "temp_dir"


@unique
class LandcoverType(ListEnum):
    """
    List of the Landcover type
    """

    CLC = "Corine Land Cover - 2018 (100m)"
    GLC = "Global Land Cover - Copernicus 2019 (100m)"
    WC = "WorldCover - ESA 2021 (10m)"
    P03 = "P03"


@unique
class LocationType(ListEnum):
    """
    List of the location
    """

    EUROPE = "Europe"
    GLOBAL = "Global"
    EUROPE_LEGACY = "Europe_legacy"
    GLOBAL_LEGACY = "Global_legacy"


@unique
class DemType(ListEnum):
    """
    List of the DEM
    """

    EUDEM = "EUDEM 25m"
    SRTM = "SRTM 30m"
    MERIT = "MERIT 5 deg"
    COPDEM_30 = "COPDEM 30m"
    OTHER = "Other"


class LandcoverStructure(ListEnum):
    """
    List of the Landcover type (Coding)
    """

    CLC = "Corine Land Cover - 2018 (100m)"
    GLC = "Global Land Cover - Copernicus 2019 (100m)"
    WC = "WorldCover - ESA 2021 (10m)"
    GC = "GlobCover - ESA 2005 (300m)"
    GL = "GlobeLand30 - China 2020 (30m)"
    P03 = "P03"


def check_parameters(input_dict: dict) -> None:
    """
     Check if parameters values are valid
    Args:
        input_dict (dict) : dict with parameters values

    Returns:

    """
    # ---- Extract parameters
    landcover_name = input_dict.get(InputParameters.LANDCOVER_NAME.value)
    p03_path = input_dict.get(InputParameters.P03_PATH.value)
    dem_name = input_dict.get(InputParameters.DEM_NAME.value)
    other_dem_path = input_dict.get(InputParameters.OTHER_DEM_PATH.value)

    # -- Check if P03 if needed
    if (landcover_name == LandcoverType.P03.value) and (p03_path is None):
        raise ValueError("P03_path is needed !")

    # --  Check if other dem path if needed
    if (dem_name == DemType.OTHER.value) and (other_dem_path is None):
        raise ValueError("Dem path is needed !")

    return


def norm_diff(xarr_1, xarr_2, new_name: str = ""):
    """
    Normalized difference

    Args:
        xarr_1 : xarray of band 1
        xarr_2 : xarray of band 2
        new_name (str): new name

    Returns:
        Normalized difference of the two bands with a new name
    """
    # Get data as np arrays
    band_1 = xarr_1.data
    band_2 = xarr_2.data

    # Compute the normalized difference
    norm = np.divide(band_1 - band_2, band_1 + band_2)

    # Create back a xarray with the proper data
    norm_xda = xarr_1.copy(data=norm)
    return rasters.set_metadata(norm_xda, norm_xda, new_name=new_name)


def produce_fcover(
    red_path: str,
    nir_path: str,
    aoi_path: str,
    tmp_dir: str,
    ref_crs: int,
    output_resolution: int,
):
    """
    Produce the fcover index
    Args:
        red_path (str): Red band path
        nir_path (str): nir band path
        aoi_path (str): AOI path
        tmp_dir (str) : Temp dir where the cropped Raster will be stored
        ref_crs (int) : ref crs
        output_resolution (int) : output resolution

    Returns:
        xarray of the fcover raster
    """
    LOGGER.info("-- Produce the fcover index --")

    # -- Open AOI
    aoi_gdf = gpd.read_file(aoi_path)

    # -- Read RED
    if isinstance(red_path, xarray.DataArray):
        red_xarr = red_path
    else:
        red_xarr = rasters.read(red_path, window=aoi_gdf).astype(
            np.float32
        )  # No need to normalize, only used in NDVI

    # -- Read NIR
    if isinstance(nir_path, xarray.DataArray):
        nir_xarr = nir_path
    else:
        nir_xarr = rasters.read(nir_path, window=aoi_gdf).astype(
            np.float32
        )  # No need to normalize, only used in NDVI

    # -- Process NDVI
    ndvi_xarr = norm_diff(nir_xarr, red_xarr, new_name="NDVI")

    # -- Reproject the raster
    # -- Re project raster and resample
    ndvi_reproj_xarr = ndvi_xarr.rio.reproject(
        ref_crs, resolution=output_resolution, resampling=Resampling.bilinear
    )

    # -- Crop the NDVI with reprojected AOI
    ndvi_crop_xarr = rasters.crop(ndvi_reproj_xarr, aoi_gdf)

    # -- Write the NDVI cropped
    ndvi_crop_path = os.path.join(files.to_abspath(tmp_dir), "ndvi.tif")
    rasters.write(ndvi_crop_xarr, ndvi_crop_path)  # , nodata=0)

    # -- Extract NDVI min and max inside the AOI
    ndvi_stat = zonal_stats(aoi_gdf, ndvi_crop_path, stats="min max")
    ndvi_min = ndvi_stat[0]["min"]
    ndvi_max = ndvi_stat[0]["max"]

    # -- Fcover calculation
    fcover_xarr = (ndvi_crop_xarr - ndvi_min) / (ndvi_max - ndvi_min)
    fcover_xarr_clean = rasters.where(ndvi_crop_xarr == -9999, np.nan, fcover_xarr)
    fcover_xarr_clean = rasters.set_metadata(
        fcover_xarr_clean, ndvi_crop_xarr, new_name="FCover"
    )

    # -- Write fcover
    fcover_path = os.path.join(tmp_dir, "fcover.tif")
    rasters.write(fcover_xarr_clean, fcover_path)  # , nodata=0)

    return fcover_xarr_clean


def produce_c_arable_europe(aoi_path: str, raster_xarr):
    """
    Produce C arable index over Europe
    Args:
        aoi_path (str): aoi path
        raster_xarr : lulc xarray

    Returns:
        xarray of the c arable raster
    """

    LOGGER.info("-- Produce C arable index over Europe --")

    arable_c_dict = {
        "Austria": 0.21800000000,
        "Belgium": 0.24500000000,
        "Bulgaria": 0.18800000000,
        "Cyprus": 0.19300000000,
        "Czech Republic": 0.19900000000,
        "Croatia": 0.25500000000,
        "Germany": 0.200000000,
        "Denmark": 0.22200000000,
        "Estonia": 0.21700000000,
        "Spain": 0.28900000000,
        "Finland": 0.23100000000,
        "France": 0.20200000000,
        "Greece": 0.2800000000,
        "Hungary": 0.27500000000,
        "Ireland": 0.20200000000,
        "Italy": 0.21100000000,
        "Lithuania": 0.24200000000,
        "Luxembourg": 0.21500000000,
        "Latvia": 0.23700000000,
        "Malta": 0.43400000000,
        "Netherlands": 0.2600000000,
        "Poland": 0.24700000000,
        "Portugal": 0.35200000000,
        "Romania": 0.29600000000,
        "Sweden": 0.23700000000,
        "Slovenia": 0.24800000000,
        "Slovakia": 0.23500000000,
        "United Kingdom": 0.17700000000,
        "the former Yugoslav Republic of Macedonia": 0.25500000000,
    }

    # -- Re project AOI to wgs84
    aoi = gpd.read_file(aoi_path)
    crs_4326 = CRS.from_epsg(4326)
    if aoi.crs != crs_4326:
        aoi = aoi.to_crs(crs_4326)

    # -- Extract europe countries
    world_countries = vectors.read(DataPath.WORLD_COUNTRIES_PATH, bbox=aoi.envelope)
    arable_c_countries = list(arable_c_dict.keys())
    europe_countries = world_countries[
        (world_countries["CONTINENT"] == "Europe")
        & (world_countries["COUNTRY"].isin(arable_c_countries))
    ]

    # -- Initialize arable arr
    arable_c_xarr = xarray.full_like(raster_xarr, fill_value=0.27)

    # -- Re project europe_countries
    crs_arable = arable_c_xarr.rio.crs
    if europe_countries.crs != crs_arable:
        europe_countries = europe_countries.to_crs(crs_arable)

    # -- Update arable_arr with arable_dict
    for key in list(europe_countries["COUNTRY"]):
        geoms = [
            feature
            for feature in europe_countries[europe_countries["COUNTRY"] == key][
                "geometry"
            ]
        ]
        arable_c_xarr = rasters.paint(arable_c_xarr, geoms, value=arable_c_dict[key])

    # -- Mask result with aoi
    arable_c_xarr = rasters.mask(arable_c_xarr, aoi)

    return arable_c_xarr


def produce_c(input_dict, post_process_dict):
    """
    Produce C index
    Args:
        lulc_xarr : lulc xarray
        fcover_xarr :fcover xarray
        aoi_path (str): aoi path
        lulc_name (str) : name of the LULC

    Returns:
        xdarray of the c index raster
    """
    LOGGER.info("-- Produce C index --")
    # --- Identify Cfactor
    # -- Cfactor dict and c_arr_arable

    fcover_path = input_dict.get(InputParameters.FCOVER_PATH.value)
    del_path = input_dict.get(InputParameters.DEL_PATH.value)
    tmp_dir = input_dict.get(InputParameters.TMP_DIR.value)
    ref_epsg = input_dict.get(InputParameters.REF_EPSG.value)
    ref_crs = CRS.from_epsg(ref_epsg)
    output_resolution = input_dict.get(InputParameters.OUTPUT_RESOLUTION.value)
    aoi_path = input_dict.get(InputParameters.AOI_PATH.value)
    aoi_gdf = vectors.read(aoi_path)
    lulc_name = input_dict.get(InputParameters.LANDCOVER_NAME.value)

    # -- Check if fcover need to be calculated or not
    if fcover_path is None:
        # -- Process fcover
        red_process_path = post_process_dict["red"]
        nir_process_path = post_process_dict["nir"]
        fcover_xarr = produce_fcover(
            red_process_path,
            nir_process_path,
            aoi_path,
            tmp_dir,
            ref_crs,
            output_resolution,
        )

    else:
        fcover_xarr = rasters.read(post_process_dict["fcover"], window=aoi_gdf)

    # -- Mask lulc if del
    if del_path:
        # -- Update the lulc with the DEL
        LOGGER.info("-- Update raster values covered by DEL --")

        # -- DEM to gdf
        del_gdf = gpd.read_file(del_path)

        # -- Update the lulc with the del
        lulc_process_path = post_process_dict["lulc"]
        lulc_del_xarr = rasters.paint(lulc_process_path, del_gdf, value=334)

        # -- Write the lulc with fire
        lulc_masked_path = os.path.join(tmp_dir, "lulc_with_fire.tif")
        rasters.write(lulc_del_xarr, lulc_masked_path)  # , nodata=0)

        lulc_xarr = lulc_del_xarr.copy()

    else:
        # -- Open the lulc raster
        lulc_xarr = rasters.read(post_process_dict["lulc"], window=aoi_gdf)

    if (lulc_name == LandcoverStructure.CLC.value) or (
        lulc_name == LandcoverType.P03.value
    ):
        cfactor_dict = {
            221: [0.15, 0.45],
            222: [0.1, 0.3],
            223: [0.1, 0.3],
            231: [0.05, 0.15],
            241: [0.07, 0.35],
            242: [0.07, 0.2],
            243: [0.05, 0.2],
            244: [0.03, 0.13],
            311: [0.0001, 0.003],
            312: [0.0001, 0.003],
            313: [0.0001, 0.003],
            321: [0.01, 0.08],
            322: [0.01, 0.1],
            323: [0.01, 0.1],
            324: [0.003, 0.05],
            333: [0.1, 0.45],
            334: [0.1, 0.55],
        }
        # -- Produce arable c
        arable_c_xarr = produce_c_arable_europe(aoi_path, lulc_xarr)
        arable_c_xarr = rasters.where(
            np.isin(
                lulc_xarr,
                [
                    211,
                    212,
                    213,
                ],
            ),
            arable_c_xarr,
            np.nan,
        )

    # -- Global Land Cover - Copernicus 2019 (100m)
    elif lulc_name == LandcoverStructure.GLC.value:
        cfactor_dict = {
            20: [0.003, 0.05],
            30: [0.01, 0.08],
            40: [0.07, 0.2],
            60: [0.1, 0.45],
            100: [0.01, 0.1],
            111: [0.0001, 0.003],
            112: [0.0001, 0.003],
            113: [0.0001, 0.003],
            114: [0.0001, 0.003],
            115: [0.0001, 0.003],
            116: [0.0001, 0.003],
            121: [0.0001, 0.003],
            122: [0.0001, 0.003],
            123: [0.0001, 0.003],
            124: [0.0001, 0.003],
            125: [0.0001, 0.003],
            126: [0.0001, 0.003],
            334: [0.1, 0.55],
        }
        # -- Produce arable c
        arable_c_xarr = rasters.where(
            np.isin(lulc_xarr, [0, 50, 70, 80, 90, 200]),
            np.nan,
            np.nan,
            lulc_xarr,
            new_name="Arable C",
        )

    # -- WorldCover - ESA 2021 (10m)
    elif lulc_name == LandcoverStructure.WC.value:
        cfactor_dict = {
            10: [0.0001, 0.003],
            20: [0.003, 0.05],
            30: [0.01, 0.08],
            40: [0.07, 0.35],
            60: [0.1, 0.45],
            100: [0.01, 0.1],
            334: [0.1, 0.55],
        }
        # -- Produce arable c
        arable_c_xarr = rasters.where(
            np.isin(lulc_xarr, [50, 70, 80, 90, 95]),
            np.nan,
            np.nan,
            lulc_xarr,
            new_name="Arable C",
        )

    # -- GlobCover - ESA 2005 (300m)
    elif lulc_name == LandcoverStructure.GC.value:
        cfactor_dict = {
            334: [0.1, 0.55],
            11: [0.07, 0.2],
            14: [0.07, 0.2],
            20: [0.07, 0.2],
            30: [0.07, 0.2],
            40: [0.0001, 0.003],
            50: [0.0001, 0.003],
            60: [0.0001, 0.003],
            70: [0.0001, 0.003],
            90: [0.0001, 0.003],
            100: [0.0001, 0.003],
            110: [0.003, 0.05],
            120: [0.003, 0.05],
            130: [0.003, 0.05],
            140: [0.01, 0.08],
            150: [0.1, 0.45],
            160: [0.01, 0.1],
            170: [0.01, 0.1],
            180: [0.01, 0.1],
            200: [0, 0],
            220: [0, 0],
        }
        # -- Produce arable c
        arable_c_xarr = rasters.where(
            lulc_xarr == 40, 0.27, np.nan, lulc_xarr, new_name="Arable C"
        )

    # -- GlobeLand30 - China 2020 (30m)
    elif lulc_name == LandcoverStructure.GL.value:
        cfactor_dict = {
            334: [0.1, 0.55],
            20: [0.00010000000, 0.00250000000],
            30: [0.01000000000, 0.07000000000],
            40: [0.01000000000, 0.08000000000],
            90: [0.00000000000, 0.00000000000],
            100: [0.00000000000, 0.00000000000],
            110: [0.10000000000, 0.45000000000],
        }
        # -- Produce arable c
        arable_c_xarr = rasters.where(
            lulc_xarr == 10, 0.27, np.nan, lulc_xarr, new_name="Arable C"
        )
    else:
        raise ValueError(f"Unknown Landcover structure {lulc_name}")

    # -- List init
    conditions = []
    choices = []

    # -- List conditions and choices for C non arable
    for key in cfactor_dict:
        conditions.append(lulc_xarr == key)
        choices.append(
            cfactor_dict[key][0]
            + (cfactor_dict[key][1] - cfactor_dict[key][0])
            * (1 - fcover_xarr.astype(np.float32))
        )

    # -- C non arable calculation
    c_arr_non_arable = np.select(conditions, choices, default=np.nan)

    # -- Merge arable and non arable c values
    c_arr = np.where(np.isnan(c_arr_non_arable), arable_c_xarr, c_arr_non_arable)

    ret = arable_c_xarr.copy(data=c_arr)
    # -- Write c raster
    c_out = os.path.join(tmp_dir, "c.tif")

    fcover_footprint = rasters_rio.get_footprint(fcover_xarr)
    ret = rasters.mask(ret, fcover_footprint)
    rasters.write(ret, c_out)  # , nodata=0)

    return ret


def produce_ls_factor_raw_wbw(dem_path: str, tmp_dir: str, ftep: bool):
    """
    Produce the LS factor raster based on the Slope function
    from sertit_utils and whitebox_workflows for the flowdir
    computation
    Args:
        dem_path (str) : dem path
        tmp_dir (str) : tmp dir path

    Returns:
        xarray of the ls factor raster
    """

    wbe = wbw.WbEnvironment()

    # When computing in the FTEP we use pysheds for fill_depressions to avoid panicking issue
    # identified here: gitlab.unistra.fr/sertit/arcgis-pro/lsi/-/issues/2
    if ftep:
        # -- Compute D8 flow directions
        grid = Grid.from_raster(dem_path)
        dem = grid.read_raster(dem_path)

        # Fill pits in DEM
        pit_filled_dem = grid.fill_pits(dem)

        # Fill depressions in DEM
        flooded_dem = grid.fill_depressions(pit_filled_dem)
        ps_flooded_dem_path = os.path.join(tmp_dir, "flooded_dem.tif")

        grid.to_raster(
            flooded_dem,
            ps_flooded_dem_path,
            # target_view=ps_flooded_dem_path.viewfinder,
            blockxsize=16,
            blockysize=16,
            dtype=np.float32,
        )

        # acc_xarr = rasters.read(ps_flooded_dem_path)
        flooded_dem = wbe.read_raster(ps_flooded_dem_path)

    else:
        # -- Compute D8 flow directions
        dem = wbe.read_raster(dem_path)

        # Fill pits in DEM
        pit_filled_dem = wbe.fill_pits(dem)

        flooded_dem = wbe.fill_depressions(pit_filled_dem)

    # Compute Flow Accumulation
    acc = wbe.fd8_flow_accum(flooded_dem, out_type="cells")

    acc_path = os.path.join(tmp_dir, "flow_acc.tif")
    wbe.write_raster(acc, acc_path, compress=False)

    acc_xarr = rasters.read(acc_path)

    # -- Make slope percentage

    # Compute Slope in degrees
    slope_p_xarr = rasters.slope(dem_path, in_rad=False)

    # -- m calculation
    conditions = [
        slope_p_xarr < 1,
        (slope_p_xarr >= 1) & (slope_p_xarr < 3),
        (slope_p_xarr >= 3) & (slope_p_xarr < 5),
        (slope_p_xarr >= 5) & (slope_p_xarr < 12),
        slope_p_xarr >= 12,
    ]
    choices = [0.2, 0.3, 0.4, 0.5, 0.6]
    m = np.select(conditions, choices, default=np.nan)

    # -- Extract cell size of the dem
    with rasterio.open(dem_path, "r") as dem_dst:
        # dem_epsg = str(dem_dst.crs)[-5:]
        cellsizex, cellsizey = dem_dst.res

    # -- Produce ls
    # -- Equation 1 : https://www.researchgate.net/publication/338535112_A_Review_of_RUSLE_Model
    ls_arr = (
        0.065 + 0.0456 * slope_p_xarr + 0.006541 * np.power(slope_p_xarr, 2)
    ) * np.power(acc_xarr.astype(np.float32) * cellsizex / 22.13, m)

    return slope_p_xarr.copy(data=ls_arr)


def produce_ls_factor_raw(dem_path: str, tmp_dir: str):
    """
    Produce the LS factor raster
    Args:
        dem_path (str) : dem path
        tmp_dir (str) : tmp dir path

    Returns:
        xarray of the ls factor raster
    """

    # -- Compute D8 flow directions
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)

    # Fill pits in DEM
    pit_filled_dem = grid.fill_pits(dem)

    # Fill depressions in DEM
    flooded_dem = grid.fill_depressions(pit_filled_dem)

    # Resolve flats in DEM
    inflated_dem = grid.resolve_flats(flooded_dem)

    # Determine D8 flow directions from DEM
    # Specify directional mapping : http://mattbartos.com/pysheds/flow-directions.html
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # Compute flow directions
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)

    # # -- Export flow directions
    dir_path = os.path.join(tmp_dir, "dir.tif")
    # Workaround because type int64 works on Windows but not Linux while it is not supposed to work on both...
    # https://github.com/mdbartos/pysheds/issues/109
    dtype = None
    if os.name == "posix":
        dtype = np.int32
    grid.to_raster(
        fdir,
        dir_path,
        target_view=fdir.viewfinder,
        blockxsize=16,
        blockysize=16,
        dtype=dtype,
    )

    # Calculate flow accumulation
    acc = grid.accumulation(fdir, dirmap=dirmap)

    # -- Export accumulation
    acc_path = os.path.join(tmp_dir, "acc.tif")
    grid.to_raster(
        acc, acc_path, target_view=acc.viewfinder, blockxsize=16, blockysize=16
    )

    # -- Open acc
    acc_xarr = rasters.read(acc_path)

    # -- Make slope percentage command
    slope_dem_p = os.path.join(tmp_dir, "slope_percent.tif")
    cmd_slope_p = [
        "gdaldem",
        "slope",
        "-compute_edges",
        strings.to_cmd_string(dem_path),
        strings.to_cmd_string(slope_dem_p),
        "-p",
    ]

    # -- Run command
    misc.run_cli(cmd_slope_p)

    # -- Open slope p
    slope_p_xarr = rasters.read(slope_dem_p)

    # -- m calculation
    conditions = [
        slope_p_xarr < 1,
        (slope_p_xarr >= 1) & (slope_p_xarr < 3),
        (slope_p_xarr >= 3) & (slope_p_xarr < 5),
        (slope_p_xarr >= 5) & (slope_p_xarr < 12),
        slope_p_xarr >= 12,
    ]
    choices = [0.2, 0.3, 0.4, 0.5, 0.6]
    m = np.select(conditions, choices, default=np.nan)

    # -- Extract cell size of the dem
    with rasterio.open(dem_path, "r") as dem_dst:
        # dem_epsg = str(dem_dst.crs)[-5:]
        cellsizex, cellsizey = dem_dst.res

    # -- Produce ls
    # -- Equation 1 : https://www.researchgate.net/publication/338535112_A_Review_of_RUSLE_Model
    ls_arr = (
        0.065 + 0.0456 * slope_p_xarr + 0.006541 * np.power(slope_p_xarr, 2)
    ) * np.power(acc_xarr.astype(np.float32) * cellsizex / 22.13, m)

    return slope_p_xarr.copy(data=ls_arr)


def produce_k_outside_europe(aoi_path: str):
    """
    Produce the K index outside Europe
    Args:
        aoi_path (str) : AOI path

    Returns:
        xarray of the K raster
    """

    LOGGER.info("-- Produce the K index outside Europe --")

    # -- Read the aoi file
    aoi_gdf = gpd.read_file(aoi_path)

    # -- Crop hwsd
    crop_hwsd_xarr = rasters.crop(DataPath.HWSD_PATH, aoi_gdf)

    # -- Extract soil information from ce access DB
    raw_db_file_path = DataPath.DBFILE_PATH_SQL
    if isinstance(DataPath.DBFILE_PATH_SQL, cloudpathlib.CloudPath):
        raw_db_file_path = DataPath.DBFILE_PATH_SQL.fspath
    conn = sqlite3.connect(raw_db_file_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, S_SILT, S_CLAY, S_SAND, T_OC, T_TEXTURE, DRAINAGE FROM HWSD_DATA"
    )

    # -- Dictionaries that store the link between the RUSLE codes (P16 methodo) and the HWSD DB codes
    # -- Key = TEXTURE(HWSD), value = b
    b_dict = {0: np.nan, 1: 4, 2: 3, 3: 2}
    # -- Key = DRAINAGE(HWSD), value = c
    c_dict = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, 7: 1}

    # -- K calculation for each type of values
    k_dict = {}
    for row in cursor.fetchall():
        if None in [row[1], row[2], row[3], row[4], row[5]]:
            k = np.nan
        else:
            # -- Silt (%) –silt fraction content (0.002 –0.05 mm)
            s_silt = row[1]
            # -- Clay (%) –clay fraction content (<0.002 mm)
            s_clay = row[2]
            # -- Sand (%) –sand fraction content (0.05 –2mm)
            s_sand = row[3]
            # -- Organic matter (%)
            a = row[4] * 1.724
            # -- Soil structure code used in soil classification
            b = b_dict[row[5]]
            # -- Profile permeability class
            c = c_dict[row[6]]

            # -- VFS (%) –very fine sand fraction content (0.05 –0.1 mm)
            vfs = (0.74 - (0.62 * s_sand / 100)) * s_sand

            # -- Textural factor
            m = (s_silt + vfs) * (100 - s_clay)

            # -- K (Soil erodibility factor)
            k = (
                (2.1 * (m**1.14) * (10**-4) * (12 - a))
                + (3.25 * (b - 2))
                + (2.5 * (c - 3))
            ) / 100

        # -- Add value in the dictionary
        k_dict[row[0]] = k

    conditions = []
    choices = []

    # -- List conditions and choices for C non arable
    for key in k_dict:
        conditions.append(crop_hwsd_xarr == key)
        choices.append(k_dict[key])

    # -- Update arr with k values
    k_arr = np.select(conditions, choices, default=np.nan)

    # Convert to xarr
    k_result_xarr = crop_hwsd_xarr.astype(np.float32).copy(data=k_arr)

    return k_result_xarr


def make_raster_list_to_pre_process(input_dict: dict) -> dict:
    """
    Make a dict with the raster images to pre process from the input dict
    Args:
        input_dict (dict) : Dict that store parameters values

    Returns:
        dict : Dict that store raster to pre process and resampling method
    """

    # --- Extract parameters ---
    aoi_path = input_dict.get(InputParameters.AOI_PATH.value)
    location = input_dict.get(InputParameters.LOCATION.value)
    fcover_path = input_dict.get(InputParameters.FCOVER_PATH.value)
    nir_path = input_dict.get(InputParameters.NIR_PATH.value)
    red_path = input_dict.get(InputParameters.RED_PATH.value)
    satellite_product_path = input_dict.get(
        InputParameters.SATELLITE_PRODUCT_PATH.value
    )
    landcover_name = input_dict.get(InputParameters.LANDCOVER_NAME.value)
    p03_path = input_dict.get(InputParameters.P03_PATH.value)
    ls_path = input_dict.get(InputParameters.LS_PATH.value)
    tmp_dir = input_dict.get(InputParameters.TMP_DIR.value)

    # -- Dict that store landcover name and landcover path
    landcover_path_dict = {
        LandcoverType.CLC.value: DataPath.CLC_PATH,
        LandcoverType.GLC.value: DataPath.GLC_PATH,
        LandcoverType.WC.value: DataPath.WC_PATH,
        LandcoverType.P03.value: p03_path,
    }

    # -- Store landcover path in a variable
    lulc_path = landcover_path_dict[landcover_name]

    raster_dict = {}
    # -- Check location
    if location == LocationType.EUROPE_LEGACY.value:
        # -- Dict that store raster to pre_process and the type of resampling
        raster_dict = {
            "r": [DataPath.R_EURO_PATH, Resampling.bilinear],
            "k": [DataPath.K_EURO_PATH, Resampling.average],
            "lulc": [lulc_path, Resampling.nearest],
            "p": [DataPath.P_EURO_PATH, Resampling.average],
        }

    if location == LocationType.GLOBAL_LEGACY.value:
        # -- Produce k
        k_xarr = produce_k_outside_europe(aoi_path)
        # -- Write k raster
        k_path = os.path.join(tmp_dir, "k_raw.tif")
        rasters.write(k_xarr, k_path)  # , nodata=0)

        # -- Dict that store raster to pre_process and the type of resampling
        raster_dict = {
            "r": [DataPath.R_GLOBAL_PATH, Resampling.bilinear],
            "lulc": [lulc_path, Resampling.nearest],
            "k": [k_path, Resampling.nearest],
        }

    if location == LocationType.EUROPE.value:
        # -- Dict that store raster to pre_process and the type of resampling
        raster_dict = {
            "r": [DataPath.R_EURO_PATH, Resampling.bilinear],
            "k": [DataPath.K_EURO_PATH, Resampling.average],
            "lulc": [lulc_path, Resampling.nearest],
            "p": [DataPath.P_EURO_PATH, Resampling.average],
        }

    if location == LocationType.GLOBAL.value:
        # -- Produce k
        k_xarr = produce_k_outside_europe(aoi_path)
        # -- Write k raster
        k_path = os.path.join(tmp_dir, "k_raw.tif")
        rasters.write(k_xarr, k_path)  # , nodata=0)

        # -- Dict that store raster to pre_process and the type of resampling
        raster_dict = {
            "r": [DataPath.R_GLOBAL_PATH, Resampling.bilinear],
            "lulc": [lulc_path, Resampling.nearest],
            "k": [DataPath.K_GLOBAL_PATH, Resampling.average],
        }

    # -- Add the ls raster to the pre process dict if provided
    if ls_path is not None:
        raster_dict["ls"] = [ls_path, Resampling.bilinear]

    # -- Add bands to the pre process dict if fcover need to be calculated or not
    if fcover_path is None:
        if satellite_product_path is not None:
            prod = Reader().open(satellite_product_path)
            bands = prod.load([NIR, RED], window=aoi_path)
            red = bands[RED]
            nir = bands[NIR]
            raster_dict["red"] = [red, Resampling.bilinear]
            raster_dict["nir"] = [nir, Resampling.bilinear]
        else:
            raster_dict["red"] = [red_path, Resampling.bilinear]
            raster_dict["nir"] = [nir_path, Resampling.bilinear]
    else:
        raster_dict["fcover"] = [fcover_path, Resampling.bilinear]

    return raster_dict


def raster_pre_processing(
    aoi_path: str,
    dst_resolution: int,
    dst_crs: CRS,
    raster_path_dict: dict,
    tmp_dir: str,
) -> dict:
    """
    Pre process a list of raster (clip, reproj, collocate)
    Args:
        aoi_path (str) : AOI path
        dst_resolution (int) : resolution of the output raster files
        dst_crs (CRS) : CRS of the output files
        raster_path_dict : dictionary that store the list of raster (key = alias : value : raster path)
        tmp_dir (str) : tmp directory

    Returns:
        dict : dictionary that store the list of pre process raster (key = alias : value : raster path)
    """
    LOGGER.info("-- RASTER PRE PROCESSING --")
    out_dict = {}

    # -- Loop on path into the dict
    ref_xarr = None
    for _, key in enumerate(raster_path_dict):
        LOGGER.info(f"********* {key} ********")
        # -- Store raster path
        # key = "lulc"
        raster_path = raster_path_dict[key][0]

        # -- Store resampling method
        resampling_method = raster_path_dict[key][1]

        # -- Read AOI
        aoi_gdf = gpd.read_file(aoi_path)

        # -- Read raster
        if isinstance(raster_path, xarray.DataArray):
            raster_xarr = raster_path
        else:
            raster_xarr = rasters.read(raster_path, window=aoi_gdf)

        # -- Add path to the dictionary
        if ref_xarr is None:
            # -- Re project raster and resample

            raster_reproj_xarr = raster_xarr.odc.reproject(
                how=dst_crs,
                resolution=dst_resolution,
                resampling=resampling_method,
                dst_nodata=rasters.FLOAT_NODATA,
            )

            # -- Crop raster with AOI
            raster_crop_xarr = rasters.crop(raster_reproj_xarr, aoi_gdf, from_disk=True)

            # -- Write masked raster
            raster_path_out = os.path.join(tmp_dir, f"{key}.tif")
            rasters.write(raster_crop_xarr, raster_path_out)

            # -- Store path result in a dict
            out_dict[key] = raster_path_out

            # -- Copy the reference xarray
            ref_xarr = raster_crop_xarr.copy()

        else:
            # -- Collocate raster
            # LOGGER.info('Collocate')
            raster_collocate_xarr = rasters.collocate(
                ref_xarr, raster_xarr, resampling_method
            )

            # -- Mask raster with AOI
            raster_masked_xarr = rasters.mask(raster_collocate_xarr, aoi_gdf)

            # -- Write masked raster
            raster_path_out = os.path.join(tmp_dir, f"{key}.tif")
            rasters.write(raster_masked_xarr, raster_path_out)

            # -- Store path result in a dict
            out_dict[key] = raster_path_out

        gc.collect()

    return out_dict


def produce_a_arr(input_dict, ftep):
    """
    Produce average annual soil loss (ton/ha/year) with the RUSLE model.
    Args:
        r_xarr : multi-annual average index xarray
        k_xarr : susceptibility of a soil to erode xarray
        ls_xarr  :combined Slope Length and Slope Steepness factor xarray
        c_xarr  : Cover management factor xarray
        p_xarr  : support practices factor xarray

    Returns:
        xarray of the average annual soil loss (ton/ha/year)
    """

    output_resolution = input_dict.get(InputParameters.OUTPUT_RESOLUTION.value)
    ref_epsg = input_dict.get(InputParameters.REF_EPSG.value)
    aoi_path = input_dict.get(InputParameters.AOI_PATH.value)
    tmp_dir = input_dict.get(InputParameters.TMP_DIR.value)

    # --- Make the list of raster to pre-process ---
    raster_dict = make_raster_list_to_pre_process(input_dict)
    gc.collect()

    # --- Pre-process raster ---
    # -- Run pre process
    post_process_dict = raster_pre_processing(
        aoi_path, output_resolution, CRS.from_epsg(ref_epsg), raster_dict, tmp_dir
    )

    ls_path = input_dict.get(InputParameters.LS_PATH.value)
    aoi_path = input_dict.get(InputParameters.AOI_PATH.value)

    aoi_gdf = vectors.read(aoi_path)

    # -- Check if ls need to be calculated or not
    if ls_path is None:
        ls_xarr = produce_ls(input_dict, post_process_dict, ftep)
    else:
        ls_xarr = rasters.read(post_process_dict["ls"], window=aoi_gdf)

    # Process C
    c_xarr = produce_c(input_dict, post_process_dict)

    # Process p
    # p_xarr = produce_p(input_dict, post_process_dict, c_xarr)

    # Open r
    r_xarr = rasters.read(post_process_dict["r"], window=aoi_gdf)

    #  Open k
    k_xarr = rasters.read(post_process_dict["k"], window=aoi_gdf)

    LOGGER.info(
        "-- Produce average annual soil loss (ton/ha/year) with the RUSLE model --"
    )

    return r_xarr * k_xarr * ls_xarr * c_xarr  # * p_xarr  # TODO Check the Pfactor


def produce_a_reclass_arr(a_xarr):
    """
    Produce reclassified a
    Args:
        a_xarr : a xarray

    Returns:
        xarray of the reclassified a raster
    """
    LOGGER.info("-- Produce the reclassified a --")

    # -- List conditions and choices
    conditions = [
        (a_xarr.data < 6.7),
        (a_xarr.data >= 6.7) & (a_xarr.data < 11.2),
        (a_xarr.data >= 11.2) & (a_xarr.data < 22.4),
        (a_xarr.data >= 22.4) & (a_xarr.data < 33.6),
        (a_xarr.data >= 33.6),
    ]

    for i, condition in enumerate(conditions):
        a_xarr.data = xarray.where(condition, i + 1, a_xarr.data)

    return a_xarr


def aoi_buffer(input_dict):
    aoi_raw_path = input_dict.get(InputParameters.AOI_PATH.value)
    tmp_dir = input_dict.get(InputParameters.TMP_DIR.value)
    ref_epsg = input_dict.get(InputParameters.REF_EPSG.value)

    # Write wkt string input to shapefile
    if aoi_raw_path.startswith("POLYGON"):
        aoi_gpd_wkt = gpd.GeoSeries.from_wkt([aoi_raw_path])
        aoi_raw_path_wkt = os.path.join(tmp_dir, "aoi_from_wkt.shp")

        aoi_gpd_wkt_4326 = aoi_gpd_wkt.set_crs(epsg=4326)
        aoi_gpd_wkt_4326.to_file(aoi_raw_path_wkt)
        aoi_raw_path = aoi_raw_path_wkt

    # - Open aoi
    aoi_gdf = vectors.read(aoi_raw_path)

    # -- Extract the epsg code from the reference system parameter
    if ref_epsg is None:
        ref_epsg = aoi_gdf.estimate_utm_crs().to_epsg()
        input_dict[InputParameters.REF_EPSG.value] = ref_epsg

    # - Reproject aoi
    aoi_gdf = aoi_gdf.to_crs(ref_epsg)

    # - Write original AOI to file
    aoi_path = os.path.join(tmp_dir, "aoi.shp")
    aoi_gdf.to_file(aoi_path)

    # - Apply buffer
    aoi_gdf.geometry = aoi_gdf.geometry.buffer(DataPath.AOI_BUFFER)

    # Export the new aoi
    aoi_path = os.path.join(tmp_dir, "aoi_buff5.shp")
    aoi_gdf.to_file(aoi_path)

    # Change the path in the input_dict
    input_dict[InputParameters.AOI_PATH.value] = aoi_path


def create_tmp_dir(input_dict):
    output_dir = input_dict.get(InputParameters.OUTPUT_DIR.value)
    tmp_dir = input_dict.get(InputParameters.TMP_DIR.value)
    # --- Create temp_dir if not exist ---
    if tmp_dir is None or not AnyPath(tmp_dir).is_absolute():
        tmp_dir = os.path.join(output_dir, "temp_dir")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    input_dict[InputParameters.TMP_DIR.value] = tmp_dir


def produce_ls(input_dict, post_process_dict, ftep):
    dem_name = input_dict.get(InputParameters.DEM_NAME.value)
    other_dem_path = input_dict.get(InputParameters.OTHER_DEM_PATH.value)
    aoi_path = input_dict.get(InputParameters.AOI_PATH.value)
    ref_epsg = input_dict.get(InputParameters.REF_EPSG.value)
    tmp_dir = input_dict.get(InputParameters.TMP_DIR.value)

    aoi_gdf = vectors.read(aoi_path)
    ref_crs = CRS.from_epsg(ref_epsg)
    # -- Dict that store dem_name with path
    dem_dict = {
        DemType.EUDEM.value: DataPath.EUDEM_PATH,
        DemType.SRTM.value: DataPath.SRTM30_PATH,
        DemType.MERIT.value: DataPath.MERIT_PATH,
        DemType.COPDEM_30.value: DataPath.COPDEM30_PATH,
        DemType.OTHER.value: other_dem_path,
    }

    # --- Pre-process the DEM ---
    # -- Extract DEM path
    dem_path = dem_dict[dem_name]

    # Read the raster
    dem_xarr = rasters.read(dem_path, window=aoi_gdf)

    # -- Reproj DEM
    dem_reproj_xarr = dem_xarr.rio.reproject(ref_crs, resampling=Resampling.bilinear)

    # -- Crop DEM
    dem_crop_xarr = rasters.crop(dem_reproj_xarr, aoi_gdf, from_disk=True)

    # --- Write reproj DEM
    dem_reproj_path = os.path.join(tmp_dir, "dem.tif")
    rasters.write(
        dem_crop_xarr,
        dem_reproj_path,
        compress="deflate",
        predictor=1,
        dtype=np.float32,
    )  # , nodata=0)

    # --- Produce ls ---
    ls_raw_xarr = produce_ls_factor_raw_wbw(dem_reproj_path, tmp_dir, ftep)

    # -- Collocate ls with the other results
    ls_xarr = rasters.collocate(
        rasters.read(
            post_process_dict[list(post_process_dict.keys())[0]], window=aoi_gdf
        ),
        ls_raw_xarr,
        Resampling.bilinear,
    )

    # -- Write ls
    ls_path = os.path.join(tmp_dir, "ls.tif")
    rasters.write(ls_xarr, ls_path)  # , nodata=0)
    return ls_xarr


def produce_p(input_dict, post_process_dict, c_xarr):
    location = input_dict.get(InputParameters.LOCATION.value)
    tmp_dir = input_dict.get(InputParameters.TMP_DIR.value)
    aoi_path = input_dict.get(InputParameters.AOI_PATH.value)

    aoi_gdf = vectors.read(aoi_path)
    # -- Produce p if location is GLOBAL
    if (
        location == LocationType.GLOBAL.value
        or location == LocationType.GLOBAL_LEGACY.value
    ):
        # -- Produce p
        p_value = 1  # Can change
        p_xarr = xarray.full_like(c_xarr, fill_value=p_value)

        # -- Write p
        p_path = os.path.join(tmp_dir, "p.tif")
        rasters.write(p_xarr, p_path)  # , nodata=0)
    elif (
        location == LocationType.EUROPE.value
        or location == LocationType.EUROPE_LEGACY.value
    ):
        p_xarr = rasters.read(post_process_dict["p"], window=aoi_gdf)
    else:
        raise ValueError(f"Unknown Location Type {location}")

    return p_xarr


def compute_statistics(input_dict, susceptibility_path):
    """
    This function allows the zonal statistics and formatting of the geodataframe
    for the LSI statistics based on Deparments level 0, 1 and 2 for the GADM layer
    Args:
        gadm_layer: geodataframe already cropped to the AOI
        raster_path: Path for the LSI raster
    Returns:
        Geodataframe with the statistics data for each of the Levels 0,1 and 2 availables
        in the AOI.
    """
    tmp_dir = input_dict.get(InputParameters.TMP_DIR.value)
    ref_epsg = input_dict.get(InputParameters.REF_EPSG.value)
    ref_crs = CRS.from_epsg(ref_epsg)

    # Load the original AOI not cropped
    original_aoi = vectors.read(os.path.join(tmp_dir, "aoi.shp"))

    # Read GADM layer and overlay with AOI
    aoi_path = input_dict.get(InputParameters.AOI_PATH.value)
    aoi = vectors.read(aoi_path)
    aoi_gadm = aoi
    gadm_buffer = 15000  # Big buffer to avoid missing departments
    aoi_gadm.geometry = aoi_gadm.geometry.buffer(gadm_buffer)

    with warnings.catch_warnings():  # For cases of polygons with more than 100 parts
        warnings.simplefilter("ignore")
        gadm = vectors.read(DataPath.GADM_PATH, window=aoi_gadm)
    gadm = gadm.to_crs(ref_crs)
    gadm_layer = gpd.clip(gadm, original_aoi)

    breaks = [0, 6.7, 11.2, 22.4, 33.6]

    # Prepare the geodataframe structure
    gadm_df = gadm_layer[["NAME_0", "NAME_1", "NAME_2", "geometry"]]

    # Prepare the three (0, 1, 2) levels of deparments:
    # Level0
    gadm_0 = gadm_df.dissolve(by="NAME_0").reset_index()
    gadm_0["LEVL_CODE"] = 0
    gadm_0 = gadm_0[["NAME_0", "LEVL_CODE", "geometry"]].rename(
        columns={"NAME_0": "NUTS_NAME"}
    )
    # Level1
    gadm_1 = gadm_df.dissolve(by="NAME_1").reset_index()
    gadm_1["LEVL_CODE"] = 1
    gadm_1 = gadm_1[["NAME_1", "LEVL_CODE", "geometry"]].rename(
        columns={"NAME_1": "NUTS_NAME"}
    )
    # Level2
    gadm_2 = gadm_df.dissolve(by="NAME_2").reset_index()
    gadm_2["LEVL_CODE"] = 2
    gadm_2 = gadm_2[["NAME_2", "LEVL_CODE", "geometry"]].rename(
        columns={"NAME_2": "NUTS_NAME"}
    )

    # GADM layer for our AOI
    rusle_stats = pd.concat([gadm_0, gadm_1, gadm_2]).reset_index()
    rusle_stats["FER_ER_ave"] = 0.0
    rusle_stats = rusle_stats[["LEVL_CODE", "NUTS_NAME", "FER_ER_ave", "geometry"]]

    # Compute zonal statistics
    stats = zonal_stats(rusle_stats, susceptibility_path, stats=["mean"])

    # Add reclassification of Code (1 to 5) and Class (Very low to Severe)
    def reclassify_code(value):
        try:
            if value > breaks[0] and value <= breaks[1]:
                return 1.0
            elif value > breaks[1] and value <= breaks[2]:
                return 2.0
            elif value > breaks[2] and value <= breaks[3]:
                return 3.0
            elif value > breaks[3] and value <= breaks[4]:
                return 4.0
            elif value > breaks[4]:
                return 5.0
            else:
                return None
        except TypeError:
            return None

    def reclassify_class(value):
        try:
            return {1: "Very low", 2: "Low", 3: "Moderate", 4: "High", 5: "Severe"}.get(
                value, "No data"
            )
        except TypeError:
            return "No data"

    rusle_code = [{"rusle_code": reclassify_code(stat["mean"])} for stat in stats]
    rusle_class = [
        {"rusle_class": reclassify_class(rusle["rusle_code"])} for rusle in rusle_code
    ]
    # Write average, code and class to GeoDataFrame
    rusle_stats["FER_ER_ave"] = pd.DataFrame(stats)
    rusle_stats["ER_code"] = pd.DataFrame(rusle_code)
    rusle_stats["ER_class"] = pd.DataFrame(rusle_class)

    return rusle_stats


def rusle_core(input_dict: dict, ftep) -> None:
    """
    Produce average annual soil loss (ton/ha/year) with the RUSLE model.

    Args:
        input_dict (dict) : Input dict containing all needed values
    """
    logging.info("Check Rusle parameters")
    create_tmp_dir(input_dict)
    aoi_buffer(input_dict)
    check_parameters(input_dict)

    # --- Extract parameters ---
    output_dir = input_dict.get(InputParameters.OUTPUT_DIR.value)
    tmp_dir = input_dict.get(InputParameters.TMP_DIR.value)

    # Produce a with RUSLE model
    a_xarr = produce_a_arr(input_dict, ftep)
    gc.collect()

    # Reload AOI to clip the output raster to the AOI
    aoi_path = os.path.join(tmp_dir, "aoi.shp")
    aoi = vectors.read(aoi_path)

    # -- Write the a raster
    a_path = os.path.join(output_dir, "MeanSoilLoss.tif")
    a_xarr_clipped = a_xarr.rio.clip(aoi.geometry.values, aoi.crs)
    rasters.write(a_xarr_clipped, a_path)
    del a_xarr_clipped
    gc.collect()

    # -- Reclass a
    a_reclas_xarr = produce_a_reclass_arr(a_xarr)
    del a_xarr
    gc.collect()

    # -- Write the a raster
    a_reclass_path = os.path.join(output_dir, "ErosionRisk.tif")
    a_reclas_xarr_clipped = a_reclas_xarr.rio.clip(aoi.geometry.values, aoi.crs)
    rasters.write(a_reclas_xarr_clipped, a_reclass_path)

    # Stats
    LOGGER.info("-- Computing RUSLE statistics (FER_ER_av)")
    a_path = os.path.join(output_dir, "MeanSoilLoss.tif")
    rusle_stats = compute_statistics(input_dict, a_path)

    LOGGER.info("-- Writing RUSLE statistics in memory")
    # Write statistics in memory
    vectors.write(rusle_stats, os.path.join(output_dir, "FER_ER_ave.shp"))

    return
