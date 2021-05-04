"""
Processing the Mean (annual) soil loss (in ton/ha/year) with the RUSLE model.

"""

__author__ = "Ledauphin Thomas"
__contact__ = "tledauphin@unistra.fr"
__python__ = "3.5.0"
__created__ = "24/02/2021"
__update__ = "04/05/2021"
__copyrights__ = "(c) SERTIT 2021"

import os
import logging
from enum import unique

import numpy as np
import xarray as xr
import rasterio
from rasterio.crs import CRS
import geopandas as gpd
from rasterio.enums import Resampling
from rasterstats import zonal_stats

from sertit import strings, misc, rasters, files, logs
from eoreader.bands.index import _norm_diff
from sertit.misc import ListEnum
from sertit.rasters import XDS_TYPE

from pysheds.grid import Grid

import pyodbc
import pyproj

import arcpy
import shapely
from shapely import speedups

# Commande pour supprimer l'erreur suivante : "ValueError: GEOSGeom_createLinearRing_r returned a NULL pointer"
shapely.speedups.disable()

np.seterr(divide='ignore', invalid='ignore')

DEBUG = False
LOGGER = logging.getLogger("rusle")

WORLD_COUNTRIES_PATH = r"\\ds2\database02\BASES_DE_DONNEES\GLOBAL\World_countries_poly\world_countries_poly.shp"
EUROPE_COUNTRIES_PATH = r"\\ds2\database02\BASES_DE_DONNEES\GLOBAL\World_countries_poly\europe_countries_poly.shp"
HWSD_PATH = r"\\ds2\database02\BASES_DE_DONNEES\GLOBAL\FAO_Harmonized_World_Soil_Database\hwsd.tif"
DBFILE_PATH = r"\\ds2\database02\BASES_DE_DONNEES\GLOBAL\FAO_Harmonized_World_Soil_Database\HWSD.mdb"

R_EURO_PATH = r"\\ds2\database02\BASES_DE_DONNEES\GLOBAL\European_Soil_Database_v2\Rfactor\Rf_gp1.tif"
K_EURO_PATH = r"\\ds2\database02\BASES_DE_DONNEES\GLOBAL\European_Soil_Database_v2\Kfactor\K_new_crop.tif"
LS_EURO_PATH = r"\\ds2\database02\BASES_DE_DONNEES\GLOBAL\European_Soil_Database_v2\LS_100m\EU_LS_Mosaic_100m.tif"
P_EURO_PATH = r"\\ds2\database02\BASES_DE_DONNEES\GLOBAL\European_Soil_Database_v2\Pfactor\EU_PFactor_V2.tif"

CLC_PATH = r"\\ds2\database02\BASES_DE_DONNEES\GLOBAL\Corine_Land_Cover\CLC_2018\clc2018_clc2018_v2018_20_raster100m\CLC2018_CLC2018_V2018_20.tif"
R_GLOBAL_PATH = r"\\ds2\database02\BASES_DE_DONNEES\GLOBAL\Global_Rainfall_Erosivity\GlobalR_NoPol.tif"

GLC_PATH = r"\\ds2\database02\BASES_DE_DONNEES\GLOBAL\Global_Land_Cover\2019\PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif"
GC_PATH = r"\\ds2\database02\BASES_DE_DONNEES\GLOBAL\Globcover_2009\Globcover2009_V2.3_Global_\GLOBCOVER_L4_200901_200912_V2.3.tif"
GL_PATH = ""  # A ajouter

EUDEM_PATH = r"\\ds2\database02\BASES_DE_DONNEES\GLOBAL\EUDEM_v2\eudem_dem_3035_europe.tif"
SRTM30_PATH = r"\\ds2\database02\BASES_DE_DONNEES\GLOBAL\SRTM_30m_v4\index.vrt"
MERIT_PATH = r"\\ds2\database02\BASES_DE_DONNEES\GLOBAL\MERIT_Hydrologically_Adjusted_Elevations\MERIT_DEM.vrt"

speedups.disable()


class ArcPyLogHandler(logging.handlers.RotatingFileHandler):
    """
    Custom logging class that bounces messages to the arcpy tool window as well
    as reflecting back to the file.
    """

    def emit(self, record):
        """
        Write the log message
        """

        try:
            msg = (record.msg % record.args)
        except:
            try:
                msg = record.msg.format(record.args)
            except:
                msg = record.msg

        if record.levelno >= logging.ERROR:
            arcpy.AddError(msg)
        elif record.levelno >= logging.WARNING:
            arcpy.AddWarning(msg)
        elif record.levelno >= logging.INFO:
            arcpy.AddMessage(msg)

        super(ArcPyLogHandler, self).emit(record)


@unique
class LandcoverType(ListEnum):
    """
    List of the Landcover type
    """
    CLC = "Corine Land Cover - 2018 (100m)"
    GLC = "Global Land Cover - Copernicus 2020 (100m)"
    GC = "GlobCover - ESA 2005 (300m)"
    GL = "GlobeLand30 - China 2020 (30m)"
    P03 = "P03"


@unique
class LocationType(ListEnum):
    """
    List of the location
    """
    EUROPE = "Europe"
    GLOBAL = "Global"


@unique
class MethodType(ListEnum):
    """
    List of the provided method
    """
    TO_BE_CALCULATED = "To be calculated"
    ALREADY_PROVIDED = "Already provided"


@unique
class DemType(ListEnum):
    """
    List of the DEM
    """
    EUDEM = "EUDEM 25m"
    SRTM = "SRTM 30m"
    MERIT = "MERIT 5 deg"
    OTHER = "Other"


class LandcoverStructure(ListEnum):
    """
    List of the Landcover type (Coding)
    """
    CLC = "Corine Land Cover - 2018 (100m)"
    GLC = "Global Land Cover - Copernicus 2020 (100m)"
    GC = "GlobCover - ESA 2005 (300m)"
    GL = "GlobeLand30 - China 2020 (30m)"
    P03 = "Corine Land Cover - 2018 (100m)"


def produce_fcover(red_path: str, nir_path: str, aoi_path: str, tmp_dir: str) -> (XDS_TYPE):  # OK
    """
    Produce the fcover index
    Args:
        red_path (str): Red band path
        nir_path (str): nir band path
        aoi_path (str): AOI path
        tmp_dir (str) : Temp dir where the cropped Raster will be store

    Returns:
        np.ndarray : ndarray of the fcover raster
        dict : metadata of the fcover raster
    """
    arcpy.AddMessage("-- Produce the fcover index --")

    # Read and resample red
    with rasterio.open(red_path) as red_dst:
        red_xarr = rasters.read(red_dst)

    # Read and resample nir
    with rasterio.open(nir_path) as nir_dst:
        nir_xarr = rasters.read(nir_dst)

    # Process ndvi
    ndvi_xarr = _norm_diff(nir_xarr, red_xarr)

    # Reprojet the aoi if needed
    with rasterio.open(red_path) as raster_dst:
        aoi_gdf = gpd.read_file(aoi_path)
        if aoi_gdf.crs != raster_dst.crs:
            aoi_gdf = aoi_gdf.to_crs(raster_dst.crs)

    # Crop the ndvi with reprojected AOI
    ndvi_crop_xarr = rasters.crop(ndvi_xarr, aoi_gdf)

    # Write the ndvi cropped
    ndvi_crop_path = os.path.join(files.to_abspath(tmp_dir), 'ndvi.tif')
    rasters.write(ndvi_crop_xarr, ndvi_crop_path, nodata=0)

    # Extract NDVI min and max inside the AOI
    ndvi_stat = zonal_stats(aoi_gdf, ndvi_crop_path, stats="min max")
    ndvi_min = ndvi_stat[0]['min']
    ndvi_max = ndvi_stat[0]['max']

    # Fcover calculation
    fcover_xarr = (ndvi_crop_xarr.astype(np.float32) - ndvi_min) / (ndvi_max - ndvi_min)

    return fcover_xarr


def produce_c_arable_europe(aoi_path: str, raster_xarr: XDS_TYPE) -> (XDS_TYPE):  # OK
    """
    Produce C arable index over Europe
    Args:
        aoi_path (str): aoi path
        raster_xarr (XDS_TYPE): lulc xarray

    Returns:
        XDS_TYPE : xarray of the c arable raster
    """

    arcpy.AddMessage("-- Produce C arable index over Europe --")

    arable_c_dict = {
        "Finland": 0.231,
        'France': 0.20200000000,
        'Germany': 0.20000000000,
        'Greece': 0.28000000000,
        'Hungary': 0.27500000000,
        'Ireland': 0.20200000000,
        'Italy': 0.21100000000,
        'Latvia': 0.23700000000,
        'Luxembourg': 0.21500000000,
        'Malta': 0.43400000000,
        'Netherlands': 0.26000000000,
        'Poland': 0.24700000000,
        'Portugal': 0.35200000000,
        'Romania': 0.29600000000,
        'Slovakia': 0.23500000000,
        'Slovenia': 0.24800000000,
        'Spain': 0.28900000000,
        'Sweden': 0.23700000000,
        'the former Yugoslav Republic of Macedonia': 0.25500000000,
        'United Kingdom': 0.17700000000,
        'Croatia': 0.25500000000
    }

    # Reproject aoi to wgs84
    aoi = gpd.read_file(aoi_path)
    crs_4326 = CRS.from_epsg(4326)
    if aoi.crs != crs_4326:
        aoi = aoi.to_crs(crs_4326)

    # Extract europe countries
    world_countries = gpd.read_file(WORLD_COUNTRIES_PATH, bbox=aoi.envelope)
    europe_countries = world_countries[world_countries['CONTINENT'] == 'Europe']

    # Initialize arable arr
    arable_c_xarr = xr.full_like(raster_xarr, fill_value=0.27)

    # Reproject europe_countries
    crs_arable = arable_c_xarr.rio.crs
    if europe_countries.crs != crs_arable:
        europe_countries = europe_countries.to_crs(crs_arable)

    # Update arable_arr with arable_dict
    for key in list(europe_countries['COUNTRY']):
        geoms = [feature for feature in europe_countries[europe_countries['COUNTRY'] == key]['geometry']]
        arable_c_xarr = rasters.paint(arable_c_xarr, geoms, value=arable_c_dict[key])

    # Mask  result with aoi
    arable_c_xarr = rasters.mask(arable_c_xarr, aoi)

    return arable_c_xarr


def produce_c(lulc_xarr: XDS_TYPE, fcover_xarr: XDS_TYPE, aoi_path: str, lulc_name: str) -> (
        XDS_TYPE):  # NOK
    """
    Produce C index
    Args:
        lulc_xarr (XDS_TYPE): lulc xarray
        fcover_xarr (XDS_TYPE) :fcover xarray
        aoi_path (str): aoi path
        lulc_name (str) : name of the LULC

    Returns:
        XDS_TYPE : xdarray of the c index raster
    """
    arcpy.AddMessage("-- Produce C index --")
    print(lulc_name)
    # Identify Cfactor
    # Cfactor dict and c_arr_arable
    if lulc_name == LandcoverStructure.CLC.value:
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
            324: [0.003, 0.05],
            331: [0, 0],
            332: [0, 0],
            333: [0.1, 0.45],
            334: [0.1, 0.55],
            335: [0, 0]
        }
        # Produce arable c
        arable_c_xarr = produce_c_arable_europe(aoi_path, lulc_xarr)
        c_xarr_arable = rasters.where(lulc_xarr == 211, arable_c_xarr, np.nan)
    # Global Land Cover - Copernicus 2020 (100m)
    elif lulc_name == LandcoverStructure.GLC.value:
        cfactor_dict = {
            111: [0.0001, 0.003],
            113: [0.0001, 0.003],
            112: [0.0001, 0.003],
            114: [0.0001, 0.003],
            115: [0.0001, 0.003],
            116: [0.0001, 0.003],
            121: [0.0001, 0.003],
            123: [0.0001, 0.003],
            122: [0.0001, 0.003],
            124: [0.0001, 0.003],
            125: [0.0001, 0.003],
            126: [0.0001, 0.003],
            20: [0.003, 0.05],
            30: [0.01, 0.08],
            90: [0.01, 0.08],
            100: [0.01, 0.1],
            60: [0.1, 0.45],
            40: [0.07, 0.2],
            70: [0, 0]
        }
        # Produce arable c
        c_arr_arable = np.where(np.isin(lulc_xarr, [11, 14, 20]), 0.27, np.nan)
        c_xarr_arable = lulc_xarr.copy(data=c_arr_arable)

    # GlobCover - ESA 2005 (300m)
    elif lulc_name == LandcoverStructure.GC.value:
        cfactor_dict = {
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
            220: [0, 0]
        }
        # Produce arable c
        c_xarr_arable = rasters.where(lulc_xarr == 40, 0.27, np.nan)

    # GlobeLand30 - China 2020 (30m)
    elif lulc_name == LandcoverStructure.GL.value:
        cfactor_dict = {
            20: [0.00010000000, 0.00250000000],
            30: [0.01000000000, 0.07000000000],
            40: [0.01000000000, 0.08000000000],
            90: [0.00000000000, 0.00000000000],
            100: [0.00000000000, 0.00000000000],
            110: [0.10000000000, 0.45000000000]
        }
        # Produce arable c
        c_xarr_arable = rasters.where(lulc_xarr == 10, 0.27, np.nan)

    # List init
    conditions = []
    choices = []

    # List conditions and choices for C non arable
    for key in cfactor_dict:
        conditions.append(lulc_xarr == key)
        choices.append(cfactor_dict[key][0] + (cfactor_dict[key][1] - cfactor_dict[key][0]) * (
                1 - fcover_xarr.astype(np.float32)))

    # C non arable calculation
    c_arr_non_arable = np.select(conditions, choices, default=np.nan)

    # Merge arable and non arable c values
    c_arr = np.where(np.isnan(c_arr_non_arable), c_xarr_arable, c_arr_non_arable)

    c_xarr = c_xarr_arable.copy(data=c_arr)

    return c_xarr


def spatial_resolution(raster_path: str) -> (float, float):  # OK
    """
    Extract the spatial resolution of a raster X, Y
    Args:
        raster_path (str) : raster path

    Returns:
    float : X resolution
    float :Y resolution
    """
    raster = rasterio.open(raster_path)
    t = raster.transform
    x = t[0]
    y = -t[4]
    return x, y


def produce_ls_factor(dem_path: str, tmp_dir: str) -> (XDS_TYPE):  # OK
    """
    Produce the LS factor raster
    Args:
        dem_path (str) : dem path
        tmp_dir (str) : tmp dir path

    Returns:
        XDS_TYPE : xarray of the ls factor raster
    """

    arcpy.AddMessage("-- Produce the LS factor --")

    # Compute D8 flow directions
    grid = Grid.from_raster(dem_path, data_name='dem')

    # Fill dem
    grid.fill_depressions(data='dem', out_name='filled_dem')

    # Resolve flat
    grid.resolve_flats(data='filled_dem', out_name='inflated_dem')

    # Produce dir
    grid.flowdir('inflated_dem', out_name='dir')

    # Export flow directions
    dir_path = os.path.join(tmp_dir, 'dir.tif')
    grid.to_raster('dir', dir_path)

    # Extract epsg code of the dem #  A voir si conserve le weight et améliorer cette partie du script
    with rasterio.open(dem_path, "r") as dem_dst:
        dem_epsg = str(dem_dst.crs)[-5:]

    # Compute areas of each cell in new projection #  A voir si conserve le weight et améliorer cette partie du script
    new_crs = pyproj.Proj('+init=epsg:{}'.format(dem_epsg))
    areas = grid.cell_area(as_crs=new_crs, inplace=False)

    # Weight each cell by its relative area
    weights = (areas / areas.max()).ravel()

    # Compute accumulation
    # grid.accumulation(data='dir', out_name='acc')
    grid.accumulation(data='dir', weights=weights, out_name='acc')

    # Export  accumulation
    acc_path = os.path.join(tmp_dir, 'acc.tif')
    grid.to_raster('acc', acc_path)

    # Extract dem spatial resolution
    cellsizex, cellsizey = spatial_resolution(dem_path)

    # Open acc
    with rasterio.open(acc_path) as acc_dst:
        acc_xarr = rasters.read(acc_dst)

    # Make slope percentage command
    slope_dem_p = os.path.join(tmp_dir, "slope_percent.tif")
    cmd_slope_p = ["gdaldem",
                   "slope",
                   "-compute_edges",
                   strings.to_cmd_string(dem_path),
                   strings.to_cmd_string(slope_dem_p), "-p"]

    # Run command
    misc.run_cli(cmd_slope_p)

    # Open slope p
    with rasterio.open(slope_dem_p) as slope_dst:
        slope_p_xarr = rasters.read(slope_dst)

    # m calculation
    conditions = [slope_p_xarr < 1, (slope_p_xarr >= 1) & (slope_p_xarr > 3), (slope_p_xarr >= 3) & (slope_p_xarr > 5),
                  (slope_p_xarr >= 5) & (slope_p_xarr > 12), slope_p_xarr >= 12]
    choices = [0.2, 0.3, 0.4, 0.5, 0.6]
    m = np.select(conditions, choices, default=np.nan)

    # Produce ls
    # Equation 1 : file:///C:/Users/TLEDAU~1/AppData/Local/Temp/Ghosal-DasBhattacharya2020_Article_AReviewOfRUSLEModel.pdf
    ls_arr = (0.065 + 0.0456 * slope_p_xarr + 0.006541 * np.power(slope_p_xarr, 2)) * np.power(
        acc_xarr.astype(np.float32) * cellsizex / 22.13, m)

    # Write ls
    ls_xarr = slope_p_xarr.copy(data=ls_arr, deep=True)

    return ls_xarr


def produce_k_outside_europe(aoi_path: str) -> (XDS_TYPE):
    """
    Produce the K index outside Europe
    Args:
        aoi_path (str) : AOI path

    Returns:
        XDS_TYPE : xarray of the K raster
    """

    arcpy.AddMessage("-- Produce the K index outside Europe --")

    # Read the aoi file
    aoi_gdf = gpd.read_file(aoi_path)

    # Crop hwsd
    crop_hwsd_xarr = rasters.crop(HWSD_PATH, aoi_gdf)

    # Extract soil information from ce access DB
    conn = pyodbc.connect(r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=' + DBFILE_PATH + ';')
    cursor = conn.cursor()
    cursor.execute('SELECT id, S_SILT, S_CLAY, S_SAND, T_OC, T_TEXTURE, DRAINAGE FROM HWSD_DATA')

    # Dictionaries that store the link between the RUSLE codes (P16 methodo) and the HWSD DB codes
    # Key = TEXTURE(HWSD), value = b
    b_dict = {
        0: np.nan,
        1: 4,
        2: 3,
        3: 2
    }
    # Key = DRAINAGE(HWSD), value = c
    c_dict = {
        1: 6,
        2: 5,
        3: 4,
        4: 3,
        5: 2,
        6: 1,
        7: 1
    }

    # K calculation for each type of values
    k_dict = {}
    for row in cursor.fetchall():
        if None in [row[1], row[2], row[3], row[4], row[5]]:
            k = np.nan
        else:
            # Silt (%) –silt fraction content (0.002 –0.05 mm)
            s_silt = row[1]
            # Clay (%) –clay fraction content (<0.002 mm)
            s_clay = row[2]
            # Sand (%) –sand fraction content (0.05 –2mm)
            s_sand = row[3]
            # Organic matter (%)
            a = row[4] * 1.724
            # Soil structure code used in soil classification
            b = b_dict[row[5]]
            # Profile permeability class
            c = c_dict[row[6]]

            # VFS (%) –very fine sand fraction content (0.05 –0.1 mm)
            vfs = (0.74 - (0.62 * s_sand / 100)) * s_sand

            # Textural factor
            m = (s_silt + vfs) * (100 - s_clay)

            # K (Soil erodibility factor)
            k = ((2.1 * (m ** 1.14) * (10 ** -4) * (12 - a)) + (3.25 * (b - 2)) + (2.5 * (c - 3))) / 100

        # Add value in the dictionary
        k_dict[row[0]] = k

    conditions = []
    choices = []

    # List conditions and choices for C non arable
    for key in k_dict:
        conditions.append(crop_hwsd_xarr == key)
        choices.append(k_dict[key])

    # Update arr with k values
    k_arr = np.select(conditions, choices, default=np.nan)

    return crop_hwsd_xarr.copy(data=k_arr, deep=True)


def raster_pre_processing(aoi_path: str, dst_resolution: int, dst_crs: str, raster_path_dict: dict,
                          tmp_dir: str) -> dict:  # OK
    """
    Pre process a list of raster (clip, reproj, collocate)
    Args:
        aoi_path (str) : AOI path
        dst_resolution (int) : resolution of the output raster files
        dst_crs (str) : CRS of the output files
        raster_path_dict : dictionary that store the list of raster (key = alias : value : raster path)
        tmp_dir (str) : tmp directory

    Returns:
        dict : dictionary that store the list of pre process raster (key = alias : value : raster path)
    """
    arcpy.AddMessage("-- RASTER PRE PROCESSING --")
    out_dict = {}

    # Loop on path into the dict
    for i, key in enumerate(raster_path_dict):

        arcpy.AddMessage('********* {} ********'.format(key))
        # Store raster path
        raster_path = raster_path_dict[key][0]

        # Crop raster
        aoi_gdf = gpd.read_file(aoi_path)
        raster_crop_xarr = rasters.crop(raster_path, aoi_gdf, from_disk=True)

        # Store resampling method
        resampling_method = raster_path_dict[key][1]

        # Add path to the dictionary
        if i == 0:

            # Reproject raster and resample
            raster_reproj_xarr = raster_crop_xarr.rio.reproject(dst_crs, resolution=dst_resolution,
                                                                resampling=resampling_method)

            # Re crop raster with AOI
            raster_recrop_xarr = rasters.crop(raster_reproj_xarr, aoi_gdf, from_disk=True)

            # Write masked raster
            raster_path = os.path.join(tmp_dir, "{}.tif".format(key))
            rasters.write(raster_recrop_xarr, raster_path, nodata=0)

            # Store path result in a dict
            out_dict[key] = raster_path

            # Copy the reference xarray
            ref_xarr = raster_recrop_xarr.copy()

        else:

            # Collocate raster
            arcpy.AddMessage('Collocate')
            raster_collocate_xarr = rasters.collocate(ref_xarr, raster_crop_xarr, resampling_method)

            # Mask raster with AOI
            raster_masked_xarr = rasters.mask(raster_collocate_xarr, aoi_gdf)

            # Write masked raster
            raster_path = os.path.join(tmp_dir, "{}.tif".format(key))
            rasters.write(raster_masked_xarr, raster_path, nodata=0)

            # Store path result in a dict
            out_dict[key] = raster_path

    return out_dict


def produce_a_arr(r_xarr: XDS_TYPE, k_xarr: XDS_TYPE, ls_xarr: XDS_TYPE, c_xarr: XDS_TYPE,
                  p_xarr: XDS_TYPE) -> XDS_TYPE:
    """
    Produce average annual soil loss (ton/ha/year) with the RUSLE model.
    Args:
        r_arr (XDS_TYPE): multi-annual average index xarray
        k_arr (XDS_TYPE): susceptibility of a soil to erode xarray
        ls_arr (XDS_TYPE) :combined Slope Length and Slope Steepness factor xarray
        c_arr (XDS_TYPE) : Cover management factor xarray
        p_arr (XDS_TYPE) : support practices factor xarray

    Returns:
        XDS_TYPE : xarray of the average annual soil loss (ton/ha/year)
    """
    arcpy.AddMessage("-- Produce average annual soil loss (ton/ha/year) with the RUSLE model --")

    return r_xarr * k_xarr * ls_xarr * c_xarr * p_xarr


def produce_a_reclass_arr(a_xarr: np.ndarray) -> (XDS_TYPE):
    """
    Produce reclassed a
    Args:
        a_arr (np.ndarray) : a array

    Returns:
        XDS_TYPEy : xarray of the reclassed a raster
    """

    arcpy.AddMessage("-- Produce the reclassed a --")

    # List conditions and choices
    conditions = [(a_xarr < 6.7), (a_xarr >= 6.7) & (a_xarr < 11.2), (a_xarr >= 11.2) & (a_xarr < 22.4),
                  (a_xarr >= 22.4) & (a_xarr < 33.6), (a_xarr >= 36.2)]
    choices = [1, 2, 3, 4, 5]

    # Update arr with k values
    a_reclass_arr = np.select(conditions, choices, default=np.nan)

    return a_xarr.copy(data=a_reclass_arr)


def epsg_from_arcgis_proj(arcgis_proj) -> None:
    """
    Extract espg code from arcgis proj
    Args:
        arcgis_proj () : Arcgis proj

    Returns:
        epsg_code : ndarray of the reclassed a raster
    """
    sr = arcpy.SpatialReference()
    sr.loadFromString(arcgis_proj)
    epsg_code = sr.factoryCode

    return epsg_code


def check_parameters(input_dict: dict) -> None:
    """
     Check if parameters values are ok
    Args:
        a_arr (np.ndarray) : a array

    Returns:

    """
    # --- Extract parameters ---
    aoi_path = input_dict.get("aoi_path")
    location = input_dict.get("location")
    fcover_method = input_dict.get("fcover_method")
    landcover_name = input_dict.get("landcover_name")
    p03_path = input_dict.get("p03_path")
    del_path = input_dict.get("del_path")
    ls_method = input_dict.get("ls_method")
    dem_name = input_dict.get("dem_name")
    other_dem_path = input_dict.get("other_dem_path")
    output_resolution = input_dict.get("output_resolution")
    ref_system = input_dict.get("ref_system")
    output_dir = input_dict.get("output_dir")

    # Check if landcover_name is valid
    if landcover_name not in LandcoverType.list_values():
        raise TypeError(f"landcover_name should be among {LandcoverType.list_values()}")
    # Check if location is valid
    if location not in LocationType.list_values():
        raise TypeError(f"location should be among {LocationType.list_values()}")
    # Check if fcover_method is valid
    if fcover_method not in MethodType.list_values():
        raise TypeError(f"fcover_method should be among {MethodType.list_values()}")
    # Check if ls_method is valid
    if ls_method not in MethodType.list_values():
        raise TypeError(f"ls_method should be among {MethodType.list_values()}")
    # Check if dem_name is valid
    if dem_name not in DemType.list_values():
        raise TypeError(f"ls_method should be among {DemType.list_values()}")

    if (landcover_name == LandcoverType.P03.value) and (p03_path == None):
        raise TypeError(f"P03_path is needed !")

    if (dem_name == DemType.OTHER.value) and (other_dem_path == None):
        raise TypeError(f"Dem path is needed !")

    return


def make_raster_list_to_pre_process(input_dict: dict):
    """
    A faire
    Args:
        input_dict (dict) :

    Returns:
        dict :
    """

    # --- Extract parameters ---
    aoi_path = input_dict.get("aoi_path")
    location = input_dict.get("location")
    fcover_method = input_dict.get("fcover_method")
    fcover_path = input_dict.get("fcover_path")
    nir_path = input_dict.get("nir_path")
    red_path = input_dict.get("red_path")
    landcover_name = input_dict.get("landcover_name")
    p03_path = input_dict.get("p03_path")
    ls_method = input_dict.get("ls_method")
    ls_path = input_dict.get("ls_path")
    output_dir = input_dict.get("output_dir")

    # Create temp_dir if not exist
    tmp_dir = os.path.join(output_dir, "temp_dir")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Dict that store landcover name, landcover path and landcover label
    landcover_path_dict = {LandcoverType.CLC.value: CLC_PATH,
                           LandcoverType.GLC.value: GLC_PATH,
                           LandcoverType.GC.value: GC_PATH,
                           LandcoverType.GL.value: GL_PATH,
                           LandcoverType.P03.value: p03_path
                           }

    # Store landcover path in a variable
    lulc_path = landcover_path_dict[landcover_name]

    if location == LocationType.EUROPE.value:

        # Dict that store raster to pre_process and the type of resampling
        raster_dict = {"r": [R_EURO_PATH, Resampling.bilinear],
                       "k": [K_EURO_PATH, Resampling.bilinear],
                       "lulc": [lulc_path, Resampling.nearest],
                       "p": [P_EURO_PATH, Resampling.bilinear]
                       }

        # Add the ls raster to the pre process dict if provided
        if ls_method == MethodType.ALREADY_PROVIDED.value:
            raster_dict["ls"] = [ls_path, Resampling.bilinear]

        # Add bands to the pre process dict if fcover need to be calculated or not
        if fcover_method == MethodType.TO_BE_CALCULATED.value:
            raster_dict["red"] = [red_path, Resampling.bilinear]
            raster_dict["nir"] = [nir_path, Resampling.bilinear]
        elif fcover_method == MethodType.ALREADY_PROVIDED.value:
            raster_dict["fcover"] = [fcover_path, Resampling.bilinear]

    elif location == LocationType.GLOBAL.value:

        # Produce k
        k_xarr = produce_k_outside_europe(aoi_path)
        # Write k raster
        k_path = os.path.join(tmp_dir, 'k_raw.tif')
        rasters.write(k_xarr, k_path, nodata=0)

        # Dict that store raster to pre_process and the type of resampling
        raster_dict = {"r": [R_GLOBAL_PATH, Resampling.bilinear],
                       "lulc": [lulc_path, Resampling.nearest],
                       "k": [k_path, Resampling.nearest]
                       }

        # Add the ls raster to the pre process dict if provided
        if ls_method == MethodType.ALREADY_PROVIDED.value:
            raster_dict["ls"] = [ls_path, Resampling.bilinear]

        # Add bands to the pre process dict if fcover need to be calculated or not
        if fcover_method == MethodType.TO_BE_CALCULATED.value:
            raster_dict["red"] = [red_path, Resampling.bilinear]
            raster_dict["nir"] = [nir_path, Resampling.bilinear]
        elif fcover_method == MethodType.ALREADY_PROVIDED.value:
            raster_dict["fcover"] = [fcover_path, Resampling.bilinear]

    return raster_dict


def produce_rusle(input_dict: dict) -> None:
    """
    A faire
    Args:
        a_arr (np.ndarray) : a array

    Returns:
        np.ndarray : ndarray of the reclassed a raster
    """

    # --- Extract parameters ---
    aoi_path = input_dict.get("aoi_path")
    location = input_dict.get("location")
    fcover_method = input_dict.get("fcover_method")
    landcover_name = input_dict.get("landcover_name")
    del_path = input_dict.get("del_path")
    ls_method = input_dict.get("ls_method")
    ls_path = input_dict.get("ls_path")
    dem_name = input_dict.get("dem_name")
    other_dem_path = input_dict.get("other_dem_path")
    output_resolution = input_dict.get("output_resolution")
    ref_system = input_dict.get("ref_system")
    output_dir = input_dict.get("output_dir")

    # --- Create temp_dir if not exist ---
    tmp_dir = os.path.join(output_dir, "temp_dir")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # --- Check if parameters values are ok ---
    check_parameters(input_dict)

    # --- Make the list of raster to pre-process ---
    raster_dict = make_raster_list_to_pre_process(input_dict)

    # --- Pre-process raster ---
    # Extract the epsg code from the reference system parameter and made the CRS
    ref_epsg = epsg_from_arcgis_proj(ref_system)
    ref_crs = CRS.from_epsg(ref_epsg)
    # Run pre process
    post_process_dict = raster_pre_processing(aoi_path, output_resolution, CRS.from_epsg(ref_epsg), raster_dict,
                                              tmp_dir)
    # AOI to gdf
    aoi_gdf = gpd.read_file(aoi_path)

    # Check if ls need to be calculated or not
    if ls_method == MethodType.TO_BE_CALCULATED.value:
        # Dict that store dem_name with path
        dem_dict = {DemType.EUDEM.value: EUDEM_PATH,
                    DemType.SRTM.value: SRTM30_PATH,
                    DemType.MERIT.value: MERIT_PATH,
                    DemType.OTHER.value: other_dem_path
                    }

        # --- Pre-process the DEM ---
        # Extract DEM path
        dem_path = dem_dict[dem_name]
        # Crop DEM
        dem_crop_xarr = rasters.crop(dem_path, aoi_gdf, from_disk=True)
        # Reproj DEM
        dem_reproj_xarr = dem_crop_xarr.rio.reproject(ref_crs, resampling=Resampling.bilinear)
        # Write reproj DEM
        dem_reproj_path = os.path.join(tmp_dir, "dem.tif")
        rasters.write(dem_reproj_xarr, dem_reproj_path, nodata=0)

        # --- Produce ls ---
        ls_raw_xarr = produce_ls_factor(dem_reproj_path, tmp_dir)
        # Collocate ls with the other results
        ls_xarr = rasters.collocate(rasters.read(post_process_dict[list(post_process_dict.keys())[0]]), ls_raw_xarr,
                                    Resampling.bilinear)
        # Write ls
        ls_path = os.path.join(tmp_dir, "ls.tif")
        rasters.write(ls_xarr, ls_path, nodata=0)
    else:
        with rasterio.open(post_process_dict['ls']) as ls_dst:
            ls_xarr = rasters.read(ls_dst)

    # Check if fcover need to be calculated or not
    if fcover_method == MethodType.TO_BE_CALCULATED.value:
        # Process fcover
        red_process_path = post_process_dict["red"]
        nir_process_path = post_process_dict["nir"]
        fcover_xarr = produce_fcover(red_process_path, nir_process_path, aoi_path, tmp_dir)

        # Write fcover
        fcover_path = os.path.join(tmp_dir, "fcover.tif")
        rasters.write(fcover_xarr, fcover_path, nodata=0)

    elif fcover_method == MethodType.ALREADY_PROVIDED.value:
        with rasterio.open(post_process_dict["fcover"]) as fcover_dst:
            fcover_xarr = rasters.read(fcover_dst)

    # Mask lulc if del
    if del_path:

        # Update the lulc with the DEL
        arcpy.AddMessage("-- Update raster values covered by DEL --")

        # DEM to gdf
        del_gdf = gpd.read_file(del_path)

        lulc_process_path = post_process_dict["lulc"]

        # Update the lulc with the del
        lulc_del_xarr = rasters.paint(lulc_process_path, del_gdf, value=334)

        # Write the lulc with fire
        lulc_masked_path = os.path.join(tmp_dir, "lulc_with_fire.tif")
        rasters.write(lulc_del_xarr, lulc_masked_path, nodata=0)

        lulc_xarr = lulc_del_xarr.copy()

    else:
        # Open the lulc raster
        with rasterio.open(post_process_dict["lulc"]) as lulc_dst:
            lulc_xarr = rasters.read(lulc_dst)

    # ------------------------------------------------------- Reprendre ici
    # Process C
    c_xarr = produce_c(lulc_xarr, fcover_xarr, aoi_path, landcover_name)

    # Write c raster
    c_out = os.path.join(tmp_dir, "c.tif")
    rasters.write(c_xarr, c_out, nodata=0)

    # Produce p if location is GLOBAL
    if location == LocationType.GLOBAL.value:
        # Produce p
        p_value = 1  # Can change
        p_xarr = xr.full_like(c_xarr, fill_value=p_value)

        # Write p
        p_path = os.path.join(tmp_dir, "p.tif")
        rasters.write(p_xarr, p_path, nodata=0)
    elif location == LocationType.EUROPE.value:
        with rasterio.open(post_process_dict["p"]) as p_dst:
            p_xarr = rasters.read(p_dst)

    # Open r
    with rasterio.open(post_process_dict["r"]) as r_dst:
        r_xarr = rasters.read(r_dst)

    # Open k
    with rasterio.open(post_process_dict["k"]) as k_dst:
        k_xarr = rasters.read(k_dst)

    # Produce a with RUSLE model
    a_xarr = produce_a_arr(r_xarr, k_xarr, ls_xarr, c_xarr, p_xarr)

    # Write the a raster
    a_path = os.path.join(output_dir, "a_rusle.tif")
    rasters.write(a_xarr, a_path, nodata=0)

    # Reclass a (Probleme avec le reclass des Na. A revoir !!)
    a_reclas_xarr = produce_a_reclass_arr(a_xarr)

    # Write the a raster
    a_reclass_path = os.path.join(output_dir, "a_rusle_reclass.tif")
    rasters.write(a_reclas_xarr, a_reclass_path, nodata=0)

    return


if __name__ == '__main__':

    logger = logging.getLogger("EO occurrence")
    handler = ArcPyLogHandler(
        "output_log.log",
        maxBytes=1024 * 1024 * 2,  # 2MB log files
        backupCount=10
    )
    formatter = logging.Formatter("%(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    # --- ENV VAR ---
    arcpy.env.overwriteOutput = True
    arcpy.CheckOutExtension("Spatial")

    # --- Parameters ---
    # Load inputs
    # input_dict = {
    #     "aoi_path": str(arcpy.GetParameterAsText(0)),
    #     "location": str(arcpy.GetParameterAsText(1)),
    #     "fcover_method": str(arcpy.GetParameterAsText(2)),
    #     "fcover_path": str(arcpy.GetParameterAsText(3)),
    #     "nir_path": str(arcpy.GetParameterAsText(4)),
    #     "red_path": str(arcpy.GetParameterAsText(5)),
    #     "landcover_name": str(arcpy.GetParameterAsText(6)),
    #     "p03_path": str(arcpy.GetParameterAsText(7)),
    #     "del_path": str(arcpy.GetParameterAsText(8)),
    #     "ls_method": str(arcpy.GetParameterAsText(9)),
    #     "ls_path": str(arcpy.GetParameterAsText(10)),
    #     "dem_name": str(arcpy.GetParameterAsText(11)),
    #     "other_dem_path": str(arcpy.GetParameterAsText(12)),
    #     "output_resolution": int(str(arcpy.GetParameterAsText(13))),
    #     "ref_system": arcpy.GetParameterAsText(14),
    #     "output_dir": str(arcpy.GetParameterAsText(15))}


    input_dict = {
        "aoi_path": r"D:\TLedauphin\02_Temp_traitement\Test_rusle\emsn073_aoi_32631.shp",
        "location": "Europe",
        "fcover_method": "To be calculated",
        "fcover_path": None,
        "nir_path": r"D:\TLedauphin\02_Temp_traitement\Test_rusle\S2A_MSIL2A_20200805T104031_N0214_R008_T31TDH_20200805T112609\S2A_MSIL2A_20200805T104031_N0214_R008_T31TDH_20200805T112609.SAFE\GRANULE\L2A_T31TDH_A026746_20200805T104810\IMG_DATA\R10m\T31TDH_20200805T104031_B08_10m.jp2",
        "red_path": r"D:\TLedauphin\02_Temp_traitement\Test_rusle\S2A_MSIL2A_20200805T104031_N0214_R008_T31TDH_20200805T112609\S2A_MSIL2A_20200805T104031_N0214_R008_T31TDH_20200805T112609.SAFE\GRANULE\L2A_T31TDH_A026746_20200805T104810\IMG_DATA\R10m\T31TDH_20200805T104031_B04_10m.jp2",
        "landcover_name": "Corine Land Cover - 2018 (100m)",
        "p03_path": None,
        "del_path": r"D:\TLedauphin\02_Temp_traitement\Test_rusle\emsn073_del_32631.shp",
        "ls_method": "To be calculated",
        "ls_path": None,
        "dem_name": "EUDEM 25m",
        "other_dem_path": None,
        "output_resolution": int(10),
        "ref_system": "PROJCS['WGS_1984_UTM_Zone_31N',GEOGCS['GCS_WGS_1984',DATUM['D_WGS_1984',SPHEROID['WGS_1984',6378137.0,298.257223563]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],PROJECTION['Transverse_Mercator'],PARAMETER['False_Easting',500000.0],PARAMETER['False_Northing',0.0],PARAMETER['Central_Meridian',3.0],PARAMETER['Scale_Factor',0.9996],PARAMETER['Latitude_Of_Origin',0.0],UNIT['Meter',1.0]]",
        "output_dir": r"D:\TLedauphin\02_Temp_traitement\Test_rusle\EMSN073_output"}
    try:
        # Compute raster RUSLE
        produce_rusle(input_dict)

        arcpy.AddMessage('RUSLE was a success.')

        # Add to the current map if exist
        # arcpy.MakeRasterLayer_management(input_dict["output_dir"])
        # aprx = arcpy.mp.ArcGISProject("CURRENT")
        # map = aprx.listMaps('*')
        # if len(map) != 0:
        #     map = aprx.listMaps('*')[0]
        #     if map is not None:
        #         raster = map.addDataFromPath(input_dict["output_file"])

    except Exception as ex:
        import traceback

        logger.error('RUSLE has failed: %s', traceback.format_exc())
    finally:
        logger.removeHandler(handler)
