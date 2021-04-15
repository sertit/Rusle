"""
Add description

"""

__author__ = "Ledauphin Thomas"
__contact__ = "tledauphin@unistra.fr"
__python__ = "3.5.0"
__created__ = "24/02/2021"
__update__ = ""
__copyrights__ = "(c) SERTIT 2021"

import os
import logging
import numpy as np
import rasterio
from rasterio import MemoryFile
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject
import geopandas as gpd
from rasterio.enums import Resampling
from rasterstats import zonal_stats

from sertit import strings, misc, rasters, files, logs
from eoreader.bands.index import _no_data_divide, _norm_diff

from pysheds.grid import Grid

import pyodbc

import pyproj

import arcpy

# Commande pour supprimer l'erreur suivante : "ValueError: GEOSGeom_createLinearRing_r returned a NULL pointer"
import shapely

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


def get_crs(raster_path: str) -> CRS:
    """
    Get CRS (and check if it is in projected CRS)
    Args:
        raster_path (str): RASTER Path

    Returns:
        CRS: RASTER CRS
    """
    with rasterio.open(raster_path, "r") as raster_dst:
        # Check if RASTER is in geo format
        if not raster_dst.crs.is_projected:
            raise Exception("Raster should be projected.")

        # Get RASTER CRS
        return raster_dst.crs


def reproj_shp(shp_path: str, raster_crs: CRS) -> str:
    """
    Reproject shp to raster crs
    Args:
        shp_path (str): Delineation vector path
        raster_crs (CRS): DEM CRS

    Returns:
        str: Reprojected delineation path
    """
    # Reproj shape vector if needed
    vec = gpd.read_file(shp_path)
    if vec.crs != raster_crs:
        arcpy.AddMessage("Reproject shp vector to CRS {}".format(raster_crs))
        # A modifier start
        if not "EPSG:" in str(raster_crs):
            epsg_code = "custom"
        else:
            epsg_code = str(raster_crs)[5:]
        # A modifier end
        reproj_vec = os.path.join(os.path.splitext(shp_path)[0] + "_reproj_{}".format(epsg_code))

        # Check if reproject shp already exist
        if os.path.isfile(reproj_vec) == False:
            # Reproject vector and write to file
            vec = vec.to_crs(raster_crs)
            vec.to_file(reproj_vec, driver="ESRI Shapefile")

        return reproj_vec

    else:
        return shp_path


def reproject_raster(src_file: str, dst_crs: str, resampling_method) -> str:
    """
    Reproject raster to dst crs
    Args:
        src_file (str): Raster path
        dst_crs (CRS): dst CRS

    Returns:
        str: Reprojected raster path
    """
    arcpy.AddMessage("Reprojecting")

    # Extract crs of the dem
    with rasterio.open(src_file, "r") as src_dst:
        src_crs = src_dst.crs

    # Check if the file does not have the good crs
    if src_crs != dst_crs:

        with rasterio.open(src_file) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            dst_file = os.path.join(os.path.splitext(src_file)[0] + "_reproj_{}.tif".format(str(dst_crs)[5:]))
            with rasterio.open(dst_file, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=resampling_method)

        return dst_file

    else:
        return src_file


def crop_raster(aoi_path: str, raster_path: str, tmp_dir: str) -> (str, np.ndarray, dict):
    """
    Crop Raster to the AOI extent
    Args:
        aoi_path (str): AOI path
        raster_path (str): DEM path
        tmp_dir (str): Temp dir where the cropped Raster will be store
    Returns:
        str: Cropped Raster path
        np.ndarray : ndarray of the cropped raster
        dict : metadata of the cropped raster
    """
    arcpy.AddMessage("Crop Raster to the AOI extent")

    with rasterio.open(raster_path) as raster_dst:
        aoi = gpd.read_file(aoi_path)
        if aoi.crs != raster_dst.crs:
            aoi = aoi.to_crs(raster_dst.crs)

        cropped_raster_arr, cropped_raster_meta = rasters.crop(raster_dst, aoi.envelope)

        cropped_raster_path = os.path.join(tmp_dir,
                                           os.path.basename(os.path.splitext(raster_path)[0]) + '_cropped.tif')
        rasters.write(cropped_raster_arr, cropped_raster_path, cropped_raster_meta, nodata=0)

        return cropped_raster_path, cropped_raster_arr, cropped_raster_meta


def produce_fcover(red_path: str, nir_path: str, aoi_path: str, tmp_dir: str) -> (np.ndarray, dict):
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
        red_band, _ = rasters.read(red_dst)

    # Read and resample nir
    with rasterio.open(nir_path) as nir_dst:
        nir_band, meta = rasters.read(nir_dst)

    # Process ndvi
    ndvi = _norm_diff(nir_band, red_band)

    # Write ndvi
    ndvi_path = os.path.join(files.to_abspath(tmp_dir), 'ndvi.tif')
    rasters.write(ndvi, ndvi_path, meta, nodata=0)

    # Extract crs of the image
    ref_crs = get_crs(ndvi_path)

    # Reproject the shp
    aoi_path = reproj_shp(aoi_path, ref_crs)

    # Crop the ndvi with reprojected AOI
    ndvi_crop_path, _, _ = crop_raster(aoi_path, ndvi_path, tmp_dir)

    # Open cropped_ndvi_path
    with rasterio.open(ndvi_crop_path) as ndvi_crop_dst:
        ndvi_crop_band, meta_crop = rasters.read(ndvi_crop_dst)

    # Extract NDVI min and max inside the AOI
    ndvi_stat = zonal_stats(aoi_path, ndvi_crop_path, stats="min max")
    ndvi_min = ndvi_stat[0]['min']
    ndvi_max = ndvi_stat[0]['max']

    # Fcover calculation
    fcover_array = _no_data_divide(ndvi_crop_band.astype(np.float32) - ndvi_min, ndvi_max - ndvi_min)

    return fcover_array, meta_crop


def update_raster(raster_path: str, shp_path: str, output_raster_path: str, value: int) -> (np.ndarray, dict):
    """
    Update raster values covered by the shape
    Args:
        raster_path (str): raster path
        shp_path (str): shape path
        output_raster_path (str): output raster path
        value (int) : value set to the updated cells

    Returns:
        np.ndarray : ndarray of the updated raster
        dict : metadata of the updated raster
    """

    # arcpy.AddMessage("-- Update raster values covered by the shape --")

    # Check if same crs
    with rasterio.open(raster_path) as raster_dst:
        shp = gpd.read_file(shp_path)
        if shp.crs != raster_dst.crs:
            shp = shp.to_crs(raster_dst.crs)

    # Store geometries of the shapefile
    geoms = [feature for feature in shp['geometry']]

    # Mask the raster with the geometries
    with rasterio.open(raster_path) as src:
        out_arr, out_meta = rasters.mask(src, geoms, invert=True, nodata=value)

    # Write raster
    rasters.write(out_arr, output_raster_path, out_meta, nodata=0)

    return out_arr, out_meta


def produce_c_arable_europe(aoi_path: str, raster_arr: np.ndarray, meta_raster: dir) -> (np.ndarray, dict):
    """
    Produce C arable index over Europe
    Args:
        aoi_path (str): aoi path
        raster_arr (np.ndarray): lulc array
        meta_raster (dict): lulc metadata

    Returns:
        np.ndarray : ndarray of the c arable raster
        dict : metadata of the c arable raster
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
    arable_c_arr = np.full_like(raster_arr, fill_value=0.27)
    arable_c_meta = meta_raster.copy()
    arable_c_meta["dtype"] = arable_c_arr.dtype

    # Reproject europe_countries
    crs_arable = arable_c_meta["crs"]
    if europe_countries.crs != crs_arable:
        europe_countries = europe_countries.to_crs(crs_arable)

    # Update arable_arr with arable_dict
    with MemoryFile() as memfile:
        with memfile.open(**arable_c_meta) as dst:
            dst.write(arable_c_arr)
            for key in list(europe_countries['COUNTRY']):
                geoms = [feature for feature in europe_countries[europe_countries['COUNTRY'] == key]['geometry']]
                arable_c_arr, meta_arable_c = mask(dst, geoms, invert=True, nodata=arable_c_dict[key])
                dst.write(arable_c_arr)

    return arable_c_arr, meta_arable_c


def produce_c(lulc_arr: np.ndarray, meta_lulc: dir, fcover_arr: np.ndarray, aoi_path: str, lulc_type: str) -> (
        np.ndarray, dict):
    """
    Produce C index
    Args:
        lulc_arr (np.ndarray): lulc array
        meta_lulc (dict): lulc metadata
        fcover_arr (np.ndarray) :fcover array
        aoi_path (str): aoi path

    Returns:
        np.ndarray : ndarray of the c index raster
        dict : metadata of the c index raster
    """
    arcpy.AddMessage("-- Produce C index --")

    # Identify Cfactor
    # Cfactor dict and c_arr_arable
    if lulc_type == 'clc':
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
        arable_c_arr, _ = produce_c_arable_europe(aoi_path, lulc_arr, meta_lulc)
        c_arr_arable = np.where(lulc_arr == 211, arable_c_arr, np.nan)
    # Global Land Cover - Copernicus 2020 (100m)
    elif lulc_type == 'glc':
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
        c_arr_arable = np.where(np.isin(lulc_arr, [11, 14, 20]), 0.27, np.nan)

    # GlobCover - ESA 2005 (300m)
    elif lulc_type == 'gc':
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
        c_arr_arable = np.where(lulc_arr == 40, 0.27, np.nan)

    # GlobeLand30 - China 2020 (30m)
    elif lulc_type == 'gl':
        cfactor_dict = {
            20: [0.00010000000, 0.00250000000],
            30: [0.01000000000, 0.07000000000],
            40: [0.01000000000, 0.08000000000],
            90: [0.00000000000, 0.00000000000],
            100: [0.00000000000, 0.00000000000],
            110: [0.10000000000, 0.45000000000]
        }
        # Produce arable c
        c_arr_arable = np.where(lulc_arr == 10, 0.27, np.nan)

    # List init
    conditions = []
    choices = []

    # List conditions and choices for C non arable
    for key in cfactor_dict:
        conditions.append(lulc_arr == key)
        choices.append(cfactor_dict[key][0] + (cfactor_dict[key][1] - cfactor_dict[key][0]) * (
                1 - fcover_arr.astype(np.float32)))

    # C non arable calculation
    c_arr_non_arable = np.select(conditions, choices, default=np.nan)

    # Merge arable and non arable c values
    c_arr = np.where(np.isnan(c_arr_non_arable), c_arr_arable, c_arr_non_arable)

    return c_arr, meta_lulc


def spatial_resolution(raster_path: str) -> (float, float):
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


def produce_ls_factor(dem_path: str, ls_path: str, tmp_dir: str) -> (np.ndarray, dict):
    """
    Produce the LS factor raster
    Args:
        dem_path (str) : dem path
        ls_path (str) : output ls index path
        tmp_dir (str) : tmp dir path

    Returns:
        np.ndarray : ndarray of the ls factor raster
        dict : metadata of the ls factor raster
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
        acc_band, meta = rasters.read(acc_dst)

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
        slope_p, _ = rasters.read(slope_dst)

    # m calculation
    conditions = [slope_p < 1, (slope_p >= 1) & (slope_p > 3), (slope_p >= 3) & (slope_p > 5),
                  (slope_p >= 5) & (slope_p > 12), slope_p >= 12]
    choices = [0.2, 0.3, 0.4, 0.5, 0.6]
    m = np.select(conditions, choices, default=np.nan)

    # Produce ls
    # Equation 1 : file:///C:/Users/TLEDAU~1/AppData/Local/Temp/Ghosal-DasBhattacharya2020_Article_AReviewOfRUSLEModel.pdf
    ls_arr = (0.065 + 0.0456 * slope_p + 0.006541 * np.power(slope_p, 2)) * np.power(
        acc_band.astype(np.float32) * cellsizex / 22.13, m)

    # Write ls
    rasters.write(ls_arr, ls_path, meta, nodata=0)

    return ls_arr, meta


def produce_k_outside_europe(aoi_path: str) -> (np.ndarray, dict):
    """
    Produce the K index outside Europe
    Args:
        aoi_path (str) : AOI path

    Returns:
        np.ndarray : ndarray of the K raster
        dict : metadata of the K raster
    """

    arcpy.AddMessage("-- Produce the K index outside Europe --")

    # Extract crs of the hwsd
    with rasterio.open(HWSD_PATH, "r") as hwsd_dst:
        hwsd_crs = hwsd_dst.crs

    # Reproject the shp
    aoi_reproj_path = reproj_shp(aoi_path, hwsd_crs)

    # Crop hwsd
    crop_hwsd_path, crop_hwsd_arr, crop_hwsd_meta = crop_raster(aoi_reproj_path, HWSD_PATH, tmp_dir)

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
        conditions.append(crop_hwsd_arr == key)
        choices.append(k_dict[key])

    # Update arr with k values
    k_arr = np.select(conditions, choices, default=np.nan)
    k_meta = crop_hwsd_meta.copy()

    return k_arr, k_meta


def raster_pre_processing(aoi_path: str, dst_resolution: int, dst_crs: str, raster_path_dict: dict,
                          tmp_dir: str) -> dict:
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

    # Reproject aoi
    aoi_ref_path = reproj_shp(aoi_path, dst_crs)

    # Loop on path into the dict
    for i, key in enumerate(raster_path_dict):

        arcpy.AddMessage('********* {} ********'.format(key))
        # Store raster path
        raster_path = raster_path_dict[key][0]

        # Crop raster
        raster_crop_path, raster_crop_arr, raster_crop_meta = crop_raster(aoi_path, raster_path, tmp_dir)

        # Store resampling method
        resampling_method = raster_path_dict[key][1]

        # Add path to the dictionary
        if i == 0:

            # Reproject raster
            raster_reproj_path = reproject_raster(raster_crop_path, dst_crs, resampling_method)

            # Resample reproj raster
            with rasterio.open(raster_reproj_path) as raster_reproj_dst:
                raster_resample_band, raster_resample_meta = rasters.read(raster_reproj_dst, dst_resolution,
                                                                          resampling_method)
            # Write resample raster
            raster_resample_path = os.path.join(tmp_dir, "{}_resample.tif".format(key))
            rasters.write(raster_resample_band, raster_resample_path, raster_resample_meta, nodata=0)

            # Re crop raster
            raster_recrop_path, _, _ = crop_raster(aoi_ref_path, raster_resample_path, tmp_dir)

            # Mask raster with AOI
            arcpy.AddMessage('Masking with AOI')
            aoi_ref_gpd = gpd.read_file(aoi_ref_path)
            geoms = [feature for feature in aoi_ref_gpd['geometry']]
            raster_recrop_arr, raster_recrop_meta = rasters.mask(raster_recrop_path, geoms)

            # Write masked raster
            raster_masked_path = os.path.join(tmp_dir, "{}_ref_masked.tif".format(key))
            rasters.write(raster_recrop_arr, raster_masked_path, raster_recrop_meta, nodata=0)

            # Store path result in a dict
            out_dict[key] = raster_masked_path

            # Update the dtype of the ref metadata
            meta_ref = raster_recrop_meta.copy()
            meta_ref.update({'dtype': np.float32})

        else:

            # Collocate raster
            arcpy.AddMessage('Collocate')
            raster_collocate_arr, raster_collocate_meta = rasters.collocate(meta_ref, raster_crop_arr, raster_crop_meta,
                                                                            resampling_method)
            # Write collocated raster
            raster_collocate_path = os.path.join(tmp_dir, "{}_collocated.tif".format(key))
            rasters.write(raster_collocate_arr, raster_collocate_path, raster_collocate_meta, nodata=0)

            # Mask raster with AOI
            arcpy.AddMessage('Masking with AOI')
            aoi_ref_gpd = gpd.read_file(aoi_ref_path)
            geoms = [feature for feature in aoi_ref_gpd['geometry']]
            raster_recrop_arr, raster_recrop_meta = rasters.mask(raster_collocate_path,
                                                                 geoms)
            # Write masked raster
            raster_masked_path = os.path.join(tmp_dir, "{}_collocated_masked.tif".format(key))
            rasters.write(raster_recrop_arr, raster_masked_path, raster_recrop_meta, nodata=0)

            # Store path result in a dict
            out_dict[key] = raster_masked_path

    return out_dict


def produce_a_arr(r_arr: np.ndarray, k_arr: np.ndarray, ls_arr: np.ndarray, c_arr: np.ndarray,
                  p_arr: np.ndarray) -> np.ndarray:
    """
    Produce average annual soil loss (ton/ha/year) with the RUSLE model.
    Args:
        r_arr (np.ndarray): multi-annual average index array
        k_arr (np.ndarray): susceptibility of a soil to erode array
        ls_arr (np.ndarray) :combined Slope Length and Slope Steepness factor array
        c_arr (np.ndarray) : Cover management factor array
        p_arr (np.ndarray) : support practices factor array

    Returns:
        np.ndarray : ndarray of the average annual soil loss (ton/ha/year)
    """
    arcpy.AddMessage("-- Produce average annual soil loss (ton/ha/year) with the RUSLE model --")

    return r_arr * k_arr * ls_arr * c_arr * p_arr


def produce_a_reclass_arr(a_arr: np.ndarray) -> (np.ndarray, dict):
    """
    Produce reclassed a
    Args:
        a_arr (np.ndarray) : a array

    Returns:
        np.ndarray : ndarray of the reclassed a raster
    """

    arcpy.AddMessage("-- Produce the reclassed a --")

    # List conditions and choices
    conditions = [(a_arr < 6.7), (a_arr >= 6.7) & (a_arr < 11.2), (a_arr >= 11.2) & (a_arr < 22.4),
                  (a_arr >= 22.4) & (a_arr < 33.6), (a_arr >= 36.2)]
    choices = [1, 2, 3, 4, 5]

    # Update arr with k values
    a_reclass_arr = np.select(conditions, choices, default=np.nan)

    return a_reclass_arr


if __name__ == '__main__':
    # Logging
    logs.init_logger(LOGGER, logging.DEBUG)

    # Load inputs
    aoi_path = str(arcpy.GetParameterAsText(0))
    location = str(arcpy.GetParameterAsText(1))

    fcover_method = str(arcpy.GetParameterAsText(2))
    fcover_path = str(arcpy.GetParameterAsText(3))
    nir_path = str(arcpy.GetParameterAsText(4))
    red_path = str(arcpy.GetParameterAsText(5))

    landcover_name = str(arcpy.GetParameterAsText(6))
    p03_path = str(arcpy.GetParameterAsText(7))
    del_path = str(arcpy.GetParameterAsText(8))

    ls_method = str(arcpy.GetParameterAsText(9))
    ls_path = str(arcpy.GetParameterAsText(10))
    dem_name = str(arcpy.GetParameterAsText(11))
    other_dem_path = str(arcpy.GetParameterAsText(12))

    output_resolution = int(str(arcpy.GetParameterAsText(13)))
    ref_system = arcpy.GetParameterAsText(14)  # get_crs(red_path)
    output_dir = str(arcpy.GetParameterAsText(15))

    # Extract the epsg code from the reference system parameter (A voir si une manière plus directe)
    sr = arcpy.SpatialReference()
    sr.loadFromString(ref_system)
    ref_epsg = sr.factoryCode
    # Define the reference CRS
    ref_crs = CRS.from_epsg(ref_epsg)

    # Create temp_dir if not exist
    tmp_dir = os.path.join(output_dir, "temp_dir")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Dict that store landcover name, landcover path and landcover label
    landcover_path_dict = {"Corine Land Cover - 2018 (100m)": [CLC_PATH, "clc"],
                           "Global Land Cover - Copernicus 2020 (100m)": [GLC_PATH, "glc"],
                           "GlobCover - ESA 2005 (300m)": [GC_PATH, "gc"],
                           "GlobeLand30 - China 2020 (30m)": [GL_PATH, "gl"],
                           "P03": [p03_path, "clc"]
                           }

    # Store landcover path in a variable
    lulc_path = landcover_path_dict[landcover_name][0]

    # Check if the AOI is located inside or outside Europe
    if location == "Europe":

        # Dict that store raster to pre_process and the type of resampling
        raster_dict = {"r": [R_EURO_PATH, Resampling.bilinear],
                       "k": [K_EURO_PATH, Resampling.bilinear],
                       "lulc": [lulc_path, Resampling.nearest],
                       "p": [P_EURO_PATH, Resampling.bilinear]
                       }

        # Add the ls raster to the pre process dict if provided
        if ls_method == "Already provided":
            raster_dict["ls"] = [ls_path, Resampling.bilinear]

        # Add bands to the pre process dict if fcover need to be calculated or not
        if fcover_method == "To be calculated":
            raster_dict["red"] = [red_path, Resampling.bilinear]
            raster_dict["nir"] = [nir_path, Resampling.bilinear]
        elif fcover_method == "Already provided":
            raster_dict["fcover"] = [fcover_path, Resampling.bilinear]

        # Run pre process
        post_process_dict = raster_pre_processing(aoi_path, output_resolution, ref_crs, raster_dict,
                                                  tmp_dir)

        # Open the r raster
        with rasterio.open(post_process_dict["r"]) as r_dst:
            r_arr, r_meta = rasters.read(r_dst)

        # Open the k raster
        with rasterio.open(post_process_dict["k"]) as k_dst:
            k_arr, _ = rasters.read(k_dst)

        # Open the p raster
        with rasterio.open(post_process_dict["p"]) as p_dst:
            p_arr, _ = rasters.read(p_dst)

    else:

        # Produce k
        k_arr, k_meta = produce_k_outside_europe(aoi_path)

        # Write k raster
        k_path = os.path.join(tmp_dir, 'k_raw.tif')
        rasters.write(k_arr, k_path, k_meta, nodata=0)

        # Dict that store raster to pre_process and the type of resampling
        raster_dict = {"r": [R_GLOBAL_PATH, Resampling.bilinear],
                       "k": [k_path, Resampling.nearest],
                       "lulc": [lulc_path, Resampling.nearest]
                       }

        # Add the ls raster to the pre process dict if provided
        if ls_method == "Already provided":
            raster_dict["ls"] = [ls_path, Resampling.bilinear]

        # Add bands to the pre process dict if fcover need to be calculated or not
        if fcover_method == "To be calculated":
            raster_dict["red"] = [red_path, Resampling.bilinear]
            raster_dict["nir"] = [nir_path, Resampling.bilinear]
        elif fcover_method == "Already provided":
            raster_dict["fcover"] = [fcover_path, Resampling.bilinear]

        # Run pre process
        post_process_dict = raster_pre_processing(aoi_path, output_resolution, ref_crs, raster_dict,
                                                  tmp_dir)

        # Open the r raster
        with rasterio.open(post_process_dict["r"]) as r_dst:
            r_arr, r_meta = rasters.read(r_dst)

        # Open the k raster
        with rasterio.open(post_process_dict["k"]) as k_dst:
            k_arr, _ = rasters.read(k_dst)

        # Produce p #--------------- A modifier
        p_value = 1  # Can change
        p_arr = r_arr.copy()
        p_arr.fill(p_value)

        # Write p
        p_meta = r_meta.copy()
        p_path = os.path.join(tmp_dir, "p.tif")
        rasters.write(p_arr, p_path, p_meta, nodata=0)

    # Check if ls need to be calculated or not
    if ls_method == "To be calculated":

        # Dict that store dem_name with path
        dem_dict = {"EUDEM 25m": EUDEM_PATH,
                    "SRTM 30m": SRTM30_PATH,
                    "MERIT 5 deg": MERIT_PATH,
                    "Other": other_dem_path
                    }

        # Extract DEM path
        dem_path = dem_dict[dem_name]

        # Crop DEM
        dem_crop_path, dem_arr, dem_meta = crop_raster(aoi_path, dem_path, tmp_dir)
        # Reproj DEM
        dem_reproj_path = reproject_raster(dem_crop_path, ref_crs, Resampling.bilinear)
        # Produce ls
        ls_raw_path = os.path.join(tmp_dir, "ls_raw.tif")
        ls_raw_arr, ls_raw_meta = produce_ls_factor(dem_reproj_path, ls_raw_path, tmp_dir)
        # Collocate ls with the other results
        ls_arr, ls_meta = rasters.collocate(r_meta, ls_raw_arr, ls_raw_meta, Resampling.bilinear)
        # Write ls
        ls_path = os.path.join(tmp_dir, "ls.tif")
        rasters.write(ls_arr, ls_path, ls_meta, nodata=0)

    elif ls_method == "Already provided":
        with rasterio.open(post_process_dict['ls']) as ls_dst:
            ls_arr, ls_meta = rasters.read(ls_dst)

    # Check if fcover need to be calculated or not
    if fcover_method == "To be calculated":
        # Process fcover
        red_process_path = post_process_dict["red"]
        nir_process_path = post_process_dict["nir"]
        fcover_arr, meta_fcover = produce_fcover(red_process_path, nir_process_path, aoi_path, tmp_dir)

        # Write fcover
        fcover_path = os.path.join(tmp_dir, "fcover.tif")
        rasters.write(fcover_arr, fcover_path, meta_fcover, nodata=0)

    elif fcover_method == "Already provided":
        with rasterio.open(post_process_dict["fcover"]) as fcover_dst:
            fcover_arr, meta_fcover = rasters.read(fcover_dst)

    # Mask lulc if del
    if del_path:
        # Reproject del
        reproj_del = reproj_shp(del_path, ref_crs)

        # Update the lulc with the DEL
        arcpy.AddMessage("-- Update raster values covered by DEL --")

        lulc_process_path = post_process_dict["lulc"]
        lulc_masked_path = os.path.join(tmp_dir, "lulc_with_fire.tif")
        _, _ = update_raster(lulc_process_path, reproj_del, lulc_masked_path,
                             334)
        # Mask raster with AOI
        aoi_ref_path = reproj_shp(aoi_path, ref_crs)
        arcpy.AddMessage(aoi_ref_path)
        aoi_ref_gpd = gpd.read_file(aoi_ref_path)
        geoms = [feature for feature in aoi_ref_gpd['geometry']]
        lulc_arr, lulc_meta = rasters.mask(lulc_masked_path, geoms)

        # Write masked raster
        raster_masked_path = os.path.join(tmp_dir, "lulc_with_fire_masked.tif")
        rasters.write(lulc_arr, raster_masked_path, lulc_meta, nodata=0)

    else:
        # Open the lulc raster
        with rasterio.open(post_process_dict["lulc"]) as lulc_dst:
            lulc_arr, lulc_meta = rasters.read(lulc_dst)

    # Process C
    lulc_alias = landcover_path_dict[landcover_name][1]
    c_arr, c_meta = produce_c(lulc_arr, lulc_meta, fcover_arr, aoi_path, lulc_alias)

    # Write c raster
    c_out = os.path.join(tmp_dir, "c.tif")
    rasters.write(c_arr, c_out, c_meta, nodata=0)

    # Produce a with RUSLE model
    a_meta = meta_fcover.copy()
    a_arr = produce_a_arr(r_arr, k_arr, ls_arr, c_arr, p_arr)

    # Write the a raster
    a_path = os.path.join(output_dir, "a_rusle.tif")
    rasters.write(a_arr, a_path, a_meta, nodata=0)

    # Reclass a (Probleme avec le reclass des Na. A revoir !!)
    a_reclas_arr = produce_a_reclass_arr(a_arr)
    a_reclass_meta = a_meta.copy()

    # Write the a raster
    a_reclass_path = os.path.join(output_dir, "a_rusle_reclass.tif")
    rasters.write(a_reclas_arr, a_reclass_path, a_reclass_meta, nodata=0)
