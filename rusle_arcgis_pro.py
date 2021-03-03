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
from typing import Union
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject
import geopandas as gpd
from rasterio.enums import Resampling
from rasterstats import zonal_stats

from sertit_utils.core import type_utils, sys_utils
from sertit_utils.eo import raster_utils

from pysheds.grid import Grid


def set_no_data(idx: np.ma.masked_array) -> np.ma.masked_array:
    """
    Set nodata to a masked array (replacing masked pixel values by the fill_value)
    Args:
        idx (np.ma.masked_array): Index

    Returns:
        np.ma.masked_array: Index with no data filled with correct values
    """
    # idx[idx.mask] = idx.fill_value
    idx[idx.mask] = np.ma.masked
    return idx


def no_data_divide(band_1: Union[float, np.ma.masked_array], band_2: np.ma.masked_array) -> np.ma.masked_array:
    """
    Get the dicision taking into account the nodata between b1 and b2

    Args:
        band_1 (np.ma.masked_array): Band 1
        band_2 (np.ma.masked_array): Band 2

    Returns:
        np.ma.masked_array: Division between band 1 and band 2
    """
    return set_no_data(np.divide(band_1, band_2))


def norm_diff(band_1: np.ma.masked_array, band_2: np.ma.masked_array) -> np.ma.masked_array:
    """
    Get normalized difference index between band 1 and band 2:
    (band_1 - band_2)/(band_1 + band_2)

    Args:
        band_1 (np.ma.masked_array): Band 1
        band_2 (np.ma.masked_array): Band 2

    Returns:
        np.ma.masked_array: Normalized Difference between band 1 and band 2

    """
    return no_data_divide(band_1.astype(np.float32) - band_2.astype(np.float32),
                          band_1.astype(np.float32) + band_2.astype(np.float32))


def to_abspath(path_str):
    """
    Return the absolute path of the specified path and check if it exists

    If not:

    - If it is a file (aka has an extension), it raises an exception
    - If it is a folder, it creates it

    Args:
        path_str (str): Path as a string (relative or absolute)

    Returns:
        str: Absolute path
    """
    abs_path = os.path.abspath(path_str)

    if not os.path.exists(abs_path):
        if os.path.splitext(abs_path)[1]:
            # If the path specifies a file (with extension), it raises an exception
            raise Exception("Non existing file: {}".format(abs_path))

        # If the path specifies a folder, it creates it
        os.makedirs(abs_path)

    return abs_path


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
    # Reproj delineation vector if nedded
    vec = gpd.read_file(shp_path)
    if vec.crs != raster_crs:
        # LOGGER.info("Reproject shp vector to CRS %s", raster_crs)
        reproj_vec = os.path.join(os.path.splitext(shp_path)[0] + "_reproj")

        # Delete reprojected vector if existing
        if os.path.isfile(reproj_vec):
            os.remove(reproj_vec)

        # Reproject vector and write to file
        vec = vec.to_crs(raster_crs)
        vec.to_file(reproj_vec, driver="ESRI Shapefile")

        return reproj_vec

    else:
        return shp_path


def reproject_raster(src_file: str, dst_crs: str) -> str:
    # Extract crs of the dem
    with rasterio.open(src_file, "r") as src_dst:
        src_crs = src_dst.crs

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

            dst_file = os.path.join(os.path.splitext(src_file)[0] + "_reproj.tif")
            with rasterio.open(dst_file, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)

        return dst_file

    else:
        return src_file


def crop_raster(aoi_path: str, raster_path: str, tmp_dir: str) -> str:
    """
    Crop Raster to the AOI extent
    Args:
        aoi_path (str): AOI path
        raster_path (str): DEM path
        tmp_dir (str): Temp dir where to store the cropped DEM
    Returns:
        str: Cropped DEM path
    """
    if aoi_path:
        #LOGGER.info("Crop Raster to the AOI extent")
        with rasterio.open(raster_path) as raster_dst:
            aoi = gpd.read_file(aoi_path)
            if aoi.crs != raster_dst.crs:
                aoi = aoi.to_crs(raster_dst.crs)

            cropped_raster_arr, cropped_raster_tr = mask(raster_dst, aoi.envelope, crop=True)
            out_meta = raster_dst.meta
            out_meta.update({"height": cropped_raster_arr.shape[1],
                             "width": cropped_raster_arr.shape[2],
                             "transform": cropped_raster_tr})

            cropped_raster_path = os.path.join(tmp_dir, os.path.basename(os.path.splitext(raster_path)[0])+'_cropped.tif')
            with rasterio.open(cropped_raster_path, "w", **out_meta) as dest:
                dest.write(cropped_raster_arr)

            return cropped_raster_path
    else:
        return raster_path


def produce_fcover(red_band, nir_band, aoi_path: str, tmp_dir : str):

    # Process ndvi
    ndvi = norm_diff(nir_band, red_band)

    # Write ndvi
    ndvi_path = os.path.join(to_abspath(tmp_dir), 'ndvi.tif')
    raster_utils.write(ndvi, ndvi_path, meta, nodata=0)

    # Extract crs of the image
    ref_crs = get_crs(ndvi_path)

    # Reproject the shp
    aoi_path = reproj_shp(aoi_path, ref_crs)

    # Crop the ndvi with reprojected AOI
    ndvi_crop_path = crop_raster(aoi_path, ndvi_path, tmp_dir)

    # Open cropped_ndvi_path
    with rasterio.open(ndvi_crop_path) as ndvi_crop_dst:
        ndvi_crop_band, _= raster_utils.read(ndvi_crop_dst)

    # Extract NDVI min and max inside the AOI
    ndvi_stat = zonal_stats(aoi_path, ndvi_crop_path, stats="min max")
    ndvi_min = ndvi_stat[0]['min']
    ndvi_max = ndvi_stat[0]['max']

    # Fcover calculation
    fcover_array = no_data_divide(ndvi_crop_band.astype(np.float32) - ndvi_min, ndvi_max - ndvi_min)

    return fcover_array, meta


def update_raster(raster_path: str, shp_path: str, output_raster_path: str, value: int) -> (np.ndarray, dict):
    # Check if same crs
    with rasterio.open(raster_path) as raster_dst:
        shp = gpd.read_file(shp_path)
        if shp.crs != raster_dst.crs:
            shp = shp.to_crs(raster_dst.crs)

    # Store geometries of the shapefile
    geoms = [feature for feature in shp['geometry']]

    # Mask the raster with the geometries
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, geoms, invert=True, nodata=value)
        out_meta = src.meta.copy()

    # Create output meta
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    # Write de raster masked
    with rasterio.open(output_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)

    return out_image, out_meta


# Ajouter les types correpondants (array ...)
def produce_c(lulc_arr: np.ndarray, fcover_arr: np.ndarray) -> np.ndarray:
    # Identify Cfactor
    # Cfactor dict 
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

    # List init
    conditions = []
    choices_min = []
    choices_range = []

    # Liste condition and choices
    for key in cfactor_dict:
        conditions.append(lulc_arr == key)
        choices_min.append(cfactor_dict[key][0])
        choices_range.append(cfactor_dict[key][1] - cfactor_dict[key][0])

    # Cfactor min processing
    cfactor_arr_min = np.select(conditions, choices_min, default=np.nan)

    # Cfactor range processing
    cfactor_arr_range = np.select(conditions, choices_range, default=np.nan)

    # C calculation 

    c_arr = cfactor_arr_min.astype(np.float32) + cfactor_arr_range.astype(np.float32) * (
                1 - fcover_arr.astype(np.float32))

    return c_arr

def spatial_resolution(raster_path : str):
    """extracts the XY Pixel Size"""
    raster = rasterio.open(raster_path)
    t = raster.transform
    x = t[0]
    y =-t[4]
    return x, y


# Process LSfactor
def produce_ls_factor(dem_path: str, ls_path: str, tmp_dir: str):
    # Get slope path
    slope_dem_d = os.path.join(tmp_dir, "slope_degrees.tif")

    # Make slope degree commande
    cmd_slope_d = ["gdaldem",
                 "slope",
                 "-compute_edges",
                 type_utils.to_cmd_string(dem_path),
                 type_utils.to_cmd_string(slope_dem_d)]

    # Run command
    sys_utils.run_command(cmd_slope_d)

    # Compute D8 flow directions
    grid = Grid.from_raster(dem_path, data_name='dem')

    # Fill dem
    grid.fill_depressions(data='dem', out_name='filled_dem')

    # Resolve flat
    grid.resolve_flats(data='filled_dem', out_name='inflated_dem')

    grid.flowdir('inflated_dem', out_name='dir')

    # Export flow directions
    dir_path = os.path.join(tmp_dir, 'dir.tif')
    grid.to_raster('dir', dir_path)

    ##Compute accumulation
    grid.accumulation(data='dir', out_name='acc')

    # Export  accumulation
    acc_path = os.path.join(tmp_dir, 'acc.tif')
    grid.to_raster('acc', acc_path)

    # Extract dem spatial resolution
    cellsizex, cellsizey = spatial_resolution(dem_path)

    # Open acc
    with rasterio.open(acc_path) as acc_dst:
        acc_band, meta = raster_utils.read(acc_dst)

    # Open slope
    with rasterio.open(slope_dem_d) as slope_dst:
        slope_band, _ = raster_utils.read(slope_dst)

    # Make slope percentage commande
    cmd_slope_p = ["gdaldem",
                 "slope",
                 "-compute_edges",
                 type_utils.to_cmd_string(dem_path),
                 type_utils.to_cmd_string(slope_dem_d), "-p"]

    # Run command
    sys_utils.run_command(cmd_slope_p)




    # Produce ls
    ls_arr = np.power(acc_band.astype(np.float32) * cellsizex / 22.13, 0.4) * np.power(
        np.sin(slope_band.astype(np.float32) * 0.0174533) / 0.0896, 1.3)  # 0.4 fixed ?

    # Write ls
    raster_utils.write(ls_arr, ls_path, meta, nodata=0)

    return ls_arr, meta


if __name__ == '__main__':

    ##### Load inputs (Only over europe yet)
    lulc_path = r"\\ds2\database02\BASES_DE_DONNEES\GLOBAL\Corine_Land_Cover\CLC_2018\clc2018_clc2018_v2018_20_raster100m\CLC2018_CLC2018_V2018_20.tif"
    dem_path = r"\\ds2\database02\BASES_DE_DONNEES\GLOBAL\EUDEM_v2\eudem_wgs84.tif"
    nir_path = r"D:\TLedauphin\02_Temp_traitement\Test_rusle\S2A_MSIL2A_20200929T113321_N0214_R080_T29TNF_20200929T143142.SAFE\GRANULE\L2A_T29TNF_A027533_20200929T113454\IMG_DATA\R10m\T29TNF_20200929T113321_B08_10m.jp2"
    red_path = r"D:\TLedauphin\02_Temp_traitement\Test_rusle\S2A_MSIL2A_20200929T113321_N0214_R080_T29TNF_20200929T143142.SAFE\GRANULE\L2A_T29TNF_A027533_20200929T113454\IMG_DATA\R10m\T29TNF_20200929T113321_B04_10m.jp2"

    aoi_path = r"D:\TLedauphin\02_Temp_traitement\Test_rusle\aoi.shp"
    del_path = r"D:\TLedauphin\02_Temp_traitement\Test_rusle\EMSR462_del.shp"

    tmp_dir = r"D:\TLedauphin\02_Temp_traitement\Test_rusle\tmp"

    output_resolution = 5

    ref_crs = get_crs(red_path)

    ##### Process Fcover
    # Read and resample red
    with rasterio.open(red_path) as red_dst:
        red_band, meta = raster_utils.read(red_dst, output_resolution, Resampling.bilinear)

    # Read and resample nir
    with rasterio.open(nir_path) as nir_dst:
        nir_band, _ = raster_utils.read(nir_dst, output_resolution, Resampling.bilinear)

    # Process fcover
    fcover_arr, meta_fcover = produce_fcover(red_band, nir_band, aoi_path, tmp_dir)

    # Write fcover
    fcover_path = os.path.join(tmp_dir, "fcover.tif")
    raster_utils.write(fcover_arr, fcover_path, meta_fcover, nodata=0)

    ##### Process Cfactor
    # Je n'arrive pas à reprojeter l'AOI en 3035 ETRS89 ...

    # Reproject LULC 
    lulc_reprojected = reproject_raster(lulc_path, ref_crs)

    # Crop the lulc with reprojected AOI
    cropped_lulc_path = crop_raster(aoi_path, lulc_reprojected, tmp_dir)

    # Resample cropped lulc
    with rasterio.open(cropped_lulc_path) as lulc_dst:
        lulc_band, meta_lulc = raster_utils.read(lulc_dst, output_resolution, Resampling.nearest)

    # Write resample lulc raster
    cropped_lulc_resampled = os.path.join(tmp_dir, "lulc_resampled.tif")
    raster_utils.write(lulc_band, cropped_lulc_resampled, meta_lulc, nodata=0)

    # Mask lulc with del 
    if dem_path :
        # Reproject del
        reproj_del = reproj_shp(del_path, ref_crs)

        # Update the lulc with the DEL
        cropped_lulc_masked = os.path.join(tmp_dir, "lulc_masked.tif")
        lulc_band, meta_lulc = update_raster(cropped_lulc_resampled, reproj_del, cropped_lulc_masked,
                                                                334)

    # Collocate both raster
    collocated_lulc_arr, meta_lulc_collocated = raster_utils.collocate(meta_fcover, lulc_band,
                                                                       meta_lulc, Resampling.nearest)

    # Write collocated lulc raster
    collocated_lulc_resampled = os.path.join(tmp_dir, "lulc_collocated.tif")
    raster_utils.write(collocated_lulc_arr, collocated_lulc_resampled, meta_lulc_collocated, nodata=0)

    #### Process C
    c_arr = produce_c(collocated_lulc_arr, fcover_arr)

    # Write c raster
    c_out = os.path.join(tmp_dir, "c.tif")
    raster_utils.write(c_arr, c_out, meta_lulc, nodata=0)


    #### Process LS Factor ==> Travailler sur le DEM resample ou resample le résultat ?

    # Extract crs of the dem
    with rasterio.open(dem_path, "r") as dem_dst:
        dem_crs = dem_dst.crs

    # Reproject the shp
    aoi_path_reproj = reproj_shp(aoi_path, dem_crs)

    # Crop DEM
    cropped_dem_path = crop_raster(aoi_path_reproj, dem_path, tmp_dir)

    # Reproject dem
    dem_reprojected = reproject_raster(cropped_dem_path, ref_crs)

    # Produce ls factor
    ls_path = os.path.join(tmp_dir, "ls.tif")
    ls_factor_arr, meta = produce_ls_factor(dem_reprojected, ls_path, tmp_dir )
