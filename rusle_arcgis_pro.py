"""
Produce RUSLE raster

"""

__author__ = "Ledauphin Thomas"
__contact__ = "tledauphin@unistra.fr"
__python__ = "3.7.0"
__created__ = "24/02/2021"
__update__ = "14/03/2023"
__copyrights__ = "(c) SERTIT 2021"

import logging.handlers
import arcpy

from sertit.arcpy import init_conda_arcpy_env, ArcPyLogHandler

init_conda_arcpy_env()

from rusle import rusle_core, InputParameters
from shapely import speedups

DEBUG = False
LOGGING_FORMAT = '%(asctime)s - [%(levelname)s] - %(message)s'
LOGGER = logging.getLogger("OSM Charter")

speedups.disable()


def epsg_from_arcgis_proj(arcgis_proj):
    """
    Extract espg code from arcgis proj
    Args:
        arcgis_proj () : Arcgis proj

    Returns:
        epsg_code : ndarray of the reclassified a raster
    """
    try:
        sr = arcpy.SpatialReference()
        sr.loadFromString(arcgis_proj)
        epsg_code = sr.factoryCode

    except:
        raise ValueError("Input coordinate system is not from Arcgis coordinate system tools")

    return epsg_code


if __name__ == '__main__':
    logger = logging.getLogger("RUSLE")
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
    input_dict = {
        InputParameters.AOI_PATH.value: str(arcpy.GetParameterAsText(0)),
        InputParameters.LOCATION.value: str(arcpy.GetParameterAsText(1)),
        InputParameters.FCOVER_METHOD.value: str(arcpy.GetParameterAsText(2)),
        InputParameters.FCOVER_PATH.value: str(arcpy.GetParameterAsText(3)),
        InputParameters.NIR_PATH.value: str(arcpy.GetParameterAsText(4)),
        InputParameters.RED_PATH.value: str(arcpy.GetParameterAsText(5)),
        InputParameters.LANDCOVER_NAME.value: str(arcpy.GetParameterAsText(6)),
        InputParameters.P03_PATH.value: str(arcpy.GetParameterAsText(7)),
        InputParameters.DEL_PATH.value: str(arcpy.GetParameterAsText(8)),
        InputParameters.LS_METHOD.value: str(arcpy.GetParameterAsText(9)),
        InputParameters.LS_PATH.value: str(arcpy.GetParameterAsText(11)),
        InputParameters.DEM_NAME.value: str(arcpy.GetParameterAsText(11)),
        InputParameters.OTHER_DEM_PATH.value: str(arcpy.GetParameterAsText(12)),
        InputParameters.OUTPUT_RESOLUTION.value: int(str(arcpy.GetParameterAsText(13))),
        InputParameters.REF_EPSG.value: epsg_from_arcgis_proj(arcpy.GetParameterAsText(14)),
        InputParameters.OUTPUT_DIR.value: str(arcpy.GetParameterAsText(15))
    }

    try:
        # Compute RUSLE charter
        rusle_core(input_dict)

        LOGGER.info('--- RUSLE was a success.')

    except Exception as ex:
        import traceback

        logger.error('RUSLE has failed: %s', traceback.format_exc())
    finally:
        logger.removeHandler(handler)
