import logging.handlers
import arcpy
from sertit.arcpy import init_conda_arcpy_env, ArcPyLogHandler

init_conda_arcpy_env()

from rusle.rusle_core import rusle_core, InputParameters, DataPath
from shapely import speedups

DEBUG = False
LOGGING_FORMAT = "%(asctime)s - [%(levelname)s] - %(message)s"
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
        raise ValueError(
            "Input coordinate system is not from Arcgis coordinate system tools"
        )

    return epsg_code

def main_arcgis(parameters, messsages):


    logger = logging.getLogger("RUSLE")
    handler = ArcPyLogHandler(
        "output_log.log", maxBytes=1024 * 1024 * 2, backupCount=10  # 2MB log files
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
        InputParameters.AOI_PATH.value: parameters[0].valueAsText,
        InputParameters.LOCATION.value: parameters[1].valueAsText,
        InputParameters.FCOVER_METHOD.value: parameters[2].valueAsText,
        InputParameters.FCOVER_PATH.value: parameters[3].valueAsText,
        InputParameters.NIR_PATH.value: parameters[4].valueAsText,
        InputParameters.RED_PATH.value: parameters[5].valueAsText,
        InputParameters.LANDCOVER_NAME.value: parameters[6].valueAsText,
        InputParameters.P03_PATH.value: parameters[7].valueAsText,
        InputParameters.DEL_PATH.value: parameters[8].valueAsText,
        InputParameters.LS_METHOD.value: parameters[9].valueAsText,
        InputParameters.LS_PATH.value: parameters[10].valueAsText,
        InputParameters.DEM_NAME.value: parameters[11].valueAsText,
        InputParameters.OTHER_DEM_PATH.value: parameters[12].valueAsText,
        InputParameters.OUTPUT_RESOLUTION.value: int(parameters[13].valueAsText),
        InputParameters.REF_EPSG.value: epsg_from_arcgis_proj(
            parameters[14].valueAsText
        ),
        InputParameters.OUTPUT_DIR.value: parameters[15].valueAsText,
    }
    DataPath.load_paths()

    try:
        # Compute RUSLE charter
        rusle_core(input_dict)

        LOGGER.info("--- RUSLE was a success.")

    except Exception as ex:
        import traceback

        logger.error("RUSLE has failed: %s", traceback.format_exc())
    finally:
        logger.removeHandler(handler)