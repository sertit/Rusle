import os

import ftep_util as ftep
import logging.handlers
from sertit import AnyPath, s3
import sys

DEBUG = False
LOGGING_FORMAT = "%(asctime)s - [%(levelname)s] - %(message)s"
LOGGER = logging.getLogger("OSM Charter")

FTEP_S3_ENDPOINT = "s3.waw2-1.cloudferro.com"


def ftep_s3_env(*args, **kwargs):
    return s3.s3_env(endpoint=FTEP_S3_ENDPOINT)(*args, **kwargs)


@ftep_s3_env
def compute_rusle():
    parameters_file_path = "/home/worker/workDir/FTEP-WPS-INPUT.properties"
    # Default parameter values
    params = ftep.Params(parameters_file_path)

    # Add parameters from file
    params.readFile(parameters_file_path)
    # --- Parameters ---
    # Load inputs
    satellite_product_path = "/home/worker/workDir/inDir/satellite_product_path/"
    satellite_product_path = (
        satellite_product_path + os.listdir(satellite_product_path)[0]
    )

    from sertit import logs
    from rusle.rusle_core import (
        LOGGER,
        LOGGING_FORMAT,
        rusle_core,
        InputParameters,
        DataPath,
    )

    logs.init_logger(LOGGER, logging.INFO, LOGGING_FORMAT)
    LOGGER.info("--- RUSLE ---")

    input_dict = {
        InputParameters.AOI_PATH.value: params.getString("aoi"),
        InputParameters.LOCATION.value: params.getString("location"),
        InputParameters.NIR_PATH.value: None,
        InputParameters.RED_PATH.value: None,
        InputParameters.SATELLITE_PRODUCT_PATH.value: satellite_product_path,
        InputParameters.LANDCOVER_NAME.value: params.getString("landcover_name"),
        InputParameters.P03_PATH.value: None,
        InputParameters.DEL_PATH.value: None,
        InputParameters.DEM_NAME.value: params.getString("dem_name"),
        InputParameters.OTHER_DEM_PATH.value: None,
        InputParameters.OUTPUT_RESOLUTION.value: params.getInt("output_resolution"),
        InputParameters.REF_EPSG.value: params.getString("epsg_code"),
        InputParameters.OUTPUT_DIR.value: "/home/worker/workDir/outDir/output",
    }
    DataPath.load_paths(ftep=True)

    try:
        # Compute RUSLE charter
        rusle_core(input_dict)
        LOGGER.info("--- RUSLE was a success.")
        sys.exit(0)

    except Exception as ex:
        LOGGER.error("RUSLE has failed:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    compute_rusle()
