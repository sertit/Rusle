import os

from main import rusle_core, InputParameters, DataPath
import ftep_util as ftep
import logging.handlers

DEBUG = False
LOGGING_FORMAT = "%(asctime)s - [%(levelname)s] - %(message)s"
LOGGER = logging.getLogger("OSM Charter")

if __name__ == "__main__":
    logger = logging.getLogger("RUSLE")
    logger.setLevel(logging.DEBUG)

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
    aoi_path = "/home/worker/workDir/inDir/aoi_path/"

    # Only one AOI is expected. It means that the only case where more than 2 files can be encountered is with shp
    sub_aoi_files = os.listdir(aoi_path)[0]
    if len(sub_aoi_files) > 1:
        for path in sub_aoi_files:
            if path.endswith("shp"):
                aoi_path = aoi_path + path
    else:
        aoi_path = aoi_path + os.listdir(sub_aoi_files)[0]

    input_dict = {
        InputParameters.AOI_PATH.value: aoi_path,
        InputParameters.LOCATION.value: params.getString("location"),
        InputParameters.FCOVER_METHOD.value: "To be calculated",
        InputParameters.FCOVER_PATH.value: None,
        InputParameters.NIR_PATH.value: None,
        InputParameters.RED_PATH.value: None,
        InputParameters.SATELLITE_PRODUCT_PATH.value: satellite_product_path,
        InputParameters.LANDCOVER_NAME.value: params.getString("landcover_name"),
        InputParameters.P03_PATH.value: None,
        InputParameters.DEL_PATH.value: None,
        InputParameters.LS_METHOD.value: "To be calculated",
        InputParameters.LS_PATH.value: None,
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

    except Exception as ex:
        import traceback

        logger.error("RUSLE has failed: %s", traceback.format_exc())
