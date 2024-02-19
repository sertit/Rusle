"""
The rusle CLI will fail if you called this file directly. Call the root file rusle.py.
"""

import logging
import argparse
import sys
from sertit import logs
from sertit.files import to_abspath
from sertit.unistra import s3_env


@s3_env
def compute_rusle():
    """
    Import osm charter with the CLI.
    Returns:

    """
    # --- PARSER ---
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-aoi",
        "--aoi_path",
        help="Path to the AOI (shp, geojson) or WKT string",
        required=True,
    )

    parser.add_argument(
        "-loc",
        "--location",
        help="Location of the AOI",
        choices=["Europe", "Global"],
        type=str,
        required=True,
    ),

    parser.add_argument(
        "-nir",
        "--nir_path",
        help="NIR band path needed if no fcover raster is provided.",
        type=to_abspath,
    )

    parser.add_argument(
        "-red",
        "--red_path",
        help="RED band path needed if no fcover raster is provided.",
        type=to_abspath,
    )

    parser.add_argument(
        "--satellite_product",
        "--sat_product",
        help="Alternative to red and nir options. Path to a satellite product with at least the Nir and Red bands",
        type=to_abspath,
    )

    parser.add_argument(
        "-lulc",
        "--landcover_name",
        help="Land Cover Name",
        choices=[
            "Corine Land Cover - 2018 (100m)",
            "Global Land Cover - Copernicus 2019 (100m)",
            "P03",
        ],
        default="Global Land Cover - Copernicus 2019 (100m)",
        type=str,
    )

    parser.add_argument(
        "-fcp",
        "--fcover_path",
        help="Path to a Fraction of green Vegetation Coverportal (Fcover) raster file. "
        "If not provided, it will be calculated from nir and red bands or satellite products",
        type=to_abspath,
    )

    parser.add_argument(
        "-p03",
        "--p03_path",
        help="P03 Path if lulc =  P03. Should have the same nomenclature as CLC",
        type=to_abspath,
    )

    parser.add_argument(
        "-del", "--del_path", help="Fire delineation path", type=to_abspath
    )

    parser.add_argument(
        "-lsp",
        "--ls_path",
        help="Optional path to the Slope angle and length (LS factor) raster. "
             "If not provided, it is calculated thanks to the DEM.",
        type=to_abspath,
    )

    parser.add_argument(
        "-dem",
        "--dem_name",
        help="DEM Name needed if ls_path option is not provided.",
        choices=["COPDEM 30m", "EUDEM 25m", "SRTM 30m", "MERIT 5 deg", "Other"],
        type=str,
        default="COPDEM 30m",
    )

    parser.add_argument(
        "-demp",
        "--other_dem_path",
        help="DEM path if dem = Other",
        type=to_abspath,
    )

    parser.add_argument(
        "-res", "--output_resolution", help="Output resolution", type=int, default=10
    )

    parser.add_argument(
        "-epsg",
        "--epsg_code",
        help="EPSG code, 4326 is not accepted. By default, it is the EPSG code of the AOI UTM zone.",
        type=int,
        default=argparse.SUPPRESS,
    )

    parser.add_argument(
        "-o", "--output", help="Output directory. ", type=to_abspath, required=True
    )

    parser.add_argument(
        "--ftep",
        help="Set this flag if the command line is run on the ftep platform. ",
        action="store_true",
    )

    # Parse args
    args = parser.parse_args()

    from rusle.rusle_core import (
        InputParameters,
        DataPath,
        rusle_core,
        LOGGER,
        LOGGING_FORMAT,
    )

    logs.init_logger(LOGGER, logging.INFO, LOGGING_FORMAT)
    LOGGER.info("--- RUSLE ---")

    # Insert args in a dict
    input_dict = {
        InputParameters.AOI_PATH.value: args.aoi_path,
        InputParameters.LOCATION.value: args.location,
        InputParameters.FCOVER_PATH.value: args.fcover_path,
        InputParameters.NIR_PATH.value: args.nir_path,
        InputParameters.RED_PATH.value: args.red_path,
        InputParameters.SATELLITE_PRODUCT_PATH.value: args.satellite_product_path,
        InputParameters.LANDCOVER_NAME.value: args.landcover_name,
        InputParameters.P03_PATH.value: args.p03_path,
        InputParameters.DEL_PATH.value: args.del_path,
        InputParameters.LS_PATH.value: args.ls_path,
        InputParameters.DEM_NAME.value: args.dem_name,
        InputParameters.OTHER_DEM_PATH.value: args.other_dem_path,
        InputParameters.OUTPUT_RESOLUTION.value: args.output_resolution,
        InputParameters.REF_EPSG.value: args.epsg_code,
        InputParameters.OUTPUT_DIR.value: args.output,
    }
    DataPath.load_paths(args.ftep)

    # --- Import osm charter
    print(DataPath.GLOBAL_DIR, DataPath.WORLD_COUNTRIES_PATH)

    try:
        rusle_core(input_dict)
        LOGGER.info("RUSLE was a success.")
        sys.exit(0)

    # pylint: disable=W0703
    except Exception as ex:
        LOGGER.error("RUSLE has failed:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    compute_rusle()
