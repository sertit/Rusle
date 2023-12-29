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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-aoi",
        "--aoi_path",
        help="AOI path as a shapefile. ",
        type=to_abspath,
        required=True,
    )

    parser.add_argument(
        "-loc",
        "--location",
        help="Location",
        choices=["Europe", "Global"],
        type=str,
        required=True,
    ),

    parser.add_argument(
        "-fc",
        "--fcover_method",
        help="Fcover Method",
        choices=["Already provided", "To be calculated"],
        type=str,
        required=True,
    ),

    parser.add_argument(
        "-fcp",
        "--fcover_path",
        help="Framework path if Already provided ",
        type=to_abspath,
    )

    parser.add_argument(
        "-nir",
        "--nir_path",
        help="NIR band path if Fcover is To be calculated",
        type=to_abspath,
    )

    parser.add_argument(
        "-red",
        "--red_path",
        help="RED band path if Fcover is To be calculated",
        type=to_abspath,
    )

    parser.add_argument(
        "--satellite_product_path",
        "--sat_product_path",
        help="Path to a satellite product with at least the Nir and Red bands",
        type=to_abspath,
    )

    parser.add_argument(
        "-lulc",
        "--landcover_name",
        help="Land Cover Name",
        choices=[
            "Corine Land Cover - 2018 (100m)",
            "Global Land Cover - Copernicus 2020 (100m)",
            "P03",
        ],
        required=True,
        type=str,
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
        "-ls",
        "--ls_method",
        help="LS Method",
        choices=["Already provided", "To be calculated"],
        type=str,
        required=True,
    )

    parser.add_argument(
        "-lsp", "--ls_path", help="LS Path if ls Already provided", type=to_abspath
    )

    parser.add_argument(
        "-dem",
        "--dem_name",
        help="DEM Name if ls To be calculated",
        choices=["EUDEM 25m", "SRTM 30m", "MERIT 5 deg", "Other"],
        type=str,
    )

    parser.add_argument(
        "-demp",
        "--other_dem_path",
        help="DEM path if ls To be calculated and dem = Other",
        type=to_abspath,
    )

    parser.add_argument(
        "-res", "--output_resolution", help="Output resolution", type=int, required=True
    )

    parser.add_argument(
        "-epsg", "--epsg_code", help="EPSG code", type=int, required=True
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

    from rusle.rusle_core import InputParameters, DataPath, rusle_core, LOGGER, LOGGING_FORMAT

    logs.init_logger(LOGGER, logging.INFO, LOGGING_FORMAT)
    LOGGER.info("--- RUSLE ---")

    # Insert args in a dict
    input_dict = {
        InputParameters.AOI_PATH.value: args.aoi_path,
        InputParameters.LOCATION.value: args.location,
        InputParameters.FCOVER_METHOD.value: args.fcover_method,
        InputParameters.FCOVER_PATH.value: args.fcover_path,
        InputParameters.NIR_PATH.value: args.nir_path,
        InputParameters.RED_PATH.value: args.red_path,
        InputParameters.SATELLITE_PRODUCT_PATH.value: args.satellite_product_path,
        InputParameters.LANDCOVER_NAME.value: args.landcover_name,
        InputParameters.P03_PATH.value: args.p03_path,
        InputParameters.DEL_PATH.value: args.del_path,
        InputParameters.LS_METHOD.value: args.ls_method,
        InputParameters.LS_PATH.value: args.ls_path,
        InputParameters.DEM_NAME.value: args.dem_name,
        InputParameters.OTHER_DEM_PATH.value: args.other_dem_path,
        InputParameters.OUTPUT_RESOLUTION.value: args.output_resolution,
        InputParameters.REF_EPSG.value: args.epsg_code,
        InputParameters.OUTPUT_DIR.value: args.output,
    }
    DataPath.load_paths(args.ftep)

    # input_dict = {
    #     "aoi_path": str(
    #         r"D:\TLedauphin\02_Temp_traitement\test_rusle\emsn073_aoi_32631.shp"),
    #     "location": str("Global"),
    #     "fcover_method": str("To be calculated"),
    #     "fcover_path": str(""),
    #     "nir_path": str(
    #         r"D:\TLedauphin\02_Temp_traitement\test_rusle\T31TDH_20200805T104031_B08_10m.jp2"),
    #     "red_path": str(
    #         r"D:\TLedauphin\02_Temp_traitement\test_rusle\T31TDH_20200805T104031_B08_10m.jp2"),
    #     "landcover_name": str("Corine Land Cover - 2018 (100m)"),
    #     "p03_path": str(r""),
    #     "del_path": str(r""),
    #     "ls_method": str("To be calculated"),
    #     "ls_path": str(""),
    #     "dem_name": str(r"SRTM 30m"),
    #     "other_dem_path": str(""),
    #     "output_resolution": int(10),
    #     "ref_epsg": 32633,
    #     "output_dir": str(r"D:\TLedauphin\02_Temp_traitement\test_rusle\EMSN158")}

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
