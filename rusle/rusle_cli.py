"""
This file is part of RUSLE.

RUSLE is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

RUSLE is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with RUSLE. If not, see <https://www.gnu.org/licenses/>.
"""

"""
The rusle CLI will fail if you called this file directly. Call the root file rusle.py.
"""

import sys

import rich_click as click


@click.command()
@click.option(
    "-aoi",
    "--aoi_path",
    help="Path to the AOI (shp, geojson) or WKT string",
    required=True,
)
@click.option(
    "-loc",
    "--location",
    help="Location of the AOI",
    type=click.Choice(["Europe", "Global", "Europe_legacy", "Global_legacy"]),
    required=True,
    show_default=True,
)
@click.option(
    "-nir",
    "--nir_path",
    help="NIR band path needed if no fcover raster is provided.",
    type=click.Path(resolve_path=True),
)
@click.option(
    "-red",
    "--red_path",
    help="RED band path needed if no fcover raster is provided.",
    type=click.Path(resolve_path=True),
)
@click.option(
    "--satellite_product",
    "--sat_product",
    help="Alternative to red and nir options. Path to a satellite product with at least the Nir and Red bands",
    type=click.Path(exists=True, resolve_path=True),
)
@click.option(
    "-lulc",
    "--landcover_name",
    help="Land Cover Name",
    type=click.Choice(
        [
            "Corine Land Cover - 2018 (100m)",
            "Global Land Cover - Copernicus 2019 (100m)",
            "WorldCover - ESA 2021 (10m)",
            "P03",
        ]
    ),
    default="WorldCover - ESA 2021 (10m)",
    show_default=True,
)
@click.option(
    "-fcp",
    "--fcover_path",
    help="Path to a Fraction of green Vegetation Coverportal (Fcover) raster file. "
    "If not provided, it will be calculated from nir and red bands or satellite products",
    type=click.Path(exists=True, resolve_path=True),
)
@click.option(
    "-p03",
    "--p03_path",
    help="P03 Path if lulc =  P03. Should have the same nomenclature as CLC",
    type=click.Path(exists=True, resolve_path=True),
)
@click.option(
    "-p03",
    "-lsp",
    "--ls_path",
    help="Optional path to the Slope angle and length (LS factor) raster. "
    "If not provided, it is calculated thanks to the DEM.",
    type=click.Path(exists=True, resolve_path=True),
)
@click.option(
    "-del",
    "--del_path",
    help="Fire delineation path",
    type=click.Path(exists=True, resolve_path=True),
)
@click.option(
    "-dem",
    "--dem_name",
    help="DEM Name needed if ls_path option is not provided.",
    type=click.Choice(["COPDEM 30m", "EUDEM 25m", "SRTM 30m", "MERIT 5 deg", "Other"]),
    default="COPDEM 30m",
    show_default=True,
)
@click.option(
    "-demp",
    "--other_dem_path",
    help="DEM path if dem = Other",
    type=click.Path(exists=True, resolve_path=True),
)
@click.option(
    "-res",
    "--output_resolution",
    help="Output resolution",
    type=click.IntRange(min=1, max=1000),
    default=10,
)
@click.option(
    "-epsg",
    "--epsg_code",
    help="EPSG code, 4326 is not accepted. By default, it is the EPSG code of the AOI UTM zone.",
    type=click.IntRange(min=1024, max=32767),
    show_default=True,
)
@click.option(
    "-o",
    "--output",
    help="Output directory. ",
    type=click.Path(file_okay=False, resolve_path=True, writable=True),
    required=True,
)
@click.option(
    "--ftep",
    help="Set this flag if the command line is run on the ftep platform. ",
    default=False,
)
@click.version_option(message="%(prog)s version %(version)s !")
def compute_rusle(
    aoi_path,
    location,
    nir_path,
    red_path,
    satellite_product,
    landcover_name,
    fcover_path,
    p03_path,
    del_path,
    ls_path,
    dem_name,
    other_dem_path,
    output_resolution,
    epsg_code,
    output,
    ftep,
):
    """
    Import osm charter with the CLI.
    Returns:

    """
    import logging

    logging.warning("Importing libraries, it may take a while...")
    from sertit import logs
    from sertit.unistra import unistra_s3

    from rusle.rusle_core import (
        LOGGER,
        LOGGING_FORMAT,
        DataPath,
        InputParameters,
        rusle_core,
    )

    with unistra_s3():
        logs.init_logger(LOGGER, logging.INFO, LOGGING_FORMAT)
        LOGGER.info("--- RUSLE ---")

        # Insert args in a dict
        input_dict = {
            InputParameters.AOI_PATH.value: aoi_path,
            InputParameters.LOCATION.value: location,
            InputParameters.FCOVER_PATH.value: fcover_path,
            InputParameters.NIR_PATH.value: nir_path,
            InputParameters.RED_PATH.value: red_path,
            InputParameters.SATELLITE_PRODUCT_PATH.value: satellite_product,
            InputParameters.LANDCOVER_NAME.value: landcover_name,
            InputParameters.P03_PATH.value: p03_path,
            InputParameters.DEL_PATH.value: del_path,
            InputParameters.LS_PATH.value: ls_path,
            InputParameters.DEM_NAME.value: dem_name,
            InputParameters.OTHER_DEM_PATH.value: other_dem_path,
            InputParameters.OUTPUT_RESOLUTION.value: output_resolution,
            InputParameters.REF_EPSG.value: epsg_code,
            InputParameters.OUTPUT_DIR.value: output,
        }
        DataPath.load_paths(ftep)

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
