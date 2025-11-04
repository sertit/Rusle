# Output Check © 2024 by Sertit is licensed under Attribution-NonCommercial-NoDerivatives 4.0 International.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/
"""Tests with output"""

from pathlib import Path

from sertit import AnyPath, ci
from sertit.types import AnyPathType
from sertit.unistra import define_s3_client, s3_env

from rusle.rusle_core import DataPath, rusle_core

ci.reduce_verbosity()

DEBUG = False


def get_ci_path() -> AnyPathType:
    """Get ci DATA path"""
    define_s3_client()
    return AnyPath("s3://sertit-ci").joinpath("rusle")


def get_output(tmp, file, debug=False, subfolder=None) -> AnyPathType:
    """Get the output file path whether we are in debug mode or not"""
    # Create base folder according the mode
    base_folder = Path(__file__).resolve().parent / "ci_output" if debug else tmp

    # Add subfolder if necessary
    if subfolder is not None:
        base_folder = base_folder / subfolder

    # Create folder if necessary
    base_folder.mkdir(parents=True, exist_ok=True)

    # Return the file path
    return base_folder / file


@s3_env
def test_rusle(tmp_path):
    ci_path = get_ci_path() / "Test_France"

    nir = ci_path / "S2" / "T32ULU_20211024T104029_B08.jp2"
    red = ci_path / "S2" / "T32ULU_20211024T104029_B04.jp2"
    aoi = ci_path / "AOI" / "32ULU-zone-test-67.shp"
    # truth
    expected_output = ci_path / "out_expected" / "ErosionRisk.tif"

    out = get_output(tmp_path, "ErosionRisk_out.tif", DEBUG, subfolder="S2_T32ULU_67")

    assert nir.exists()
    assert red.exists()
    assert aoi.exists()
    assert expected_output.exists()

    # with tempfile.TemporaryDirectory() as output:
    input_dict = {
        "aoi_path": str(aoi),
        "location": "Europe",
        "nir_path": str(nir),
        "red_path": str(red),
        "dem_name": "COPDEM 30m",
        "landcover_name": "WorldCover - ESA 2021 (10m)",
        "output_resolution": 10,
        "output_dir": out,
    }
    DataPath.load_paths()
    rusle_core(input_dict=input_dict, ftep=False)

    ci.assert_raster_max_mismatch(
        expected_output, AnyPath(out, "ErosionRisk.tif"), max_mismatch_pct=3
    )
