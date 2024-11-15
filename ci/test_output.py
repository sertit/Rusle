# -*- coding: utf-8 -*-
# Output Check © 2024 by Sertit is licensed under Attribution-NonCommercial-NoDerivatives 4.0 International.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/
""" Tests with output"""
import os
import tempfile

from sertit import AnyPath, ci
from sertit.unistra import s3_env

from rusle.rusle_core import DataPath, rusle_core

ci.reduce_verbosity()

os.environ["USE_S3_STORAGE"] = "1"
os.environ["AWS_S3_ENDPOINT"] = "s3.unistra.fr"


@s3_env
def test_rusle():
    ci_path = AnyPath("s3://sertit-ci") / "rusle" / "Test_France"
    nir = ci_path / "S2" / "T32ULU_20211024T104029_B08.jp2"
    red = ci_path / "S2" / "T32ULU_20211024T104029_B04.jp2"
    aoi = ci_path / "AOI" / "32ULU-zone-test-67.shp"
    expected_output = ci_path / "out_expected" / "ErosionRisk.tif"

    assert nir.exists()
    assert red.exists()
    assert aoi.exists()
    assert expected_output.exists()

    with tempfile.TemporaryDirectory() as output:
        input_dict = {
            "aoi_path": str(aoi),
            "location": "Europe",
            "nir_path": str(nir),
            "red_path": str(red),
            "dem_name": "COPDEM 30m",
            "landcover_name": "WorldCover - ESA 2021 (10m)",
            "output_resolution": 10,
            "output_dir": output,
        }
        DataPath.load_paths()
        rusle_core(input_dict=input_dict)

        output_classification = os.path.join(output, "ErosionRisk.tif")
        ci.assert_raster_max_mismatch(
            expected_output, output_classification, max_mismatch_pct=3
        )
