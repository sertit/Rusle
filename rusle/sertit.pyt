import arcpy

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Sertit"
        self.alias = "Sertit"

        # List of tool classes associated with this toolbox
        self.tools = [Rusle]


class Rusle(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Rusle"
        self.description = ""
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        # Define parameter definitions

        # AOI
        aoi = arcpy.Parameter(
            displayName="Aoi",
            name="aoi",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )

        # Location
        location = arcpy.Parameter(
            displayName="Location",
            name="location",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )

        location.filter.type = "ValueList"
        location.filter.list = ["Europe", "Global"]
        location.value = "Global"

        # Nir path
        nir_path = arcpy.Parameter(
            displayName="Nir infrared band",
            name="nir_path",
            datatype="GPRasterLayer",
            parameterType="Optional",
            direction="Input",
        )

        # Red path
        red_path = arcpy.Parameter(
            displayName="Red band",
            name="red_path",
            datatype="GPRasterLayer",
            parameterType="Optional",
            direction="Input",
        )

        # Fcover path
        fc_path = arcpy.Parameter(
            displayName="Fcover",
            name="fc_path",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
            category="Advanced",
        )

        # Landcover
        landcover = arcpy.Parameter(
            displayName="Landcover",
            name="landcover",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
            category="Advanced",
        )

        landcover.filter.type = "ValueList"
        landcover.filter.list = [
            "Corine Land Cover - 2018 (100m)",
            "Global Land Cover - Copernicus 2019 (100m)",
            "P03",
        ]
        landcover.value = "Global Land Cover - Copernicus 2019 (100m)"

        # P03 Raster
        p03_path = arcpy.Parameter(
            displayName="P03 Raster",
            name="p03",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
            category="Advanced",
        )

        # Fire delineation
        fire_delineation = arcpy.Parameter(
            displayName="Fire delineation",
            name="fire_delineation",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
            category="Advanced",
        )

        # LS Raster
        ls_raster_path = arcpy.Parameter(
            displayName="LS Raster",
            name="ls_raster",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
            category="Advanced",
        )

        # DEM
        dem = arcpy.Parameter(
            displayName="DEM",
            name="dem",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            category="Advanced",
        )

        dem.filter.type = "ValueList"
        dem.filter.list = ["COPDEM 30m", "EUDEM 25m", "SRTM 30m", "MERIT 5 deg", "Other"]
        dem.value = "COPDEM 30m"

        # Dem Raster path
        dem_raster_path = arcpy.Parameter(
            displayName="Dem Raster",
            name="dem_raster",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
            category="Advanced",
        )

        # Output resolution
        output_resolution = arcpy.Parameter(
            displayName="Output resolution",
            name="output_resolution",
            datatype="GPDouble",
            direction="Input",
            category="Advanced",
        )
        output_resolution.value = 10

        # Output folder
        output_folder = arcpy.Parameter(
            displayName="Output Folder",
            name="output_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input",
        )

        params = [
            aoi,
            location,
            fc_path,
            nir_path,
            red_path,
            landcover,
            p03_path,
            fire_delineation,
            ls_raster_path,
            dem,
            dem_raster_path,
            output_resolution,
            output_folder,
        ]

        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        if parameters[9].value == "Other":
            parameters[10].enabled = True
        else:
            parameters[10].enabled = False

        if parameters[5].value == "P03":
            parameters[6].enabled = True
        else:
            parameters[6].enabled = False

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""

        import pathlib
        import sys

        # Don't remove these lines
        tools_path = pathlib.Path(__file__).parent

        # The tool is run from sertit_atools so add sertit_atools to python path
        if tools_path.name == "sertit_atools":
            tools_path = str(tools_path.absolute())
        # The tool is run from this project so only add the root folder to python path
        else:
            tools_path = str(tools_path.parent.absolute())
        if tools_path not in sys.path:
            sys.path.append(tools_path)

        main_arcgis(parameters, messages)
        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return

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

    import logging
    import arcpy
    from sertit.arcpy import init_conda_arcpy_env, ArcPyLogger, feature_layer_to_path

    init_conda_arcpy_env()

    from rusle.rusle_core import rusle_core, InputParameters, DataPath
    # Don't concatenate or edit variable names of the two following lines unless
    # you don't want logs
    arcpy_logger = ArcPyLogger("RUSLE")
    logger = arcpy_logger.logger
    logger.setLevel(logging.DEBUG)

    # --- ENV VAR ---
    arcpy.env.overwriteOutput = True
    arcpy.CheckOutExtension("Spatial")

    aoi_path = feature_layer_to_path(parameters[0].value)
    nir_path = feature_layer_to_path(parameters[3].value)
    red_path = feature_layer_to_path(parameters[4].value)

    # --- Parameters ---
    # Load inputs
    input_dict = {
        InputParameters.AOI_PATH.value: aoi_path,
        InputParameters.LOCATION.value: parameters[1].valueAsText,
        InputParameters.FCOVER_PATH.value: parameters[2].valueAsText,
        InputParameters.NIR_PATH.value: nir_path,
        InputParameters.RED_PATH.value: red_path,
        InputParameters.LANDCOVER_NAME.value: parameters[5].valueAsText,
        InputParameters.P03_PATH.value: parameters[6].valueAsText,
        InputParameters.DEL_PATH.value: parameters[7].valueAsText,
        InputParameters.LS_PATH.value: parameters[8].valueAsText,
        InputParameters.DEM_NAME.value: parameters[9].valueAsText,
        InputParameters.OTHER_DEM_PATH.value: parameters[10].valueAsText,
        InputParameters.OUTPUT_RESOLUTION.value: int(parameters[11].valueAsText),
        InputParameters.OUTPUT_DIR.value: parameters[12].valueAsText,
    }

    # Little trick because rusle_core interprets empty string as real value
    for key in input_dict.keys():
        if input_dict[key] == "":
            input_dict[key] = None

    DataPath.load_paths()

    try:
        # Compute RUSLE charter
        rusle_core(input_dict)

        logger.info("--- RUSLE was a success.")

    except Exception as ex:
        import traceback

        logger.error("RUSLE has failed: %s", traceback.format_exc())
