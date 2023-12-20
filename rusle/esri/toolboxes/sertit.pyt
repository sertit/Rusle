import arcpy

import pathlib
import sys

tools_path = pathlib.Path(__file__).parent.parent.parent.absolute()

if tools_path not in sys.path:
    sys.path.append(str(tools_path))

# Dummy comment2

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
            datatype="DEFile",
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

        # FC method
        fc_method = arcpy.Parameter(
            displayName="FCover method",
            name="fc_method",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )

        fc_method.filter.type = "ValueList"
        fc_method.filter.list = ["Already Provided", "To be calculated"]

        # Nir path
        nir_path = arcpy.Parameter(
            displayName="Nir infrared band",
            name="nir_path",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
        )

        # Red path
        red_path = arcpy.Parameter(
            displayName="Red band",
            name="red_path",
            datatype="DEFile",
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
        )

        # Landcover
        landcover = arcpy.Parameter(
            displayName="Landcover",
            name="landcover",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )

        landcover.filter.type = "ValueList"
        landcover.filter.list = [
            "Corine Land Cover - 2018 (100m)",
            "Global Land Cover - Copernicus 2020 (100m)",
            "P03",
        ]

        # P03 Raster
        p03_path = arcpy.Parameter(
            displayName="P03 Raster",
            name="p03",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
        )

        # Fire delineation
        fire_delineation = arcpy.Parameter(
            displayName="Fire delineation",
            name="fire_delineation",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
        )

        # LS method
        ls_method = arcpy.Parameter(
            displayName="LS method",
            name="ls_method",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )

        ls_method.filter.type = "ValueList"
        ls_method.filter.list = ["Already Provided", "To be calculated"]

        # LS Raster
        ls_raster_path = arcpy.Parameter(
            displayName="LS Raster",
            name="ls_raster",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
        )

        # DEM
        dem = arcpy.Parameter(
            displayName="DEM",
            name="dem",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )

        dem.filter.type = "ValueList"
        dem.filter.list = ["EUDEM 25m", "SRTM 30m", "MERIT 5 deg", "Other"]

        # Dem Raster path
        dem_raster_path = arcpy.Parameter(
            displayName="Dem Raster",
            name="dem_raster",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
        )

        # Output resolution
        output_resolution = arcpy.Parameter(
            displayName="Output resolution",
            name="output_resolution",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
        )

        # Output Coordinate System
        output_coordinate_system = arcpy.Parameter(
            displayName="Output Coordinate System",
            name="output_coordinate_system",
            datatype="GPCoordinateSystem",
            parameterType="Required",
            direction="Input",
        )

        # Third parameter
        output_folder = arcpy.Parameter(
            displayName="Output Folder",
            name="output_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Output",
        )

        params = [
            aoi,
            location,
            fc_method,
            fc_path,
            nir_path,
            red_path,
            landcover,
            p03_path,
            fire_delineation,
            ls_method,
            ls_raster_path,
            dem,
            dem_raster_path,
            output_resolution,
            output_coordinate_system,
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

        if parameters[2].value == "To be calculated":
            parameters[3].enabled = False
            parameters[4].enabled = True
            parameters[5].enabled = True

        elif parameters[2].value == "Already provided":
            parameters[3].enabled = True
            parameters[4].enabled = False
            parameters[5].enabled = False
        else:
            parameters[3].enabled = False
            parameters[4].enabled = False
            parameters[5].enabled = False

        if parameters[6].value == "P03":
            parameters[7].enabled = True
        else:
            parameters[7].enabled = False

        # ls_method
        if parameters[9].value == "To be calculated":
            parameters[10].enabled = False
            parameters[11].enabled = True

            # dem
            if parameters[11].value == "Other":
                parameters[12].enabled = True
            else:
                parameters[12].enabled = False

        # ls_method
        elif parameters[9].value == "Already provided":
            parameters[10].enabled = True
            parameters[11].enabled = False
        else:
            parameters[10].enabled = False
            parameters[11].enabled = False
            parameters[11].enabled = False

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""

        import logging.handlers
        import arcpy

        from sertit.arcpy import init_conda_arcpy_env, ArcPyLogHandler

        init_conda_arcpy_env()

        from rusle.main import rusle_core, InputParameters, DataPath
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

        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return
