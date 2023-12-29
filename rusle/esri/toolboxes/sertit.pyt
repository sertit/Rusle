import arcpy

import pathlib
import sys

tools_path = pathlib.Path(__file__).parent.parent.parent.absolute()

if tools_path not in sys.path:
    sys.path.append(str(tools_path))

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

        tools_path = str(pathlib.Path(__file__).parent.parent.parent.absolute())
        if tools_path not in sys.path:
            sys.path.append(tools_path)

        from rusle.rusle_arcgis import main_arcgis
        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return
