import os
import shutil
import tempfile
import unittest

from osgeo import osr, ogr
import pygeoprocessing
import shapely


def create_vector(geom_list, spatial_ref_epsg, target_filepath):
    driver = 'GPKG'
    projection = osr.SpatialReference()
    projection.ImportFromEPSG(spatial_ref_epsg)
    pygeoprocessing.shapely_geometry_to_vector(
        geom_list,
        target_filepath,
        projection.ExportToWkt(),
        driver,
        ogr_geom_type=ogr.wkbPolygon)


class SDRScenarioCompareTests(unittest.TestCase):
    """Tests for the SDR Scenario Comparison Plugin."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        self.workspace_dir = tempfile.mkdtemp(suffix='\U0001f60e')  # smiley

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_validate_comparable_scenarios(self):
        """Test vectors with identical geometries pass validation."""
        from invest_sdr_scenario_compare import sdr_scenario_compare

        epsg = 3116
        geom_list = [
            shapely.geometry.Polygon([(0, 0), (1.23456789, 0), (1, -1), (0, -1), (0, 0)]),
            shapely.geometry.Polygon([(1, 0), (2, 0), (2, -1), (1, -1), (1, 0)])]
        vector_path_list = [
            os.path.join(self.workspace_dir, x)
            for x in ['a.gpkg', 'b.gpkg', 'c.gpkg']]
        for target in vector_path_list:
            create_vector(geom_list, epsg, target)

        sdr_scenario_compare.validate_comparable_scenarios(vector_path_list)

    def test_validate_comparable_scenarios_mismatched_srs(self):
        """Test vectors with different spatial ref systems raise exception."""
        from invest_sdr_scenario_compare import sdr_scenario_compare

        geom_list = [
            shapely.geometry.Polygon([(0, 0), (1.23456789, 0), (1, -1), (0, -1), (0, 0)]),
            shapely.geometry.Polygon([(1, 0), (2, 0), (2, -1), (1, -1), (1, 0)])]
        epsg_list = [3116, 4326, 3116]
        vector_path_list = [
            os.path.join(self.workspace_dir, x)
            for x in ['a.gpkg', 'b.gpkg', 'c.gpkg']]
        for target, epsg in zip(vector_path_list, epsg_list):
            create_vector(geom_list, epsg, target)

        with self.assertRaises(ValueError) as cm:
            sdr_scenario_compare.validate_comparable_scenarios(vector_path_list)
        self.assertIn('Projection mismatch between', str(cm.exception))

    def test_validate_comparable_scenarios_mismatched_extents(self):
        """Test vectors with different extents raise exception."""
        from invest_sdr_scenario_compare import sdr_scenario_compare

        geom_list_a = [
            shapely.geometry.Polygon([(0, 0), (1.23456789, 0), (1, -1), (0, -1), (0, 0)]),
            shapely.geometry.Polygon([(1, 0), (2, 0), (2, -1), (1, -1), (1, 0)])]
        geom_list_b = [
            shapely.geometry.Polygon([(0, 0), (1.23456789, 0), (1, -1), (0, -1), (0, 0)]),
            shapely.geometry.Polygon([(10, 0), (2, 0), (2, -1), (1, -1), (10, 0)])]
        vector_path_list = [
            os.path.join(self.workspace_dir, x)
            for x in ['a.gpkg', 'b.gpkg']]
        epsg_list = [3116] * len(vector_path_list)
        for target, epsg, geoms in zip(
                vector_path_list, epsg_list, [geom_list_a, geom_list_b]):
            create_vector(geoms, epsg, target)

        with self.assertRaises(ValueError) as cm:
            sdr_scenario_compare.validate_comparable_scenarios(vector_path_list)
        self.assertIn('Extent mismatch between', str(cm.exception))

    def test_validate_comparable_scenarios_mismatched_geometries(self):
        """Test vectors with different geometries raise exception."""
        from invest_sdr_scenario_compare import sdr_scenario_compare

        # An edge case where extents of layers are identical, but feature geometries are not
        geom_list_a = [
            shapely.geometry.Polygon([(0, 0), (2, 0), (2, -2), (0, -2), (0, 0)])]
        geom_list_b = [
            shapely.geometry.Polygon([(0, 0), (1, -0.5), (2, 0), (2, -2), (0, -2), (0, 0)])]
        vector_path_list = [
            os.path.join(self.workspace_dir, x)
            for x in ['a.gpkg', 'b.gpkg']]
        epsg_list = [3116] * len(vector_path_list)
        for target, epsg, geoms in zip(
                vector_path_list, epsg_list, [geom_list_a, geom_list_b]):
            create_vector(geoms, epsg, target)

        with self.assertRaises(ValueError)as cm:
            sdr_scenario_compare.validate_comparable_scenarios(vector_path_list)
        self.assertIn('Geometry mismatch between', str(cm.exception))

    def test_validate_comparable_scenarios_mismatched_features(self):
        """Test vectors with different number of features raise exception."""
        from invest_sdr_scenario_compare import sdr_scenario_compare

        geom_list_a = [
            shapely.geometry.Polygon([(0, 0), (1.23456789, 0), (1, -1), (0, -1), (0, 0)]),
            shapely.geometry.Polygon([(1, 0), (2, 0), (2, -1), (1, -1), (1, 0)])]
        geom_list_b = [
            shapely.geometry.Polygon([(0, 0), (1.23456789, 0), (1, -1), (0, -1), (0, 0)])]
        vector_path_list = [
            os.path.join(self.workspace_dir, x)
            for x in ['a.gpkg', 'b.gpkg']]
        epsg_list = [3116] * len(vector_path_list)
        for target, epsg, geoms in zip(
                vector_path_list, epsg_list, [geom_list_a, geom_list_b]):
            create_vector(geoms, epsg, target)

        with self.assertRaises(ValueError) as cm:
            sdr_scenario_compare.validate_comparable_scenarios(vector_path_list)
        self.assertIn('Feature count mismatch between', str(cm.exception))
