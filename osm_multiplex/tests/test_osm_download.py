# third-party libraries
import networkx as nx
import pytest

# local imports
from .. import osm_download

class TestDownloadOsmLayer:
    """
    Tests downloading a single-mode network layer from OSM
    """
    def test_geocode_download(self):
        """
        Check if location 'Corvallis, Oregon' for mode 'walk' will successfully download
        """
        test_area = 'Corvallis, Oregon'
        test_mode = 'walk'

        test_layer = osm_download.download_osm_layer(test_area, test_mode)

        assert isinstance(test_layer, nx.classes.multidigraph.MultiDiGraph) == True

    def test_bbox_download(self):
        """
        Check if box (44.6, 44.55, -123.25, -123.3) for mode 'walk' will successfully download
        """
        test_area = [44.6, 44.55, -123.25, -123.3]
        test_mode = 'bike'

        test_layer = osm_download.download_osm_layer(test_area, test_mode)

        assert isinstance(test_layer, nx.classes.multidigraph.MultiDiGraph) == True

    def test_bad_area_download(self):
        """
        Check if exception is raised when neither a string for a geocoded location nor a bbox is entered
        """

        test_area = [44.6]
        test_mode = 'drive'
        with pytest.raises(Exception):
            osm_download.download_osm_layer(test_area, test_mode)

class TestGenerateMultiplex:
    """
    Tests the creation of multilayer transportation networks
    """
    def test_geocode_multiplex(self):
        """
        Check if location 'Albany, Oregon' for modes ('walk', 'bike') will be successfully created
        """
        test_area = 'Albany, Oregon'
        test_modes = ['walk', 'bike']

        test_multiplex = osm_download.generate_multiplex(test_area, test_modes)

        assert isinstance(test_multiplex, nx.classes.multidigraph.MultiDiGraph) == True

    def test_bbox_multiplex(self):
        """
        Check if box (44.65, 44.60, -123.10, -123.15) for modes ('walk', 'drive') will be successfully created
        """
        test_area = [44.65, 44.60, -123.10, -123.15]
        test_modes = ['walk', 'drive']

        test_multiplex = osm_download.generate_multiplex(test_area, test_modes)

        assert isinstance(test_multiplex, nx.classes.multidigraph.MultiDiGraph) == True

class TestMergeMultiplexNodes:
    """
    Tests if colocated nodes have edges connecting them
    """

    def test_merge_multiplex_nodes(self):
        separated_graph = nx.MultiDiGraph()
        separated_graph.add_nodes_from(['A-1', 'A-2', 'B-1', 'B-2'])
        separated_graph.add_edges_from([('A-1', 'A-2'), ('B-1', 'B-2')])

        connected_graph = osm_download.merge_multiplex_nodes(separated_graph)

        connected_edges = list(connected_graph.edges)
        assert ('A-1', 'B-1', 0) in connected_edges
        assert ('B-1', 'A-1', 0) in connected_edges
        assert ('A-2', 'B-2', 0) in connected_edges
        assert ('B-2', 'A-2', 0) in connected_edges