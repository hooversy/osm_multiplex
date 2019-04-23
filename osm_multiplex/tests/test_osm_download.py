import networkx as nx

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
		test_area = 'Corvalls, Oregon'
		test_mode = 'walk'

		test_layer = osm_download.download_osm_layer(test_area, test_mode)

		assert isinstance(test_layer, networkx.classes.multidigraph.MultiDiGraph) == True

	def test_bbox_download(self):
		"""
		Check if box (44.6, 44.55, -123.25, -123.3) for mode 'walk' will successfully download
		"""
		test_area = [44.6, 44.55, -123.25, -123.3]
		test_mode = 'bike'

		test_layer = osm_download.download_osm_layer(test_area, test_mode)

		assert isinstance(test_layer, networkx.classes.multidigraph.MultiDiGraph) == True

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

		assert isinstance(test_multiplex, networkx.classes.multidigraph.MultiDiGraph) == True)

	def test_bbox_multiplex(self):
		"""
		Check if box (44.65, 44.60, -123.10, -123.15) for modes ('walk', 'drive') will be successfully created
		"""
		test_area = [44.65, 44.60, -123.10, -123.15]
		test_modes = ['walk', 'drive']

		test_multiplex = osm_download.generate_multiplex(test_area, test_modes)

		assert isinstance(test_multiplex, networkx.classes.multidigraph.MultiDiGraph) == True)