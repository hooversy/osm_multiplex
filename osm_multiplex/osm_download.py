"""
.. moduleauthor:: Sylvan Hoover <hooversy@oregonstate.edu>
"""

import osmnx as ox
import networkx as nx

def download_osm_layer(area, mode):
	"""Download a single-mode layer from OSM

	Parameters
	----------
	area : string or list 
		String of geocoded place or list of [north, south, east, west]
		
	mode : string
		Mode choice of  {‘walk’, ‘bike’, ‘drive’, ‘drive_service’, ‘all’, ‘all_private’, ‘none’}

	Returns
	-------
	layer : networkx multidigraph
		OSM map layer of specific mode
	"""

	if isinstance(area, str):
		layer = ox.graph_from_place(area, network_type=mode)
	elif isinstance(area, list):
		layer = ox.graph_from_bbox(area[0], area[1], area[2], area[3], network_type=mode)
	else:
		raise Exception('Graph area not geocoded place nor bounding box')

	return layer

def generate_multiplex(area, modes):
	"""Create multiplex transportation network graph from OSM

	Parameters
	----------
	area : string or list
		String of geocoded place or list of [north, south, east, west]

	modes : list
		Modes included in multiplex graph

	Returns
	-------
	multiplex : networkx multidigraph
		Multiplex graph of merged OSM layers for all specified modes
	"""

	multiplex = nx.Graph()

	for mode in modes:
		layer = download_osm_layer(area, mode)
		multiplex = nx.union(layer, multiplex, rename=(mode, None))

	return multiplex

def merge_multiplex_nodes(multiplex_separated):
	"""In the multiplex graph, each mode has its own layer of nodes and edges. Nodes are
	represented by a mode prefix and a node number. In order to allow inter-mode movement,
	a zero-cost edge needs to be created between co-located nodes for different modes.

	Parameters
	----------
	multiplex_separated : networkx multidigraph
		Multiplex network w/ each mode in an isolated layer

	Returns
	-------
	multiplex_connected : networkx multidigraph
		Multiplex network with co-located nodes connected
	"""