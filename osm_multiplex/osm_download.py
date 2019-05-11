"""
.. moduleauthor:: Sylvan Hoover <hooversy@oregonstate.edu>
"""

# standard libraries
import copy
import re

# third-party libraries
import osmnx as ox
import networkx as nx

def generate_multiplex(area, modes):
    """Create multiplex transportation network graph from OSM

    Parameters
    ----------
    area : str or list
        String of geocoded place or list of [north, south, east, west]

    modes : list
        Modes included in multiplex graph

    Returns
    -------
    multiplex : networkx multidigraph
        Multiplex graph of merged OSM layers for all specified modes
    """

    separated_multiplex = nx.MultiDiGraph()

    for mode in modes:
        layer = download_osm_layer(area, mode)
        separated_multiplex = nx.union(layer, separated_multiplex, rename=(mode, None))

    multiplex = merge_multiplex_nodes(separated_multiplex)

    return multiplex

def download_osm_layer(area, mode):
    """Download a single-mode layer from OSM

    Parameters
    ----------
    area : str or list 
        String of geocoded place or list of [north, south, east, west]
        
    mode : str
        Mode choice of  {‘walk’, ‘bike’, ‘drive’, ‘drive_service’, ‘all’, ‘all_private’, ‘none’}

    Returns
    -------
    layer : networkx multidigraph
        OSM map layer of specific mode
    """

    if isinstance(area, str):
        layer = ox.graph_from_place(area, network_type=mode)
    elif isinstance(area, list) and len(area) == 4:
        layer = ox.graph_from_bbox(area[0], area[1], area[2], area[3], network_type=mode)
    else:
        raise Exception('Graph area not geocoded place nor bounding box')

    return layer

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

    node_list = list(multiplex_separated.nodes)
    node_list_all = copy.deepcopy(node_list)
    node_num = []
    for node in node_list:
        node_num.append(re.sub('^.*?-', '', node))
    node_set = set(node_num) # returns set of distinct OSM node id

    for node in node_set:
        colocated_nodes = [mode_node for mode_node in node_list_all if node in mode_node]
        for start_node in colocated_nodes:
            for end_node in colocated_nodes:
                multiplex_separated.add_edge(start_node, end_node)
            multiplex_separated.remove_edge(start_node, start_node)

    multiplex_connected = multiplex_separated # only after above loop added the necessary edges

    return multiplex_connected