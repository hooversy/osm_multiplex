from argparse import ArgumentParser

from osm_download import generate_multiplex

parser = ArgumentParser(description='Utility for multiplex graph generation '
									'and anomaly detection for collected counts.'
						)

parser.add_argument('-g', '--graph',
                    action='store_true',
                    required=False,
                    help='Output multiplex transportation graph'
                    )

parser.add_argument('-a', '--graph-area',
					dest='graph_area',
					required=False,
					help='Specify area for multiplex graph generation'
					)

parser.add_argument('-a', '--graph-modes',
					dest='graph_modes',
					required=False,
					help='Specify modes for multiplex graph generation'
					)

parser.add_argument('-ad', '--anomaly-detect',
					dest='anomaly_detect',
                    action='store_true',
                    required=False,
                    help='Run mode counts anomaly detection'
                    )

parser.add_argument('-fd', '--first-data',
					type=str,
                    required=False,
                    dest='data1',
                    help='First dataset (e.g., bus.csv).'
					)

parser.add_argument('fm', '--first-mode',
					type=str,
                    required=False,
                    dest='mode1',
                    help='First mode (e.g., bus).'
					)

parser.add_argument('-sd', '--second-data',
					type=str,
                    required=False,
                    dest='data2',
                    help='Second dataset (e.g., bike.csv).'
					)

parser.add_argument('sm', '--second-mode',
					type=str,
                    required=False,
                    dest='mode2',
                    help='Second mode (e.g., bike).'
					)

args = parser.parse_args()

if args.graph == True:
	generate_multiplex(graph_area, graph_modes)
elif args.anomaly_detect == True:
	print('Still under development')
else:
	raise Exception('No action specified')