from argparse import ArgumentParser

from .osm_download import generate_multiplex
from .anomaly_detection import anomaly_detection

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
                    type=list,
                    required=False,
                    help='Specify modes for multiplex graph generation'
                    )

parser.add_argument('-ad', '--anomaly-detect',
                    dest='anomaly_detect',
                    action='store_true',
                    required=False,
                    help='Run mode counts anomaly detection'
                    )

parser.add_argument('-d', '--data',
                    type=str,
                    required=False,
                    help='Mobility counts'
                    )

args = parser.parse_args()

if args.graph == True:
    generate_multiplex(args.graph_area, args.graph_modes)
elif args.anomaly_detect == True:
    anomaly_detection(args.data)
else:
    raise Exception('No action specified')