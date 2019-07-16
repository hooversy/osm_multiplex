# third-party libraries
from argparse import ArgumentParser
from networkx import write_gpickle
import pickle

# local imports
from osm_download import generate_multiplex
from count_data import process_data
from lstm_preprocessing import preprocess
from lstm import anomaly_detect

parser = ArgumentParser(description='Utility for multiplex graph generation '
                                    'and anomaly detection for collected counts.'
                        )

parser.add_argument('-g', '--graph',
                    action='store_true',
                    required=False,
                    help='Output multiplex transportation graph'
                    )

parser.add_argument('-ga', '--graph-area',
                    dest='graph_area',
                    required=False,
                    help='Specify area for multiplex graph generation'
                    )

parser.add_argument('-gm', '--graph-modes',
                    dest='graph_modes',
                    type=str,
                    required=False,
                    help='Specify modes for multiplex graph generation'
                    )

parser.add_argument('-ad', '--anomaly-detect',
                    dest='anomaly_detect',
                    action='store_true',
                    required=False,
                    help='Run mode counts anomaly detection'
                    )

parser.add_argument('-d1', '--dataset1',
                    type=str,
                    default=None,
                    required=False,
                    help='Mobility counts'
                    )

parser.add_argument('-d2', '--dataset2',
                    type=str,
                    default=None,
                    required=False,
                    help='Mobility counts'
                    )

parser.add_argument('-e1', '--element-id1',
                    dest='element_id1',
                    type=str,
                    default=None,
                    required=False,
                    help='Element ID for dataset1'
                    )

parser.add_argument('-e2', '--element-id2',
                    dest='element_id2',
                    type=str,
                    default=None,
                    required=False,
                    help='Element ID for dataset2'
                    )

parser.add_argument('-ts1', '--timestamp1',
                    type=str,
                    default=None,
                    required=False,
                    help='Timestamp for dataset1'
                    )

parser.add_argument('-ts2', '--timestamp2',
                    type=str,
                    default=None,
                    required=False,
                    help='Timestamp for dataset2'
                    )

parser.add_argument('-ss1', '--session-start1',
                    dest='session_start1',
                    type=str,
                    default=None,
                    required=False,
                    help='Session start for dataset1'
                    )

parser.add_argument('-ss2', '--session-start2',
                    dest='session_start2',
                    type=str,
                    default=None,
                    required=False,
                    help='Session start for dataset2'
                    )

parser.add_argument('-se1', '--session-end1',
                    dest='session_end1',
                    type=str,
                    default=None,
                    required=False,
                    help='Session end for dataset1'
                    )

parser.add_argument('-se2', '--session-end2',
                    dest='session_end2',
                    type=str,
                    default=None,
                    required=False,
                    help='Session end for dataset2'
                    )

parser.add_argument('-b1', '--boardings1',
                    type=str,
                    default=None,
                    required=False,
                    help='Boardings for dataset1'
                    )

parser.add_argument('-b2', '--boardings2',
                    type=str,
                    default=None,
                    required=False,
                    help='Boardings for dataset2'
                    )

parser.add_argument('-a1', '--alightings1',
                    type=str,
                    default=None,
                    required=False,
                    help='Alightings for dataset1'
                    )

parser.add_argument('-a2', '--alightings2',
                    type=str,
                    default=None,
                    required=False,
                    help='Alightings for dataset2'
                    )

parser.add_argument('-o1', '--occupancy1',
                    type=str,
                    default=None,
                    required=False,
                    help='Occupancy for dataset1'
                    )

parser.add_argument('-o2', '--occupancy2',
                    type=str,
                    default=None,
                    required=False,
                    help='Occupancy for dataset2'
                    )

parser.add_argument('-lat1', '--latitude1',
                    type=str,
                    default=None,
                    required=False,
                    help='Latitude for dataset1'
                    )

parser.add_argument('-lat2', '--latitude2',
                    type=str,
                    default=None,
                    required=False,
                    help='Latitude for dataset2'
                    )

parser.add_argument('-lon1', '--longitude1',
                    type=str,
                    default=None,
                    required=False,
                    help='Longitude for dataset1'
                    )

parser.add_argument('-lon2', '--longitude2',
                    type=str,
                    default=None,
                    required=False,
                    help='Longitude for dataset2'
                    )

parser.add_argument('-l', '--location-names',
                    dest='locations',
                    type=str,
                    default=None,
                    required=False,
                    help='Name for locations in datasets'
                    )

parser.add_argument('-np', '--npmi',
                    dest='npmi',
                    action='store_true',
                    required=False,
                    help='Filter datasets using NMPI'
                    )

parser.add_argument('-r', '--rolling',
                    dest='rolling',
                    action='store_true',
                    required=False,
                    help='Process using rolling detection'
                    )

args = parser.parse_args()

if args.graph == True:
    multiplex = generate_multiplex(args.graph_area, [args.graph_modes])
    write_gpickle(multiplex, './osm_multiplex/data/multiplex.gpickle')
elif args.anomaly_detect == True:
    print("Merging datasets")
    likely_pairs = process_data(args.dataset1, args.dataset2, run_npmi=args.npmi,
    element_id1=args.element_id1, timestamp1=args.timestamp1, session_start1=args.session_start1, session_end1=args.session_end1,
    boardings1=args.boardings1, alightings1=args.alightings1, occupancy1=args.occupancy1,
    lat1=args.latitude1, lon1=args.longitude1,
    element_id2=args.element_id2, timestamp2=args.timestamp2, session_start2=args.session_start2, session_end2=args.session_end2,
    boardings2=args.boardings2, alightings2=args.alightings2, occupancy2=args.occupancy2,
    lat2=args.latitude2, lon2=args.longitude2
    )
    print("Preprocessing for LSTM")
    preprocessed = preprocess(likely_pairs)
    print("LSTM processing")
    if args.rolling == True:
        anomalies = anomaly_detect(preprocessed, "rolling", locations=args.locations)
    else:
        anomalies = anomaly_detect(preprocessed, locations=args.locations)
    pickle.dump(anomalies, open('./osm_multiplex/data/anomaly_results.pickle', 'wb'))
else:
    raise Exception('No action specified')