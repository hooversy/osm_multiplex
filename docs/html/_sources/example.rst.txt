.. Example documentation

Examples
========

OSM Muliplex Graph
------------------

To generate an OSM multiplex graph for an OSM geocoded place from command line use
``-g`` to indicate graph generation, ``-ga`` to specify graph area, and ``-gm`` to
specify graph modes. For example, if seeking the bike network for Corvallis, Oreogn::

    python osm_multiplex -g -ga 'Corvallis, Oregon' -gm 'bike'

The resulting networkx multidigraph is pickled as ``./osm_multiplex/data/multiplex.gpickle``

Anomaly Detection
-----------------
To detect anomalies in independently sourced human mobility data from command line,
specify the datasets and relevant fields. The anomalous weeks will be printed on 
screen as the location is processed::

    python osm_multiplex -ad -d1 './osm_multiplex/data/dataset1.csv' -e1 'tagID' -ts1 'timestamp' -lat1 'lat' -lon1 'lon'
    -d2 './osm_multiplex/data/dataset2.csv' -e2 'PIN' -ss2 'SessionStart_Epoch' -se2 'SessionEnd_Epoch'
    -lat2 'GPS_LAT' -lon2 'GPS_LONG'