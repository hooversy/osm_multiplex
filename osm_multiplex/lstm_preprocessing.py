"""
.. moduleauthor:: Sylvan Hoover <hooversy@oregonstate.edu>
"""

# third-party libraries
import pandas as pd
import osmnx as ox
import numpy as np

# local imports
import count_data
import osm_download

def spatial_grouping(dataframe, location_selection='1', mode='all'):
    """Assigns a single location to the data records. The data records can choose the location of dataset 1,
    the location of dataset 2, or get a location assignment based on an osm-derived network.

    Parameters
    ----------
    dataframe : pandas DataFrame
        The data records to be spatially grouped

    location_selection : str
        Selection of how to spatially group the data
            - '1' : Use dataset 1's location
            - '2' : Use dataset 2's location
            - 'osm' : assign a location based on an osm-derived network

    mode : str
        Mode choice of  {‘walk’, ‘bike’, ‘drive’, ‘drive_service’, ‘all’, ‘all_private’, ‘none’}

    Returns
    -------
    single_location : pandas DataFrame
        Data records grouped to a single spatial construct
    """
    if location_selection == '1':
        single_location = dataframe.drop(columns=['lat2', 'lon2']).rename(index=str, columns={"lat1": "lat", "lon1": "lon"})
    if location_selection == '2':
        single_location = dataframe.drop(columns=['lat1', 'lon1']).rename(index=str, columns={"lat2": "lat", "lon2": "lon"})
    if location_selection == 'osm':
        single_location = assign_osm(dataframe, mode)

    return single_location.reset_index(drop=True)

def assign_osm(dataframe, mode='all'):
    """Assigns an OSM node by taking the average location of the two datasets and finding the nearest node present in the
    mode layer.

    Parameters
    ----------
    dataframe : pandas DataFrame
        The data records to be used to find the nearest OSM node

    mode : str
        Mode choice of  {‘walk’, ‘bike’, ‘drive’, ‘drive_service’, ‘all’, ‘all_private’, ‘none’}

    Returns
    -------
    df_osm_location : pandas DataFrame
        Data records with the OSM ID of the nearest node and its respective lat/lon
    """

    max_lat = dataframe[['lat1', 'lat2']].max().max()
    min_lat = dataframe[['lat1', 'lat2']].min().min()
    max_lon = dataframe[['lon1', 'lon2']].max().max()
    min_lon = dataframe[['lon1', 'lon2']].min().min()

    osm_layer = osm_download.download_osm_layer([max_lat, min_lat, max_lon, min_lon], mode)
    osm_nodes = pd.DataFrame([[node[0], node[1]['y'], node[1]['x']] for node in osm_layer.nodes(data=True)],
                             columns=['osm_id', 'lat', 'lon']).set_index('osm_id')

    dataframe['avg_lat'] = dataframe[['lat1', 'lat2']].mean(axis=1)
    dataframe['avg_lon'] = dataframe[['lon1', 'lon2']].mean(axis=1)

    nearest_node = ox.get_nearest_nodes(osm_layer, dataframe['avg_lon'], dataframe['avg_lat'], method='balltree')
    dataframe['osm_id'] = nearest_node
    df_osm_location = dataframe.join(osm_nodes, on=['osm_id']).drop(columns=['lat1', 'lon1', 'lat2', 'lon2', 'avg_lat', 'avg_lon'])

    return df_osm_location

def occupancy_level(dataframe):
    """Calculates the count to be attributed to a record. If connected to an individual, then should be assigned
    a value of 1. If grouped boardings and alightings, then a running sum of boardings and alightings 
    will be used to determine occupancy

    Parameters
    ----------
    dataframe : pandas DataFrame
        Records either containing individuals or grouped data

    Returns
    -------
    dataframe : pandas DataFrame
        Records that now include an occupancy value indicative of the presence that detected by that system
    """

    if 'boardings1' in dataframe.columns:
        dataframe = daily_cumulative(dataframe, '1')
    else:
        dataframe['occupancy1'] = 1

    if 'boardings2' in dataframe.columns:
        dataframe = daily_cumulative(dataframe, '2')
    else:
        dataframe['occupancy2'] = 1

    return dataframe

def daily_cumulative(dataframe, identifier):
    """Cumulative daily sum of boardings and alightings to indicate vehicle occupancy. Currently lacks any
    tuning or preprocessing, but that can be implemented in later versions

    Parameters
    ----------
    dataframe : pandas DataFrame
        Records with at least one dataset being grouped

    identifier : str
        Indicates which dataset is the grouped data over which the cumulative occupancy is to be calculated
            - '1' : Calculate for dataset 1
            - '2' : Calculate for dataset 2
    Returns
    -------
    sum_occupancy : pandas DataFrame
        Dataframe with one dataset's boardings/alightings replaced by occupancy
    """
    datetime_df = count_data.standardize_datetime(dataframe)
    if identifier == '1':
        occupancy = 'occupancy1'
        element = 'element_id1'
        boardings = 'boardings1'
        alightings = 'alightings1'
        if 'timestamp1' in datetime_df.columns:
            time = 'timestamp1'
        else:
            time = 'session_start1'
    elif identifier == '2':
        occupancy = 'occupancy2'
        element = 'element_id2'
        boardings = 'boardings2'
        alightings = 'alightings2'
        if 'timestamp2' in datetime_df.columns:
            time = 'timestamp2'
        else:
            time = 'session_start2'
    else:
        raise Exception('Need valid dataset identifier')

    datetime_df['date'] = datetime_df[time].dt.date
    datetime_df['boarding_sum'] = datetime_df.sort_values([time]).groupby([element, 'date'])[boardings].cumsum()
    datetime_df['alighting_sum'] = datetime_df.sort_values([time]).groupby([element, 'date'])[alightings].cumsum()
    datetime_df[occupancy] = datetime_df['boarding_sum'] - datetime_df['alighting_sum']
    sum_occupancy = datetime_df.drop(columns=['date', boardings, alightings, 'boarding_sum', 'alighting_sum'])

    return sum_occupancy

def time_grouping(dataframe, interval='15T', time_selection='1'):
    """Groups data by temporal interval

    Parameters
    ----------
    dataframe : pandas DataFrame
        Records to be temporally grouped

    interval : str
        Interval over which records should be grouped. String options as specified by 
        `http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#time-date-components`
        The default is 15 minutes

    time_selection : str
        Source of time to be used
            - '1' : Use dataset 1's time
            - '2' : Use dataset 2's time
            - 'avg' : Use time that is the mean value of the two datasets' times

    Returns
    -------
    grouped_time : pandas DataFrame
        Returns recorded mode occupancy levels grouped by specified time interval
    """
    epoch_df = count_data.standardize_epoch(dataframe)

    if time_selection == '1':
        try:
            epoch_df['time'] = epoch_df['timestamp1']
        except:
            epoch_df['time'] = epoch_df['session_start1']
    elif time_selection == '2':
        try:
            epoch_df['time'] = epoch_df['timestamp2']
        except:
            epoch_df['time'] = epoch_df['session_start2']
    elif time_selection == 'avg':
        try:
            epoch_df['time1'] = epoch_df['timestamp1']
        except:
            epoch_df['time1'] = epoch_df['session_start1']
        try:
            epoch_df['time2'] = epoch_df['timestamp2']
        except:
            epoch_df['time2'] = epoch_df['session_start2']

        epoch_df['time'] = epoch_df[['time1', 'time2']].mean(axis=1).astype(int)
    else:
        raise Exception('Time selection not valid')

    datetime_df = count_data.standardize_datetime(epoch_df[['time', 'lat', 'lon', 'occupancy1', 'occupancy2']])
    grouped_time = datetime_df.groupby([pd.Grouper(key='time', freq=interval), 'lat', 'lon']).sum()

    return grouped_time

def weekly_dataframes(dataframe, interval='15T'):
    """Generates a dictionary of dataframes with each k,v pair representing a location and the difference between the two
    datasource counts.

    Parameters
    ----------
    dataframe : pandas DataFrame
        Contains count and difference values for all locations

    interval : str
        The time interval to be represented in the resulting dataframe. The default in 15 minutes, which results
        in 672 entries for every week

    Returns
    -------
    dataframes : dict
        A dictionary of DataFrames with each k,v pair representing a location and the difference between the two
        datasource counts.
    """
    grouped_dataframes = {}
    pivoted_dataframes = {}
    
    for name, group in dataframe.groupby(['lat', 'lon']):
        grouped_dataframes['location' + str(name)] = group.reset_index(level=['lat', 'lon']).drop(columns=['lat', 'lon'])
    for location, counts in grouped_dataframes.items():
        gaps_filled = counts[['occupancy1', 'occupancy2']].asfreq(freq=interval, method='pad').reset_index(drop=False)
        pivoted = pd.pivot_table(gaps_filled,
                                 index=[gaps_filled['time'].dt.year, gaps_filled['time'].dt.week],
                                 columns=gaps_filled.groupby(pd.Grouper(key='time', freq='W')).cumcount().add(1),
                                 values=['occupancy1', 'occupancy2'],
                                 aggfunc='sum')
        pivoted.index.names = ['year', 'week']
        useful_data = pivoted.iloc[1:-1] # removes likely incomplete first and last weeks
        if not (useful_data.empty or useful_data.shape[0]<5):
            pivoted_dataframes[location] = useful_data

    return pivoted_dataframes

def preprocess(dataframe):
    spatial_grouped = spatial_grouping(dataframe)
    occupancy = occupancy_level(spatial_grouped)
    time_grouped = time_grouping(occupancy)
    preprocessed = weekly_dataframes(time_grouped)

    return preprocessed
