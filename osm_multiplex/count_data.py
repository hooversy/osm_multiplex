"""
.. moduleauthor:: Sylvan Hoover <hooversy@oregonstate.edu>
"""

# third-party libraries

import numpy as np
import pandas as pd
import sqlite3

def csv_to_df(data, element_id=None, timestamp=None, session_start=None, session_end=None,
              boardings=None, alightings=None, lat=None, lon=None):
    """Imports counter data as csv and creates pandas dataframe

    Parameters
    ----------
    data : str
        File path of counter date

    id : str
        Header of id column

    timestamp : str
        Header of timestamp column. Optional if session times are used.

    session_start : str
        Header of session start time column. Optional if timestamp is used.

    session_end : str
        Header of session end time column. Optional if timestamp is used.

    boardings : int
        For grouped data, count of number boarding vehicle

    alightings : int
        For grouped data, count of number alighting vehicle

    lat : str
        Header of latitude column.

    lon : str
        Header of longitude column.
    
    Returns
    -------
    df_fixed_headers : pandas DataFrame
        DataFrame of counter data with headers set for further processing
    """

    if timestamp == None:
        df_imported_headers = pd.read_csv(data, parse_dates=[session_start, session_end], infer_datetime_format=True)
        if boardings == None:
            df_selected_columns = df_imported_headers[[element_id, session_start, session_end, lat, lon]].copy()
            df_fixed_headers = df_selected_columns.rename(index=str, 
                columns={element_id: "element_id", session_start: "session_start", session_end: "session_end",
                         lat: "lat", lon: "lon"})
        else:
            df_selected_columns = df_imported_headers[[element_id, session_start, session_end, boardings, alightings, lat, lon]].copy()
            df_fixed_headers = df_selected_columns.rename(index=str, 
                columns={element_id: "element_id", session_start: "session_start", session_end: "session_end",
                         boardings:'boardings', alightings:'alightings', lat: "lat", lon: "lon"})

    else:
        df_imported_headers = pd.read_csv(data, parse_dates=[timestamp], infer_datetime_format=True)
        if boardings == None:
            df_selected_columns = df_imported_headers[[element_id, timestamp, lat, lon]].copy()
            df_fixed_headers = df_selected_columns.rename(index=str, 
                columns={element_id:'element_id', timestamp:'timestamp', lat:'lat', lon:'lon'})
        else:
            df_selected_columns = df_imported_headers[[element_id, timestamp, boardings, alightings, lat, lon]].copy()
            df_fixed_headers = df_selected_columns.rename(index=str, 
                columns={element_id:'element_id', timestamp: 'timestamp', boardings:'boardings', alightings:'alightings',
                         lat:'lat', lon:'lon'})
    
    return df_fixed_headers

def standardize_datetime(dataframe):
    """Converts epoch times to datetime format. The import of the csv infers datetime format except in the
    case of epoch times. If epoch times are present, this converts them to datetime.

    Parameters
    ----------
    dataframe : pandas DataFrame
        DataFrame of the imported csv potentially with epoch times
    
    Returns
    -------
    dataframe : pandas DataFrame
        DataFrame with all times now datetime format
    """
    time_list = ['timestamp', 'timestamp1', 'timestamp2', 'time', 
                 'session_start', 'session_end', 'session_start1', 'session_end1', 'session_start2', 'session_end2']
    for time in time_list:
        try:
            if dataframe[time].dtype != 'datetime64[ns]':
                dataframe[time] = pd.to_datetime(dataframe[time],unit='s')
        except:
            pass

    return dataframe

def standardize_epoch(dataframe):
    """Converts datetimes times to unix time format.

    Parameters
    ----------
    dataframe : pandas DataFrame
        DataFrame of the imported csv potentially with datetimes
    
    Returns
    -------
    dataframe : pandas DataFrame
        DataFrame with all times now unix time int64 format
    """
    time_list = ['timestamp', 'timestamp1', 'timestamp2', 'time', 
                 'session_start', 'session_end', 'session_start1', 'session_end1', 'session_start2', 'session_end2']
    for time in time_list:
        try:
            if dataframe[time].dtype != 'int64':
                dataframe[time] = dataframe[time].astype(np.int64)
                if dataframe[time][0] > 10000000000:
                    dataframe[time] = dataframe[time] // 10**9
        except:
            pass

    return dataframe

def session_length_filter(dataframe, session_max=600):
    """If dataset has start and stop times for a session, filters out sessions exceeding max defined length
    
    Parameters
    ----------
    dataframe : pandas DataFrame
        DataFrame of all the records including single timestamp and session duration

    session_max : int
        The max length of a session in seconds to be included as a mobility candidate session
    
    Returns
    -------
    filtered_dataframe : pandas DataFrame
        DataFrame with all the records with a single timestamp and those with sessions shorter than defined
        by `session_max`
    """
    try:
        dataframe['session_length'] = dataframe['session_end'] - dataframe['session_start']
        mobility_sessions = dataframe['session_length'] <= session_max
        filtered_dataframe = dataframe[mobility_sessions].drop(columns=['session_length']).reset_index(drop=True)
    except:
        filtered_dataframe = dataframe

    return filtered_dataframe

def time_range_join_np(data1, data2, time_range=60):
    """Performs a range join based on indicated time plus/minus `time_range` buffer. This approach is quick and leverages
    numpy, but may fail on larger datasets due to exceeding memory limits

    Parameters
    ----------
    data1 : pandas DataFrame
        DataFrame of the first dataset

    data2 : pandas DataFrame
        DataFrame of the second dataset

    time_range : int
        Value for time buffer indicating range in join
    
    Returns
    -------
    df_range_join : pandas DataFrame
        DataFrame with a range join of the two datasets based on time
    """
    try:
        d1_time = data1.timestamp1.values
    except:
        d1_time = data1.session_start1.values

    try:
        d2_time = data2.timestamp2.values
    except:
        d2_time = data2.session_start2.values

    d2_time_high = d2_time + time_range
    d2_time_low = d2_time - time_range

    i, j = np.where((d1_time[:, None] >= d2_time_low) & (d1_time[:, None] <= d2_time_high))
    df_range_join_objects = pd.DataFrame(
                        np.column_stack([data1.values[i], data2.values[j]]),
                        columns=data1.columns.append(data2.columns)
                        )

    df_range_join = df_range_join_objects.infer_objects()

    return df_range_join

def time_range_join_sql(data1, data2, time_range=60):
    """Performs a range join based on indicated time plus/minus `time_range` buffer. This function uses the SQL API, which is
    much slower than the numpy based approach, but it more memory efficient so will work on larger datasets where the numpy 
    approach may quickly exceed available memory and fail

    Parameters
    ----------
    data1 : pandas DataFrame
        DataFrame of the first dataset

    data2 : pandas DataFrame
        DataFrame of the second dataset

    time_range : int
        Value for time buffer indicating range in join
    
    Returns
    -------
    df_range_join : pandas DataFrame
        DataFrame with a range join of the two datasets based on time
    """
    try:
        data1['time'] = data1['timestamp1']
    except:
        data1['time'] = data1['session_start1']

    try:
        data2['time'] = data2['timestamp2']
    except:
        data2['time'] = data2['session_start2']

    data2['time_high'] = data2['time'] + time_range
    data2['time_low'] = data2['time'] - time_range
    data2 = data2.drop(columns=['time'])

    conn = sqlite3.connect(':memory:')

    data1.to_sql('data1', conn, index=False)
    data2.to_sql('data2', conn, index=False)

    qry = '''
        select
            *
        from
            data1 join data2 on
            time between time_low and time_high
          '''

    df_range_join = pd.read_sql_query(qry, conn).drop(columns=['time', 'time_high', 'time_low'])

    return df_range_join

# def time_range_join_intervalindex(data1, data2, time_range=60):
#     """Performs a range join based on indicated time plus/minus `time_range` buffer. This is an attempt to use interval
#     indexing that exists in pandas, but for the moment can only serve as a merge and not a join so multiple records matches
#     don't get their full representation

#     Parameters
#     ----------
#     data1 : pandas DataFrame
#         DataFrame of the first dataset

#     data2 : pandas DataFrame
#         DataFrame of the second dataset

#     time_range : int
#         Value for time buffer indicating range in join
    
#     Returns
#     -------
#     df_range_join : pandas DataFrame
#         DataFrame with a range join of the two datasets based on time
#     """
#     try:
#         data1['time'] = data1['timestamp1']
#     except:
#         data1['time'] = data1['session_start1']

#     try:
#         data2['time'] = data2['timestamp2']
#     except:
#         data2['time'] = data2['session_start2']

#     data2['time_high'] = data2['time'] + time_range
#     data2['time_low'] = data2['time'] - time_range
#     data2 = data2.drop(columns=['time'])

#     time_range = pd.IntervalIndex.from_arrays(data2.time_low, data2.time_high, 'both')
#     data1_interval = data1.set_index('time')
#     data2_interval = data2.set_index(time_range)
#     df_range_join = # can't join w/ int and interval type indexes

#     return df_range_join

def haversine_dist_filter(dataframe, dist_max=100):
    """Returns dataframe with filtered for distance between recorded points using haversine distance

    Parameters
    ----------
    dataframe : pandas DataFrame
        DataFrame of the recorded candidate pairs

    dist_max : int
        Maximum distance between records
    
    Returns
    -------
    df_dist : pandas DataFrame
        DataFrame filtered for only records within defined distance
    """
    radius = 6378137 # meters

    dataframe['dlat'] = np.radians(dataframe['lat2']-dataframe['lat1'])
    dataframe['dlon'] = np.radians(dataframe['lon2']-dataframe['lon1'])
    dataframe['a'] = np.sin(dataframe['dlat']/2)**2 + np.cos(np.radians(dataframe['lat1'])) * np.cos(np.radians(dataframe['lat2'])) * np.sin(dataframe['dlon']/2)**2
    dataframe['c'] = 2 * np.arcsin(np.sqrt(dataframe['a']))
    dataframe['dist'] = radius * dataframe['c']

    close_distance = dataframe['dist'] <= dist_max
    df_dist = dataframe[close_distance].drop(columns=['dlat', 'dlon', 'a', 'c', 'dist']).reset_index(drop=True)

    return df_dist

def pairwise_filter(data1, data2, session_limit=600, detection_distance=100, detection_time=60):
    """Takes two datasets with identifiers to range join and filter to produce a list of probable joint identifiers

    Parameters
    ----------
    data1 : pandas DataFrame
        DataFrame of the first dataset

    data2 : pandas DataFrame
        DataFrame of the second dataset

    session_limit : int
        For records with session times, the maximum amount of time in seconds for a session to have occurred.
        This is to filter for only sessions while in transit

    detection_distance : int
        The maximum distance in meters for two detections to have occurred

    detection_time : int
        The maximum time difference in the initial recording of a detection
    
    Returns
    -------
    candidate_pairs : pandas DataFrame
        DataFrame of possible shared identifiers
    """
    data1_epoch = standardize_epoch(data1)
    data2_epoch = standardize_epoch(data2)

    data1_session_filter = session_length_filter(data1_epoch, session_max=session_limit)
    data2_session_filter = session_length_filter(data2_epoch, session_max=session_limit)
    
    # appends column names to distinguish between the two datasets
    data1_suffix = data1_session_filter.add_suffix('1')
    data2_suffix = data2_session_filter.add_suffix('2')

    # range join includes filtering for time proximity of recorded event
    df_range_join = time_range_join_sql(data1_suffix, data2_suffix, time_range=detection_time)

    candidate_pairs = haversine_dist_filter(df_range_join, dist_max=detection_distance)

    return candidate_pairs

def npmi(dataframe):
    """Takes a dataset with identifiers and calculates the Normalize Pointwise Mutual Information value
    for every pair of identifiers

    Parameters
    ----------
    dataframe : pandas DataFrame
        DataFrame of the candidate identifiers
    
    Returns
    -------
    npmi : pandas DataFrame
        DataFrame of NMPI values for all pairs
    """
    id_only = dataframe[['element_id1', 'element_id2']]
    ids_size = id_only.groupby(['element_id1', 'element_id2']).size().to_frame('ids_count')
    id1_size = id_only.groupby(['element_id1']).size().to_frame('id1_count')
    id2_size = id_only.groupby(['element_id2']).size().to_frame('id2_count')
    total_id_count = id_only.shape[0]
    id_distinct = id_only.drop_duplicates().reset_index(drop=True)

    calculations = id_distinct.join(ids_size, on=['element_id1', 'element_id2']).join(id1_size, on=['element_id1']).join(id2_size, on=['element_id2'])
    calculations['pmi'] = np.log((calculations['ids_count'] / total_id_count) / ((calculations['id1_count'] / total_id_count) * (calculations['id2_count'] / total_id_count)))
    calculations['npmi'] = calculations['pmi'] / (-1 * np.log(calculations['ids_count'] / total_id_count))

    npmi = calculations[['element_id1', 'element_id2', 'npmi']]

    return npmi

def npmi_data_filter(count_data, npmi_results, min_npmi=0.5):
    """Returns only data records where the npmi for the id pair exceeds a minimum threshold

    Parameters
    ----------
    count_data : pandas DataFrame
        Records of candidate pairs likely output from pairwise_filer

    npmi_results : pandas DataFrame
        All the pairs of identifiers and their respective nmpi values from candidate pairs

    min_nmpi : float
        Threshold for accepted pairs recognizing "-1 for never occurring together, 
        0 for independence, and +1 for complete co-occurrence."

    Returns
    -------
    selected_data : pandas DataFrame
        Only the records of identifier pairs that exceed the npmi threshold
    """
    selected_pairs = npmi_results['npmi'] >= min_npmi
    pair_list = npmi_results[selected_pairs].drop(columns=['npmi']).reset_index(drop=True)
    selected_data = pd.merge(count_data, pair_list, on=['element_id1', 'element_id2'])

    return selected_data

def process_data(data1, data2, run_npmi=False, 
    element_id1=None, timestamp1=None, session_start1=None, session_end1=None, boardings1=None, alightings1=None, lat1=None, lon1=None,
    element_id2=None, timestamp2=None, session_start2=None, session_end2=None, boardings2=None, alightings2=None, lat2=None, lon2=None):
    
    df1 = csv_to_df(data1, element_id=element_id1, timestamp=timestamp1, session_start=session_start1, session_end=session_end1,
                    boardings=boardings1, alightings=alightings1, lat=lat1, lon=lon1)
    df2 = csv_to_df(data2, element_id=element_id2, timestamp=timestamp2, session_start=session_start2, session_end=session_end2,
                    boardings=boardings2, alightings=alightings2, lat=lat2, lon=lon2)
    paired = pairwise_filter(df1, df2)
    # if only individually identified data, then process for npmi tagging
    if run_npmi==True:
        npmi_results = npmi(paired)
        likely_pairs = npmi_data_filter(paired, npmi_results)
        return likely_pairs
    # otherwise return all pairs that meet filter requirements without npmi assessment
    else:
        return paired
