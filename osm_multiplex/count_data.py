"""
.. moduleauthor:: Sylvan Hoover <hooversy@oregonstate.edu>
"""

# third-party libraries

import numpy as np
import pandas as pd

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
	try:
		if dataframe['timestamp'].dtype != 'datetime64[ns]':
			dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'],unit='s')
	except:
		pass
	try:
		if dataframe['session_start'].dtype != 'datetime64[ns]' or dataframe['session_end'].dtype != 'datetime64[ns]':
			dataframe['session_start'] = pd.to_datetime(dataframe['session_start'],unit='s')
			dataframe['session_end'] = pd.to_datetime(dataframe['session_end'],unit='s')
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
	try:
		if dataframe['timestamp'].dtype != 'int64':
			dataframe['timestamp'] = dataframe['timestamp'].astype(np.int64) // 10**9
	except:
		pass
	try:
		if dataframe['session_start'].dtype != 'int64' or dataframe['session_end'].dtype != 'int64':
			dataframe['session_start'] = dataframe['session_start'].astype(np.int64) // 10**9
			dataframe['session_end'] = dataframe['session_end'].astype(np.int64) // 10**9
	except:
		pass

	return dataframe

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

	# appends column names to distinguish between the two datasets
	data1_epoch.add_suffix('1')
	data2_epoch.add_suffix('2')



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


	