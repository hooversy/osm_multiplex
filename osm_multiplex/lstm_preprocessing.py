"""
.. moduleauthor:: Sylvan Hoover <hooversy@oregonstate.edu>
"""

# third-party libraries
import pandas as pd
import numpy as np

# local imports
from . import count_data

def spatial_grouping(dataframe, location_selection='1'):
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
		single_location = assign_osm(dataframe)

	return single_location.reset_index(drop=True)

def osm_location_assignment(dataframe, assignment_method='1'):
	# still under development
	return dataframe

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
	try:
		if dataframe['boardings1'] != None:
			dataframe['occupancy1'] = 0 # need running sum of boardings and alightings; keep in mind processing for data errors
	except:
		dataframe['occupancy1'] = 1

	try:
		if dataframe['boardings2'] != None:
			dataframe['occupancy2'] = 0 # need running sum of boardings and alightings; keep in mind processing for data errors
	except:
		dataframe['occupancy2'] = 1

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

		epoch_df['time'] = np.mean(['time1', 'time2'], dtype=int)
	else:
		raise Exception('Time selection not valid')

	datetime_df = count_data.standardize_datetime(epoch_df[['time', 'lat', 'lon', 'occupancy1', 'occupancy2']])
	grouped_time = datetime_df.groupby([pd.Grouper(key='time', freq=interval), 'lat', 'lon']).sum()

	return grouped_time
