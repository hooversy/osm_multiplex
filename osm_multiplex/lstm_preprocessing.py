"""
.. moduleauthor:: Sylvan Hoover <hooversy@oregonstate.edu>
"""

# third-party libraries
import pandas as pd

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

def assign_osm(dataframe, assignment_method='1'):
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
