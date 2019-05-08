# third-party libraries
import pandas as pd

# local imports
from .. import lstm_preprocessing

class TestSpatialGrouping:
	"""Tests the output of a single location for the record"""
	def test_selection_1(self):
		"""Select the default, which is choosing the location from dataset 1"""
		data_list = [[11.22, 33.44, 55.66, 77.88], [99.00, 11.22, 33.44, 55.66]]
		target_list = [[11.22, 33.44], [99.00, 11.22]]
		data = pd.DataFrame(data_list, columns=['lat1', 'lon1', 'lat2', 'lon2'])
		target = pd.DataFrame(target_list, columns=['lat', 'lon'])

		test = lstm_preprocessing.spatial_grouping(data)

		assert test.equals(target)

	def test_selection_2(self):
		"""Select the location from dataset 2"""
		data_list = [[11.22, 33.44, 55.66, 77.88], [99.00, 11.22, 33.44, 55.66]]
		target_list = [[55.66, 77.88], [33.44, 55.66]]
		data = pd.DataFrame(data_list, columns=['lat1', 'lon1', 'lat2', 'lon2'])
		target = pd.DataFrame(target_list, columns=['lat', 'lon'])

		test = lstm_preprocessing.spatial_grouping(data, location_selection='2')

		assert test.equals(target)

#class TestOsmLocationAssignment:
	# pending development of this function

#class TestOccupancyLevel:
	"""Tests the output of occupancy levels for both grouped and single user data"""
	# pending development of the running sum for grouped data

class TestTimeGrouping:
	"""Tests the grouping of records into specified time intervals"""
	def test_timestamp1_session2_selection1(self):
		data_list = [[1519330080, 1519330090, 44.44, 55.55, 2, 3], [1519330081, 1519330030, 44.44, 55.55, 1, 4]]
		target_list = [['2018-02-22 20:00:00', 44.44, 55.55, 3, 7]]
		data = pd.DataFrame(data_list, columns=['timestamp1', 'session_start2', 'lat', 'lon', 'occupancy1', 'occupancy2'])
		target = pd.DataFrame(target_list, columns=['time', 'lat', 'lon', 'occupancy1', 'occupancy2'])
		target['time'] =  pd.to_datetime(target['time'])
		target_multi = target.set_index(['time', 'lat', 'lon'])

		test = lstm_preprocessing.time_grouping(data)

		assert test.equals(target_multi)