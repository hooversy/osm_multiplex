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

class TestOccupancyLevel:
	"""Tests the output of occupancy levels for both grouped and single user data"""
	def test_both_individual(self):
		"""Both datasets have individual identifiers"""
		data_list = [['bike1', 'scooter1']]
		target_list = [['bike1', 'scooter1', 1, 1]]
		data = pd.DataFrame(data, columns=['element_id1', 'element_id2'])
		target = pd.DataFrame(target_list, columns=['element_id1', 'element_id2', 'occupancy1', 'occupancy2'])

		test = lstm_preprocessing.occupancy_level(data)

		assert test.equals(target)

	#def test_both_grouped(self):
		"""Both datasets have grouped counts"""

class TestDailyCumulative:
	"""Test the cumulative sum of grouped data to derive occupancy"""
		def test_summing_1_timestamp:
			"""Test cumulative sum for dataset 1 with timestamp"""
			data_list = [['bob1', 1519330080, 2, 1], ['bob1', 1519330085, 3, 0], ['bob1', 1519430080, 3, 1], ['bob1', 1519430085, 1, 2]]
			target_list = [['bob1', '2018-02-22 20:08:00', 1], ['bob1', '2018-02-22 20:08:05', 4], ['bob1', '2018-02-23 23:54:40', 2], ['bob1', '2018-02-23 23:54:45', 1]]
			data = pd.DataFrame(data_list, columns=['element_id1', 'timestamp1', 'boardings1', 'alightings1'])
			target = pd.DataFrame(target_list, columns=['element_id1', 'timestamp1', 'occupancy1'])
			target['timestamp1'] =  pd.to_datetime(target['timestamp1'])

			test = lstm_preprocessing.daily_cumulative(data, '1')

			assert test.equals(target)

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