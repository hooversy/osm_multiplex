# standard libraries
import os

# third-party libraries
import pandas as pd

# local imports
from .. import count_data

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestCsvToDf:
	"""
	Tests converting a csv with various headers into a processible DataFrame
	"""
	def test_timestamp(self):
		"""
		Check if a csv w/ a timestamp is properly converted to the desired DataFrame
		"""
		data = os.path.join(THIS_DIR, 'test_timestamp.csv')
		element_id = 'tagID'
		timestamp = 'timestamp'
		lat = 'lat'
		lon = 'lon'

		test_df = count_data.csv_to_df(data, element_id=element_id, timestamp=timestamp, lat=lat, lon=lon)

		assert pd.util.hash_pandas_object(test_df).sum() == -6761865716520410554

	def test_session(self):
		"""
		Check if a csv w/ session times is properly converted to the desired DataFrame
		"""
		data = os.path.join(THIS_DIR, 'test_session.csv')
		element_id = 'MacPIN'
		session_start = 'SessionStart_Epoch'
		session_end = 'SessionEnd_Epoch'
		lat = 'GPS_LAT'
		lon = 'GPS_LONG'

		test_df = count_data.csv_to_df(data, element_id=element_id, session_start=session_start, session_end=session_end, lat=lat, lon=lon)

		assert pd.util.hash_pandas_object(test_df).sum() == 7098407329788286247

class TestStandardizeTimestamp:
	"""
	Tests ensuring all times are datetime format
	"""
	def test_no_change_needed(self):
		"""
		Tests if all timestamps are already datetime and no change is needed
		"""
		test_times = ['2018-02-22 20:08:00', '2018-02-09 18:05:00', '2018-02-09 18:26:00']
		test_df = pd.DataFrame(test_times, columns=['timestamp'])
		test_df['timestamp'] =  pd.to_datetime(test_df['timestamp'])

		processed_df = count_data.standardize_timestamp(test_df)

		assert processed_df['timestamp'].dtype == 'datetime64[ns]'

	def test_timestamp_epoch(self):
		"""
		Tests if timestamp is an epoch time
		"""
		test_times = ['1519330080', '1518199500', '1518200760']
		test_df = pd.DataFrame(test_times, columns=['timestamp'])

		processed_df = count_data.standardize_timestamp(test_df)

		assert processed_df['timestamp'].dtype == 'datetime64[ns]'

	def test_session_epoch(self):
		"""
		Tests if session times are epoch times
		"""
		test_times = [['1519330080', '1518199500'], ['1518200760', '1519330080'], ['1518199500', '1518200760']]
		test_df = pd.DataFrame(test_times, columns=['session_start', 'session_end'])

		processed_df = count_data.standardize_timestamp(test_df)

		assert processed_df['session_start'].dtype == 'datetime64[ns]'
		assert processed_df['session_end'].dtype == 'datetime64[ns]'
		