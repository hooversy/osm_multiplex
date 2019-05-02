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