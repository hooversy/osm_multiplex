"""
.. moduleauthor:: Sylvan Hoover <hooversy@oregonstate.edu>
"""

# third-party libraries

import pandas as pd

def csv_to_df(data, element_id=None, timestamp=None, session_start=None, session_end=None, lat=None, lon=None):
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

	lat : str
		Header of latitude column.

	lon : str
		Header of longitude column.
	
	Returns
	-------
	df_fixed_headers : pandas DataFrame
		DataFrame of counter data with headers set for further processing
	"""

	if timestamp != None:
		df_imported_headers = pd.read_csv(data, parse_dates=[timestamp], infer_datetime_format=True)
		df_selected_columns = df_imported_headers[[element_id, timestamp, lat, lon]].copy()
		df_fixed_headers = df_selected_columns.rename(index=str, 
			columns={element_id: "element_id", timestamp: "timestamp", lat: "lat", lon: "lon"})
	else:
		df_imported_headers = pd.read_csv(data, parse_dates=[session_start, session_end], infer_datetime_format=True)
		df_selected_columns = df_imported_headers[[element_id, session_start, session_end, lat, lon]].copy()
		df_fixed_headers = df_selected_columns.rename(index=str, 
			columns={element_id: "element_id", session_start: "session_start", session_end: "session_end", lat: "lat", lon: "lon"})

	return df_fixed_headers