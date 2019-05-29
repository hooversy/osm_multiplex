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

    def test_timestamp_ba(self):
        """
        Check if a csv w/ a timestamp and grouped counts is properly converted to the desired DataFrame
        """
        data = os.path.join(THIS_DIR, 'test_timestamp_ba.csv')
        element_id = 'tagID'
        timestamp = 'timestamp'
        boardings = 'boardings'
        alightings = 'alightings'
        lat = 'lat'
        lon = 'lon'

        test_df = count_data.csv_to_df(data, element_id=element_id, timestamp=timestamp,
                                       boardings=boardings, alightings=alightings, lat=lat, lon=lon)

        assert pd.util.hash_pandas_object(test_df).sum() == 7008548250528393651

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

    def test_session_ba(self):
        """
        Check if a csv w/ session times and grouped counts is properly converted to the desired DataFrame
        """
        data = os.path.join(THIS_DIR, 'test_session_ba.csv')
        element_id = 'MacPIN'
        session_start = 'SessionStart_Epoch'
        session_end = 'SessionEnd_Epoch'
        boardings = 'boardings'
        alightings = 'alightings'
        lat = 'GPS_LAT'
        lon = 'GPS_LONG'

        test_df = count_data.csv_to_df(data, element_id=element_id, session_start=session_start, session_end=session_end,
                                       boardings=boardings, alightings=alightings, lat=lat, lon=lon)

        assert pd.util.hash_pandas_object(test_df).sum() == 2589903708124850504

class TestStandardizeDatetime:
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

        processed_df = count_data.standardize_datetime(test_df)

        assert processed_df['timestamp'].dtype == 'datetime64[ns]'

    def test_timestamp_epoch(self):
        """
        Tests if timestamp is an epoch time
        """
        test_times = ['1519330080', '1518199500', '1518200760']
        test_df = pd.DataFrame(test_times, columns=['timestamp'])

        processed_df = count_data.standardize_datetime(test_df)

        assert processed_df['timestamp'].dtype == 'datetime64[ns]'

    def test_session_epoch(self):
        """
        Tests if session times are epoch times
        """
        test_times = [['1519330080', '1518199500'], ['1518200760', '1519330080'], ['1518199500', '1518200760']]
        test_df = pd.DataFrame(test_times, columns=['session_start', 'session_end'])

        processed_df = count_data.standardize_datetime(test_df)

        assert processed_df['session_start'].dtype == 'datetime64[ns]'
        assert processed_df['session_end'].dtype == 'datetime64[ns]'

class TestStandardizeEpoch:
    """
    Tests ensuring all times are unix epoch
    """
    def test_no_change_needed(self):
        """
        Tests if all timestamps are already epochs and no change is needed
        """
        test_times = [1519330080, 1518199500, 1518200760]
        test_df = pd.DataFrame(test_times, columns=['timestamp'])

        processed_df = count_data.standardize_epoch(test_df)

        assert processed_df['timestamp'].dtype == 'int64'

    def test_timestamp_datetime(self):
        """
        Tests if timestamp is a datetime
        """
        test_times = ['2018-02-22 20:08:00', '2018-02-09 18:05:00', '2018-02-09 18:26:00']
        test_df = pd.DataFrame(test_times, columns=['timestamp'])
        test_df['timestamp'] =  pd.to_datetime(test_df['timestamp'])

        processed_df = count_data.standardize_epoch(test_df)

        assert processed_df['timestamp'].dtype == 'int64'

    def test_session_datetime(self):
        """
        Tests if session times are datetimes
        """
        test_times = [['2018-02-22 20:08:00', '2018-02-09 18:05:00'], ['2018-02-09 18:26:00', '2018-02-22 20:08:00'],
                      ['2018-02-09 18:05:00', '2018-02-09 18:26:00']]
        test_df = pd.DataFrame(test_times, columns=['session_start', 'session_end'])
        test_df['session_start'] =  pd.to_datetime(test_df['session_start'])
        test_df['session_end'] =  pd.to_datetime(test_df['session_end'])

        processed_df = count_data.standardize_epoch(test_df)

        assert processed_df['session_start'].dtype == 'int64'
        assert processed_df['session_end'].dtype == 'int64'

class TestSessionLengthFilter:
    """
    Tests limiting the length of sessions to be included in candidate sessions
    """
    def test_filter_sessions(self):
        """
        Tests if dataframes with sessions are correctly filtered
        """
        session_max = 100
        test_sessions = [[1519330080, 1519330090], [151899500, 1518209500], [1518200760, 1518200770]]
        filtered_sessions = [[1519330080, 1519330090], [1518200760, 1518200770]]

        test_df = pd.DataFrame(test_sessions, columns=['session_start', 'session_end'])
        filtered_df = pd.DataFrame(filtered_sessions, columns=['session_start', 'session_end'])

        filtered_test_df = count_data.session_length_filter(test_df, session_max)

        assert filtered_test_df.equals(filtered_df)

    def test_no_sessions(self):
        """
        Tests if dataframes with single timestamps are correctly not changed
        """
        session_max = 100
        test_timestamps = [1519330080, 1518199500, 1518200760]
        test_df = pd.DataFrame(test_timestamps, columns=['timestamp'])

        filtered_test_df = count_data.session_length_filter(test_df, session_max)

        assert filtered_test_df.equals(test_df)

class TestTimeRangeJoinNp:
    """
    Tests range joining two dataframes based on time
    """
    def test_d1timestamp_d2session_np(self):
        """
        Tests with data1 having a timestamp and data2 having session times
        """
        time_range = 100
        data1_list = [[1519330080, 'bob1'], [1519330030, 'bob1'], [1518200760, 'sue1']]
        data2_list = [[1519330050, 1519330150, 'bob2'], [1518200780, 1518200980, 'sue2'], [1529200760, 1529200790, 'earl2']]
        target_list = [[1519330080, 'bob1', 1519330050, 1519330150, 'bob2'],
                       [1519330030, 'bob1', 1519330050, 1519330150, 'bob2'],
                       [1518200760, 'sue1', 1518200780, 1518200980, 'sue2']]

        data1 = pd.DataFrame(data1_list, columns=['timestamp1', 'name1'])
        data2 = pd.DataFrame(data2_list, columns=['session_start2', 'session_end2', 'name2'])
        target = pd.DataFrame(target_list, columns=['timestamp1', 'name1', 'session_start2', 'session_end2', 'name2'])

        df_range_join = count_data.time_range_join_np(data1, data2, time_range)

        assert df_range_join.equals(target)

    def test_d1session_d2timestamp_np(self):
        """
        Tests with data1 having session times and data2 having a timestamp
        """
        time_range = 100
        data1_list = [[1519330050, 1519330150, 'bob1'], [1518200780, 1518200980, 'sue1'], [1529200760, 1529200790, 'earl1']]
        data2_list = [[1519330080, 'bob2'], [1519330030, 'bob2'], [1518200760, 'sue2']]
        target_list = [[1519330050, 1519330150, 'bob1', 1519330080, 'bob2'],
                       [1519330050, 1519330150, 'bob1', 1519330030, 'bob2'],
                       [1518200780, 1518200980, 'sue1', 1518200760, 'sue2']]

        data1 = pd.DataFrame(data1_list, columns=['session_start1', 'session_end1', 'name1'])
        data2 = pd.DataFrame(data2_list, columns=['timestamp2', 'name2'])
        target = pd.DataFrame(target_list, columns=['session_start1', 'session_end1', 'name1', 'timestamp2', 'name2'])

        df_range_join = count_data.time_range_join_np(data1, data2, time_range)

        assert df_range_join.equals(target)

class TestTimeRangeJoinSql:
    """
    Tests range joining two dataframes based on time
    """
    def test_d1timestamp_d2session_sql(self):
        """
        Tests with data1 having a timestamp and data2 having session times
        """
        time_range = 100
        data1_list = [[1519330080, 'bob1'], [1519330030, 'bob1'], [1518200760, 'sue1']]
        data2_list = [[1519330050, 1519330150, 'bob2'], [1518200780, 1518200980, 'sue2'], [1529200760, 1529200790, 'earl2']]
        target_list = [[1519330080, 'bob1', 1519330050, 1519330150, 'bob2'],
                       [1519330030, 'bob1', 1519330050, 1519330150, 'bob2'],
                       [1518200760, 'sue1', 1518200780, 1518200980, 'sue2']]

        data1 = pd.DataFrame(data1_list, columns=['timestamp1', 'name1'])
        data2 = pd.DataFrame(data2_list, columns=['session_start2', 'session_end2', 'name2'])
        target = pd.DataFrame(target_list, columns=['timestamp1', 'name1', 'session_start2', 'session_end2', 'name2'])

        df_range_join = count_data.time_range_join_sql(data1, data2, time_range)

        assert df_range_join.equals(target)

    def test_d1session_d2timestamp_sql(self):
        """
        Tests with data1 having session times and data2 having a timestamp
        """
        time_range = 100
        data1_list = [[1519330050, 1519330150, 'bob1'], [1518200780, 1518200980, 'sue1'], [1529200760, 1529200790, 'earl1']]
        data2_list = [[1519330080, 'bob2'], [1519330030, 'bob2'], [1518200760, 'sue2']]
        target_list = [[1519330050, 1519330150, 'bob1', 1519330080, 'bob2'],
                       [1519330050, 1519330150, 'bob1', 1519330030, 'bob2'],
                       [1518200780, 1518200980, 'sue1', 1518200760, 'sue2']]

        data1 = pd.DataFrame(data1_list, columns=['session_start1', 'session_end1', 'name1'])
        data2 = pd.DataFrame(data2_list, columns=['timestamp2', 'name2'])
        target = pd.DataFrame(target_list, columns=['session_start1', 'session_end1', 'name1', 'timestamp2', 'name2'])

        df_range_join = count_data.time_range_join_sql(data1, data2, time_range)

        assert df_range_join.equals(target)

class TestHaversineDistFilter:
    """
    Tests filtering using haversine distance
    """
    def test_distance_filter(self):
        dist_max = 3000
        test_locations = [[44.49, -123.51, 44.51, -123.49], [44.0, -123.0, 43.0, -124.0]]
        target_list = [[44.49, -123.51, 44.51, -123.49]]
        dataframe = pd.DataFrame(test_locations, columns=['lat1', 'lon1', 'lat2', 'lon2'])
        target = pd.DataFrame(target_list, columns=['lat1', 'lon1', 'lat2', 'lon2'])

        filtered_df = count_data.haversine_dist_filter(dataframe, dist_max)

        assert filtered_df.equals(target)

class TestPairwiseFilter:
    """
    Tests the creation of candidate pairs of identifiers based on spatiotemporal filters
    """
    def test_pairwise_filter(self):
        data1_list = [['bob1', 1519330050, 44.4999, -123.5001], ['bob1', 1519330080, 44.5001, -123.4999], ['sue1', 1519330150, 43.0, -124.0]]
        data2_list = [['bob2', 1519330040, 1519330070, 44.50, -123.50], ['jake2', 1519333150, 1519333320, 44.0, -123.0]]
        target_list = [['bob1', 1519330050, 44.4999, -123.5001, 'bob2', 1519330040, 1519330070, 44.50, -123.50],
                       ['bob1', 1519330080, 44.5001, -123.4999, 'bob2', 1519330040, 1519330070, 44.50, -123.50]]

        data1 = pd.DataFrame(data1_list, columns=['element_id', 'timestamp', 'lat', 'lon'])
        data2 = pd.DataFrame(data2_list, columns=['element_id', 'session_start', 'session_end', 'lat', 'lon'])
        target = pd.DataFrame(target_list, columns=['element_id1', 'timestamp1', 'lat1', 'lon1',
                                                    'element_id2', 'session_start2', 'session_end2', 'lat2', 'lon2'])

        paired_records = count_data.pairwise_filter(data1, data2)

        assert paired_records.equals(target)

class TestNpmi:
    """
    Tests the calculation of the normalized pointwise mutual information value for two identifiers in a dataset
    """
    def test_npmi(self):
        data_list = [['bob', 'sue'], ['bob', 'sue'], ['bob', 'sandy'], ['bill', 'sandy'], ['biff', 'sandy'], ['jeff', 'mike']]
        data = pd.DataFrame(data_list, columns=['element_id1', 'element_id2'])

        npmi = count_data.npmi(data)

        assert pd.util.hash_pandas_object(npmi).sum() == -7751402083798698346

class TestNpmiDataFilter:
    """
    Tests if data is properly filtered by the value of the nmpi for the id pair
    """
    def test_default_filter(self):
        npmi_list = [['bob1', 'bob2', 0.6], ['sue1', 'susan1', 0.4], ['sue1', 'bob2', -0.4]]
        data_list = [['bob1', 'bob2', 'ate cake'],['bob1', 'bob2', 'swam in the river'], ['sue1', 'susan1', 'chilled']]
        target_list = [['bob1', 'bob2', 'ate cake'],['bob1', 'bob2', 'swam in the river']]

        npmi = pd.DataFrame(npmi_list, columns=['element_id1', 'element_id2', 'npmi'])
        data = pd.DataFrame(data_list, columns=['element_id1', 'element_id2', 'activity'])
        target = pd.DataFrame(target_list, columns=['element_id1', 'element_id2', 'activity'])

        test = count_data.npmi_data_filter(data, npmi)

        assert test.equals(target)