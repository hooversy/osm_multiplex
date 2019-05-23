# third-party libraries
import pandas as pd
import pytest

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

    def test_selection_osm(self):
        """Select the location by finding the nearest OSM node to the average"""
        data_list = [[44.594487, -123.262589, 44.562769, -123.267733],
             [44.594528, -123.261476, 44.563046, -123.268784]]
        target_list = [[36921149, 44.57822, -123.264745],
                       [36921149, 44.57822, -123.264745]]
        data = pd.DataFrame(data_list, columns=['lat1', 'lon1', 'lat2', 'lon2'])
        target = pd.DataFrame(target_list, columns=['osm_id', 'lat', 'lon'])

        test = lstm_preprocessing.spatial_grouping(data, location_selection='osm')

        assert test.equals(target)

class TestAssignOsm:
    """Find the nearest OSM node to the average of the two dataset locations"""
    def test_find_nearest(self):
        data_list = [[44.594487, -123.262589, 44.562769, -123.267733],
                     [44.594528, -123.261476, 44.563046, -123.268784]]
        target_list = [[36921149, 44.57822, -123.264745],
                       [36921149, 44.57822, -123.264745]]
        data = pd.DataFrame(data_list, columns=['lat1', 'lon1', 'lat2', 'lon2'])
        target = pd.DataFrame(target_list, columns=['osm_id', 'lat', 'lon'])

        test = lstm_preprocessing.assign_osm(data)

        assert test.equals(target)

class TestOccupancyLevel:
    """Tests the output of occupancy levels for both grouped and single user data"""
    def test_both_individual(self):
        """Both datasets have individual identifiers"""
        data_list = [['bike1', 'scooter1']]
        target_list = [['bike1', 'scooter1', 1, 1]]
        data = pd.DataFrame(data_list, columns=['element_id1', 'element_id2'])
        target = pd.DataFrame(target_list, columns=['element_id1', 'element_id2', 'occupancy1', 'occupancy2'])

        test = lstm_preprocessing.occupancy_level(data)

        assert test.equals(target)

    def test_both_grouped(self):
        """Both datasets have grouped counts"""
        data_list = [['bike1', 'scooter1', 1519330080, 1519330081, 2, 1, 3, 1],
                     ['bike1', 'scooter1', 1519330085, 1519330086, 3, 0, 2, 1],
                     ['bike1', 'scooter1', 1519430080, 1519430081, 3, 1, 4, 2],
                     ['bike1', 'scooter1', 1519430085, 1519430086, 1, 2, 0, 1]]
        target_list = [['bike1', 'scooter1', '2018-02-22 20:08:00', '2018-02-22 20:08:01', 1, 2],
                       ['bike1', 'scooter1', '2018-02-22 20:08:05', '2018-02-22 20:08:06', 4, 3],
                       ['bike1', 'scooter1', '2018-02-23 23:54:40', '2018-02-23 23:54:41', 2, 2],
                       ['bike1', 'scooter1', '2018-02-23 23:54:45', '2018-02-23 23:54:46', 1, 1]]
        data = pd.DataFrame(data_list, columns=['element_id1', 'element_id2', 'timestamp1', 'timestamp2', 'boardings1', 'alightings1', 'boardings2', 'alightings2'])
        target = pd.DataFrame(target_list, columns=['element_id1', 'element_id2', 'timestamp1', 'timestamp2', 'occupancy1', 'occupancy2'])
        target['timestamp1'] =  pd.to_datetime(target['timestamp1'])
        target['timestamp2'] =  pd.to_datetime(target['timestamp2'])

        test = lstm_preprocessing.occupancy_level(data)

        assert test.equals(target)

class TestDailyCumulative:
    """Test the cumulative sum of grouped data to derive occupancy"""
    def test_summing_1_timestamp(self):
        """Test cumulative sum for dataset 1 with timestamp"""
        data_list = [['bob1', 1519330080, 2, 1], ['bob1', 1519330085, 3, 0], ['bob1', 1519430080, 3, 1], ['bob1', 1519430085, 1, 2]]
        target_list = [['bob1', '2018-02-22 20:08:00', 1], ['bob1', '2018-02-22 20:08:05', 4], ['bob1', '2018-02-23 23:54:40', 2], ['bob1', '2018-02-23 23:54:45', 1]]
        data = pd.DataFrame(data_list, columns=['element_id1', 'timestamp1', 'boardings1', 'alightings1'])
        target = pd.DataFrame(target_list, columns=['element_id1', 'timestamp1', 'occupancy1'])
        target['timestamp1'] =  pd.to_datetime(target['timestamp1'])

        test = lstm_preprocessing.daily_cumulative(data, '1')

        assert test.equals(target)

    def test_summing_2_timestamp(self):
        """Test cumulative sum for dataset 2 with timestamp"""
        data_list = [['bob2', 1519330080, 2, 1], ['bob2', 1519330085, 3, 0], ['bob2', 1519430080, 3, 1], ['bob2', 1519430085, 1, 2]]
        target_list = [['bob2', '2018-02-22 20:08:00', 1], ['bob2', '2018-02-22 20:08:05', 4], ['bob2', '2018-02-23 23:54:40', 2], ['bob2', '2018-02-23 23:54:45', 1]]
        data = pd.DataFrame(data_list, columns=['element_id2', 'timestamp2', 'boardings2', 'alightings2'])
        target = pd.DataFrame(target_list, columns=['element_id2', 'timestamp2', 'occupancy2'])
        target['timestamp2'] =  pd.to_datetime(target['timestamp2'])

        test = lstm_preprocessing.daily_cumulative(data, '2')

        assert test.equals(target)

    def test_summing_1_session(self):
        """Test cumulative sum for dataset 1 with session times"""
        data_list = [['bob1', 1519330080, 1519330081, 2, 1],
                     ['bob1', 1519330085, 1519330086, 3, 0],
                     ['bob1', 1519430080, 1519430081, 3, 1],
                     ['bob1', 1519430085, 1519430086, 1, 2]]
        target_list = [['bob1', '2018-02-22 20:08:00', '2018-02-22 20:08:01', 1],
                       ['bob1', '2018-02-22 20:08:05', '2018-02-22 20:08:06', 4],
                       ['bob1', '2018-02-23 23:54:40', '2018-02-23 23:54:41', 2],
                       ['bob1', '2018-02-23 23:54:45', '2018-02-23 23:54:46', 1]]
        data = pd.DataFrame(data_list, columns=['element_id1', 'session_start1', 'session_end1', 'boardings1', 'alightings1'])
        target = pd.DataFrame(target_list, columns=['element_id1', 'session_start1', 'session_end1', 'occupancy1'])
        target['session_start1'] =  pd.to_datetime(target['session_start1'])
        target['session_end1'] =  pd.to_datetime(target['session_end1'])

        test = lstm_preprocessing.daily_cumulative(data, '1')

        assert test.equals(target)

    def test_summing_2_session(self):
        """Test cumulative sum for dataset 2 with session times"""
        data_list = [['bob2', 1519330080, 1519330081, 2, 1],
                     ['bob2', 1519330085, 1519330086, 3, 0],
                     ['bob2', 1519430080, 1519430081, 3, 1],
                     ['bob2', 1519430085, 1519430086, 1, 2]]
        target_list = [['bob2', '2018-02-22 20:08:00', '2018-02-22 20:08:01', 1],
                       ['bob2', '2018-02-22 20:08:05', '2018-02-22 20:08:06', 4],
                       ['bob2', '2018-02-23 23:54:40', '2018-02-23 23:54:41', 2],
                       ['bob2', '2018-02-23 23:54:45', '2018-02-23 23:54:46', 1]]
        data = pd.DataFrame(data_list, columns=['element_id2', 'session_start2', 'session_end2', 'boardings2', 'alightings2'])
        target = pd.DataFrame(target_list, columns=['element_id2', 'session_start2', 'session_end2', 'occupancy2'])
        target['session_start2'] =  pd.to_datetime(target['session_start2'])
        target['session_end2'] =  pd.to_datetime(target['session_end2'])

        test = lstm_preprocessing.daily_cumulative(data, '2')

        assert test.equals(target)

    def test_invalid_identifier(self):
        """Tests if exception is raised when identifier parameter is not valid"""
        data_list = [['bob2', 1519330080, 1519330081, 2, 1],
                     ['bob2', 1519330085, 1519330086, 3, 0],
                     ['bob2', 1519430080, 1519430081, 3, 1],
                     ['bob2', 1519430085, 1519430086, 1, 2]]
        target_list = [['bob2', '2018-02-22 20:08:00', '2018-02-22 20:08:01', 1],
                       ['bob2', '2018-02-22 20:08:05', '2018-02-22 20:08:06', 4],
                       ['bob2', '2018-02-23 23:54:40', '2018-02-23 23:54:41', 2],
                       ['bob2', '2018-02-23 23:54:45', '2018-02-23 23:54:46', 1]]
        data = pd.DataFrame(data_list, columns=['element_id2', 'session_start2', 'session_end2', 'boardings2', 'alightings2'])

        with pytest.raises(Exception):
            lstm_preprocessing.daily_cumulative(data, '3')

class TestTimeGrouping:
    """Tests the grouping of records into specified time intervals"""
    def test_timestamp1_session2_interval15_selection1(self):
        data_list = [[1519330080, 1519330090, 44.44, 55.55, 2, 3],
                     [1519330081, 1519330030, 44.44, 55.55, 1, 4],
                     [1519430080, 1519430090, 44.44, 55.55, 3, 2],
                     [1519430081, 1519430030, 44.44, 55.55, 2, 6]]
        target_list = [['2018-02-22 20:00:00', 44.44, 55.55, 3, 7, 4],
                       ['2018-02-23 23:45:00', 44.44, 55.55, 5, 8, 3]]
        data = pd.DataFrame(data_list, columns=['timestamp1', 'session_start2', 'lat', 'lon', 'occupancy1', 'occupancy2'])
        target = pd.DataFrame(target_list, columns=['time', 'lat', 'lon', 'occupancy1', 'occupancy2', 'difference'])
        target['time'] =  pd.to_datetime(target['time'])
        target_multi = target.set_index(['time', 'lat', 'lon'])

        test = lstm_preprocessing.time_grouping(data, interval='15T', time_selection='1')

        assert test.equals(target_multi)

    def test_timestamp1_session2_interval15_selection2(self):
        data_list = [[1519330080, 1519330090, 44.44, 55.55, 2, 3],
                     [1519330081, 1519330030, 44.44, 55.55, 1, 4],
                     [1519430080, 1519430090, 44.44, 55.55, 3, 2],
                     [1519430081, 1519430030, 44.44, 55.55, 2, 6]]
        target_list = [['2018-02-22 20:00:00', 44.44, 55.55, 3, 7, 4],
                       ['2018-02-23 23:45:00', 44.44, 55.55, 5, 8, 3]]
        data = pd.DataFrame(data_list, columns=['timestamp1', 'session_start2', 'lat', 'lon', 'occupancy1', 'occupancy2'])
        target = pd.DataFrame(target_list, columns=['time', 'lat', 'lon', 'occupancy1', 'occupancy2', 'difference'])
        target['time'] =  pd.to_datetime(target['time'])
        target_multi = target.set_index(['time', 'lat', 'lon'])

        test = lstm_preprocessing.time_grouping(data, interval='15T', time_selection='2')

        assert test.equals(target_multi)

    def test_session1_timestamp2_interval60_selection2(self):
        data_list = [[1519330080, 1519330090, 44.44, 55.55, 2, 3],
                     [1519330081, 1519330030, 44.44, 55.55, 1, 4],
                     [1519430080, 1519430090, 44.44, 55.55, 3, 2],
                     [1519430081, 1519430030, 44.44, 55.55, 2, 6]]
        target_list = [['2018-02-22 20:00:00', 44.44, 55.55, 3, 7, 4],
                       ['2018-02-23 23:00:00', 44.44, 55.55, 5, 8, 3]]
        data = pd.DataFrame(data_list, columns=['session_start1', 'timestamp2', 'lat', 'lon', 'occupancy1', 'occupancy2'])
        target = pd.DataFrame(target_list, columns=['time', 'lat', 'lon', 'occupancy1', 'occupancy2', 'difference'])
        target['time'] =  pd.to_datetime(target['time'])
        target_multi = target.set_index(['time', 'lat', 'lon'])

        test = lstm_preprocessing.time_grouping(data, interval='60T', time_selection='2')

        assert test.equals(target_multi)

    def test_session1_timestamp2_interval60_selection1(self):
        data_list = [[1519330080, 1519330090, 44.44, 55.55, 2, 3],
                     [1519330081, 1519330030, 44.44, 55.55, 1, 4],
                     [1519430080, 1519430090, 44.44, 55.55, 3, 2],
                     [1519430081, 1519430030, 44.44, 55.55, 2, 6]]
        target_list = [['2018-02-22 20:00:00', 44.44, 55.55, 3, 7, 4],
                       ['2018-02-23 23:00:00', 44.44, 55.55, 5, 8, 3]]
        data = pd.DataFrame(data_list, columns=['session_start1', 'timestamp2', 'lat', 'lon', 'occupancy1', 'occupancy2'])
        target = pd.DataFrame(target_list, columns=['time', 'lat', 'lon', 'occupancy1', 'occupancy2', 'difference'])
        target['time'] =  pd.to_datetime(target['time'])
        target_multi = target.set_index(['time', 'lat', 'lon'])

        test = lstm_preprocessing.time_grouping(data, interval='60T', time_selection='1')

        assert test.equals(target_multi)

    def test_session1_timestamp2_interval30_selectionavg(self):
        data_list = [[1519330080, 1519330090, 44.44, 55.55, 2, 3],
                     [1519330081, 1519330030, 44.44, 55.55, 1, 4],
                     [1519430080, 1519430090, 44.44, 55.55, 3, 2],
                     [1519430081, 1519430030, 44.44, 55.55, 2, 6]]
        target_list = [['2018-02-22 20:00:00', 44.44, 55.55, 3, 7, 4],
                       ['2018-02-23 23:30:00', 44.44, 55.55, 5, 8, 3]]
        data = pd.DataFrame(data_list, columns=['session_start1', 'timestamp2', 'lat', 'lon', 'occupancy1', 'occupancy2'])
        target = pd.DataFrame(target_list, columns=['time', 'lat', 'lon', 'occupancy1', 'occupancy2', 'difference'])
        target['time'] =  pd.to_datetime(target['time'])
        target_multi = target.set_index(['time', 'lat', 'lon'])

        test = lstm_preprocessing.time_grouping(data, interval='30T', time_selection='avg')

        assert test.equals(target_multi)

    def test_timestamp1_session2_interval30_selectionavg(self):
        data_list = [[1519330080, 1519330090, 44.44, 55.55, 2, 3],
                     [1519330081, 1519330030, 44.44, 55.55, 1, 4],
                     [1519430080, 1519430090, 44.44, 55.55, 3, 2],
                     [1519430081, 1519430030, 44.44, 55.55, 2, 6]]
        target_list = [['2018-02-22 20:00:00', 44.44, 55.55, 3, 7, 4],
                       ['2018-02-23 23:30:00', 44.44, 55.55, 5, 8, 3]]
        data = pd.DataFrame(data_list, columns=['timestamp1', 'session_start2', 'lat', 'lon', 'occupancy1', 'occupancy2'])
        target = pd.DataFrame(target_list, columns=['time', 'lat', 'lon', 'occupancy1', 'occupancy2', 'difference'])
        target['time'] =  pd.to_datetime(target['time'])
        target_multi = target.set_index(['time', 'lat', 'lon'])

        test = lstm_preprocessing.time_grouping(data, interval='30T', time_selection='avg')

        assert test.equals(target_multi)

    def test_invalid_time_selection(self):
        data_list = [[1519330080, 1519330090, 44.44, 55.55, 2, 3],
             [1519330081, 1519330030, 44.44, 55.55, 1, 4],
             [1519430080, 1519430090, 44.44, 55.55, 3, 2],
             [1519430081, 1519430030, 44.44, 55.55, 2, 6]]
        data = pd.DataFrame(data_list, columns=['timestamp1', 'session_start2', 'lat', 'lon', 'occupancy1', 'occupancy2'])

        with pytest.raises(Exception):
            lstm_preprocessing.time_grouping(data, interval='30T', time_selection='3')

class TestWeeklyDifferenceDataframes:
    def test_two_weeks(self):
        data_list = [['2018-02-22 20:00:00', 44.44, 55.55, 3, 7, 4],
                     ['2018-02-23 23:30:00', 44.44, 55.55, 5, 8, 3],
                     ['2018-02-24 00:00:00', 44.44, 55.55, 4, 3, 1],
                     ['2018-02-24 00:30:00', 44.44, 55.55, 2, 3, 1],
                     ['2018-02-24 01:00:00', 66.66, 77.77, 5, 5, 0],
                     ['2018-02-24 01:30:00', 66.66, 77.77, 3, 3, 0],
                     ['2018-02-24 02:00:00', 66.66, 77.77, 7, 8, 1],
                     ['2018-02-24 02:30:00', 66.66, 77.77, 3, 5, 2],
                     ['2018-02-24 03:00:00', 66.66, 77.77, 4, 5, 1],
                     ['2018-03-24 03:30:00', 44.44, 55.55, 7, 8, 1],
                     ['2018-03-24 04:00:00', 44.44, 55.55, 6, 5, 1],
                     ['2018-03-24 04:30:00', 44.44, 55.55, 9, 8, 1],
                     ['2018-03-24 05:00:00', 44.44, 55.55, 2, 2, 0],
                     ['2018-03-24 05:30:00', 44.44, 55.55, 8, 8, 0],
                     ['2018-03-24 06:00:00', 44.44, 55.55, 6, 5, 1],
                     ['2018-03-24 06:30:00', 44.44, 55.55, 7, 8, 1],
                     ['2018-03-24 07:00:00', 44.44, 55.55, 2, 4, 2],
                     ['2018-03-24 07:30:00', 44.44, 55.55, 5, 4, 1],]
        data = pd.DataFrame(data_list, columns=['time', 'lat', 'lon', 'occupancy1', 'occupancy2', 'difference'])
        data['time'] =  pd.to_datetime(data['time'])
        data_multi = data.set_index(['time', 'lat', 'lon'])

        test = lstm_preprocessing.weekly_difference_dataframes(data_multi)

        assert test != None # need a better assertion, but can't find how to hash a dictionary of dataframes

