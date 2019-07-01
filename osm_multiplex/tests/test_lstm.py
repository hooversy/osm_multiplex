# third-party libraries
import pandas as pd
import numpy as np

# local imports
from .. import lstm

class TestAnomalyDetect:
    def test_anomaly_detect(self):
        dataframes = {}
        for location in ['location(11.11, 22.22)', 'location(33.33, 44.44)']:
            data1 = pd.DataFrame(np.random.randint(0,10,size=(10, 672)), columns=range(0, 672))
            data1.columns = pd.MultiIndex.from_product([['occupancy1'], data1.columns])
            data2 = pd.DataFrame(np.random.randint(0,10,size=(10, 672)), columns=range(0, 672))
            data2.columns = pd.MultiIndex.from_product([['occupancy2'], data2.columns])
            data = pd.concat([data1, data2], axis=1)
            data['year'] = 2018
            data['week'] = range(10)
            data_multi = data.set_index(['year', 'week'])
            dataframes[location] = data_multi

        test = lstm.anomaly_detect(dataframes)

        assert len(test) == 2

class TestWeeklyDataframes:
    def test_weeks(self):
        data_list = [['2018-02-22 20:00:00', 44.44, 55.55, 3, 7],
                     ['2018-02-23 23:30:00', 44.44, 55.55, 5, 8],
                     ['2018-02-24 00:00:00', 44.44, 55.55, 4, 3],
                     ['2018-02-24 00:30:00', 44.44, 55.55, 2, 3],
                     ['2018-02-24 01:00:00', 66.66, 77.77, 5, 5],
                     ['2018-02-24 01:30:00', 66.66, 77.77, 3, 3],
                     ['2018-02-24 02:00:00', 66.66, 77.77, 7, 8],
                     ['2018-02-24 02:30:00', 66.66, 77.77, 3, 5],
                     ['2018-04-24 03:00:00', 66.66, 77.77, 4, 5],
                     ['2018-03-24 03:30:00', 44.44, 55.55, 7, 8],
                     ['2018-03-24 04:00:00', 44.44, 55.55, 6, 5],
                     ['2018-03-24 04:30:00', 44.44, 55.55, 9, 8],
                     ['2018-03-24 05:00:00', 44.44, 55.55, 2, 2],
                     ['2018-03-24 05:30:00', 44.44, 55.55, 8, 8],
                     ['2018-03-24 06:00:00', 44.44, 55.55, 6, 5],
                     ['2018-04-24 06:30:00', 44.44, 55.55, 7, 8]]
        data = pd.DataFrame(data_list, columns=['time', 'lat', 'lon', 'occupancy1', 'occupancy2'])
        data['time'] =  pd.to_datetime(data['time'])
        data_multi = data.set_index(['time', 'lat', 'lon'])

        sampler = lstm.datasamples()
        test_weekly = sampler.weekly_sample(data_multi)

        assert test_weekly != None # need a better assertion, but can't find how to hash a dictionary of dataframes

class TestRollingDataframes:
    def test_roll(self):
        data_list = [['2018-02-22 20:00:00', 44.44, 55.55, 3, 7],
                     ['2018-02-23 23:30:00', 44.44, 55.55, 5, 8],
                     ['2018-02-24 00:00:00', 44.44, 55.55, 4, 3],
                     ['2018-02-24 00:30:00', 44.44, 55.55, 2, 3],
                     ['2018-02-24 01:00:00', 66.66, 77.77, 5, 5],
                     ['2018-02-24 01:30:00', 66.66, 77.77, 3, 3],
                     ['2018-02-24 02:00:00', 66.66, 77.77, 7, 8],
                     ['2018-02-24 02:30:00', 66.66, 77.77, 3, 5],
                     ['2018-04-24 03:00:00', 66.66, 77.77, 4, 5],
                     ['2018-03-24 03:30:00', 44.44, 55.55, 7, 8],
                     ['2018-03-24 04:00:00', 44.44, 55.55, 6, 5],
                     ['2018-03-24 04:30:00', 44.44, 55.55, 9, 8],
                     ['2018-03-24 05:00:00', 44.44, 55.55, 2, 2],
                     ['2018-03-24 05:30:00', 44.44, 55.55, 8, 8],
                     ['2018-03-24 06:00:00', 44.44, 55.55, 6, 5],
                     ['2018-04-24 06:30:00', 44.44, 55.55, 7, 8]]
        data = pd.DataFrame(data_list, columns=['time', 'lat', 'lon', 'occupancy1', 'occupancy2'])
        data['time'] =  pd.to_datetime(data['time'])
        data_multi = data.set_index(['time', 'lat', 'lon'])

        sampler = lstm.datasamples()
        test_rolling = sampler.rolling_sample(data_multi)

        assert test_rolling != None # need a better assertion, but can't find how to hash a dictionary of dataframes