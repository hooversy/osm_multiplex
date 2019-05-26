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