# third-party libraries
import pandas as pd
import numpy as np

# local imports
from .. import lstm

class TestAnomalyDetect:
    def test_anomaly_detect(self):
        dataframes = {}
        locations = ['location(11.11, 22.22)', 'location(33.33, 44.44)']
        for location in locations:
            data = pd.DataFrame(np.random.randint(0,10,size=(10, 672)), columns=range(0, 672))
            data['year'] = 2018
            data['week'] = range(10)
            data_multi = data.set_index(['year', 'week'])
            dataframes[location] = data_multi

        test = lstm.anomaly_detect(dataframes)

        assert len(test) == 2