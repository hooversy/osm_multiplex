"""
.. moduleauthor:: Sylvan Hoover <hooversy@oregonstate.edu>

based on: https://github.com/chen0040/keras-anomaly-detection/blob/master/keras_anomaly_detection/library/recurrent.py
"""

# standard libraries
import os

# third-party libraries

from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# modified LstmAutoEncoder class from reference for comparative time-series of mobility data
class LstmAutoEncoder(object):
    model_name = 'lstm-auto-encoder'
    VERBOSE = 0

    def __init__(self):
        self.model = None
        self.time_window_size = None
        self.config = None
        self.metric = None
        self.threshold = None

    @staticmethod
    def create_model(time_window_size, metric):
        model = Sequential()
        model.add(LSTM(units=128, input_shape=(time_window_size, 3), return_sequences=False))

        model.add(Dense(units=time_window_size, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
        #print(model.summary())
        return model

    def load_model(self, model_dir_path):
        config_file_path = LstmAutoEncoder.get_config_file(model_dir_path)
        self.config = np.load(config_file_path, allow_pickle=True).item()
        self.metric = self.config['metric']
        self.time_window_size = self.config['time_window_size']
        self.threshold = self.config['threshold']
        self.model = LstmAutoEncoder.create_model(self.time_window_size, self.metric)
        weight_file_path = LstmAutoEncoder.get_weight_file(model_dir_path)
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder.model_name + '-config.npy'

    @staticmethod
    def get_weight_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder.model_name + '-architecture.json'

    def fit(self, timeseries_dataset, model_dir_path, batch_size=None, epochs=None, validation_split=None, metric=None,
            std_dev_threshold=None):
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 20
        if validation_split is None:
            validation_split = 0.2
        if metric is None:
            metric = 'mean_absolute_error'
        if std_dev_threshold is None:
            std_dev_threshold = 1.5

        self.metric = metric
        self.time_window_size = timeseries_dataset.shape[1]

        weight_file_path = LstmAutoEncoder.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = LstmAutoEncoder.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = LstmAutoEncoder.create_model(self.time_window_size, metric=self.metric)
        open(architecture_file_path, 'w').write(self.model.to_json())
        self.model.fit(x=timeseries_dataset, y=timeseries_dataset[:,:,2],
                       batch_size=batch_size, epochs=epochs,
                       verbose=LstmAutoEncoder.VERBOSE, validation_split=validation_split,
                       callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

        scores = self.predict(timeseries_dataset)
        self.threshold = np.mean(scores) + std_dev_threshold * np.std(scores)

        #print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold
        config_file_path = LstmAutoEncoder.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

    def predict(self, timeseries_dataset):
        target_timeseries_dataset = self.model.predict(x=timeseries_dataset)
        dist = np.linalg.norm(timeseries_dataset[:,:,2] - target_timeseries_dataset, axis=-1)
        return dist

    def anomaly(self, timeseries_dataset, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(timeseries_dataset)
        return zip(dist >= self.threshold, dist)

def anomaly_detect(data):
    """Takes a dictionary of location, data pairs containing a series of temporal sample and assesses likely
    anomalous samples. The resulting dictionary can be used to identify locations and time periods of likely
    anomalous data collection.

    Parameters
    ----------
    data : dict
        Each key in the dictionary is a specific location with the value being a DataFrame of two time-series
        representing collected data from each source


    Returns
    -------
    reconstruction_dict : dict
        Each key is the location and error threshold with the value being a list of the reconstruction error 
        for each sample in the location
    """
    reconstruction_dict = {}
    for location, dataframe in data.items():
        model_dir_path = os.path.join(THIS_DIR, './models')
        print(location)
        samples = len(dataframe.index.codes[0])
        timesteps = len(dataframe.columns.levels[1])
        np_data_o1 = dataframe[['occupancy1']].values
        np_data_o2 = dataframe[['occupancy2']].values
        np_data_diff = np.abs(np_data_o1 - np_data_o2)
        scaler = MinMaxScaler()
        np_data_o1 = scaler.fit_transform(np_data_o1)
        np_data_o2 = scaler.fit_transform(np_data_o2)
        np_data_diff = scaler.fit_transform(np_data_diff)
        np_data = np.stack((np_data_o1, np_data_o2, np_data_diff), axis=-1)
        print(str(np_data.shape[0]) + ' weeks processing') 

        ae = LstmAutoEncoder()

        # fit the data and save model into model_dir_path
        ae.fit(np_data[:, :, :], model_dir_path=model_dir_path, std_dev_threshold=1.5)

        # load back the model saved in model_dir_path detect anomaly
        ae.load_model(model_dir_path)
        anomaly_information = ae.anomaly(np_data[:, :, :])
        reconstruction_error = {}
        if ae.threshold == 0.0:
            continue
        else:
            reconstruction_error['threshold'] = ae.threshold
        years = list(dataframe.index.get_level_values(0))
        weeks = list(dataframe.index.get_level_values(1))
        for idx, (is_anomaly, dist) in enumerate(anomaly_information):
            reconstruction_error[str(years[idx]) + ', ' +str(weeks[idx])] = [is_anomaly, dist]
            if is_anomaly == True:
                print(location + ' year ' + str(years[idx]) + ', week ' + str(weeks[idx]) + ' is anomalous')
        reconstruction_dict[location] = reconstruction_error

    return reconstruction_dict
