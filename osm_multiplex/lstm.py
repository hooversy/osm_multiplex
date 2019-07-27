"""
.. moduleauthor:: Sylvan Hoover <hooversy@oregonstate.edu>

based on: https://github.com/chen0040/keras-anomaly-detection/blob/master/keras_anomaly_detection/library/recurrent.py
"""

# standard libraries
import os
import sys

# third-party libraries

from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import csv
import pickle

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
        model.add(LSTM(units=256, input_shape=(time_window_size, 3), return_sequences=True))
        model.add(LSTM(units=256, return_sequences=True))
        model.add(LSTM(units=256, return_sequences=False))
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
            std_dev_threshold = 1.0

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

        scores, _ = self.predict(timeseries_dataset)
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
        return dist, target_timeseries_dataset

    def anomaly(self, timeseries_dataset, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist, predicted_timeseries_dataset = self.predict(timeseries_dataset)
        return zip(dist >= self.threshold, dist), predicted_timeseries_dataset

class anomalydetect(object):

    def __init__(self):
        self.ae = LstmAutoEncoder()
        self.model_dir_path = os.path.join(THIS_DIR, './models')

    def construct_npdata(self, dataframe, testing=False):
        np_data_o1 = dataframe[['occupancy1']].values
        if testing == True:
            np_data_o2 = np.concatenate((dataframe[['occupancy2']].values[:-5]*0.5, dataframe[['occupancy2']].values[-5:]*0.1))
        else:
            np_data_o2 = dataframe[['occupancy2']].values
        np_data_diff = np.abs(np_data_o1 - np_data_o2)
        
        scaler = MinMaxScaler()
        for table in [np_data_o1, np_data_o2, np_data_diff]:
            table = scaler.fit_transform(table)
        np_data = np.stack((np_data_o1, np_data_o2, np_data_diff), axis=-1)

        return np_data

    def fit_detect(self, np_data):
        # fit the data and save model into model_dir_path
        self.ae.fit(np_data[:, :, :], model_dir_path=self.model_dir_path)

        # load back the model saved in model_dir_path detect anomaly
        self.ae.load_model(self.model_dir_path)
        anomaly_information, predicted_timeseries_dataset = self.ae.anomaly(np_data[:, :, :])

        return anomaly_information, predicted_timeseries_dataset

    def week_anomaly_detect(self, data, testing=False):
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
        collected_target = {}
        predicted_target = {}

        pickle.dump(data, open('./osm_multiplex/data/lstm_input_data.pickle', 'wb'))

        for location, dataframe in data.items():
            # samples = len(dataframe.index.codes[0])
            # timesteps = len(dataframe.columns.levels[1])

            np_data = self.construct_npdata(dataframe.dropna(), testing)
            
            print('For ' + str(location) + ', ' + str(np_data.shape[0]) + ' weeks processing') 

            anomaly_information, predicted_timeseries_dataset = self.fit_detect(np_data)

            reconstruction_error = {}
            if self.ae.threshold == 0.0:
                continue
            else:
                reconstruction_error['threshold'] = self.ae.threshold
            years = dataframe.index.get_level_values(0).tolist()
            weeks = dataframe.index.get_level_values(1).tolist()
            for idx, (is_anomaly, dist) in enumerate(anomaly_information):
                reconstruction_error[str(years[idx]) + ', ' +str(weeks[idx])] = [is_anomaly, dist]
                if is_anomaly == True:
                    print(location + ' year ' + str(years[idx]) + ', week ' + str(weeks[idx]) + ' is anomalous')
            reconstruction_dict[location] = reconstruction_error
            collected_target[location] = np_data[:,:,2]
            predicted_target[location] = predicted_timeseries_dataset

        pickle.dump(reconstruction_dict, open('./osm_multiplex/data/lstm_dist.pickle', 'wb'))
        pickle.dump(collected_target, open('./osm_multiplex/data/lstm_collected_target.pickle', 'wb'))
        pickle.dump(predicted_target, open('./osm_multiplex/data/lstm_predicted_target.pickle', 'wb'))
        return reconstruction_dict

    # def rolling_anomaly_detect(self, data):
    #     """Takes a dictionary of location, data pairs containing a series of temporal sample and assesses likely
    #     anomalous samples. The resulting dictionary can be used to identify locations and time periods of likely
    #     anomalous data collection.

    #     Parameters
    #     ----------
    #     data : dict
    #         Each key in the dictionary is a specific location with the value being a DataFrame of two time-series
    #         representing collected data from each source


    #     Returns
    #     -------
    #     reconstruction_dict : dict
    #         Each key is the location and error threshold with the value being a list of the reconstruction error 
    #         for each sample in the location
    #     """
        
    #     reconstruction_dict = {}
    #     for location, dataframe in data.items():
    #         # samples = len(dataframe.index.codes[0])
    #         # timesteps = len(dataframe.columns.levels[1])

    #         np_data = self.construct_npdata(dataframe)
            
    #         print('For ' + str(location) + ', ' + str(np_data.shape[0]) + ' steps processing') 

    #         anomaly_information = self.fit_detect(np_data)

    #         reconstruction_error = {}
    #         if self.ae.threshold == 0.0:
    #             continue
    #         else:
    #             reconstruction_error['threshold'] = self.ae.threshold
    #         time = dataframe.index.tolist()
    #         for idx, (is_anomaly, dist) in enumerate(anomaly_information):
    #             reconstruction_error[str(time[idx])] = [is_anomaly, dist]
    #             if is_anomaly == True:
    #                 print(location + ' at ' + str(time[idx]) + ' is anomalous')
    #         reconstruction_dict[location] = reconstruction_error

    #     return reconstruction_dict

class datasamples(object):
    
    def create_loc_dic(self, locations):
        with open(locations, mode='r') as f:
            reader = csv.reader(f)
            loc_dic = {(float(row[1]), float(row[2])):row[0] for row in reader}
        return loc_dic

    def group_dataframes(self, dataframe, locations):
        grouped_dataframes = {}

        if locations is not None:
            location_dict = self.create_loc_dic(locations)
        
        for name, group in dataframe.groupby(['lat', 'lon']):
            try:
                grouped_dataframes['location ' + str(location_dict[name])] = group.reset_index(level=['lat', 'lon']).drop(columns=['lat', 'lon'])
            except:
                grouped_dataframes['location ' + str(name)] = group.reset_index(level=['lat', 'lon']).drop(columns=['lat', 'lon'])

        return grouped_dataframes

    def gaps_filler(self, dataframe, interval, method=None, fill_value=None):
        gaps_filled = dataframe[['occupancy1', 'occupancy2']].asfreq(freq=interval, method=method, fill_value=fill_value).reset_index(drop=False)

        return gaps_filled

    def weekly_sample(self, dataframe, interval='60T', locations=None):
        """Generates a dictionary of dataframes with each k,v pair representing a location and the difference between the two
        datasource counts.

        Parameters
        ----------
        dataframe : pandas DataFrame
            Contains count and difference values for all locations

        interval : str
            The time interval to be represented in the resulting dataframe. The default in 15 minutes, which results
            in 672 entries for every week

        Returns
        -------
        dataframes : dict
            A dictionary of DataFrames with each k,v pair representing a location and the difference between the two
            datasource counts.
        """
        
        pivoted_dataframes = {}
        
        grouped_dataframes = self.group_dataframes(dataframe, locations)
        
        for location, counts in grouped_dataframes.items():
            gaps_filled = self.gaps_filler(counts, interval)
            pivoted = pd.pivot_table(gaps_filled,
                                    index=[gaps_filled['time'].dt.year, gaps_filled['time'].dt.week],
                                    columns=gaps_filled.groupby(pd.Grouper(key='time', freq='W')).cumcount().add(1),
                                    values=['occupancy1', 'occupancy2'],
                                    aggfunc='sum')
            pivoted.index.names = ['year', 'week']
            useful_data = pivoted.iloc[1:-1].dropna() # removes likely incomplete first and last weeks
            if not (useful_data.empty or useful_data.shape[0]<5):
                pivoted_dataframes[location] = useful_data

        return pivoted_dataframes

    # def rolling_sample(self, dataframe, interval='60T', length=672, locations=None):
    #     """Generates a dictionary
    #     """
        
    #     pivoted_dataframes = {}
        
    #     grouped_dataframes = self.group_dataframes(dataframe, locations)

    #     for location, counts in grouped_dataframes.items():
    #         gaps_filled = self.gaps_filler(counts, interval, fill_value=0)

    #         pivoted1 = pd.DataFrame(index=gaps_filled['time'], columns=range(length))
    #         pivoted2 = pd.DataFrame(index=gaps_filled['time'], columns=range(length))
    #         pivoted1.columns = pd.MultiIndex.from_product([['occupancy1'], pivoted1.columns])
    #         pivoted2.columns = pd.MultiIndex.from_product([['occupancy2'], pivoted2.columns])
    #         pivoted = pd.concat([pivoted1, pivoted2], axis = 1)
    #         # inefficient looping; need to find better approach
    #         index_length = len(pivoted.index)
    #         for time_index in range(index_length):
    #             if time_index % 50 == 0:
    #                 print("Processing row " + str(time_index) + " of " + str(index_length))
    #             pivot1 = gaps_filled.occupancy1.iloc[time_index:time_index+length].tolist()
    #             pivot2 = gaps_filled.occupancy2.iloc[time_index:time_index+length].tolist()
    #             if len(pivot1) == length and len(pivot2) == length:
    #                 pivoted.occupancy1.iloc[time_index] = pivot1
    #                 pivoted.occupancy2.iloc[time_index] = pivot2
    #             else:
    #                 break
    #         useful_data = pivoted.dropna()[:-1]
    #         if not (useful_data.empty or useful_data.shape[0]<5):
    #             pivoted_dataframes[location] = useful_data

    #     return pivoted_dataframes


def anomaly_detect(preprocessed_dataframe, detection_type="weekly", locations=None, testing=False):
    sampler = datasamples()
    ad = anomalydetect()

    if detection_type == "weekly":
        sampled = sampler.weekly_sample(preprocessed_dataframe, locations=locations)
        anomaly_detection = ad.week_anomaly_detect(sampled, testing=testing)
    # elif detection_type == "rolling":
    #     sampled = sampler.rolling_sample(preprocessed_dataframe, locations=locations)
    #     anomaly_detection = ad.rolling_anomaly_detect(sampled)

    return anomaly_detection