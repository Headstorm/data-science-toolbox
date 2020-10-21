from os import listdir
from os.path import isfile, join
import os
import glob
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
from os import listdir
import numpy as np
from numpy import array
from datetime import datetime, date, time, timedelta
import multiprocessing
# Import models
# grid search holt winter's exponential smoothing
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose

from tensorflow.python.client import device_lib
import tensorflow.compat.v1.keras as tf
import sys
from keras import layers
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.layers import Masking
import threading
from sklearn.utils import class_weight
import tensorflow

from numba import cuda
from keras.models import load_model
import h5py

# sess = tensorflow.Session(config=tensorflow.ConfigProto(log_device_placement=True))

from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Masking, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.framework import ops
from keras.optimizers import RMSprop, Adam
from keras.layers import Masking
from sklearn.utils import class_weight

from sklearn.preprocessing import MinMaxScaler

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import gc

import ast

# config = tensorflow.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tensorflow.compat.v1.Session(config=config)


EXCLUDE_WEATHER_COLUMN = ['lat', 'lon', 'wban_x', 'call', 'elev', 'begin', 'end', 'usaf', 'wban_y', 'name',
'count_temp', 'count_dewp', 'count_slp', 'count_stp', 'count_visib', 'count_wdsp', 'flag_max', 'flag_min', 'flag_prcp', 'fog',
'rain_drizzle','snow_ice_pellets', 'hail','thunder','tornado_funnel_cloud', 'sndp', 'stp', 'visib', 'wdsp', 'slp', 'min', 'max', 'mxpsd',
'year', 'mo', 'da', 'stn', 'country']

import warnings
warnings.filterwarnings("ignore")


# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def root_mean_squared_error(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def generate_model(n_steps, n_features, data):
	RNN = tf.layers.LSTM
	# RNN = tf.layers.CuDNNLSTM
	model = Sequential()
	# model.add(RNN(50,  batch_input_shape=(training_data.shape[0], training_data.shape[1], training_data.shape[2])))
	model.add(RNN(50,  batch_input_shape=(data.shape[0], data.shape[1], data.shape[2])))
	model.add(Dense(1, activation="tanh"))
	model.compile(optimizer='adam', loss='mse')
	return model

def limit_mem():
	K.get_session().close()
	gc.collect()
	config = tensorflow.compat.v1.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tensorflow.compat.v1.Session(config=config)

if __name__ == '__main__':

	### Read Station data and select only relevant stations

	print("----Load Station----")
	
	weather_station_list = pd.read_csv("data/weather_meta_data/ghcnd_stations.csv")[['id', 'state']]
	weather_station_list['state'] = weather_station_list['state'].str.strip()
	weather_station_list['state'] = weather_station_list['state'].replace('', 'UNK')
	print(weather_station_list)

	weather_country_list = pd.read_csv("data/weather_meta_data/ghcnd_countries.csv")
	print(weather_country_list)
	weather_country_list.rename(columns={"code": "country_region", "name" : "country_name"}, inplace=True)
	print(weather_country_list)

	weather_file_list = []
	li = []

	year_list = ['2017', '2018', '2019', '2020']

	for file in glob.glob("./data/weather_data/*.csv"):
		if any(ele in file for ele in year_list):
			weather_file_list.append(file)
		
	for file in tqdm(weather_file_list):
		df = pd.read_csv(file, index_col=None, header=0, low_memory=False)
		df = df[['id', 'date', 'element', 'value']]
		df['date'] = df['date'].astype(str)
		df['date'] = df['date'].str.replace('-', '')
		df['date'] = df['date'].astype(int)
		df = pd.pivot_table(df, values='value', index=['id', 'date'], columns='element')
		df = df[['TAVG']]
		df.reset_index(drop=False, inplace=True)
		df['country_region'] = df['id'].str[:2]
		li.append(df)

	weather_frame = pd.concat(li, axis=0, ignore_index=True)
	
	weather_frame = weather_frame.merge(weather_station_list, on=['id'], how='left')
	weather_frame = weather_frame[['date', 'country_region', 'state', 'TAVG']]
	print(weather_frame)
	print(weather_frame.columns)

	### Aggregate features

	weather_frame = weather_frame.groupby(['date', 'country_region', 'state']).mean().reset_index(drop=False)

	weather_frame = weather_frame.merge(weather_country_list, on=['country_region'])

	weather_frame = weather_frame[['date', 'country_name', 'state', 'TAVG']]
	weather_frame.rename(columns={"country_name": "country_region", "state" : "province_state"}, inplace=True)
	weather_frame['country_region'] = weather_frame['country_region'].str.strip()

	cut_small_location = pd.DataFrame()
	# remove all location that has small dataset
	list_country = set(weather_frame['country_region'])
	for country in list_country:
		list_state = set(weather_frame.loc[weather_frame['country_region']==country, 'province_state'])
		temp_df = weather_frame.loc[weather_frame['country_region']==country, :]
		for state in list_state:
			temp_df2 = temp_df.loc[temp_df['province_state']==state, :].sort_values(by=['date'], ascending=True).reset_index(drop=True)
			if len(temp_df2) < (365 * 2):
				continue
			if temp_df2['date'].iloc[-1] < 20200401:
				continue
			cut_small_location = cut_small_location.append(temp_df2)

	# weather_frame.to_csv('during_covid_weather.csv')
	cut_small_location.reset_index(drop=True, inplace=True)

	dataset = cut_small_location.copy()

	dataset['date'] = dataset['date'].astype(str)

	interpolate_data = pd.DataFrame()

	import math

	### Interpolate Missing Value (NaN)

	list_country = set(dataset['country_region'])

	state_abs_max_df = pd.DataFrame()

	for country in list_country:
		list_state = set(dataset.loc[dataset['country_region']==country, 'province_state'])
		temp_df = dataset.loc[dataset['country_region']==country, :]
		for state in list_state:
			temp_df2 = temp_df.loc[temp_df['province_state']==state, :].sort_values(by=['date'], ascending=True).reset_index(drop=True)
			interpolate_feature_list = ['TAVG']
			for f in interpolate_feature_list:
				if pd.isnull(temp_df2[f]).all():
					temp_df2[f].fillna(0, inplace=True)
				else:
					for i in range(len(temp_df2[f])):
						if i == 0 or i == len(temp_df2[f])-1:
							if math.isnan(temp_df2[f][i]):
								temp_df2[f].iloc[i] = 0
						else:
							previous_v = temp_df2[f].iloc[i-1]
							next_v = temp_df2[f].iloc[i+1]
							if math.isnan(next_v):
								temp_df2[f].iloc[i] = previous_v/2
							else:
								temp_df2[f].iloc[i] = (previous_v+next_v)/2
				abs_max_value = abs(temp_df2[f].max())
				state_abs_max_df = state_abs_max_df.append({'state': state, 'country': country, 'feature': f, 'value': abs_max_value}, ignore_index=True)
				temp_df2[f] = temp_df2[f]/abs_max_value
			
			interpolate_data = interpolate_data.append(temp_df2)

	### Model Training and Future Weather Prediction

	pred_actual = pd.DataFrame()
	future_pred_df = pd.DataFrame()
	rsme_score_df = pd.DataFrame()

	# Data Parameter
	n_steps = 365
	n_test = 30
	n_features = 1
	n_future_pred = 180

	# Train model for each state and feature

	list_country = set(interpolate_data['country_region'])

	for country in list_country:
		temp_df = interpolate_data.loc[interpolate_data['country_region']==country, :]
		list_state = set(interpolate_data.loc[interpolate_data['country_region']==country, 'province_state'])
		for state in list_state:
			temp_df2 = temp_df.loc[temp_df['province_state']==state, :].sort_values(by=['date'], ascending=True).reset_index(drop=True)
			location_df = temp_df2[temp_df2.columns[~temp_df2.columns.isin(['province_state', 'country_region'])]]
			location_df.dropna(axis='columns', inplace=True)
			location_df.set_index('date', inplace=True)

			print('Location : ', country, state)
			if len(location_df.columns) == 0:
				continue
			print(location_df.describe())

			# List the date that will be used for future prediction

			date_test_list = list(location_df.index.values)[-n_test:]
			first_date_for_future = datetime.strptime(date_test_list[-1], '%Y%m%d')
			date_future_list = [datetime.strftime(d, '%Y%m%d') for d in (first_date_for_future + timedelta(n) for n in range(n_future_pred))]

			# Open the log file for that state		

			# file = open('output/weather_log/'+country+"_"+state+"_log.txt","w")
			# file.write('Location : ' + country + ", " + state + "\n")
			# file.write(location_df.describe().to_string() + "\n")

			temp_pred_actual = pd.DataFrame()
			temp_future_pred = pd.DataFrame()
			temp_rsme_score = pd.DataFrame()

			for f in location_df.columns:
				# define dataset
				data = location_df[f]
				train_data = data[:-n_test]
				test_data = data[-n_test:]

				# split into train/test samples
				X, y = split_sequence(data, n_steps)
				train_x = X[:-n_test]
				train_y = y[:-n_test]
				test_x = X[-n_test:]
				test_y = y[-n_test:]

				train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], n_features))
				test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], n_features))

				# print(train_x.shape)
				# print(test_x.shape)

				# create weather prediction model
				model = generate_model(n_steps, n_features, train_x)
				earlystopping = EarlyStopping(monitor='loss', patience=25, mode='min', restore_best_weights=True)
				callbacks = [earlystopping]

				# train the model
				history = model.fit(train_x, train_y, epochs=500, batch_size=train_x.shape[0], verbose=2, callbacks=callbacks)

				train_weights = model.get_weights()

				# save model
				# model.save('output/weather_model/'+country+"_"+state+"_"+f+"_model.h5")

				# limit_mem()

				# # load model
				# model = tf.models.load_model('output/weather_model/'+country+"_"+state+"_"+f+"_model.h5")
				train_weights = model.get_weights()

				test_model = generate_model(n_steps, n_features, test_x)
				test_model.set_weights(train_weights)

				
				# history = model.fit(train_x, train_y, epochs=100, verbose=2, callbacks=callbacks)

				# predict the test dataset
				pred_y = test_model.predict(test_x, verbose=0)

				max_abs = state_abs_max_df.loc[(state_abs_max_df['state']==state)&(state_abs_max_df['country']==country)&(state_abs_max_df['feature']==f), 'value'].iloc[0]

				print(max_abs)

				temp_pred_actual[f+'_actual'] = [p * max_abs for p in test_y]
				temp_pred_actual[f+'_pred'] = [pred[0] * max_abs for pred in pred_y] 

				# calculate RMSE score for that feature of the state and record to log file
				rmse_score = measure_rmse(test_y, [pred[0] for pred in pred_y])
				# file.write("RMSE of " + f + " : " + str(rmse_score) + "\n")
				temp_rsme_score = temp_rsme_score.append({'state': state, 'country': country, f+'_rmse': rmse_score}, ignore_index=True)

				# plot the actual vs prediction graph of the feature of that state
				# pyplot.title("Prediction/Actual of "+f+" in "+country+", "+state)
				# pyplot.plot([p * max_abs for p in test_y], color='blue', label="prediction" )
				# pyplot.plot([pred[0] * max_abs for pred in pred_y], color='red', label="prediction")
				# pyplot.savefig(os.path.join('output/weather_plot/'+country+"_"+state+"_"+f+'_plot.png'))
				# pyplot.clf()

				# Predict the next n days
				# print("---Prediction for Future Date---")

				future_x = X[-1:][0]
				future_pred = list()

				x_input = array(future_x)
				x_input = x_input.reshape((1, n_steps, n_features))

				# limit_mem()

				# # load model
				# model = tf.models.load_model('weather_model/'+country+"_"+state+"_"+f+"_model.h5")
				train_weights = model.get_weights()

				future_model = generate_model(n_steps, n_features, x_input)
				future_model.set_weights(train_weights)

				for i in range(n_future_pred):
					y = future_model.predict(x_input, verbose=0)
					future_pred.append(y[0][0])
					future_x = np.append(future_x[1:], [y])
					x_input = array(future_x)
					x_input = x_input.reshape((1, n_steps, n_features))

				temp_future_pred[f+'_pred'] = [p * max_abs for p in future_pred]

				# print(temp_future_pred)
				# # reset graph for the next model (to avoid out of memory)
				# limit_mem()
				
				model.reset_states()
				ops.reset_default_graph()

				gc.collect()
				

			# record all result to files

			temp_pred_actual['state'] = state
			temp_future_pred['state'] = state

			temp_pred_actual['date'] = date_test_list
			temp_future_pred['date'] = date_future_list

			temp_pred_actual['country'] = country
			temp_future_pred['country'] = country

			temp_future_pred['date_idx'] = range(n_future_pred)

			# print(temp_future_pred)

			pred_actual = pred_actual.append(temp_pred_actual)
			future_pred_df = future_pred_df.append(temp_future_pred)
			rsme_score_df = rsme_score_df.append(temp_rsme_score)

			# print(future_pred_df)

			pred_actual.to_csv('output/weather_output/'+'pred_actual.csv', index=True)
			future_pred_df.to_csv('output/weather_output/'+'future_pred.csv', index=True)
			rsme_score_df.to_csv('output/weather_model/'+'rsme_score.csv', index=True)

		print(pred_actual)
		print(future_pred_df)

		# file.close()

		pred_actual.to_csv('output/weather_output/'+'pred_actual.csv', index=True)
		future_pred_df.to_csv('output/weather_output/'+'future_pred.csv', index=True)
		rsme_score_df.to_csv('output/weather_model/'+'rsme_score.csv', index=True)