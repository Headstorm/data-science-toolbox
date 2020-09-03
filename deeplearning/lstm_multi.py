#===================================================================
# Time series forecast uing Deep Learning - Reccurent Neural Network
# Any model with time dependent sequence can be modeled using this
#
#===================================================================
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from datetime import datetime
import pandas as pd
import numpy as np
import pandasql as psql
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def prep_dataset(path ,year ,month ,day ,hour ,
    x_names,
    y_name ,
    drops,max_lags):
    dataset = read_csv(path) # ,  parse_dates = [[year, month, day, hour]], index_col=0, date_parser=parse)
    dataset = dataset.reset_index()
    dd = dataset.copy()
    for c in dd.columns:
        if (c  in drops) :
            dataset.drop(c, axis=1, inplace=True)
    dataset = psql.sqldf("""
    select a1.* from dataset a1 order by a1.{year},a1.{month},a1.{day},a1.{hour}
    """.format(year=year,month=month,day=day,hour=hour)) 
    dataset[ y_name] = dataset[y_name].apply(lambda x:x if x>0 else 0.0)#fillna(0, inplace=True)
    # drop the first max_lags observations
    dataset = dataset[max_lags:]
    dataset.to_csv('./datasets/pollution.csv')
    return dataset



def main():
    
    final_x = [y_name]+num_features+cat_features
    n_features = len(num_features+cat_features)
    if data_is_hourly:
        n_train_hours = n_years_used*360 * 24  # 2 years hourly used - sample size
    else:  
        n_train_hours = 2*360 
    batch_size= int(n_train_hours/10) #72
    dataset = pd.read_csv(data_path+'/'+input_file_name)
    dataset[y_name].fillna(0, inplace=True)
    encoder = LabelEncoder()
    for c in cat_features:
        dataset[c] = encoder.fit_transform(dataset[c])

    values = dataset[final_x].values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)


    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_hours, 1)
    print(reframed.shape)
    
    # split into train and test sets
    values = reframed.values
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    n_obs = n_hours * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    # design network
    model = Sequential()
    model.add(LSTM(n_layers, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    
    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

if __name__=="__main__":

    #======= INPUT REQUIREMENTS
    # year
    # month
    # day
    # hour
    # <dependent_variable name>
    # <numerical feature1>
    # <numerical feature2>
    # <numerical featureN>
    # <caterical feature1>
    # <caterical feature2>
    # <caterical featureM>
    #
    # Additional requirements
    # - year, month, day, hour must be present in the data for hourly data,  
    #    if daily data set hour =0 for all rows
    # - time series data must have the same sequential interval of time
    # - there should not be any gap in the date sequence
    #==============================data information
    #final_x = ["pollution" ,"dew" ,"temp"  ,"press","wnd_dir", "wnd_spd" ,"snow",  "rain"]
    cat_features = ["wnd_dir"]
    num_features = ["dew" ,"temp","press","wnd_dir" ,"snow",  "rain"]
    y_name = "pollution"

    ################INPUT PARAMETERS=====================================
    data_is_hourly = True
    n_years_used = 2  # number of years to use in training
    #cat_indices = [4]   # position number of features (and label) in the dataset
    epochs  = 50   # number of time to scan data
    n_layers = 50   # number of Recurrent Neural network Layers
    max_lags = 1    # max number of lags
    pred_lead = 1   #number of prediction steps - in hour for hourly data

    n_hours = 3   # number of lag steps for each feature

    data_path = '/Users/aokossi/Documents/ds-projects/datasets'   # data folder
    input_file_name = 'pollution_input_data.csv'  # input file name - must abide with input requirements
    #==================================================

    main()