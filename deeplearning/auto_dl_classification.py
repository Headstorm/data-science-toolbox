# use autokeras to find a model for the sonar dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from autokeras import StructuredDataClassifier
import pandas as pd
import numpy as np

def auto_ml_classify(url,x_names,y_name,max_trials,test_size,X_new):
    # load dataset
    dataframe = read_csv(url)
    X = dataframe[x_names].values
    y = dataframe[y_name].values
    print(X.shape, y.shape)
    # basic data preparation
    X = X.astype('float32')
    y = LabelEncoder().fit_transform(y)
    # separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # define the search
    search = StructuredDataClassifier(max_trials=max_trials)
    # perform the search
    search.fit(x=X_train, y=y_train, verbose=0)
    # evaluate the model
    loss, acc = search.evaluate(X_test, y_test, verbose=0)
    print('Accuracy: %.3f' % acc)
    # use the model to make a prediction
    
    yhat = search.predict(X_new)
    print('Predicted: %.3f' % yhat[0])
    # get the best performing model
    model = search.export_model()
    # summarize the loaded model
    model.summary()
    # save the best performing model to file for next time you dont have to train
    model.save('model_sonar.h5')


if __name__=="__main__":
    # provide a test x data : can be one row or a set of rows, must match X data
    row = [0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032]
    x_names = [ 'var'+str(i) for i in range(1,len(row)+1)] # explicitly provide X column names - all must be numeric
    y_name = "y"  # provide name of the categorical class label
    url = './raw_sonar_data.csv'  # provide model data set 
    max_trials = 15    # maximum number of trials for AutoKeras
    test_size = 0.33    # size of test dataset
    

    #================do not change==============================
    X_new = asarray([row]).astype('float32')
    auto_ml_classify(url,x_names,y_name,max_trials,test_size,X_new)