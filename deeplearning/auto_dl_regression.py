# use autokeras to find a model for the insurance dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from autokeras import StructuredDataRegressor
 

def auto_ml_regress(url,x_names,y_name,max_trials,test_size,X_new):
    # load dataset
    dataframe = read_csv(url)
    X = dataframe[x_names].values
    y = dataframe[y_name].values
    print(X.shape, y.shape)
    # separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # define the search
    search = StructuredDataRegressor(max_trials=max_trials, loss='mean_absolute_error')
    # perform the search
    search.fit(x=X_train, y=y_train, verbose=0)
    # evaluate the model
    mae, _ = search.evaluate(X_test, y_test, verbose=0)
    print('MAE: %.3f' % mae)
    # use the model to make a prediction
    #X_new = asarray([[108]]).astype('float32')
    yhat = search.predict(X_new)
    print('Predicted: %.3f' % yhat[0])
    # get the best performing model
    model = search.export_model()
    # summarize the loaded model
    model.summary()
    # save the best performing model to file
    model.save('reg_model_insurance.h5')

if __name__=="__main__":
    X_new = asarray([[108]]).astype('float32')  # provide rows to use to test data in prediction
    x_names = ['n_claims']   # name of independent variables
    url = './agg_sonar_data.csv'  # location of model input file
    
    y_name = "total_payment"  # name of dependent variables 
    max_trials = 20   # number of AutoKeras trials
    test_size = 0.33  # proportion of obs used for test data
    auto_ml_regress(url,x_names,y_name,max_trials,test_size,X_new)