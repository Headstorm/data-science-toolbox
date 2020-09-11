
## Goal: 
#### Given a CSV file or any other source file, we want to create (1) classification model, (2) regression model and (3) the specific case of a time series model using deep learning machine learning approach using auto keras and keras.

### Input: create a data frame from text file reading or from whatever other source, then use the one of the 3 example provided here to create predictions.

### Classification example:
```python
# sample data row used to rest a prediction
row = [0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032]

x_names = [ 'var'+str(i) for i in range(1,len(row)+1)] # explicitly provide X column names - all must be numeric
y_name = "y"  # provide name of the categorical class label
url = './raw_sonar_data.csv'  # provide model data set 
max_trials = 15    # maximum number of trials for AutoKeras
test_size = 0.33    # size of test dataset
    

    #================do not change==============================
X_new = asarray([row]).astype('float32')
auto_ml_classify(url,x_names,y_name,max_trials,test_size,X_new)

```

### Regression example:
```python
X_new = asarray([[108]]).astype('float32')  # provide rows to use to test data in prediction
x_names = ['n_claims']   # name of independent variables
url = './agg_sonar_data.csv'  # location of model input file 
y_name = "total_payment"  # name of dependent variables 
max_trials = 20   # number of AutoKeras trials
test_size = 0.33  # proportion of obs used for test data
auto_ml_regress(url,x_names,y_name,max_trials,test_size,X_new)
```
 ### Time Series Regression example:

 
 ```python
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
    ```