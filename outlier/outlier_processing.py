#===============================================
# Coded by Alphonse Okossi
#===============================================
import numpy as np
import pandas as pd   
import pandasql as psql
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
# the dataset used in this example can be found in confluence at: https://headstorm.atlassian.net/wiki/spaces/DATASIG/pages/1135509505/Datasets
def quantile_outlier(df,low_prob,high_prob,excludes):  
    data1 = df.copy()
    if not excludes:
        data = df[df[excludes]]
    else:
        data=df.copy()     
    for c in data.columns:      
            if (is_string_dtype(data[c])):
                print("categorical column: "+c)
                print(data[c].value_counts())
            elif (is_numeric_dtype(data[c])):
                print("numeric column: "+c)
                print(data[c].describe())
                z = (data[c]-np.mean(data[c]))/np.std(data[c])
                q_low = df[c].quantile(low_prob)
                q_hi  = df[c].quantile(high_prob)        
                # quantile outlier flagging
                data1[c+'_quantile_outlier_flag'] = 1
                dd = data[(data[c] < q_hi) & (data[c] > q_low)]
                data1.loc[(data[c] < q_hi) & (data[c] > q_low),c+'_quantile_outlier_flag'] = 0
    return data1

def flag_outliers(df,excludes):
    data1 = df.copy()
    if not excludes:
        data = df[df[excludes]]
    else:
        data=df.copy()   
    print(df.describe()) 
    for c in data.columns:
        if (is_string_dtype(data[c])):
            print("categorical column: "+c)
            print(data[c].value_counts())
        elif (is_numeric_dtype(data[c])):
            print("numeric column: "+c)
            print(data[c].describe())
            z = (data[c]-np.mean(data[c]))/np.std(data[c])
            data['z'] = z
            # six sigma outlier flagging
            data1[c+'outlier_flag'] = 0
            dd = data1[((data1[c]-np.mean(data1[c]))/np.std(data1[c]))< -3.0]
            if len(dd)>0:
                data1.loc[((data1[c]-np.mean(data1[c]))/np.std(data1[c]))< -3.0,c+'outlier_flag'] = 1
                print("lower outliers:"+c)
                print(dd)       
            dd = data1.loc[((data1[c]-np.mean(data1[c]))/np.std(data1[c]))>3.0]
            if len(dd)>0:
                data1.loc[((data1[c]-np.mean(data1[c]))/np.std(data1[c]))>3.0,c+'outlier_flag'] = 1
                print("upper outliers"+c)
                print(dd)
    return data1

if __name__=="__main__":

    # preparing the data
    path = "/Users/aokossi/Documents/datasets/Predictive-Maintenance-Modelling-Datasets-master"
    telemetry = "telemetry.csv"
    main_data = "maint.csv"
    machines = "machines.csv"
    failures = "failures.csv"
    errors = "errors.csv"

    telemetry_df = pd.read_csv(path+"/"+telemetry)
    main_data_df = pd.read_csv(path+"/"+main_data)
    machines_df = pd.read_csv(path+"/"+machines)
    failures_df = pd.read_csv(path+"/"+failures)
    errors_df = pd.read_csv(path+"/"+errors)

    print("telemetry data preview")
    print(telemetry_df.head())
    print("main data preview")
    print(main_data_df.head())
    print("machines data preview")
    print(machines_df.head())
    print("failures data preview")
    print(failures_df.head())
    print("errors data preview")
    print(errors_df.head())

    # build analytic dataset
    df = psql.sqldf("""
    select a1.datetime,a1.machineID,volt,rotate,pressure,vibration,
    comp,model,age,failure,errorID
    from telemetry_df a1 
    left join main_data_df a2 on a1.datetime=a2.datetime and a1.machineID=a2.machineID
    left join machines_df a3 on a1.machineID=a3.machineID
    left join failures_df a4 on a1.machineID=a4.machineID and a1.datetime=a4.datetime
    left join errors_df a5 on a1.machineID=a5.machineID and a1.datetime=a5.datetime
    """)
    print("duplicates check")

    print("Count with drop duplicates")
    print(df.drop_duplicates().count())
    print("Count without drop duplicates")
    print(df.count())

    # six sigma outlier detection method
    out_df = flag_outliers(df=df,excludes=['datetime'])
    for c in out_df.columns:
        if "outlier_flag" in c:
            if  np.max(out_df[c])==1 and np.min(out_df[c])==0:
                print("outlier frequency count for "+c)
                print(out_df[c].value_counts())

    # quantile outlier detection method
    out_df = quantile_outlier(df=df,low_prob=0.01,high_prob=0.99,excludes=['datetime'])
    for c in out_df.columns:
        if "quantile_outlier_flag" in c:
            if  np.max(out_df[c])==1 and np.min(out_df[c])==0:
                print("quantile-outlier frequency count for "+c)
                print(out_df[c].value_counts())

    
    print("telemetry data preview")
    print(telemetry_df.head())
    print("main data preview")
    print(main_data_df.head())
    print("machines data preview")
    print(machines_df.head())
    print("failures data preview")
    print(failures_df.head())
    print("errors data preview")
    print(errors_df.head())
