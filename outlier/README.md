// instructions about running the code and the dataset and other relevant information. I recommend adding a brief about what the goal of the file is and what is the expected output.

Goal: given a CSV file or any other source file, we want to flag numerical column values if they are outside of normal bounds or not. Two methods are implemented 

Input: create a data frame from text file reading or from whatever other source, then use the flag_outliers function or the quantile_outlier function to flag each numerical column.

Function: flag_outliers
parameters:
    - df: a dataframe having numerical and categorical columns
    - excludes: a list of column names to exclude
return: a pandas data frame with the flags for each column, flag=1 means outlier detected and 0 = no outlier

Example:

out_df = flag_outliers(df=df,excludes=['datetime'])
    for c in out_df.columns:
        if "outlier_flag" in c:
            if  np.max(out_df[c])==1 and np.min(out_df[c])==0:
                print("outlier frequency count for "+c)
                print(out_df[c].value_counts())

    # quantile outlier detection method


Function: quantile_outlier
parameters:
    - df: a dataframe having numerical and categorical columns
    - excludes: a list of column names to exclude
    - low_prob: probability for low outlier
    - high_prob: probability for high outlier
return: a pandas data frame with the flags for each column, flag=1 means outlier detected and 0 = no outlier

Example:
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

