# Header

The purpose of this project is to demonstrate some of the data cleaning and aggregating capability of the Python pandas library

## How to Run
1. Run the shell script `startup.sh`. This will unzip the raw input data and create the necessary directory structure.
2. Run `cleaner.py`. This script takes in the raw property tax information, extracts the specific columns we're interested in, and saves the result in the folder `2.cleaned_data`
3. Run `aggregator.py` This script takes the result of the previous step and aggregates the property tax data by zip code, calculating the average home value for each zip code. The result is saved in the folder `3.aggregated data`
