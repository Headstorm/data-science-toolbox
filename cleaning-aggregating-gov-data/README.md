# Header

The purpose of this project is to demonstrate some of the data cleaning and aggregating capability of the Python pandas library

## Initial setup

To begin, run the shell script `startup.sh`. This will unzip the raw input data and create the necessary directory structure for this project. Next, install the required Python packages from the `requirements.txt` file by navigating to this directory and running the command `pip install -r requirements.txt`.

## Part 1: Finding the average home values of Dallas-area zip codes

The input data can be found in the directory `1.desired_subset_of_raw_data`. These files were downloaded from the Dallas Central Appraisal District's website and contain various pieces of information about properties in the Dallas area.

However, we are only interested in a few columns within those tables. Running the python script `cleaner.py` will take in that raw property tax information, extract the specific columns we're interested in, and save the result in the folder `2.cleaned_data`.

We now have a single table where each row represents a particular address. The next step is to aggregate the data by ZIP code and calculate the average assessed home value. The Python script `aggregator.py` does this and saves the result in the directory `3.aggregated data`.

## Part 2: Getting the geo-location of each address

A cool possible extension to this project would be to have a more granular view of property values: instead of aggregating by zip code, you could do so by small areas defined by lat-long coordinates. To do this, we need to know the lat-long coordinates of each address.

We can use the US Census Bureau's [geocoding tool](https://geocoding.geo.census.gov/) to accomplish this task. The first step is to use the Python script `format_for_geocoding.py` to transform the cleaned data in `2.cleaned_data`. It will reorder the columns and divide the addresses into chunks of 10,000 (the geocoding tool can only do 10,000 at a time), which are saved in the folder `to_geocode`.

Then, the shell script `geocode_requests.sh` will send the API requests necessary and handle the response. Be warned, each chunk takes about 20 minutes to do.

## Part 3: Tableau Visualization 

To see what the data speaks, we created a visualization using Tableau. By using the "Map" feature in Tableau, we were able to draw boundaries across each zip-code. For varying prices, different colors were choosen (which can be altered as preffered) and displayed on the map. 
