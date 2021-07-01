import pandas as pd

usecols = ["ACCOUNT_NUM", 'street_num', 'full_street_name', 'property_city', 'property_zipcode']

chunksize = 10000   #C ensus bureau geocoder can do 10k addresses at a time
chunk_idx = 0
for chunk in pd.read_csv('2.cleaned_data/prop_tax_data.csv', 
                                index_col="ACCOUNT_NUM", usecols=usecols, chunksize=chunksize):
    
    chunk['num_and_street'] = chunk['street_num'].astype(str) + ' ' + chunk['full_street_name']
    chunk['state'] = 'TX'
    chunk.dropna(axis=0, subset=['property_zipcode'], inplace=True)
    chunk['property_zipcode'] = chunk['property_zipcode'].astype('int32')
    chunk.drop(columns=['street_num', 'full_street_name'], inplace=True)

    # Make sure columns are in the proper order for census geocoder
    chunk = chunk [['num_and_street', 'property_city', 'state', 'property_zipcode']]

    print(chunk.head())
    fname = f'./to_geocode/chunk_{chunk_idx}.csv'
    chunk.to_csv(fname)

    chunk_idx +=1