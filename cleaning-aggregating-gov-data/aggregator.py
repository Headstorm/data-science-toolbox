import pandas as pd

prop_tax_data = pd.read_csv('./2.cleaned_data/prop_tax_data.csv')
prop_tax_data.dropna(axis='rows', subset=['property_zipcode'], inplace=True)

print(prop_tax_data.head(20))
print(prop_tax_data.info())

prop_tax_data = prop_tax_data[['property_zipcode', 'tot_val']]
prop_tax_data = prop_tax_data.groupby(['property_zipcode']).mean()

print(prop_tax_data.head(20))
print(prop_tax_data.info())

prop_tax_data.to_csv('./3.aggregated_data/by_zip.csv')