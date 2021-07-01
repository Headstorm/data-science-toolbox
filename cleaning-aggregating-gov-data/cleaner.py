import pandas as pd

def get_raw_data(fname, cols_to_read, limit_rows=True, nrows=100):
    path = f"./1.desired_subset_of_raw_data/{fname}.csv"
    if limit_rows:
        df = pd.read_csv(path, nrows=nrows, index_col="ACCOUNT_NUM", usecols=cols_to_read)
    else:
        df = pd.read_csv(path, index_col="ACCOUNT_NUM", usecols=cols_to_read)

    return df

def clean_final_df(final):
    final = final.rename(columns=str.lower)

    final['bldg_class_cd'] = pd.to_numeric(final['bldg_class_cd'], errors='coerce')
    final.dropna(axis='rows', subset=['bldg_class_cd'], inplace=True)
    final = final.astype({
                            'land_val': 'int32',
                            'tot_val': 'int32',
                            'bldg_class_cd': 'int32',
                            'property_zipcode': 'string',
                        })
    
    get_first_five = lambda zip: str(zip[:5])
    first_five_of_zip = final.apply(lambda row: get_first_five(row['property_zipcode']) if type(row['property_zipcode']) == str else row['property_zipcode'], axis=1)
    final['property_zipcode'] = first_five_of_zip

    return final

def filter_non_residential(final):
    final = final [
                    ((final['bldg_class_cd'] >= 1) & (final['bldg_class_cd'] <= 16)) |
                    (final['bldg_class_cd'] == 18) |
                    (final['bldg_class_cd'] == 32) |
                    ((final['bldg_class_cd'] >= 23) & (final['bldg_class_cd'] <= 26))
                ]
    return final


account_apprl_year = get_raw_data("account_apprl_year", 
                                    ["ACCOUNT_NUM", "LAND_VAL", "TOT_VAL", "GIS_PARCEL_ID", "BLDG_CLASS_CD"],
                                    False)
account_info = get_raw_data("account_info", 
                                ["ACCOUNT_NUM", "STREET_NUM", "FULL_STREET_NAME", "BLDG_ID", "UNIT_ID", "PROPERTY_CITY", "PROPERTY_ZIPCODE", "GIS_PARCEL_ID"],
                                False)
res_detail = get_raw_data("res_detail", ["ACCOUNT_NUM", "TOT_LIVING_AREA_SF"], False)

final = account_apprl_year.join(account_info, how="inner", lsuffix="_apprl_year", rsuffix="_info")
final = final.join(res_detail, how="left")

print("\n\n\nInitial")
print(final.head(20))
print(final.info())

final = clean_final_df(final)

print("\n\n\nClean")
print(final.head(20))
print(final.info())

final = filter_non_residential(final)

print("\n\n\nFiltered")
print(final.head(20))
print(final.info())

final.to_csv('./2.cleaned_data/prop_tax_data.csv')