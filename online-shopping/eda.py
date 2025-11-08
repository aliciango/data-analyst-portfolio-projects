#! C:\Users\jingn\Documents\SQL\online_shoppers\myenv\Scripts\python.exe
import pandas as pd
import numpy as np
import matplotlib
import seaborn
import hashlib

################################
## A. IMPORT AND DATA STRUCTURE
################################
shoppers_copy = pd.read_csv("online_shoppers_intention.csv")

shoppers = shoppers_copy.copy()
shoppers.info()
shoppers.columns = shoppers.columns.str.lower()
shoppers.columns = ['admin_page_cnt', 'admin_duration', 'info_page_cnt',
    'info_duration', 'product_page_cnt', 'product_duration',
    'bounce_rate', 'exit_rate', 'avg_page_val', 'special_day_0_1', 'month',
    'op_sys', 'browser_type', 'region', 'traffic_type', 'visitor_type',
    'weekend', 'revenue']

## A - Data Type Inspecting 
# Drop unnecessary columns
shoppers = shoppers.drop(['op_sys', 'region'], axis=1)

# Convert columns to categorical
categorical_cols = ['month', 'visitor_type', 'browser_type', 'traffic_type']
shoppers[categorical_cols] = shoppers[categorical_cols].astype('category')

# Convert columns to boolean
boolean_cols = ['weekend', 'revenue']
shoppers[boolean_cols] = shoppers[boolean_cols].astype(bool)

# Convert columns to float
float_cols = ['admin_duration', 'info_duration', 'product_duration', \
              'bounce_rate', 'exit_rate', 'avg_page_val', 'special_day_0_1']
shoppers[float_cols] = shoppers[float_cols].astype(float)

# Convert columns to integer
int_cols = ['admin_page_cnt', 'info_page_cnt', 'product_page_cnt']
shoppers[int_cols] = shoppers[int_cols].astype(int)

shoppers.shape # > 12,330 x 16

## A - Clean each data type
# Go through each of these variable and use value_counts to count the number of record for each unique categories
categorical_cols = ['month', 'visitor_type', 'browser_type', 'traffic_type']
# > 13 browser types, 20 traffic types, 3 visitor types, and missing april, january
for col in categorical_cols: 
    print(shoppers[col].value_counts().sort_index())
    
# A - month column
shoppers.rename(columns={'month': 'month_raw'}, inplace=True)
month_map = { 'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 'Jul': 7, 'Aug': 8, \
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12 }
month_map = { 'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, \
              'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12 }
shoppers['month_num'] = shoppers['month_raw'].map(month_map)

# A - visitor_type column
shoppers['visitor_type'] = shoppers['visitor_type'].str.strip().str.lower()

################################
## B. DATA CLEANING
################################
# generate boxplot for all integer and float columns
# int_cols = ['admin_page_cnt', 'info_page_cnt', 'product_page_cnt']
# float_cols = ['admin_duration', 'info_duration', 'product_duration', \
#               'bounce_rate', 'exit_rate', 'avg_page_val', 'special_day_0_1']


# def remove_outliers_iqr(df, column_name):
#     """
#     Removes outliers from a specific column of a pandas DataFrame using the IQR method.

#     Args:
#         df (pd.DataFrame): The input DataFrame.
#         column_name (str): The name of the column from which to remove outliers.

#     Returns:
#         pd.DataFrame: A new DataFrame with outliers for the specified column removed.
#     """
#     # Calculate Q1 (25th percentile) and Q3 (75th percentile)
#     q1, q3 = df[column_name].quantile([0.25, 0.75])
    
#     # Calculate the Interquartile Range (IQR)
#     iqr = q3 - q1
    
#     # Determine the lower and upper bounds for outlier detection
#     lower_whisker = q1 - 1.5 * iqr
#     upper_whisker = q3 + 1.5 * iqr
    
#     # Create a mask to filter out rows with outliers
#     mask = (df[column_name] >= lower_whisker) & (df[column_name] <= upper_whisker)
    
#     return df[mask].copy()

# new_num_cols = remove_outliers_iqr(shoppers, 'avg_page_val')
# new_num_stats = new_num_cols.describe()
# new_num_cols.shape

# ## Missing value
# shoppers.isnull().sum()
# shoppers.describe()

################################
## C. UNIQUE IDENTIFIER
################################
## The problem of this dataset is that it does not have a single SessionID or a precise Timestamp column to uniquely identify that session.
## I solved this problem by create surrogated key using hash
def generate_hash_id(row):
    # Combine all relevant column values into a single string for hashing
    combined_string = "".join(str(row[col]) for col in shoppers.columns)
    return hashlib.sha256(combined_string.encode('utf-8')).hexdigest()

# Apply the function to each row to create a unique session ID
shoppers['session_id'] = shoppers.apply(generate_hash_id, axis=1)

# Verify the number of unique session IDs
print(f"Number of unique session IDs: {shoppers['session_id'].nunique()}")
print(f"Total number of rows: {len(shoppers)}")

# Check for duplicates in the generated session_id
if shoppers['session_id'].nunique() == len(shoppers):
    print("All generated session IDs are unique.")
else:
    print("Warning: Duplicate session ID found.")
    
shoppers[shoppers.duplicated(keep=False)]
# drop duplicated sessions
shoppers = shoppers.drop_duplicates(subset=['session_id'], keep="first")
shoppers['session_id'].nunique() == len(shoppers) # returned True
shoppers.shape # > 11,998 x 18

################################
## D. Feature Engineering
################################
## D - pivot table, create [page_type, page_cnt, page_duration]
dfs = []
for page in ['admin', 'info', 'product']:
    temp = shoppers[['session_id', f'{page}_page_cnt', f'{page}_duration']].copy()
    temp['page_type'] = page
    temp = temp.rename(columns={
        f'{page}_page_cnt': 'page_cnt',
        f'{page}_duration': 'page_duration'
    })
    dfs.append(temp)

shoppers_long = pd.concat(dfs, ignore_index=True)
shoppers_long.shape # > 35,994 * 4

# validate the result of pivoting
shoppers_long[shoppers_long['session_id']=="e44d48f5e0081681306f100e95561343b80ae722d7a339652e9b043ea10d302a"]
shoppers.columns

## D - create a copy of the shoppers before merging
shoppers_to_merge = shoppers.copy()
# Drop the wide-format columns
cols_to_drop = [
    'product_page_cnt', 'product_duration',
    'info_page_cnt', 'info_duration',
    'admin_page_cnt', 'admin_duration'
]
shoppers_to_merge.drop(columns=cols_to_drop, inplace=True)
# Merge with long format
shoppers_long_wide = shoppers_to_merge.merge(shoppers_long, on='session_id', how='left')
shoppers_long_wide.shape # > 35,994 x 14


## D - sum the total of page visited
pages_per_session = (
    shoppers_long_wide
    .groupby('session_id', as_index=False)
    .agg(total_pages=('page_cnt', 'sum'))
)
shoppers_long_wide = shoppers_long_wide.merge(pages_per_session, on='session_id', how='left')
# validating
shoppers_long_wide[shoppers_long_wide['session_id'] == "000132741cbac4ce5301bcb2cbd7e8d27482d514f299c71bf4de3743403b5353"]

duration_per_session = (
    shoppers_long_wide
    .groupby('session_id', as_index=False)
    .agg(total_duration=('page_duration', 'sum'))
)
shoppers_long_wide = shoppers_long_wide.merge(duration_per_session, on='session_id', how='left')
# validating
shoppers_long_wide[shoppers_long_wide['session_id'] == "000132741cbac4ce5301bcb2cbd7e8d27482d514f299c71bf4de3743403b5353"]
shoppers_long_wide.info()
shoppers_long_wide['page_type'] = shoppers_long_wide['page_type'].astype('category')

shoppers_long_wide.to_csv('./cleaned_data/shoppers_long_wide.csv', index=False)