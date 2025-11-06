import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_2022_copy = pd.read_csv('Cleaned_Unicorn_Companies_2022.csv')
df_2023_copy = pd.read_csv('Cleaned_Unicorn_Companies_2023.csv')
df_2024_copy = pd.read_csv('unicorn_companies_2024.csv', index_col=0)

df_2023 = df_2023_copy.copy()
df_2024 = df_2024_copy.copy()
df_2022 = df_2022_copy.copy()
df_2024.head()
df_2022.head()
df_2023.head()

######################
## A. Data Discovery and Initial Cleaning
######################
# A. Column Standardization: map 2023 columns' names to match 2022 columns' names
df_2024.columns = ['country', 'country/region', 'select_investors', 'company_link', 'img_src', 'company', 'valuation_raw', 'funding_raw']
# country/region in 2024 datasets is actually just region
df_2024.shape # 1549 x 8
df_2024.info()
df_2024.head(5)

# A. drop null values in valuation and company columns of 2023 dataset
df_2024 = df_2024[(df_2024['valuation_raw'].notnull()) & (df_2024['company'].notnull())]
# inspect whether if only the 'B' character is included or any other character
df_2024['valuation_currency'] = df_2024['valuation_raw'].str[-1].astype('string')
df_2024['valuation_currency'].value_counts()
df_2024.query('valuation_currency == "0"') # webull company has missing valuation
# drop company 'Webull' and country is 'United States' (in case there is another company called Webull, we use the unique identifier by using both company and country)
df_2024 = df_2024.query('not (company == "Webull" and `country` == "United States")')
df_2024.shape # 1548 x 9


# A. string cleaning
# Clean 'valuation_raw' column: remove $ and B, keep numeric values
df_2024['valuation'] = df_2024['valuation_raw'].str.extract('(\d+\.?\d*)')
df_2024['valuation'] = pd.to_numeric(df_2024['valuation'], errors='coerce')
df_2024['valuation'] = df_2024['valuation'].astype('int64')

# Clean 'funding_raw' column:  remove $ and B, keep numeric values
df_2024['funding'] = df_2024['funding_raw'].str.extract('(\d+\.?\d*)')
df_2024['funding'] = pd.to_numeric(df_2024['funding'], errors='coerce')
df_2024['funding'] = df_2024['funding'].astype('int64')

# Clean 'company' column: strip leading and trailing spaces
df_2024['company_cleaned'] = df_2024['company'].str.strip().str.upper()


# A. find duplicates in company names in 2023 dataset
duplicate_companies_24 = df_2024[df_2024.duplicated(subset=['company'], keep=False)]
duplicate_companies_24.sort_values(by='company')
# companies have same name, same country, but different valuation
# join later using ('company', 'valuation')
# 4 companies have same name: Branch and Figure
# check if they have the same companies in 2022 and 2023 data
df_2022.query('company')

# A. missing values handling
# Identified missing values in df_2024
df_2024.isnull().sum()

# some companies have missing information about their 'investors' / 'country' / 'country/region'
# we will see if we can fill that information by finding their investors in 2022 and 2023 datasets
companies_to_check = pd.array(df_2024[ df_2024.query('country.isnull()').index ]['company_cleaned'])

# rows in df_2023 that match any name in the companies_to_check
lookup_table = df_2023[df_2023['company_cleaned'].isin(companies_to_check)][ ['company_cleaned','country/region'] ]
lookup_table.columns = ['company_cleaned', 'country']
lookup_table
# found the countries ('United States') for two companies: BREX and CHARACTER.AI
# because df_2023 does not include information about the country/region, we have to look 
# back at the 2024 dataset to see which country/region that United States is identified with
# Get unique combinations of country and region, showing counts for each pair
print(df_2024.groupby(['country', 'country/region']).size().reset_index(name='count'))

## >> Both of these companies are in the 'United States' and 'North America'
# add 'North America' as a 'country/region' to the lookup_table
lookup_table['country/region'] = 'North America'
lookup_table = lookup_table.drop_duplicates() # unique lookup table


# fill up the empty values in the (country, country/region) columns with the values from lookup table
df_2024 = \
    df_2024.merge(
                lookup_table,
                on = 'company_cleaned',
                how = 'left',
                suffixes = ('','_lookup'))
df_2024['country'] = df_2024['country'].fillna(df_2024['country_lookup'])
df_2024['country/region'] = df_2024['country/region'].fillna(df_2024['country/region_lookup'])
# validate
df_2024.loc[df_2024['company_cleaned'].isin(['BREX','CHARACTER.AI'])][['company_cleaned','country','country/region']]


######################
## B. First Normal Form  (1NF) Transformation
######################
# splitting investors data
# Step 1: Split the investors column into lists
df_24_1nf = df_2024.copy()
df_24_1nf['select_investors'] = df_2024['select_investors'].str.split(',')

# Step 2: Explode the list into separate rows (1NF)
df_24_1nf = df_24_1nf.explode('select_investors')

# Step 3: Clean up whitespace
df_24_1nf['select_investors'] = df_24_1nf['select_investors'].str.strip()
df_24_1nf.head(10)

df_24_1nf['select_investors'].nunique() #  889 unique investors
investor_counts_24 = df_24_1nf['select_investors'].value_counts()
investor_counts_24.head(10)

######################
## C. Feature Engineering
# ######################


######################
## D. Data Export
# ######################
df_24_1nf = df_24_1nf.drop(['valuation_raw','funding_raw','country_lookup', 'country/region_lookup'], axis=1)
df_24_1nf.info() 
df_24_1nf.shape # 1931 x 10
# before saving
# convert to the right data type for each column to save resources

# columns that can be converted to string type
string_cols = ['company', 'country', 'country/region', 'select_investors','company_link', 'img_src', 'company_cleaned']
for col in string_cols:
    df_24_1nf[col] = df_24_1nf[col].astype('string')



df_24_1nf.to_csv('Cleaned_Unicorn_Companies_2024.csv', index=False)
