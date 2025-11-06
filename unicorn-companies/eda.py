import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Data Loading
df_o = pd.read_csv('Modified_Unicorn_Companies.csv')
df = df_o.copy()
######################
## A. Data Discovery
######################
# A. data shape
df.shape
# > 1074 x 10

# A. transform column names to lowercase and replace spaces with underscores
df.columns = [col.lower() for col in df.columns]
df.columns = [col.replace(' ', '_') for col in df.columns]
df.columns

# A. missing values
df.isnull().sum()
# > city has 17 missing values
# A-missing value handling: fill missing city with 'Unknown'
df['city'] = df['city'].fillna('Unknown')


# A. data types
df.info()
# A-data type handling: transform date
df['date_joined'] = pd.to_datetime(df['date_joined'])
# A-data type handling: split funding column into 'funding' of numeric type and 'currency' of string type
df = df[(df['funding'] != 'Unknown') & (df['funding'].notnull())]
# strip dollar sign
df['funding'] = df['funding'].str.strip('$')
# extract the last character as currency (M: million, B: billion)
df['funding_currency'] = df['funding'].str[-1].astype('string')
# remove the last character from funding to keep only numeric part
df['funding'] = df['funding'].str[:-1]
df['funding'] = df['funding'].fillna('0')
df = df[ df['funding'] != 0 ]


# columns that can be converted to string type
string_cols = ['company', 'city', 'country/region', 'continent']
df[string_cols] = df[string_cols].astype('string')
    
# some columns can be converted to categorical type for efficient use of memory
categorical_cols = ['industry', 'funding_currency']
df[categorical_cols] = df[categorical_cols].astype('category')


# A. statistics and data distribution
numeric_cols = ['valuation', 'funding']
# A. check if there is any missing value
df.isnull().sum() # no missing values

# A. scatter plot between valuation and funding
plt.figure(figsize=(8,6))
plt.scatter(df['funding'], df['valuation'], alpha=0.6)

# A. Primary key cleaning: inspect the company names which may contain duplicates
df['company'].nunique() # 1059 unique company names
df.shape # 1061, but there are only 1059 unique company names

# inspect the rows that have duplicated company names
df_duplicates = df[df.duplicated(subset=['company'], keep=False)]
df_duplicates.shape # 6 X 11

# remove duplicated company names because all rows have identical values
df = df.drop_duplicates(subset=['company'], keep='first')
df.info

# trim whitespace from all string columns
str_cols = df.select_dtypes(include=['object', 'string']).columns
for col in str_cols:
    df[col] = df[col].str.strip()
df.info()
df.isnull()


# A. industry column cleaning: inspect unique values
pd.DataFrame(df['industry'].unique()).sort_values(by=0).reset_index(drop=True)
# clean the industry names: lowercase, replace 'and' with '&', strip whitespace
df['industry_cleaned'] = (df['industry']
                  .astype('string')
                  .str.lower()
                  .str.strip()
                  .str.replace(' and ', ' & ')
)
# Drop the old 'industry' column
df = df.drop(columns=['industry'])
# Rename your new column to take its place
df = df.rename(columns={'industry_cleaned': 'industry'})
# Convert to categorical type
df['industry'] = df['industry'].astype('category')
# validating: pd.DataFrame(df['industry'].unique()).sort_values(by=0).reset_index(drop=True)
df.head(10)
df.info()


######################
## B. Data cleaning
######################
# B. 1F transformation, the column Select Investors contain a list of investors for each record, split the column
df['select_investors_transformed'] = df['select_investors'].astype(str).str.split(',').tolist()
df_long = (
    df.explode('select_investors_transformed')
      .assign(select_investors=lambda x: x['select_investors_transformed'].str.strip())
      .reset_index(drop=True)
)
df_long.shape
df_long.head()
df_long.drop('select_investors_transformed', axis=1)




######################
## C. The time it took each company to reach unicorn status
######################
df_long['date_joined_cleaned'] = pd.to_datetime(df_long['date_joined'], format='%Y-%m-%d')
df_long['years_to_unicorn'] = 2022 - df_long['date_joined_cleaned'].dt.year

# visualize using a boxplot 
df_long['years_to_unicorn'] = pd.to_numeric(df_long['years_to_unicorn'], errors='coerce')
df_long['years_to_unicorn'].dropna().plot.box()
plt.ylabel('Years to unicorn')
plt.title('Boxplot of Years to Unicorn')
plt.show()


######################
## D. Maximum unicorn company valuation for each industry
######################
# Average valuation for each industry
valuation_df_mean = (df_22_1nf[["industry", "valuation"]]
           .groupby("industry")
           .mean() 
           .sort_values(by="valuation", ascending=False)
          )


# Create bar plot
plt.figure(figsize=(12, 6))
plt.bar(valuation_df_mean.index, valuation_df_mean['valuation'])
plt.xticks(rotation=45, ha='right')
plt.title('Mean Valuation by Industry')
plt.xlabel('Industry')
plt.ylabel('Valuation (Billions)')
plt.tight_layout()
plt.show()
valuation_df_mean


# Max valuation for each industry
valuation_df_max = (df[["industry", "valuation"]]
           .groupby("industry")
           .max() 
           .sort_values(by="valuation", ascending=False)
          )
# Create bar plot
plt.figure(figsize=(12, 6))
plt.bar(valuation_df_mean.index, valuation_df_mean['valuation'])
plt.xticks(rotation=45, ha='right')
plt.title('Max Valuation by Industry')
plt.xlabel('Industry')
plt.ylabel('Valuation (Billions)')
plt.tight_layout()
plt.show()
valuation_df_mean

######################
## E. Investigate an industry in depth: valuation, years_to_unicorn, funding 
######################
df_long['company_cleaned'] = df_long['company'].str.upper()
df_long.to_csv('Cleaned_Unicorn_Companies_2022.csv', index=False)


df_long[df_long['company_cleaned'].isin(['BOLT', 'BRANCH', 'FABRIC', 'FIGURE'])]
