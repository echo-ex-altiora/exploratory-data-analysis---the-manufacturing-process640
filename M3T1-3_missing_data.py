from db_utils import load_dataframe
from EDA_classes import DataTransform
from EDA_classes import DataFrameInfo
from EDA_classes import Plotter
from EDA_classes import DataframeTransform

import pandas as pd
from scipy.stats import chi2_contingency # type: ignore
from scipy.stats import normaltest # type: ignore
pd.options.mode.chained_assignment = None  # default='warn'

'''
Task 1 : Convert columns to correct format

Now that you have familiarised yourself with the data we want to alter any columns that aren't in the correct format. 

If there are columns which you think should be converted into a different format, create a DataTransform class to 
handle these conversions. 
Within the DataTransform class add methods which you can apply to your DataFrame columns to perform any required conversions.
'''

failure_df = load_dataframe()

# The columns Machine failure, TWF, HDF, PWF, OSF, RNF are Dtype int64 but would be better stored as boolean data 
# The column Type is Dtype object but should be category

data_transform = DataTransform(failure_df)
data_transform.auto_to_boolean()
data_transform.manual_to_categorical('Type')

'''
Task 2 : Create a class to get information from the dataframe

Create a DataFrameInfo class which will contain methods that generate useful information about your DataFrame.

Some useful utility methods you might want to create that are often used for EDA tasks are:
    - Describe all columns in the DataFrame to check their data types
    - Extract statistical values: median, standard deviation and mean from the columns and the DataFrame
    - Count distinct values in categorical columns
    - Print out the shape of the DataFrame
    - Generate a count/percentage count of NULL values in each column
    Any other methods you may find useful

'''
info = DataFrameInfo(failure_df)


'''
Task 3 : Remove/Impute missing values in data

An important EDA task is to impute or remove missing values from the dataset. 
Missing values can occur due to a variety of reasons such as data entry errors or incomplete information.

You will first identify the variables with missing values and determine the percentage of missing values in each variable. 
Depending on the extent of missing data, you may choose to either impute the missing values or remove them from the dataset.

If the percentage of missing data is relatively small, you may choose to impute the missing values 
using statistical methods such as mean or median imputation. 

Alternatively, if the percentage of missing data is large or if the missing data is not missing at random, 
you may choose to remove the variables or rows with missing data entirely.
'''

'''
Task 3 Step 1 : Create Plotter and DataTransform Classes (find in EDA_classes)
    - A Plotter class to visualise insights from the data
    - A DataFrameTransform class to perform EDA transformations on your data
'''
plot = Plotter(failure_df)
transform = DataframeTransform(failure_df)

'''
Task 3 Step 2 : Dropping columns
Use a method/function to determine the amount of NULLs in each column.
Determine which columns should be dropped and drop them.
'''
info.percentage_null()
print(info.find_columns_with_missing_values())

# The three columns with missing values are Tool wear, Air temp and Process temp which are all numerical columns
# The percentage of missing values does not exceed 10% in any of the columns so there is no reason to drop any columns

'''
Task 3 Step 3: Within your DataFrameTransform class create a method which can impute your DataFrame columns. 
Decide whether the column should be imputed with the median or the mean and impute the NULL values.
'''

# We need to determine if the missing data is missing at random or not missing at random 
# as this will determine whether we choose to remove rows or impute data.

# first lets get an overview of the correlation between the numerical columns
plot.correlation_matrix()
# From the heatmap we can see that:
# Air Temperature and Process Temperature are strongly correlated at 87% as expected
# Tool wear doesn't have strong correlation with any other values

# Due to having no significant correlation, we can probably risk concluding that Tool wear is missing at random
# To decide what to impute the missing value with, lets look at the distribution...
plot.kde_plot('Tool wear [min]')
info.manual_extract_stats('Tool wear [min]')
# The distribution looks fairly flat and the mean and median are very similar so we could impute with either
transform.impute_with_mean('Tool wear [min]')


# Lets further investigate the correlation between process and air temp 
plot.scatter_plot('Air temperature [K]', 'Process temperature [K]')

# To check if your data are MAR, 
# take each column with missingness and recode it as one if it is missing and zero otherwise. 
# Then regress each of the the other variables onto it using a logistic regression. 
# A significant p-value indicates an association between the regressor and missingness, meaning your data are MAR.

# One potential approach we could take, since they are strongly correlated, 
# is to use the mean of the difference to impute the missing values
# This method would require us to drop rows where both air and process temp are missing
temp = ['Air temperature [K]', 'Process temperature [K]']
temp_df = failure_df[temp]
temp_df['difference'] = temp_df['Process temperature [K]'] - temp_df['Air temperature [K]']
temp_info = DataFrameInfo(temp_df)
temp_info.extract_stats()
temp_df = temp_df.dropna(subset=['Process temperature [K]','Air temperature [K]'], how='all')
temp_df['Process temperature [K]'] = temp_df['Process temperature [K]'].fillna(temp_df['Air temperature [K]'] + temp_df['difference'].mean())
temp_df['Air temperature [K]'] = temp_df['Air temperature [K]'].fillna(temp_df['Process temperature [K]'] - temp_df['difference'].mean())


# Alternatively ...
# Use the D’Agostino’s K^2 Test (goodness-of-fit measure of departure from normality) to test for normal distribution
data = failure_df['Air temperature [K]']
info.manual_extract_stats('Air temperature [K]')
# D’Agostino’s K^2 Test
stat, p = normaltest(data, nan_policy='omit')
print('Statistics=%.3f, p=%.3f' % (stat, p))
# Air temp looks to be normally distributed
plot.kde_plot('Air temperature [K]')
# the median and mean are very similar so we can use them interchangeably
transform.impute_with_mean('Air temperature [K]')

data = failure_df['Process temperature [K]']
info.manual_extract_stats('Process temperature [K]')
# D’Agostino’s K^2 Test
stat, p = normaltest(data, nan_policy='omit')
print('Statistics=%.3f, p=%.3f' % (stat, p))
# Process temp looks normally distributed
plot.kde_plot('Process temperature [K]') 
# the median and mean are very similar so we can use them interchangeably
transform.impute_with_mean('Process temperature [K]')


'''
Task 3 Step 4: Run your NULL checking method/function again to check that all NULLs have been removed. 
Generate a plot by creating a method in your Plotter class to visualise the removal of NULL values.
'''

info.percentage_null()

