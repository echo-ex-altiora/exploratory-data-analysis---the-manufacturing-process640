from db_utils import RDSDatabaseConnector
from EDA_classes import DataTransform
from EDA_classes import DataFrameInfo
from EDA_classes import Plotter
from EDA_classes import DataframeTransform

import pandas as pd
from scipy.stats import chi2_contingency # type: ignore
from scipy.stats import normaltest # type: ignore
pd.options.mode.chained_assignment = None  # default='warn'

connect = RDSDatabaseConnector()
failure_df = connect.extract_data()
transform_dataset = DataTransform(failure_df)
transform_dataset.auto_to_boolean()
transform_dataset.manual_to_categorical('Type')
info = DataFrameInfo(failure_df)

# Step 1 : Create Plotter and DataTransform Classes (find in EDA_classes)

plot = Plotter(failure_df)
transform = DataframeTransform(failure_df)


# Step 2 : Use a method/function to determine the amount of NULLs in each column. 
# Determine which columns should be dropped and drop them.

info.percentage_null()
print(info.find_columns_with_missing_values())
# The three columns with null values are Tool wear, Air temp and Process temp which are all numerical columns
# None of the columns have a high number of null value so no need to drop any columns


# Step 3: Within your DataFrameTransform class create a method which can impute your DataFrame columns. 
# Decide whether the column should be imputed with the median or the mean and impute the NULL values.


# first lets get an overview of the correlation between the numerical columns
plot.correlation_matrix()
# from the heatmap we can see that:
# Air Temperature and Process Temperature are strongly correlated as expected
# and that Tool wear doesn't have strong correlation with any other values


# lets further investigate the correlation between process and air temp
temp = ['Air temperature [K]', 'Process temperature [K]']
temp_df = failure_df[temp]

temp_df['missing_air_temp'] = temp_df['Air temperature [K]'].isnull()
contingency_table = pd.crosstab(temp_df['missing_air_temp'], temp_df['Process temperature [K]'])
# Perform chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic = {chi2}")
print(f"p-value = {p}")
# as the p-value is much greater than 0.05, 
# we can assume that the null values for air temp are randomly distributed with respect to process temperature 

temp_df['missing_process_temp'] = temp_df['Process temperature [K]'].isnull()
contingency_table = pd.crosstab(temp_df['missing_process_temp'], temp_df['Air temperature [K]'])
# Perform chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic = {chi2}")
print(f"p-value = {p}")
# the p-value is actually less that 0.05 here suggesting some kind of dependency
# so plot Air temp keeping only records where process temp is null
data = temp_df[temp_df["missing_process_temp"] == True]
plot_2 = Plotter(data)
plot_2.kde_plot("Air temperature [K]")
# this looks like a normal distribution


# next we want to check each of the columns with null values for normal distribution
data = failure_df['Tool wear [min]']
# D’Agostino’s K^2 Test
stat, p = normaltest(data, nan_policy='omit')
print('Statistics=%.3f, p=%.3f' % (stat, p))
# A p-value close to zero means the data are normally distributed
plot.kde_plot('Tool wear [min]')
plot.qq_plot('Tool wear [min]')
info.manual_extract_stats('Tool wear [min]')
# From the Q-Q plot it is clear that while the data are normally distributed through the middle of the range, 
# there is a slight deviation from normality at the upper bound 
# but a significant deviation at the lower bound because the normal distribution would extend into the negative numbers 
# while the minimum tool wear cannot do below 0.
# From the Q-Q ploty we can see that the mean and median will be very similar so we could impute by either.
transform.impute_with_median('Tool wear [min]')


data = failure_df['Air temperature [K]']
# D’Agostino’s K^2 Test
stat, p = normaltest(data, nan_policy='omit')
print('Statistics=%.3f, p=%.3f' % (stat, p))
# air temp is definetly normally distributed
plot.kde_plot('Air temperature [K]')
plot.qq_plot('Air temperature [K]')
# the qq_plot shows that the data are normally distributed through the middle and upper range but deviates at the lower range 
info.manual_extract_stats('Air temperature [K]')
# the median and mean are very similar so we can use them interchangeably
transform.impute_with_mean('Air temperature [K]')

data = failure_df['Process temperature [K]']
# D’Agostino’s K^2 Test
stat, p = normaltest(data, nan_policy='omit')
print('Statistics=%.3f, p=%.3f' % (stat, p))
# air temp looks normally distributed
plot.kde_plot('Process temperature [K]')
plot.qq_plot('Process temperature [K]')
#the qq_plot shows that the data are normally distributed through the middle range but deviates at the lower range 
info.manual_extract_stats('Process temperature [K]')
# the median and mean are very similar so we can use them interchangeably
transform.impute_with_mean('Process temperature [K]')


# Step 4: Run your NULL checking method/function again to check that all NULLs have been removed. 
# Generate a plot by creating a method in your Plotter class to visualise the removal of NULL values. ???

info.percentage_null()

