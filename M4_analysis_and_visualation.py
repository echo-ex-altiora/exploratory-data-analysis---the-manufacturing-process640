# Milestone 3

from db_utils import RDSDatabaseConnector
from EDA_classes import DataTransform
from EDA_classes import DataFrameInfo
from EDA_classes import Plotter
from EDA_classes import DataframeTransform

connect = RDSDatabaseConnector()
failure_df = connect.extract_data()
transform_dataset = DataTransform(failure_df)
transform_dataset.auto_to_boolean()
transform_dataset.manual_to_categorical('Type')


transform = DataframeTransform(failure_df)
transform.impute_with_median('Tool wear [min]')
transform.impute_with_mean('Air temperature [K]')
transform.impute_with_mean('Process temperature [K]')
# transform.remove_rows_with_null()
# transform.correct_skew_boxcox("Rotational speed [rpm]") # i'm removing this as it makes it difficult to understand the data
transform.remove_outliers_IQR('Rotational speed [rpm]', 1.5)
transform.remove_outliers_IQR('Torque [Nm]', 1.5)
failure_df.drop('Air temperature [K]', axis=1, inplace=True)

info = DataFrameInfo(failure_df)
plot = Plotter(failure_df)

'''
# Task 1 : Current Operating Ranges

In this task if you're unsure what each column represents you may want to keep your data dictionary on hand as a reference. 
The business would like to understand at what ranges the machine is operating at currently. 
Create a table which displays to operating ranges of: Air Temperature, process temperature, Rotational speed, Torque and Tool wear

Then breakdown the same data to understand the ranges for each of the different product quality types.

The management would also like to know the upper limits of tool wear the machine tools have been operating at. 
Create a visualisation displaying the number of tools operating at different tool wear values.
'''

column_names = list(failure_df.columns)
numeric_features = [col for col in failure_df.columns[1:]
                    if failure_df[col].dtype == 'float64' or failure_df[col].dtype == 'int64']

operating_ranges_df = failure_df[numeric_features].agg(['min', 'max'])
print(operating_ranges_df.head(5))

operating_ranges_by_type = failure_df.groupby(['Type'], observed=False)[numeric_features].agg(['min', 'max'])
print(operating_ranges_by_type.head(10))

# plot.countplot("Tool wear [min]")
# plot.histogram("Tool wear [min]")

'''
Task 2 : Determine the failure rate in the process 

You've been tasked with determining how many failures there are in the manufacturing process and the leading causes of failure.
Determine and visualise how many failures have happened in the process, what percentage is this of the total? 
Check if the failures are being caused based on the quality of the product.

What seems to be the leading causes of failure in the process? 
Create a visualisation of the number of failures due to each possible cause during the manufacturing process.
'''

print(f'\n{failure_df['Machine failure'].value_counts()} \n')
# plot.countplot('Machine failure')
print(f'{failure_df['Machine failure'].value_counts(normalize=True) * 100} \n')

machine_failure_by_type = failure_df.groupby(['Type'], observed=False)['Machine failure'].value_counts()
print(f'{machine_failure_by_type} \n')

machine_failure_by_type_normalised = failure_df.groupby(['Type'], observed=False)['Machine failure'].value_counts(normalize=True) * 100
print(f'{machine_failure_by_type_normalised} \n')
# There are over 10x the amount of machine failures on low quality products as high quality products, however 
# there are far more low quality roducts produced and when we look at the percentage of machine failure by type...
failure_df['Machine failure'].value_counts(normalize=True) * 100
# Machine failure is actually less than twice as likely to occur on low quality products than high


import seaborn as sns # type: ignore
import matplotlib.pyplot as plt
import pandas as pd


temp_df = failure_df.loc[failure_df['Machine failure'] == True]
temp_df.drop(temp_df.columns[[0,1,2,3,4,5,6,7]], axis=1, inplace=True)

temp_df_count = temp_df.sum()
print(f'{temp_df_count} \n')

df = pd.melt(temp_df)
sns.countplot(data=df, x='variable', hue='value')
# plt.show()
# heat dissipation failure (HDF) is the leading cause of failures

'''
# Task 3 : A deeper understanding of failures

With the failures identified you will need to dive deeper into what the possible causes of failure might be in the process.

For each different possible type of failure try to investigate if there is any correlation between any of the settings 
the machine was running at. Do the failures happen at certain torque ranges, processing temperatures or rpm?

Try to identify these risk factors so that the company can make more informed decisions about what settings to run the machines at. 
If you find any insight into when the machine is likely to fail then develop a strategy on how the machine might be setup 
to avoid this.
'''

possible_failures = {'TWF':'Tool wear failure', 'HDF':'Heat dissipation failure', 'PWF':'Power failure', 'OSF':'Overstrain failure', 'RNF':'Random failure'}
for key, value in possible_failures.items():
    print('\n', value)
    operating_ranges_by_failure = failure_df.groupby([key], observed=False)[[numeric_features]].agg(['min', 'max', 'mean'])
    print(operating_ranges_by_failure)

'''
TWF 
- Process temperature : range and mean are similar
- Rotational speed : range and mean are similar
- Torque : range and mean are similar
- Tool wear : When TWF = True, the mean is ~211 and minimum value is 108 compared to a mean of ~108 and minimum value of 0 when TWF=False
Summary : TWF is far more likely to occur when Tool wear is high

HDF
- Process temperature : When HDF = True, the mean is higher by ~1 degrees and the minimum value is ~ 4 degrees higher
- Rotational speed : When HDF = True, the mean and max are both significantly lower
- Torque : When HDF = True, the mean is higher by ~12 and the min is higher by ~21 
- Tool wear : range and mean are similar
Summary : HDF is not likely to occur when Process temperature and Torque are kept lower and rotational speed is higher

PWF - 
- Process temperature : the means are very similar, although the minimum value is higher by ~3 when PWF = True
- Rotational speed : When PWF = True, the mean and max are both significantly lower
- Torque : When PWF = True, the mean is higher by ~22 and the min is higher by ~38 
- Tool wear : range and mean are similar
Summary : PWF more likely to occur when Torque is high and rotational speed is low

OSF - 
- Process temperature : range and mean are very similar
- Rotational speed : When OSF = True, the mean and max are both lower
- Torque : When OSF = True, the mean is higher by ~16 and the min is higher by ~26 
- Tool wear : When OSF = True, the mean is ~205 and minimum value is 108 compared to a mean of ~107 and minimum value of 0 when OSF = False

RNF - 
- Process temperature : range and mean are very similar
- Rotational speed : range and mean are similar
- Torque : range and mean are similar
- Tool wear : range and mean are similar
Summary : Nothing much of note

Risk Factors are:
- High Torque : exceeding 40 Nm
- High Tool wear : exceeding 108 min
- Low Rotational speed : below 1500 rpm

'''