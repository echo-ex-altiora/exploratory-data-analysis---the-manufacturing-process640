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
info = DataFrameInfo(failure_df)
plot = Plotter(failure_df)
transform = DataframeTransform(failure_df)
transform.impute_with_median('Tool wear [min]')
transform.impute_with_mean('Air temperature [K]')
transform.impute_with_mean('Process temperature [K]')
transform.correct_skew_boxcox("Rotational speed [rpm]")


column_names = list(failure_df.columns)
numeric_features = [col for col in failure_df.columns[1:]
                    if failure_df[col].dtype == 'float64' or failure_df[col].dtype == 'int64']

'''
Step 1 : First visualise your data using your Plotter class to determine if the columns contain outliers.
'''

# Visual methods for detecting outliers are box plots and histograms

plot.box_plot('Air temperature [K]') 
plot.histogram('Air temperature [K]', 100) 
plot.box_plot('Process temperature [K]')
plot.histogram('Process temperature [K]', 100)
plot.box_plot('Rotational speed [rpm]')
plot.histogram('Rotational speed [rpm]', 100)
plot.box_plot('Torque [Nm]')
plot.histogram('Torque [Nm]', 100)
plot.box_plot('Tool wear [min]')
plot.histogram('Tool wear [min]', 100)

# From initial visual inspection, we can determine that:
# - Air temperature [K] and Tool wear [min] have 0 potential outliers 
# - Process temperature [K] may have a couple on the lower end
# - Rotation speed [rpm] and Torque [Nm] both seem to have quite alot of potential outliers on the boxplot

# Statistical methods are z-score and Interquartile range

# Calculate the Z-Scores
for column in numeric_features:
    mean = failure_df[column].mean()
    std = failure_df[column].std()
    print(f'Mean of {column} is {mean} and standard deviation is {std}')
    z_scores = (failure_df[column] - mean) / std
    failure_df_z_scores = failure_df
    failure_df_z_scores['z_scores'] = z_scores
    subset = failure_df_z_scores.loc[failure_df_z_scores['z_scores'] > 3, [column, 'z_scores']]
    if len(subset) == 0:
        print(f'{column} has 0 potential outliers \n')
    else:
        print(f'Number of potential outliers is : {len(subset)} \n')


# The Z-Score is a statistical measurement that describes a data point's position relative to the mean of a group of values,
# measured in terms of standard deviations. 
# Data points with a Z-Score having a high absolute value, typically beyond a threshold like 2 or 3, 
# are often considered outliers as they significantly deviate from the average

# Summary Torque has 14 potential outliers

# Calculate IQR
for column in numeric_features:
    Q1 = failure_df[column].quantile(0.25)
    Q3 = failure_df[column].quantile(0.75)
    IQR = Q3 - Q1
    print(f"Q1 (25th percentile): {Q1}")
    print(f"Q3 (75th percentile): {Q3}")
    print(f"IQR: {IQR}")
    outliers = failure_df[(failure_df[column] < (Q1 - 1.5 * IQR)) | (failure_df[column] > (Q3 + 1.5 * IQR))]
    if len(outliers) > 0:
        print(f'The number of potential outliers : {len(outliers)}')
        print(f"Outliers of {column} column:")
        print('\n')
    else:
        print(f'{column} has 0 outliers \n')

# Values that fall below the first quartile minus 1.5 times the IQR or above the third quartile plus 1.5 times the IQR 
# are typically classified as outliers, as they lie outside the common range of variability in the data.

# Summary Process temp has 10 potential outliers, rotational speed has 90 potential outliers, torque has 69 potential outliers

'''
Step 2: Once identified use a method to transform or remove the outliers from the dataset. Build this method in your DataFrameTransform class.
'''
transform.remove_outliers_IQR('Rotational speed [rpm]', 1.5)
transform.remove_outliers_IQR('Torque [Nm]', 1.5)

'''
Step 3: With the outliers transformed/removed re-visualise your data with you Plotter class to check that the outliers have been correctly removed.
'''
plot.box_plot('Rotational speed [rpm]')
plot.histogram('Rotational speed [rpm]', 100)

plot.box_plot('Torque [Nm]')
plot.histogram('Torque [Nm]', 100)



# Secondly, compare the columns for correlation and
# investigate any data points that deviate significantly from the general trend, indicating potential outliers
# Visual methods for detecting outliers re scatter plots and regression lines

plot.pair_plot()
plot.correlation_matrix()
# there is a strong correlation between Air and Process temp and Rotational speed and Torque

plot.scatter_plot('Rotational speed [rpm]', 'Torque [Nm]')
plot.scatter_with_lin_regression('Rotational speed [rpm]', 'Torque [Nm]')

plot.scatter_plot('Air temperature [K]', 'Process temperature [K]')
plot.scatter_with_lin_regression('Air temperature [K]', 'Process temperature [K]')
# there are significant ouliers due to filling in null values

connect.save_edited_dataframe(failure_df)

