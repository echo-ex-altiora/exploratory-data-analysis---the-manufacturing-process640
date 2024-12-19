# Highly correlated columns in a dataset can lead to multicollinearity issues, 
# which can affect the accuracy and interpretability of models built on the data. 
# In this task, you will identify highly correlated columns and remove them to improve the quality of the data.

from db_utils import RDSDatabaseConnector
from db_utils import load_dataframe
from EDA_classes import DataTransform
from EDA_classes import DataFrameInfo
from EDA_classes import Plotter
from EDA_classes import DataframeTransform
from scipy.stats import linregress # type: ignore

connect = RDSDatabaseConnector()
failure_df = load_dataframe()
transform_dataset = DataTransform(failure_df)
transform_dataset.auto_to_boolean()
transform_dataset.manual_to_categorical('Type')
info = DataFrameInfo(failure_df)
info.percentage_null()

plot = Plotter(failure_df)
plot.correlation_matrix()

transform = DataframeTransform(failure_df)
transform.impute_with_median('Tool wear [min]')
transform.impute_with_mean('Air temperature [K]')
transform.impute_with_mean('Process temperature [K]')
transform.correct_skew_boxcox("Rotational speed [rpm]")
transform.remove_outliers_IQR('Rotational speed [rpm]', 2)
transform.remove_outliers_IQR('Torque [Nm]', 2)
transform.remove_outliers_z_score('Torque [Nm]')

# Step 1: First compute the correlation matrix for the dataset and visualise it.

plot.correlation_matrix()

# Step 2: Identify which columns are highly correlated. 
# You will need to decide on a correlation threshold and to remove all columns above this threshold.

# There is a strong positive correlation between Air and Process temp 
# and a strong negative correlation between Rotational speed and Torque
column_names = list(failure_df.columns)
numeric_features = [col for col in failure_df.columns[1:]
                    if failure_df[col].dtype == 'float64' or failure_df[col].dtype == 'int64']

columns_with_high_correlation = []
for column_1 in numeric_features:
    for column_2 in numeric_features:
        if column_1 == column_2:
            continue
        else:
            slope, intercept, r_value, p_value, std_err = linregress(failure_df[column_1], failure_df[column_2])
            if round(r_value, 1) >= 0.8:
                columns_with_high_correlation.append(column_1)
print(columns_with_high_correlation)

plot.scatter_with_lin_regression('Air temperature [K]', 'Process temperature [K]')
plot.scatter_with_lin_regression('Rotational speed [rpm]', 'Torque [Nm]')


# Step 3: Decide which columns can be removed based on the results of your analysis.

# We can remove either Air or Process temp in this case, 
# since Process temp is the temp the machine was operating at during production while air temp is just the room temp
# We should keep Process temp


# Step 4: Remove the highly correlated columns from the dataset.

failure_df.drop('Air temperature [K]', axis=1, inplace=True)
print(failure_df.head(5))

