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

# Step 1: Firstly you will need to identify the skewed columns in the data. 
# This can be done using standard Pandas methods. 
# You then need to determine a threshold for the skewness of the data, over which a column will be considered skewed. 
# You should also visualise the data using your Plotter class to analyse the skew.

# Step 2: Once the skewed columns are identified, you should perform transformations on these columns 
# to determine which transformation results in the biggest reduction in skew. 
# Create the the method to transform the columns in your DateFrameTransform class.

# Step 3: Apply the identified transformations to the columns to reduce their skewness.

# Step 4: Visualise the data to check the results of the transformation have improved the skewness of the data.

# Step 5: At this point you may want to save a separate copy of your DataFrame to compare your results.

import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns # type: ignore
import pandas as pd
import numpy as np
from statsmodels.graphics.gofplots import qqplot # type: ignore
from scipy import stats # type: ignore


column_names = list(failure_df.columns[2:])
column_names = list(failure_df.columns)
numeric_features = [col for col in failure_df.columns[1:]
                    if failure_df[col].dtype == 'float64' or failure_df[col].dtype == 'int64']

columns_with_high_skew = []
for column in numeric_features:
    print(f"Skew of {column} column is {failure_df[column].skew()}")
    if failure_df[column].skew() > 0.1 or failure_df[column].skew() < -0.1:
        plot.histogram(column)
        plot.qq_plot(column)
        columns_with_high_skew.append(column)

print(columns_with_high_skew)
# the main contenders for skew are Air temp and rotational speed which are both right skewed

plot.skew_log("Air temperature [K]")
plot.skew_log("Rotational speed [rpm]")
plot.skew_boxcox("Rotational speed [rpm]")

transform.correct_skew_log("Air temperature [K]")
transform.correct_skew_boxcox("Rotational speed [rpm]")

for column in columns_with_high_skew:
    print(f"Skew of {column} column is {failure_df[column].skew()}")
    plot.histogram(column)
    plot.qq_plot(column)

connect.save_edited_dataframe(failure_df)

