from db_utils import RDSDatabaseConnector
from db_utils import load_dataframe
from EDA_classes import DataTransform
from EDA_classes import DataFrameInfo
from EDA_classes import Plotter
from EDA_classes import DataframeTransform

connect = RDSDatabaseConnector()
failure_df = load_dataframe()
transform_dataset = DataTransform(failure_df)
transform_dataset.auto_to_boolean()
transform_dataset.manual_to_categorical('Type')
info = DataFrameInfo(failure_df)
plot = Plotter(failure_df)
transform = DataframeTransform(failure_df)
transform.impute_with_median('Tool wear [min]')

transform.impute_with_mean('Air temperature [K]')
transform.impute_with_mean('Process temperature [K]')
# transform.impute_with_correlated_column('Process temperature [K]', 'Air temperature [K]')

'''
# Step 1: Firstly you will need to identify the skewed columns in the data. 
# This can be done using standard Pandas methods. 
# You then need to determine a threshold for the skewness of the data, over which a column will be considered skewed. 
# You should also visualise the data using your Plotter class to analyse the skew.
'''
column_names = list(failure_df.columns[2:])
column_names = list(failure_df.columns)
numeric_features = [col for col in failure_df.columns[1:]
                    if failure_df[col].dtype == 'float64' or failure_df[col].dtype == 'int64']

columns_with_high_skew = []
for column in numeric_features:
    print(f"Skew of {column} column is {failure_df[column].skew()}\n")
    if failure_df[column].skew() > 1 or failure_df[column].skew() < -1:
        print(failure_df[column].describe())
        plot.histogram(column)
        plot.qq_plot(column)
        columns_with_high_skew.append(column)

print(columns_with_high_skew)
# From looking at the histograms of the different columns, I've decided that the threshold for skewness should be 1/-1
# The main contender for skew is Rotational speed which has a big gap between the 3rd quartile and the max
# Looking at the histogram and qq-plot we can see is heavily right skewed

'''
# Step 2: Once the skewed columns are identified, you should perform transformations on these columns 
# to determine which transformation results in the biggest reduction in skew. 
# Create the the method to transform the columns in your DateFrameTransform class.
'''
plot.skew_log("Rotational speed [rpm]")
plot.skew_boxcox("Rotational speed [rpm]")

'''
# Step 3: Apply the identified transformations to the columns to reduce their skewness.
'''
transform.correct_skew_boxcox("Rotational speed [rpm]")
# transform.correct_skew_log("Rotational speed [rpm]")

'''
# Step 4: Visualise the data to check the results of the transformation have improved the skewness of the data.
'''
for column in columns_with_high_skew:
    print(f"Skew of {column} column is {failure_df[column].skew()}")
    plot.histogram(column)
    plot.qq_plot(column)

'''
# Step 5: At this point you may want to save a separate copy of your DataFrame to compare your results.
'''
connect.save_edited_dataframe(failure_df)

