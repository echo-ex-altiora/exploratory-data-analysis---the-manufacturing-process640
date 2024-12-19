from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns # type: ignore
import pandas as pd
import numpy as np
from scipy.stats import linregress # type: ignore
import plotly.express as px # type: ignore
from scipy.stats import normaltest # type: ignore
from statsmodels.graphics.gofplots import qqplot # type: ignore
from scipy.stats import chi2_contingency # type: ignore
from scipy import stats # type: ignore
pd.options.mode.chained_assignment = None  # default='warn'

plt.rc("axes.spines", top=False, right=False)
sns.set_style(style='darkgrid', rc=None)
style.use('fivethirtyeight')
five_thirty_eight = [
    "#30a2da",
    "#fc4f30",
    "#e5ae38",
    "#6d904f",
    "#8b8b8b",
]
sns.set_palette(five_thirty_eight)


class DataTransform:

    def __init__(self, dataframe):
        self.df = dataframe
        self.column_names = list(self.df.columns)

    def auto_to_boolean(self):

        for column in self.column_names:
            unique = list(self.df[column].unique())
            if unique == [0, 1]:
                self.df[column] = self.df[column].astype('bool')
    
    def manual_to_boolean(self, column_name):

        if column_name not in self.column_names:
            print('Not a valid column name')
        else:
            num_unique = self.df[column_name].nunique()
            if num_unique < 3:
                self.df[column_name] = self.df[column_name].astype('bool')
            else:
                print('Data cannot be converted to boolean type')
    
    def manual_to_categorical(self, column_name):

        if column_name not in self.column_names:
            print('Not a valid column name')
        else:
            self.df[column_name] = self.df[column_name].astype('category')


class DataFrameInfo:

    def __init__(self, dataframe):
        self.df = dataframe
        self.column_names = list(self.df.columns)
        self.numeric_features = [col for col in self.df.columns[1:]
                                 if self.df[col].dtype == 'float64' or self.df[col].dtype == 'int64']
        self.categorical_features = [col for col in self.df.columns[2:] if col not in self.numeric_features]
    
    def describe_columns(self):
        # Describe all columns in the DataFrame to check their data types

        data_types = list(self.df.dtypes)       
        for column in self.column_names:
            print(column, self.df[column].dtype, self.df[column].nunique())
    
    def extract_stats(self):
        # Extract statistical values: median, standard deviation and mean from the columns and the DataFrame

        for column in self.column_names:
            if is_numeric_dtype(self.df[column]) == True:
                print(f'\n{column} \nmean : {self.df[column].mean()} \nmedian : {self.df[column].median()} \nstandard deviation : {self.df[column].std()}')
    
    def manual_extract_stats(self, column_name):

        if column_name not in self.column_names:
            print('Not a valid column name')
        else:
            if is_numeric_dtype(self.df[column_name]) == True:
                print(f'\n{column_name} \nmean : {self.df[column_name].mean()} \nmedian : {self.df[column_name].median()} \nstandard deviation : {self.df[column_name].std()}')
            else:
                print('Not a numerical column')

    def count_category_columns(self):
        # Count distinct values in categorical columns

        for column in self.column_names:
            if self.df[column].dtype == 'category':
                count = self.df[column].value_counts()
                print(count)
    
    def count_boolean_columns(self):
        # Count distinct values in boolean columns

        for column in self.column_names:
            if self.df[column].dtype == 'bool':
                count = self.df[column].value_counts()
                print(count)
  
    def print_shape(self):
        # Print out the shape of the DataFrame

        shape = self.df.shape
        print(f'This dataset has {shape[0]} rows and {shape[1]} columns')
    
    def percentage_null(self):
        # Generate a count/percentage count of NULL values in each column

        print("Percentage of missing values in each column:")
        print(self.df.isna().mean() * 100)
    
    def find_columns_with_missing_values(self):

        columns_with_null_values = []
        for column in self.column_names:
            if self.df[column].isnull().values.any() == True:
                columns_with_null_values.append(column)
        return columns_with_null_values
    
    def chi_squared_test(self, column_with_nulls, column):
        '''used for categorical data only'''
        temp_df = self.df[[column_with_nulls, column]]
        temp_df['missing_values'] = temp_df[column_with_nulls].isnull()
        contingency_table = pd.crosstab(temp_df['missing_values'], temp_df[column])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f'\nFrequency of NaNs in {column_with_nulls} as a function of {column}')
        print(f'Chi-square statistic = {chi2}')
        print(f'p-value = {p}')

class Plotter:

    def __init__(self, dataframe):
        self.df = dataframe
        self.column_names = list(self.df.columns)
        self.numeric_features = [col for col in self.df.columns[1:]
                                 if self.df[col].dtype == 'float64' or self.df[col].dtype == 'int64']
        self.categorical_features = [col for col in self.df.columns[2:] if col not in self.numeric_features]
    
    def histogram(self, column_name, bins = 50):
        self.df[column_name].hist(bins=bins)
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.title('Histogram')
        plt.show()
    
    def kde_plot(self, column_name):
        sns.histplot(data=self.df, x=column_name, kde=True)
        sns.despine()
        plt.xlabel('Count')
        plt.ylabel(column_name)
        plt.title('Probability Density Function Plot')
        plt.show()

    def multi_kde(self):
        sns.set_theme(font_scale=0.7)
        f = pd.melt(self.df, value_vars=self.numeric_features)
        g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False)
        g = g.map(sns.histplot, "value", kde=True)
        plt.show()
    
    def qq_plot(self, column):
        if column not in self.column_names:
            print('Not a valid column name')
        qq_plot = qqplot(self.df[column] , scale=1 ,line='q', fit=True)
        plt.show()
    
    def box_and_whiskers(self, column_name):
        fig = px.box(self.df, y=column_name,width=600, height=500)
        fig.show()
    
    def box_plot(self, column):
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=self.df, y=column, color='lightgreen', showfliers=True)
        plt.title(f'Box plot of {column}')
        plt.show()
    
    def violin(self, column_name):
        sns.violinplot(data=self.df, y=column_name)
        sns.despine
        plt.show()

    def countplot(self, column):
        sns.countplot(self.df, x=column)
        plt.show()
    
    def discrete_prob_dist(self, column_name):
        if column_name not in self.column_names:
            print('Not a valid column name')
        elif column_name in self.categorical_features:
            plt.rc("axes.spines", top=False, right=False)
            # Calculate value counts and convert to probabilities
            probs = self.df[column_name].value_counts(normalize=True)
            # Create bar plot
            dpd=sns.barplot(y=probs.index, x=probs.values, color='b')
            plt.xlabel('Values')
            plt.ylabel('Probability')
            plt.title('Discrete Probability Distribution')
            plt.show()   
    
    def cumulative_density(self, column_name):
        if column_name not in self.column_names:
            print('Not a valid column name')
        elif column_name in self.numeric_features:       
            sns.histplot(self.df[column_name], cumulative=True, stat='density', element="poly")
            plt.title('Cumulative Distribution Function (CDF)')
            plt.xlabel(column_name)
            plt.ylabel('Cumulative Probability')
            plt.show()
            
    def boxplot_by_category(self, column_1, column_2):
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=self.df, x=column_1, y=column_2, color='lightgreen', showfliers=True)
        plt.title(f'Box plot of {column_2} by {column_1}')
        plt.show()
        
    def scatter_plot(self, column_1, column_2):
        if column_1 not in self.column_names or column_2 not in self.column_names:
            print('Not a valid column name')
        else:
            sns.scatterplot(data=self.df, x=column_1, y=column_2)
            plt.xlabel(column_1)
            plt.ylabel(column_2)
            plt.title('ScatterPlot')
            plt.show()
    
    def scatter_with_lin_regression(self, column_1, column_2):
        sns.regplot(data=self.df, x=column_1, y=column_2)
        slope, intercept, r_value, p_value, std_err = linregress(self.df[column_1], self.df[column_2])
        print(linregress(self.df[column_1], self.df[column_2]))
        predicted_values = slope * self.df[column_1] + intercept
        residuals = self.df[column_2] - predicted_values
        mse = np.mean(residuals**2)
        print("slope of regression line: ", slope)
        print("Mean Squared Error (MSE) of the regression line: ", mse)
        plt.xlabel(column_1)
        plt.ylabel(column_2)
        plt.title('ScatterPlot with linear regression line')
        plt.show()
        return r_value

    def bar_chart(self, column_1, column_2):
        if column_1 not in self.column_names or column_2 not in self.column_names:
            print('Not a valid column name')
        elif column_1 in self.categorical_features:
            sns.barplot(data=self.df, y=column_1, x=column_2)
    
    def correlation_matrix(self):
        temp_df = self.df[self.numeric_features]
        corr = temp_df.corr()
        mask = np.zeros_like(corr, dtype=np.bool_)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    def full_correlation_matrix(self):
        temp_df = self.df[self.df.columns[3:]]
        corr = temp_df.corr()
        mask = np.zeros_like(corr, dtype=np.bool_)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm')
        plt.show()
    
    def pair_plot(self):
        temp_df = self.df[self.numeric_features]
        sns.pairplot(temp_df)        
        plt.show()    

    def skew_log(self, column):
        if column not in self.column_names:
            print('Not a valid column name')
        else:
            log_column = self.df[column].map(lambda i: np.log(i) if i > 0 else 0)
            t=sns.histplot(log_column,label="Skewness: %.2f"%(log_column.skew()), kde=True )
            t.legend()
            qq_plot = qqplot(log_column , scale=1 ,line='q', fit=True)
            plt.show()
    
    def skew_boxcox(self, column):
        if column not in self.column_names:
            print('Not a valid column name')
        else:
            boxcox_column = self.df[column]
            boxcox_column = stats.boxcox(boxcox_column)
            boxcox_column = pd.Series(boxcox_column[0])
            t=sns.histplot(boxcox_column,label="Skewness: %.2f"%(boxcox_column.skew()))
            t.legend()
            qq_plot = qqplot(boxcox_column, scale=1 ,line='q', fit=True)
            plt.show()
    
    def skew_yeojohnson(self, column):
        if column not in self.column_names:
            print('Not a valid column name')
        else:
            yeojohnson_column = self.df[column]
            yeojohnson_column = stats.yeojohnson(yeojohnson_column)
            yeojohnson_column = pd.Series(yeojohnson_column[0]) 
            t=sns.histplot(yeojohnson_column,label="Skewness: %.2f"%(yeojohnson_column.skew()))
            t.legend()
            qq_plot = qqplot(yeojohnson_column, scale=1 ,line='q', fit=True)
            plt.show()


    
class DataframeTransform:

    def __init__(self, dataframe):
        self.df = dataframe
        self.column_names = list(self.df.columns)
    
    def impute_with_mean(self, column):
        if column not in self.column_names:
            print('Not a valid column name')
        else:
            self.df[column] = self.df[column].fillna(self.df[column].mean())
    
    def impute_with_median(self, column):
        if column not in self.column_names:
            print('Not a valid column name')
        else:
            self.df[column] = self.df[column].fillna(self.df[column].median())
    
    def impute_with_correlated_column(self, column_1, column_2):
        self.df['difference'] = self.df[column_1] - self.df[column_2]
        self.df[column_1] = self.df[column_1].fillna(self.df[column_2] + self.df['difference'].mean())
        self.df[column_2] = self.df[column_2].fillna(self.df[column_1] - self.df['difference'].mean())
        self.df.drop('difference', axis=1, inplace=True)
        self.df.dropna(subset=[column_1, column_2], how='all', inplace=True)
    
    def remove_rows_with_null(self):
        self.df.dropna(how='any', inplace=True) 
        
    def correct_skew_log(self, column):
        if column not in self.column_names:
            print('Not a valid column name')
        else:
            self.df[column] = self.df[column].map(lambda i: np.log(i) if i > 0 else 0)
    
    def correct_skew_boxcox(self, column):
        if column not in self.column_names:
            print('Not a valid column name')
        else:
            boxcox_column = self.df[column]
            boxcox_column = stats.boxcox(boxcox_column)
            self.df[column] = pd.Series(boxcox_column[0])
    
    def correct_skew_yeojohnson(self, column):
        if column not in self.column_names:
            print('Not a valid column name')
        else:
            yeojohnson_column = self.df[column]
            yeojohnson_column = stats.yeojohnson(yeojohnson_column)
            self.df[column] = pd.Series(yeojohnson_column[0]) 
   
    def remove_outliers_IQR(self, column, multiplier = 1.5):
        if column not in self.column_names:
            print('Not a valid column name')
        else:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            self.df.drop(self.df[(self.df[column] < (Q1 - multiplier * IQR)) | (self.df[column] > (Q3 + multiplier * IQR))].index, inplace=True)

    def remove_outliers_z_score(self, column, z = 3):
        if column not in self.column_names:
            print('Not a valid column name')
        else:
            mean = self.df[column].mean()
            std = self.df[column].std()
            z_scores = (self.df[column] - mean) / std
            self.df['z_scores'] = z_scores
            self.df.drop(self.df[self.df['z_scores'] > 3].index, inplace=True)
            self.df.drop('z_scores', axis=1, inplace=True)
            

