# Exploratory data analysis---the manufacturing process

You have been tasked with to optimising a manufacturing machine process for a large industrial company. The manufacturing process is a crucial aspect of the company's operations, and the optimisation of the process has a significant impact on the organisation's overall efficiency and profitability.


## Installation instructions

```git clone https://github.com/echo-ex-altiora/exploratory-data-analysis---the-manufacturing-process640.git```

Libraries you need in your python environment
conda install jupyter
conda install -n base ipykernel --update-deps --force-reinstall
pip install yaml
pip install sqlalchemy 
pip install pandas
pip install matplotlib
pip install seaborn
pip install numpy
pip install scipy
conda install -c conda-forge statsmodels
pip import statsmodels.graphics.gofplots


## Usage instructions

This repository is used to for exploratory data analysis of the manufactoring process.
To access this repository, clone the repository to your computer and open in any software that supports python. 

If you are not already in the correct folder, move to it using ```cd TheManufactoringProcess```.
In order to load the database, you must have the credentials stored as a YAML file in the same folder
Run files from terminal using `python {filename}`


## File structure of the project

There are seven python files in the main folder named TheManufactoringProcess.

The file db_utils.py contains two functions and one class used to load credentials, connect to a database, extract and save a dataframe as a csv file to your local computer and load a csv file as a pandas dataframe. It does not need to be run at any point.

The file EDA_classes.py contains the classes DataTransform, DataFrameInfo, Plotter and DataFrame Transform. It also contains a short piece of code to set up how plots will look. This file does not need to be run at any point as the classes are imported into the other python files as needed.

There are five python files for each major step in the exploratory data analysis process:
- M3T1-3_missing_data.py is used first to convert columns to correct format and remove and impute missing values
- M3T4_skewed_columns.py is used to identify and correct skewed data
- M3T5_remove_outliers.py is used to remove outliers in the data
- M3T6_Drop_correlated_columns.py is used to drop overly correlated columns
- M4_analysis_and_visualisation.py is used last to draw deeper insights from the data

There are two python notebooks:
- Exploratory_Data_Analysis.ipynb
- Analysis_and_Visualisation.ipynb
These contain the same code as the python files but are formatted to be easier to read

There is one text file:
- terminology.txt
This contains information on what the terms from the database mean.

There are two csv files:
- the_manufactoring_process_dataframe.csv contains the original, unedited dataframe
- new_manufactoring_process_dataframe.csv contains the an edited copy of the dataframe

Finally, there is this README.md file and an .gitignore file containing credentials.yaml file.


## License information
No license information.
