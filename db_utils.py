import yaml
from sqlalchemy import create_engine # type: ignore
from sqlalchemy import inspect # type: ignore
import pandas as pd

def load_credentials():
    with open('credentials.yaml', 'r') as file:
        credentials = yaml.safe_load(file)
    return credentials

class RDSDatabaseConnector:
    '''
    RDS Database Connecter

    Parameters:
    ----------
    credentials : dictionary
        credentials for Database. can be manually entered as dictionary or defaults to function load_credentials
    
    Attributes:
    ----------
    host : str
    port : int
    database : str
    user : str
    password : str

    Methods:
    -------
    initialise_engine
        Sets up engine
    extract_data
        Connects to engine to database and returns data as a pandas dataframe
    save_data
        Saves data as a CSV file
    save_edited_data(database)
        Saves the inputted database as a CSV file 
    '''

    def __init__(self, credentials = load_credentials()):
        self.host = credentials['RDS_HOST']
        self.port = credentials['RDS_PORT']
        self.database = credentials['RDS_DATABASE']
        self.user = credentials['RDS_USER']
        self.password = credentials['RDS_PASSWORD']
    
    def initilise_engine(self):
        '''
        Initialises a SQLAlchemy engine from the credentials provided to class.
        '''
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}")
        return engine
    
    def extract_data(self):
        '''
        Extracts data from the RDS database and returns it as a Pandas DataFrame. 
        The data is stored in a table called failure_data.
        '''
        engine = self.initilise_engine()
        engine.connect()
        inspector = inspect(engine)
        table_name = inspector.get_table_names()
        failure_data = pd.read_sql_table(table_name[0], engine)
        return failure_data

    def save_data(self):
        '''
        Saves the data as a CSV file to local machine.
        '''
        dataframe = self.extract_data()
        dataframe.to_csv('the_manufacturing_process_dataframe.csv', index=False)
    
    def save_edited_dataframe(self, dataframe):
        '''
        Saves the data as a CSV file to local machine.
        Parameters:
        ----------
        dataframe : pandas dataframe
        '''
        dataframe.to_csv('new_manufactoring_process_dataframe.csv', index=False)

def load_dataframe():
    dataframe = pd.read_csv('the_manufacturing_process_dataframe.csv')
    return dataframe