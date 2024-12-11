import yaml
from sqlalchemy import create_engine # type: ignore
from sqlalchemy import inspect # type: ignore
import pandas as pd
from pandas.api.types import is_numeric_dtype


def load_credentials():
    with open('credentials.yaml', 'r') as file:
        credentials = yaml.safe_load(file)
    return credentials


class RDSDatabaseConnector:

    def __init__(self, credentials = load_credentials()):
        self.host = credentials['RDS_HOST']
        self.port = credentials['RDS_PORT']
        self.database = credentials['RDS_DATABASE']
        self.user = credentials['RDS_USER']
        self.password = credentials['RDS_PASSWORD']
    
    def print_credentials(self):
        print(self.host, self.port, self.database, self.user, self.password)
    
    def initilise_engine(self):
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}")
        return engine
    
    def extract_data(self):
        engine = self.initilise_engine()
        engine.connect()
        inspector = inspect(engine)
        table_name = inspector.get_table_names()
        failure_data = pd.read_sql_table(table_name[0], engine)
        return failure_data

    def save_data(self):
        dataframe = self.extract_data()
        dataframe.to_csv('the_manufacturing_process_dataframe.csv', index=False)
    
    def save_edited_dataframe(self, dataframe):
        dataframe.to_csv('new_manufactoring_process_dataframe.csv', index=False)

