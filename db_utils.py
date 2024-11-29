import yaml
from sqlalchemy import create_engine
from sqlalchemy import inspect
import pandas as pd


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

load = RDSDatabaseConnector()

# load.save_data()

failure_df = load.extract_data()

print(failure_df.head())

shape = failure_df.shape
print(f'This dataset has {shape[0]} rows and {shape[1]} columns')

failure_df.info()