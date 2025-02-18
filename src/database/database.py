"""
database.py
16 Fev 2025

This module defines the database connector for the application.

Dependencies:
    - python-dotenv
    - sqlalchemy
    - 
                                                         
"""

from dotenv import load_dotenv
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Company

load_dotenv()

class SingletonMeta(type):
    """
    This is a defintion of a Meta Class
    It will allow us to create only one instance of DB_Engine (see below)
    
    Do not worry if you can not fully grasp what it is doing
    
    If cursious, ask chatGPT, he willl explain it better than me
    """
    
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class DB_Engine(metaclass=SingletonMeta):
    """
    One of our best friends in the program
    
    DB_Engine is the bridge between our python code and the postgresql database
    Every query, insert, ... that we will do in the code will use this as the "middleman"
    
    Without this, it would not be possible to connect to the database and "talk" with it
    """
    
    
    def __init__(self):
        self.dsn = f"{os.getenv('DB_DRIVER')}://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        
        # The engine is responsible for managing the connection to the database.
        self.engine = None
        
        # The sessionmaker is a factory that generates session objects.
        # A session object is responsible for interacting with the database, 
        # executing queries, and committing transactions.
        self.sessionmaker = None
        
    def connect(self):
        """
        Descr
            Builds the engine and the session that will allow us to communicate with the database
        Output
            sqlalchemy engine
        """
        
        if self.engine is None:
            # Create the engine
            self.engine = create_engine(self.dsn)
            # Create sessionmaker bound to the engine
            self.sessionmaker = sessionmaker(bind=self.engine)
            print("Database connected.")
        return self.engine
        




# Usage

if __name__ == '__main__':  

    db1 = DB_Engine()
    db2 = DB_Engine()

    print(db1 is db2)
    
    db1.connect()