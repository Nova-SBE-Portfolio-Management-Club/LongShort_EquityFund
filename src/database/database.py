"""
database.py
16 Fev 2025

This module defines the database connector for the application.

Dependencies:
    - python-dotenv
    - sqlalchemy
    - yfinance
    - pandas
    -
                                                         
"""

from dotenv import load_dotenv
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

import yfinance as yf
import pandas as pd

from models import Company, Pair, PriceData

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
    
    def insert_pair(self, pair: Pair):
        # Inserts a pair object into database
        with Session(self.engine) as session:
            session.add(pair)
            session.commit()

    def exists_pair(self, pair: Pair) -> bool:
        # Checks if a Pair object already exists in database
        with Session(self.engine) as session:
            session.query(Pair).filter_by(tickerA=pair.tickerA, tickerB=pair.tickerB).first() is not None

    def get_all_pairs(self) -> list[Pair]:
        # Returns list of all Pair objects stored in database
        with Session(self.engine) as session:
            return session.query(Pair).all()
        
    def exists_pricedata(self, pricedata: PriceData) -> bool:
        # Checks if a PriceData object already exists in database
        with Session(self.engine) as session:
            return session.query(PriceData).filter_by(
                ticker=pricedata.ticker, price=pricedata.price
            ).first() is not None
        
    def get_all_pricedata(self, company: str, to_pandas: bool = False):
        # Returns all PriceData for a company as a list or pandas DataFrame
        with Session(self.engine) as session:
            data = session.query(PriceData).filter_by(ticker=company).all()
            if to_pandas:
                return pd.DataFrame([{"ticker": d.ticker, "price": d.price, "date": d.date} for d in data])
            return data
    
    def insert_company_pricedata(self, company: str, n_years: int):
        # Fetches historical stock data for a company using yfinance and saves it to database
        stock = yf.Ticker(company)
        df = stock.history(period=f"{n_years}y")

        if df.empty:
            print(f"No data found for {company}.")
            return

        with Session(self.engine) as session:
            price_data_objects = [
                PriceData(ticker=company, date=index, price=row["Close"]) for index, row in df.iterrows()
            ]
            session.bulk_save_objects(price_data_objects)
            session.commit()
            
    def get_company_pricedata(self, company: str, start: str, end: str):
        # Retrieves PriceData for a company within a specified date range
        with Session(self.engine) as session:
            data = session.query(PriceData).filter(
                PriceData.ticker == company,
                PriceData.date >= start,
                PriceData.date <= end
            ).all()
            if not data:
                print(f"Warning: No data found for {company} between {start} and {end}.")
            return data
        
    def get_company_pricedata_period(self, company: str):
        # Returns the oldest and newest dates for which we have PriceData for a company
        with Session(self.engine) as session:
            oldest = session.query(PriceData).filter_by(ticker=company).order_by(PriceData.date.asc()).first()
            newest = session.query(PriceData).filter_by(ticker=company).order_by(PriceData.date.desc()).first()

            return {"oldest": oldest.date, "newest": newest.date} if oldest and newest else None



        




# Usage

if __name__ == '__main__':  

    db1 = DB_Engine()
    db2 = DB_Engine()

    print(db1 is db2)
    
    db1.connect()