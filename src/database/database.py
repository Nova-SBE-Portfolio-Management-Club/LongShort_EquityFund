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

from models import Company, Pair, PriceData, Country, Asset

from typing import List

load_dotenv()

class SingletonMeta(type):
    """
    This is a definition of a Meta Class.
    It will allow us to create only one instance of DB_Engine (see below).

    Do not worry if you cannot fully grasp what it is doing.
    If curious, ask chatGPT, he will explain it better than me.
    """
    
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class DB_Engine(metaclass=SingletonMeta):
    """
    DB_Engine is the bridge between our python code and the postgresql database.
    Every query, insert, ... that we will do in the code will use this as the "middleman".
    
    Without this, it would not be possible to connect to the database and "talk" with it.
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
        Builds the engine and the session that will allow us to communicate with the database.
        Output: sqlalchemy engine.
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

    def get_all_pairs(self) -> List[Pair]:
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



        
    def insert_country(self, country: Country) -> None:
        """
        Inserts a Country object into the database.
        """
        session = self.sessionmaker()
        try:
            session.add(country)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error inserting country {country.name}: {e}")
        finally:
            session.close()

    def exists_country(self, country: Country) -> bool:
        """
        Checks if the given Country object already exists in the database
        by comparing its unique attributes (e.g., name or ISIN).
        Returns True if it exists, False otherwise.
        """
        session = self.sessionmaker()
        try:
            # Example: check by 'name' or 'isin_code'
            existing = (
                session.query(Country)
                .filter_by(name=country.name, isin_code=country.isin_code)
                .first()
            )
            return existing is not None
        except Exception as e:
            print(f"Error checking existence of country {country.name}: {e}")
            return False
        finally:
            session.close()

    def get_all_countries(self) -> List[Country]:
        """
        Returns a list of all Country objects stored in the database.
        """
        session = self.sessionmaker()
        try:
            countries = session.query(Country).all()
            return countries
        except Exception as e:
            print(f"Error retrieving all countries: {e}")
            return []
        finally:
            session.close()

    def insert_asset(self, asset) -> None:
        """
        Inserts an Asset (or Company) into the database.
        """
        session = self.sessionmaker()
        try:
            session.add(asset)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error inserting asset {asset.ticker}: {e}")
        finally:
            session.close()
    
    def exists_asset(self, asset) -> bool:
        """
        Checks if the given Asset/Company object already exists in the database
        by comparing its unique attributes (e.g., ticker, ISIN).
        Returns True if it exists, False otherwise.
        """
        session = self.sessionmaker()
        try:
            # Example filter: check by ticker or ISIN code
            existing = (
                session.query(Asset)
                .filter_by(ticker=asset.ticker)
                .first()
            )
            return existing is not None
        except Exception as e:
            print(f"Error checking existence of asset {asset.ticker}: {e}")
            return False
        finally:
            session.close()

    def get_all_assets(self) -> list:
        """
        Returns a list of all Asset objects stored in the database (including Companies).
        """
        session = self.sessionmaker()
        try:
            assets = session.query(Asset).all()
            return assets
        except Exception as e:
            print(f"Error retrieving all assets: {e}")
            return []
        finally:
            session.close()

    def insert_all_country_companies(self, country: Country) -> None:
        """
        Retrieves all companies for the given country (using investpy or other library),
        creates Company objects, and inserts them into the database in bulk.
        """
        import investpy
        
        session = self.sessionmaker()
        try:
            # 1. Fetch all companies from investpy
            companies_data = investpy.stocks.get_stocks(country=country.name)
            
            # 2. Create a list of Company objects
            company_objects = []
            for _, row in companies_data.iterrows():
                # Adjust fields to match your Company model.
                print(row['symbol'])
                company = Company(
                    ticker=row["symbol"],
                    name=row['name'],
                    country=country.name,
                    isin=row["isin"],
                )
                company_objects.append(company)
            
            # 3. Insert them in bulk
            session.add_all(company_objects)
            session.commit()
            
            print(f"Inserted {len(company_objects)} companies for country {country.name}.")
        except Exception as e:
            session.rollback()
            print(f"Error inserting all companies for country {country.name}: {e}")
        finally:
            session.close()

    def get_all_country_companies(self, country) -> list:
        """
        Returns a list of all Company objects for the given Country that are stored in the DB.
        """
        session = self.sessionmaker()
        try:
            # Assuming you have a relationship or a foreign key "country_id" in Company.
            companies = (
                session.query(Company)
                .filter_by(country_id=country.id)
                .all()
            )
            return companies
        except Exception as e:
            print(f"Error retrieving companies for country {country.name}: {e}")
            return []
        finally:
            session.close()


# Usage

if __name__ == '__main__':  
    db1 = DB_Engine()
    db2 = DB_Engine()

    print(db1 is db2)
    
    db1.connect()
    
    c1 = Country('portugal','PT','PL','LS','EUR')
    
    print('->',db1.get_all_assets())
    #print('->',db1.insert_all_country_companies(c1))
    
    
