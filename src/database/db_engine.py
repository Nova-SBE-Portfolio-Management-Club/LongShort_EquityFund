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

from sqlalchemy import select, delete, create_engine, func
from sqlalchemy.orm import sessionmaker, Session, aliased
from sqlalchemy.dialects.postgresql import insert

import yfinance as yf
import pandas as pd
import datetime as dt

from database import Company, Pair, PriceData, Country, Asset

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
        """
        Inserts a pair object into database
        """
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
        
    def insert_price_data(self, price_data: List[PriceData]) -> bool:
        """
        Inserts a list of PriceData objects into the database.
        If a conflict occurs on (date, ticker), it will do nothing.
        """
        session = self.sessionmaker()
        try:
            # Convert to dicts and strip SQLAlchemy internals
            if not price_data:
                return False

            # Convert to dicts using the model's to_dict() method
            price_data_dicts = [row.to_dict() for row in price_data]
            # Create insert statement with ON CONFLICT DO NOTHING
            insert_stmt = insert(PriceData).values(price_data_dicts).on_conflict_do_nothing(
                index_elements=["date", "ticker"]
            )

            session.execute(insert_stmt)
            session.commit()
            return True # Successfully inserted
        except Exception as e:
            session.rollback()
            return False # Insertion failed
        finally:
            session.close()
    
    def __insert_company_pricedata(self, company: str, n_years: int):
        """
        == DEPRECATED ==
        To Delete in Future Reviews
        
        Fetches historical stock data for a company using yfinance and saves it to database
        """
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
        """
        Returns the oldest and newest dates for which we have PriceData for a company
        """
        with Session(self.engine) as session:
            oldest = session.query(PriceData).filter_by(ticker=company).order_by(PriceData.date.asc()).first()
            newest = session.query(PriceData).filter_by(ticker=company).order_by(PriceData.date.desc()).first()

            return {"oldest": oldest.date, "newest": newest.date} if oldest and newest else None
        
    def get_company_country(self, company: Company) -> Country:
        """
        Retrives the Country Object of that Company
        """
        session = self.sessionmaker()
        try:
            country = session.query(Country).filter_by(name=company.country).first()
            return country
        except Exception as e:
            print(f"Error retrieving country for company {company.ticker}: {e}")
            return None
        finally:
            session.close()


        
    def insert_country(self, country: Country) -> None:
        """
        Inserts a Country object into the database.
        """
        session = self.sessionmaker()
        print(f'Inserting Country {country}')
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
            existing = (
                session.query(Country)
                .filter(Country.name == country.name)
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

    def get_all_assets(self) -> List[Asset]:
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
            
            
    def update_last_update_date(self, ticker: str, new_date: dt.datetime) -> None:
        """
        Update the Last Update Date for a Company (in the Assets table) by ticker.
        
        ticker      (str):  The ticker of the company.
        new_date    (datetime): The new date to set as the last update.
        """
        session = self.sessionmaker()
        try:
            # Query the Asset using the ticker
            asset = session.query(Asset).filter(Asset.ticker == ticker).first()
            
            # Check if the asset exists
            if asset:
                # Update the last_update_date
                asset.last_update_date = new_date
                
                # Commit the changes
                session.commit()
                print(f"Successfully updated last update date for ticker {ticker}")
            else:
                print(f"No asset found with ticker {ticker}")
        
        except Exception as e:
            # Rollback in case of an error
            session.rollback()
            print(f"Error updating last update date for ticker {ticker}: {e}")
        
        finally:
            # Close the session
            session.close()
            
    def set_invalid_asset(self, ticker: str) -> None:
        """
        Marks an asset as invalid by setting its last_update_date to None.
        
        ticker      (str):  The ticker of the asset to mark as invalid.
        """
        session = self.sessionmaker()
        try:
            # Query the Asset using the ticker
            asset = session.query(Asset).filter(Asset.ticker == ticker).first()
            if asset:
                asset.valid = False
                session.commit()
        except Exception as e:
            session.rollback()
        finally:
            session.close()
            
    def is_invalid_asset(self, ticker: str) -> bool:
        """
        Checks if an asset is marked as invalid.
        
        ticker      (str):  The ticker of the asset to check.
        
        Returns:
            bool: True if the asset is invalid, False otherwise.
        """
        session = self.sessionmaker()
        try:
            # Query the Asset using the ticker
            asset = session.query(Asset).filter(Asset.ticker == ticker).first()
            return not(asset.valid) if asset else True
        except Exception as e:
            return True
        finally:
            session.close()
            
    def delete_asset(self, asset: Asset) -> None:
        """
        Deletes the specified asset from the database
        First it check if it exists
        """
        pass
            
    def delete_assets(self, assets: List[Asset]) -> None:
        """
        Deletes the assets in the 'assets' list
        First it checks if the assets exists
        """
        pass
        
        
    def delete_all_assets(self) -> None:
        """
        Deletes all the assets stored in the database
        """
        pass
    
    def get_all_companies(self) -> List[Company]:
        """
        Returns a list of all Companies objects stored in the database
        """
        session = self.sessionmaker()
        try:
            assets = session.query(Company).all()
            return assets
        except Exception as e:
            print(f"Error retrieving all assets: {e}")
            return []
        finally:
            session.close()
            
    def get_all_companies_with_last_update_date(self) -> List[tuple]:
        """
        Returns a list of tuples (Company, Company_Last_Update_Date), for all companies
        NOTE: Very important function for UPDATE command
        """
        session = self.sessionmaker()  # Create session using the sessionmaker
        try:
            # Create an aliased Asset table to avoid table name conflicts
            asset_alias = aliased(Asset)

            # Query the Company and the aliased Asset table, getting the relevant fields (Company and Asset's last_update_date)
            companies = session.query(Company, asset_alias.last_update_date).join(asset_alias, Company.ticker == asset_alias.ticker).all()

            # Return the results as a list of tuples
            result = [(company, last_update_date) for company, last_update_date in companies]
            return result

        except Exception as e:
            print(f"Error retrieving companies with last update date: {e}")
            return []
        finally:
            session.close()
            
    def get_companies_by_attr(self, attr: str, attr_value: str, ) -> List[Company]:
        """
        Returns a list of all Companies objects stored in the database, by attribute, where:
        
        attr        (str):      the attribute to group the companies by (country, sector, industry, ...)
        attr_value  (str):      the attribute's value
        """
        session = self.sessionmaker()
        try:
            # Dynamically access the attribute and apply the condition
            filter_condition = getattr(Company, attr) == attr_value
            
            # Query the Company table and apply the filter condition
            companies = session.query(Company).filter(filter_condition).all()
            return companies
        except Exception as e:
            print(f"Error retrieving companies by {attr}: {e}")
            return []
        finally:
            session.close()
            
    
    def get_companies_by_attr_with_last_update_date(self, attr: str, attr_value: str) -> List[tuple]:
        """
        Returns a list of tuples (Company, Company_Last_Update_Date), by attribute, where:
        
        attr        (str):      the attribute to group the companies by (country, sector, industry, ...)
        attr_value  (str):      the attribute's value
        
        NOTE: Very important function for UPDATE command
        """
        session = self.sessionmaker()  # Create session using the sessionmaker
        try:
            # Dynamically access the attribute and apply the condition
            filter_condition = getattr(Company, attr) == attr_value
            
            # Create an aliased Asset table to avoid table name conflicts
            asset_alias = aliased(Asset)

            # Query the Company and the aliased Asset table, getting the relevant fields (Company and Asset's last_update_date)
            companies = session.query(Company, asset_alias.last_update_date).join(asset_alias, Company.ticker == asset_alias.ticker).filter(filter_condition).all()

            # Return the results as a list of tuples
            result = [(company, last_update_date) for company, last_update_date in companies]
            return result

        except Exception as e:
            print(f"Error retrieving companies by {attr}: {e}")
            return []
        finally:
            session.close()
            
            
    def count_companies_by_attr(self, attr: str) -> dict:
        """
        Count the number of Companies grouped by a specific attribute (e.g., country, sector, industry).
        
        attr(str): the attribute to group by (country, sector, industry, ...)
        
        Returns:
            dict: a dictionary with the attribute values as keys and counts as values.
        """
        session = self.sessionmaker()
        try:
            # Dynamically access the attribute to group by
            group_by_attr = getattr(Company, attr)
            
            # Query to count companies by the specified attribute and group by it
            result = session.query(group_by_attr, func.count(Company.ticker)) \
                            .group_by(group_by_attr) \
                            .all()
            
            # Convert the result into a dictionary
            count_dict = {row[0]: row[1] for row in result}
            
            return count_dict
            
        except Exception as e:
            print(f"Error counting companies by {attr}: {e}")
            return {}
        finally:
            session.close()
            
    def delete_company(self, cmpny: Company) -> None:
        """
        Deletes the specified Company from the database
        First it check if it exists
        """
        ticker = cmpny.ticker
        
        session = self.sessionmaker()
        try:
            # Prepare the delete statement
            delete_stmt = delete(Asset).where(Asset.ticker == ticker)

            # Execute the delete statement
            session.execute(delete_stmt)
            session.commit()  # Commit the transaction to the database
                
        except Exception as e:
            print(f"Error deleting asset {ticker}: {e}")
            return False
        finally:
            session.close()
            
            
    def delete_companies_from_country(self, country: Country) -> None:
        """
        Deletes all assets which are companies and have country == country.name
        """
        session = self.sessionmaker()
        try:
            # Prepare the delete statement
            delete_stmt = delete(Asset).where(Asset.asset_child_company.any(Company.country == country.name))

            # Execute the delete statement
            session.execute(delete_stmt)
            session.commit()  # Commit the transaction to the database
                
        except Exception as e:
            print(f"Error deleting asset {country.name}: {e}")
            return False
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
            asset_dicts     = []
            company_dicts   = []
            for _, row in companies_data.iterrows():
                
                # Adjust fields to match Company model.
                company = Company(
                    ticker=row["symbol"],
                    name=row['name'],
                    country=country.name,
                    isin=row["isin"],
                )
                
                # Check if company is from country
                if company.isin[:2] != country.isin_code:
                    # Company not from Country - SKIP IT
                    continue
                
                company_dicts.append(company.to_dict())
                asset_dicts.append({'ticker': row['symbol'], 'atype':'company'})
            
            # 3. Insert Assets in bulk
            insert_statement_assets = insert(Asset).values(asset_dicts).on_conflict_do_nothing(index_elements=["ticker"]) 
            session.execute(insert_statement_assets)
            session.commit()
            
            # 4. Insert Companies in bulk
            insert_statement_comps = insert(Company).values(company_dicts).on_conflict_do_nothing(index_elements=["ticker"]) 
            session.execute(insert_statement_comps)
            session.commit()
            
            #print(f"Inserted {len(company_dicts)} companies for country {country.name}.")
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
        
    def get_bloomberg_codes(self, option: str, country: str = None):
        """
        Fetch Bloomberg company codes based on user selection.
        """
        with Session(self.engine) as session:
            if option == "all":
                result = session.execute(
                    select(Company.ticker, Country.bloomberg_code)
                    .join(Country, Company.country == Country.name)  # Join condition
                ).all()
            elif option == "country":
                result = session.execute(
                    select(Company.ticker, Country.bloomberg_code)
                    .join(Country, Company.country == Country.name)
                    .where(Country.name == country)
                ).all()
            else:
                raise ValueError("Invalid option. Choose 'all' or 'country'.")
        return [f"{ticker} {country_code} Equity" for ticker, country_code in result]
    


# Usage

if __name__ == '__main__':  
    db1 = DB_Engine()
    db2 = DB_Engine()

    print(db1 is db2)
    
    db1.connect()
    
    
    c1 = Country('portugal','PT','PL','LS','EUR')
    c2 = Country('spain','ES','SM','MC','EUR')
    c3 = Country('united states','US','US',None,'USD')
    
    db1.delete_companies_from_country(c3)
    print('Done')
            
    comp1 = Company('AAPL','Apple','usa')
    comp2 = Company('NVDA','Nvidia','usa')
    comp3 = Company('GALP','GALP','portugal')
    
    #db1.insert_asset(comp1)
    #db1.insert_asset(comp2)
    #db1.insert_asset(comp3)

    
