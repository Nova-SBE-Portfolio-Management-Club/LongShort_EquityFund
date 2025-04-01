"""
Functions to handle the UPGRADE command
"""

import sys
import os
import time
import yfinance as yf
from datetime import datetime, timedelta

from typing import List

sys.path.append(os.path.abspath("src/database"))

from database.db_engine import DB_Engine
from database.models    import PriceData, Company


def fetch_assets(db: DB_Engine, option: str, filter_value: str = None):
    """Fetches assets based on the user selection (all, country, or sector)."""
    
    if option == "all":
        assets = db.get_all_assets()
    elif option == "country":
        assets = db.get_companies_by_attr_with_last_update_date("country", filter_value)
        print("AAAAAAAAAAAAAAAAAAAAAAA")
    elif option == "sector":
        assets = db.get_companies_by_attr_with_last_update_date("sector", filter_value)
    else:
        raise ValueError("Invalid option. Choose 'all', 'country', or 'sector'.")
    
    return assets

def fetch_price_data(company: Company, last_update_date: datetime):
    """Fetches daily adjusted open and close prices from Yahoo Finance."""
    
    print(111)
    start_date = (last_update_date + timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        print(company.get_yfin_ticker(), start_date, end_date)
        data = yf.download(company.get_yfin_ticker(), start=start_date, end=end_date, interval='1d', progress=False, auto_adjust=True)
        if data.empty:
            print(f"No new price data for {company.ticker}.")
            return []
        
        return [PriceData(index.to_pydatetime(),company.ticker, 1, row['Open'][0]) for index, row in data.iterrows()] +\
            [PriceData(index.to_pydatetime(),company.ticker, 0, row['Close'][0]) for index, row in data.iterrows()]
            
    except Exception as e:
        print(f"Error fetching data for {company.ticker}: {e}")
        return []

def update_database(db: DB_Engine, price_data: List[PriceData], company: Company):
    """Updates the database with new price data and last update date."""
    if not price_data:
        return
    
    last_date = price_data[-1].date  # Get the last date from the price data
    
    db.insert_price_data(price_data)
    
    db.update_last_update_date(company.ticker, last_date)
    

def handle_update_command(db: DB_Engine):
    """Handles the update process based on user selection."""
    print("\n• Update (0)")
    print("  ○ All (0)")
    print("  ○ Country (1)")
    print("  ○ Sector (2)")
    # TODO: Add the option to Update Non-Company Assets (Futures)
    
    choice = input("Choose an option (0: All, 1: Country, 2: Sector): ")
    
    if choice == "0":
        assets = fetch_assets(db, "all")
    elif choice == "1":
        country = input("Enter the country name: ")
        assets = fetch_assets(db, "country", country)
    elif choice == "2":
        sector = input("Enter the sector name: ")
        assets = fetch_assets(db, "sector", sector)
    else:
        print("Invalid choice. Exiting.")
        return
    
    for company, last_update_date in assets:
        if last_update_date is None:
            last_update_date = datetime(1998, 1, 1)  # Default if never updated
        
        price_data = fetch_price_data(company, last_update_date)
        update_database(db, price_data, company)

if __name__ == "__main__":
    db = DB_Engine()
    db.connect()
    handle_update_command(db)