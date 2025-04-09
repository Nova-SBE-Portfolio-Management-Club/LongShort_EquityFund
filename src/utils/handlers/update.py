"""
Functions to handle the UPGRADE command
"""

import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm  # Progress bar
from time import perf_counter as pc

from typing import List

from database import DB_Engine, PriceData, Company, Asset

from sqlalchemy import update



def fetch_assets(db: DB_Engine, option: str, filter_value: str = None):
    """Fetches assets based on the user selection (all_comps, country, or sector)."""
    
    if option == "all_comps":
        assets = db.get_all_companies_with_last_update_date()
    elif option == "country":
        assets = db.get_companies_by_attr_with_last_update_date("country", filter_value)
    elif option == "sector":
        assets = db.get_companies_by_attr_with_last_update_date("sector", filter_value)
    else:
        raise ValueError("Invalid option.")
    
    return assets

def fetch_price_data(db: DB_Engine, company: Company, last_update_date: datetime):
    """Fetches daily adjusted open and close prices from Yahoo Finance."""
    
    if last_update_date is None:
        is_none = True
        last_update_date = datetime(1998, 1, 1)
    else:
        is_none = False
    
    start_date = (last_update_date + timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    if start_date == end_date:
        return []
    
    try:
        t0 = pc()
        data = yf.download(company.get_yfin_ticker(db), start=start_date, end=end_date, interval='1d', progress=False, auto_adjust=True)
        t1 = pc()
        print(f"Time taken to fetch data for {company.ticker}: {t1 - t0:.2f} seconds")
        if data.empty:
            
            if is_none:
                db.set_invalid_asset(company.ticker)
            
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
    

def handle_update_command(db: DB_Engine, batches = 20):
    """Handles the update process based on user selection."""
    print("\n• Update (0)")
    print("  ○ All - Companies (0)")
    print("  ○ Country (1)")
    print("  ○ Sector (2)")
    # TODO: Add the option to Update Non-Company Assets (Futures)
    
    choice = input("Choose an option (0: All - Companies, 1: Country, 2: Sector): ")
    
    if choice == "0":
        assets = fetch_assets(db, "all_comps")
    elif choice == "1":
        country = input("Enter the country name: ")
        assets = fetch_assets(db, "country", country)
    elif choice == "2":
        sector = input("Enter the sector name: ")
        assets = fetch_assets(db, "sector", sector)
    else:
        print("Invalid choice. Exiting.")
        return
    
    with db.sessionmaker() as session:
        all_price_data = []
        last_update_dict = {}  # Dictionary to store the last update date per company
        total_assets = len(assets)

        # Initialize progress bar
        with tqdm(total=total_assets, desc="Fetching price data", unit="company") as pbar:
            for idx, (company, last_update_date) in enumerate(assets, start=1):
                
                to = pc()
                
                if db.is_invalid_asset(company.ticker):
                    continue

                # Fetch price data
                price_data = fetch_price_data(db, company, last_update_date)
                
                t1 = pc()
                print(f"Time taken to call fetch_price_data {company.ticker}: {t1 - to:.2f} seconds")
                
                # Update progress bar
                pbar.update(1)

                if not price_data:
                    continue  # Skip if no price data

                # Get last date of price data
                last_date = price_data[-1].date
                last_update_dict[company.ticker] = last_date

                # Collect price data
                all_price_data.extend(price_data)

                # Save data every `batches` companies
                if idx % batches == 0 or idx == total_assets:
                    t2 = pc()
                    if all_price_data:
                        status = db.insert_price_data(all_price_data)
                        all_price_data.clear()  # Free memory
                    
                    if last_update_dict:
                        if status:
                            # Only update, if the insert was successful
                            for ticker, date in last_update_dict.items():
                                stmt = update(Asset).where(Asset.ticker == ticker).values(last_update_date=date)
                                session.execute(stmt)  
                        last_update_dict.clear()  # Free memory

                    # Commit the batch
                    session.commit()
                    
                    t3 = pc()
                    print(f"Time taken to save data for {idx} companies: {t3 - t2:.2f} seconds")
    
    #for company, last_update_date in assets:
    #    if last_update_date is None:
    #        last_update_date = datetime(1998, 1, 1)  # Default if never updated
    #    
    #    price_data = fetch_price_data(db, company, last_update_date)
    #    update_database(db, price_data, company)

if __name__ == "__main__":
    db = DB_Engine()
    db.connect()
    handle_update_command(db)