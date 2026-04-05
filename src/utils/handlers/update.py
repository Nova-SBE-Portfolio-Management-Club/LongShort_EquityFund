"""
Functions to handle the UPGRADE command
"""

import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm  # Progress bar
from time import perf_counter as pc

from typing import List

from database import DB_Engine, PriceData, Company, Asset, Futures, MiscellData

from sqlalchemy import update



def fetch_assets(db: DB_Engine, option: str, filter_value: str = None):
    """Fetches assets based on the user selection (all_comps, country, sector, futures, miscellaneous data )."""
    
    if option == "all_comps":
        assets = db.get_all_companies_with_last_update_date()
    elif option == "country":
        assets = db.get_companies_by_attr_with_last_update_date("country", filter_value)
    elif option == "sector":
        assets = db.get_companies_by_attr_with_last_update_date("sector", filter_value)
    elif option == "futures":
        assets = db.get_all_futures_with_last_update_date()
    elif option == "miscell":
        assets = db.get_all_misc_data()

    else:
        raise ValueError("Invalid option.")
    
    return assets

def fetch_company_price_data(db: DB_Engine, company: Company, last_update_date: datetime):
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
        yfin_ticker = company.get_yfin_ticker(db)
        t0 = pc()
        data = yf.download(yfin_ticker, start=start_date, end=end_date, interval='1d', progress=False, auto_adjust=True)
        t1 = pc()
        print(f"Time taken to fetch data for {company.ticker}: {t1 - t0:.2f} seconds")
        if data.empty:
            
            if is_none:
                db.set_invalid_asset(company.ticker)
            
            return []
        price_data = [
            PriceData(dt, company.ticker, open_, close_)
            for dt, open_, close_ in zip(
                data.index.to_pydatetime(),
                data[('Open', yfin_ticker)].values,
                data[('Close', yfin_ticker)].values
            )
        ]
        return price_data
            
    except Exception as e:
        print(f"Error fetching data for {company.ticker}: {e}")
        return []


def check_valid_company(db: DB_Engine,company: Company, last_update_date: datetime ) -> bool:
    """Uses 3 month average volume to check validity."""
    
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
        yfin_ticker = company.get_yfin_ticker(db)
        t0 = pc()
        data = yf.download(yfin_ticker, period='3mo', interval='1d', progress=False, auto_adjust=True)
        avg_volume = data['Volume'].mean()  
        if avg_volume < 500000:  # Example threshold
            db.set_invalid_asset(company.ticker)
            return []
        t1 = pc()
        print(f"Time taken to fetch data for {company.ticker}: {t1 - t0:.2f} seconds")
        if data.empty:
            
            if is_none:
                db.set_invalid_asset(company.ticker)
            
            return []
    except Exception as e:
        print(f"Error fetching data for {company.ticker}: {e}")
        return []
def fetch_futures_price_data(db: DB_Engine, future: Futures, last_update_date: datetime):
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
        yfin_ticker = future.ticker
        t0 = pc()
        data = yf.download(yfin_ticker, start=start_date, end=end_date, interval='1d', progress=False, auto_adjust=True)
        t1 = pc()
        print(f"Time taken to fetch data for {future.ticker}: {t1 - t0:.2f} seconds")
        if data.empty:
            
            if is_none:
                db.set_invalid_asset(future.ticker)
            
            return []
        price_data = [
            PriceData(dt, future.ticker, open_, close_)
            for dt, open_, close_ in zip(
                data.index.to_pydatetime(),
                data[('Open', yfin_ticker)].values,
                data[('Close', yfin_ticker)].values
            )
        ]
        return price_data
            
    except Exception as e:
        print(f"Error fetching data for {future.ticker}: {e}")
        return []

def fetch_miscell_data(db: DB_Engine, misc: MiscellData, last_update_date: datetime):
    """Calculates the current moon phase."""

    if last_update_date is None:
        is_none = True
        last_update_date = datetime(1998, 1, 1)
    else:
        is_none = False
    
    
    try:
        # Calculate the current moon phase
        current_date = datetime.today()
        moon_phase = get_moon_phase(current_date)
        
        return [MiscellData(current_date, moon_phase)]
    except Exception as e:
        print(f"Error fetching data for {misc.moonphase}: {e}")
        return []

    
def get_julian_date(date: datetime) :
    """Calculate the Julian Date for a given date."""
    a = int((14 - datetime.month) / 12)
    y = date.year + 4800 - a
    m = date.month + 12 * a - 3
    jd = date.day + int((153 * m + 2) / 5) + 365 * y + int(y / 4) - int(y / 100) + int(y / 400) - 32045
    jd = jd + (date.hour - 12) / 24 + date.minute / 1440 + date.second / 86400
    return jd

def get_moon_phase(date: datetime) -> str:
    """Determine the moon phase based on the given date."""
    # Get the Julian Date for the reference New Moon and the current date
    SYNODIC_MONTH = 29.53059  
    REFERENCE_NEW_MOON_DATE = datetime(2025, 2, 28)  

    reference_jd = get_julian_date(REFERENCE_NEW_MOON_DATE)
    current_jd = get_julian_date(date)
    

    days_since_new_moon = current_jd - reference_jd
    lunar_cycle = days_since_new_moon % SYNODIC_MONTH
    return round(lunar_cycle)
    
    




    
    

def update_company_database(db: DB_Engine, price_data: List[PriceData], company: Company):
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
    print("  ○ Futures (3)")
    print("  ○ Miscellaneous Data (4)")
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
    elif choice == "3":
        assets = fetch_assets(db, "futures")
    elif choice == "4":
        assets = fetch_assets(db, "miscell")
   
    else:
        print("Invalid choice. Exiting.")
        return
    
    with db.sessionmaker() as session:
        all_price_data = []
        last_update_dict = {}  # Dictionary to store the last update date per company
        total_assets = len(assets)

        # Initialize progress bar
        if choice in ["0", "1", "2"]:  # Companies
            with tqdm(total=total_assets, desc="Fetching price data", unit="company") as pbar:
             for idx, (company, last_update_date) in enumerate(assets, start=1):
                to = pc()

                if db.is_invalid_asset(company.ticker):
                   continue

                # Fetch price data
                price_data = fetch_company_price_data(db, company, last_update_date)

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

        elif choice == "3":  # Futures
            with tqdm(total=total_assets, desc="Fetching futures data", unit="future") as pbar:
             for idx, (future, last_update_date) in enumerate(assets, start=1):
                to = pc()

                if db.is_invalid_asset(future.ticker):
                  continue

                # Fetch futures price data
                price_data = fetch_futures_price_data(db, future, last_update_date)

                t1 = pc()
                print(f"Time taken to call fetch_futures_price_data {future.ticker}: {t1 - to:.2f} seconds")

                # Update progress bar
                pbar.update(1)

                if not price_data:
                 continue  # Skip if no price data

                # Get last date of price data
                last_date = price_data[-1].date
                last_update_dict[future.ticker] = last_date

                # Collect price data
                all_price_data.extend(price_data)

                # Save data every `batches` futures
                if idx % batches == 0 or idx == total_assets:
                 t2 = pc()
                if all_price_data:
                    status = db.insert_price_data(all_price_data)
                    all_price_data.clear()  # Free memory

                if last_update_dict:
                    if status:
                    # Only update, if the insert was successful
                     for ticker, date in last_update_dict.items():
                        stmt = update(Futures).where(Futures.ticker == ticker).values(last_update_date=date)
                        session.execute(stmt)
                    last_update_dict.clear()  # Free memory

                # Commit the batch
                session.commit()

                t3 = pc()
                print(f"Time taken to save data for {idx} futures: {t3 - t2:.2f} seconds")

        elif choice == "4":  # Miscellaneous Data
            with tqdm(total=total_assets, desc="Fetching miscellaneous data", unit="misc") as pbar:
             for idx, (misc, last_update_date) in enumerate(assets, start=1):
                to = pc()

                # Fetch miscellaneous data
                misc_data = fetch_miscell_data(db, misc, last_update_date)

                t1 = pc()
                print(f"Time taken to call fetch_miscell_data: {t1 - to:.2f} seconds")

                # Update progress bar
                pbar.update(1)

                if not misc_data:
                  continue  # Skip if no data

                # Collect miscellaneous data
                all_price_data.extend(misc_data)

                # Save data every `batches` items
                if idx % batches == 0 or idx == total_assets:
                  t2 = pc()
                if all_price_data:
                    status = db.insert_miscell_data(all_price_data)
                    all_price_data.clear()  # Free memory

                # Commit the batch
                session.commit()

                t3 = pc()
                print(f"Time taken to save data for {idx} miscellaneous items: {t3 - t2:.2f} seconds")
        with tqdm(total=total_assets, desc="Fetching price data", unit="company") as pbar:
            for idx, (company, last_update_date) in enumerate(assets, start=1):
                
                to = pc()
                
                if db.is_invalid_asset(company.ticker):
                    continue

                # Fetch price data
                price_data = fetch_company_price_data(db, company, last_update_date)

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