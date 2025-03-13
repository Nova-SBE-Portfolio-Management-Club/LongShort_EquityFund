"""
Functions to handle the UPGRADE command
"""


import yfinance as yf
import psycopg2
from datetime import datetime, timedelta

def get_database_connection():
    """Establish connection to the PostgreSQL database."""
    return psycopg2.connect(
        dbname="pmc_ls",
        user="postgres",
        password="postgres",
        host="localhost",
        port=5432
    )

def fetch_assets(option: str, filter_value: str = None):
    """Fetches assets based on the user selection (all, country, or sector)."""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    if option == "all":
        query = "SELECT ticker, last_update_date FROM assets;"
    elif option == "country":
        query = "SELECT ticker, last_update_date FROM companies WHERE country = %s;"
    elif option == "sector":
        query = "SELECT ticker, last_update_date FROM companies WHERE sector = %s;"
    else:
        raise ValueError("Invalid option. Choose 'all', 'country', or 'sector'.")
    
    cursor.execute(query, (filter_value,) if filter_value else ())
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        raise ValueError(f"No assets found for the selected {option} '{filter_value}'.")
    
    return rows

def fetch_price_data(ticker: str, last_update_date: datetime):
    """Fetches daily adjusted open and close prices from Yahoo Finance."""
    start_date = (last_update_date + timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            print(f"No new price data for {ticker}.")
            return []
        
        return [(index.strftime('%Y-%m-%d'), ticker, True, row['Open'], row['Close']) for index, row in data.iterrows()]
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return []

def update_database(price_data, ticker):
    """Updates the database with new price data and last update date."""
    if not price_data:
        return
    
    conn = get_database_connection()
    cursor = conn.cursor()
    
    insert_query = """
        INSERT INTO prices_data (date, ticker, open_status, price, close_price)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (date, ticker) DO NOTHING;
    """
    cursor.executemany(insert_query, price_data)
    
    last_date = price_data[-1][0]  # Get the last date from the price data
    update_query = "UPDATE assets SET last_update_date = %s WHERE ticker = %s;"
    cursor.execute(update_query, (last_date, ticker))
    
    conn.commit()
    conn.close()
    print(f"Updated {ticker} with {len(price_data)} new records.")

def handle_update_command():
    """Handles the update process based on user selection."""
    print("\n• Update (0)")
    print("  ○ All (0)")
    print("  ○ Country (1)")
    print("  ○ Sector (2)")
    
    choice = input("Choose an option (0: All, 1: Country, 2: Sector): ")
    
    if choice == "0":
        assets = fetch_assets("all")
    elif choice == "1":
        country = input("Enter the country name: ")
        assets = fetch_assets("country", country)
    elif choice == "2":
        sector = input("Enter the sector name: ")
        assets = fetch_assets("sector", sector)
    else:
        print("Invalid choice. Exiting.")
        return
    
    for ticker, last_update_date in assets:
        if last_update_date is None:
            last_update_date = datetime(2000, 1, 1)  # Default if never updated
        
        price_data = fetch_price_data(ticker, last_update_date)
        update_database(price_data, ticker)

if __name__ == "__main__":
    handle_update_command()