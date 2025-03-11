"""
Functions to handle the BLOOMBERG command
"""


def handle_bloomberg_command():
    pass


from ..messages             import *
import pandas as pd
import psycopg2  # PostgreSQL database connector
import os

def get_database_connection():
    """Establish connection to the PostgreSQL database."""
    return psycopg2.connect(
        dbname="your_database",
        user="your_username",
        password="your_password",
        host="your_host",
        port="your_port"
    )

def fetch_bloomberg_codes(option: str, country: str = None):
    """Fetches Bloomberg company codes based on user selection."""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    if option == "all":
        query = "SELECT ticker, country_code FROM companies;"
        filename = "bl_all.csv"
    elif option == "country":
        query = "SELECT ticker, country_code FROM companies WHERE country_code = %s;"
        filename = f"bl_{country}.csv"
    else:
        raise ValueError("Invalid option. Choose 'all' or 'country'.")
    
    cursor.execute(query, (country,) if country else ())
    rows = cursor.fetchall()
    conn.close()
    
    # Format Bloomberg codes
    companies = [f"{ticker} {country_code} Equity" for ticker, country_code in rows]
    return companies, filename

def save_to_csv(companies, filename):
    """Saves the fetched company codes to a CSV file."""
    export_path = "src/bloomberg/exports/"
    os.makedirs(export_path, exist_ok=True)  # Ensure directory exists
    
    df = pd.DataFrame(companies, columns=["company"])
    df.to_csv(os.path.join(export_path, filename), index=False)
    print(f"Exported {len(df)} companies to {filename}")

def main():
    print("\n• Export (0)")
    print("  ○ All (0)")
    print("  ○ Country (1)")
    print("• Import (1)")
    
    choice = input("Export (0)\n  All (0)\n  Country (1)\nImport (1)\n>> ")
    
    if choice == "0":
        export_choice = input("\n  All (0)\n  Country (1)\n>> ")
        
        if export_choice == "0":
            companies, filename = fetch_bloomberg_codes("all")
        elif export_choice == "1":
            country = input("Enter the country code (e.g., 'PL' for Poland): ")
            companies, filename = fetch_bloomberg_codes("country", country)
        else:
            print("Invalid export option. Exiting.")
            return
        
        save_to_csv(companies, filename)
    elif choice == "1":
        print("Import functionality will be implemented later.")
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
