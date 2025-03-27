"""
Functions to handle the BLOOMBERG command
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.abspath("src/database"))

from database import DB_Engine

def save_to_csv(companies, filename):
    """Saves the fetched company codes to a CSV file."""
    export_path = "src/bloomberg/exports/"
    os.makedirs(export_path, exist_ok=True)
    
    df = pd.DataFrame(companies, columns=["company"])
    df.to_csv(os.path.join(export_path, filename), index=False)
    print(f"Exported {len(df)} companies to {filename}")

def handle_bloomberg_command():
    """Initiates the Bloomberg command handling process."""
    db = DB_Engine()
    db.connect()
    
    choice = input("Export (0)\n  All (0)\n  Country (1)\nImport (1)\n>> ")
    
    if choice == "0":
        export_choice = input("\n  All (0)\n  Country (1)\n>> ")
        
        if export_choice == "0":
            companies = db.get_bloomberg_codes("all")
            filename = "bl_all.csv"
        elif export_choice == "1":
            country = input("Enter the country code (e.g., 'PL' for Poland): ")
            companies = db.get_bloomberg_codes("country", country)
            filename = f"bl_{country}.csv"
        else:
            print("Invalid export option. Exiting.")
            return
        
        save_to_csv(companies, filename)
    
    elif choice == "1":
        print("Import functionality will be implemented later.")
    
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    handle_bloomberg_command()
