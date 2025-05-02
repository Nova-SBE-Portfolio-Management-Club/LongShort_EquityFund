"""
Functions to handle the BLOOMBERG command
"""

import os
import pandas as pd
import datetime as dt
import csv


from database import DB_Engine

def save_to_csv(companies, filename):
    """Saves the fetched company codes to a CSV file."""
    export_path = "bloomberg/exports/"
    os.makedirs(export_path, exist_ok=True)
    
    df = pd.DataFrame(companies, columns=["company"])
    df.to_csv(os.path.join(export_path, filename), index=False)
    print(f"Exported {len(df)} companies to {filename}")

def export_yc_data(db: DB_Engine):
    data = db.get_yc_data()
    
    export_path = "bloomberg/exports/"
    os.makedirs(export_path, exist_ok=True)

    df = pd.DataFrame(data, columns=["YC_Ticker", "Last Update Date"])
    filename = "yc_data.csv"
    df.to_csv(os.path.join(export_path, filename), index=False)
    print(f"Exported yield curve data for {len(df)} countries to {filename}")

def import_from_csv(db: DB_Engine, filename: str):
    """Imports data from a CSV file into the database."""
    import_path = "bloomberg/exports/"
    file_path = os.path.join(import_path, filename)
    
    if not os.path.exists(file_path):
        print(f"File {filename} does not exist.")
        return

    # Read CSV file
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if there is one

        if filename == "bl_all.csv":
            companies = [row[0] for row in reader]
            db.insert_bloomberg_codes(companies)  # Call insert function for Bloomberg codes
        elif filename == "yc_data.csv":
            yc_data = [(row[0], dt.datetime.strptime(row[1], "%Y-%m-%d")) for row in reader]
            db.insert_yc_data(yc_data)  # Call insert function for yield curve data
        else:
            print("Unsupported file type.")

def handle_bloomberg_command(db):
    """Initiates the Bloomberg command handling process."""
    choice = input("Export (0)\nImport (1)\nGet (2)\n>> ").strip()

    if choice == "0":
        export_choice = input("\n  All (0)\n  Country (1)\n  Yield Curve (2)\n>> ").strip()
        
        if export_choice == "0":
            companies = db.get_bloomberg_codes("all")
            save_to_csv(companies, "bl_all.csv")
        elif export_choice == "1":
            country = input("Enter the country code (e.g., 'usa', 'portugal', ...): ")
            companies = db.get_bloomberg_codes("country", country)
            save_to_csv(companies, f"bl_{country}.csv")
        elif export_choice == "2":
            export_yc_data(db)
        else:
            print("Invalid export option.")

    elif choice == "1":
        print("\nAvailable files to import:")
        files = [f for f in os.listdir("bloomberg/exports/") if f.endswith(".csv")]
        
        for idx, file in enumerate(files):
            print(f"  ({idx}) {file}")
        
        file_choice = int(input("Select a file to import by number: ").strip())
        
        if 0 <= file_choice < len(files):
            selected_file = files[file_choice]
            import_from_csv(db, selected_file)
        else:
            print("Invalid file choice.")

    elif choice == "2":
        get_choice = input("\n  All (0)\n  Country (1)\n  Yield Curve (2)\n>> ").strip()
        
        if get_choice == "0":
            data = db.get_yc_data()
            for yc_code, last_update in data:
                print(f"{yc_code} | Last Update: {last_update.strftime('%Y-%m-%d') if last_update else 'None'}")
        else:
            print("Invalid get option.")

    else:
        print("Invalid main option. Exiting.")



if __name__ == "__main__":
    handle_bloomberg_command()
