"""
Functions to handle the POPULATE command
"""


def handle_populate_command():
    pass

# Imports

import os
import pandas as pd
import numpy as np

from database import DB_Engine, Country


def handle_populate_command(db: DB_Engine):
    """
    Adds to the database the companies of a set of countries the user provides through an CSV file
    """
    file_path = "database/imports/countries.csv"
    template_path = "database/imports/countries_template.csv"
    
    print("[INFO] Please upload the 'countries.csv' file to the following location:")
    print(f"       {file_path}")
    print(f"A template file is available at: {template_path}")

    while True:
        input("[\nPress Enter after uploading the file.") 
        print(file_path, os.path.exists(file_path),os.path.exists(os.path.abspath("src/"+file_path)))
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                required_columns = {"name", "isin_code", "bloomberg_code", "yfinance_code", "currency"}
                if not required_columns.issubset(df.columns):
                    print(f"\nIncorrect format. The CSV file must contain the following columns: {', '.join(required_columns)}")
                    print(f"A template for a correctly formatted file is available at: {template_path}")
                    continue
                break
            except Exception as e:
                print(f"\nFailed to read '{file_path}': {e}")
                print("Please upload a CSV file.")
                continue
        print("\nFile not found. Please upload 'countries.csv' and try again.")

    for _, row in df.iterrows():
        print(row["yfinance_code"],type(row["yfinance_code"]))
        country_name = row["name"].strip()
        isin_code = row["isin_code"].strip()
        bloomberg_code = row["bloomberg_code"].strip()
        yfinance_code = row["yfinance_code"].strip() if type(row["yfinance_code"])==str else None
        currency = row["currency"].strip()
    
        
        country = Country(
            name=country_name,
            isin_code=isin_code,
            bloomberg_code=bloomberg_code,
            yfinance_code=yfinance_code,
            currency=currency
        )
        
        if not db.exists_country(country):
            db.insert_country(country)
            
        try:
            db.insert_all_country_companies(country) # Talvez seja necessário alterar a função em caso de já existirem empresas e haver perigo de duplicação
        except Exception as e:
            print(f"\n[WARNING] Could not fetch companies for {country_name}: {e}") 
        
if __name__ == "__main__":  # From Bloomberg command
    
    db = DB_Engine()
    db.connect()
    
    handle_populate_command(db)