"""
Functions to handle the GET command
"""

import os
from ..messages      import *
import pandas as pd


from database import DB_Engine


def handle_get_country(db: DB_Engine):
    """Handle the GET country command."""
    option = input("Count (0)\n Show All (1)\n ")
    
    countries = db.get_all_countries()
    if option == "1":
        #Show all countries.
        for country in countries:
            print(country)
        
    elif option == "0":
        print(f'Total number of countries: {len(countries)}\n')
        
        
        
def handle_get_company(db: DB_Engine):
    """Handle the GET company command."""   
    option = input("Count (0)\nCount By Country (1)\nCount By Sector (2)\n")
    
    print()
    if option == "0":
        print(f"Total companies: {len(db.get_all_companies())}")

    elif option == "1":
        # Group By Country
        dix = db.count_companies_by_attr('country')
        for country in dix:
            print(f'{country:<15} {dix[country]:>5} companies')
        print()
            
    elif option == "2":
        # Group By Sector
        dix = db.count_companies_by_attr('sector')
        for country in dix:
            print(f'{country:<15} {dix[country]:>5} companies')
        print()
    
    
def handle_get_pairs():
    
    return
    # TODO: Fix this
    
    """Handle the GET pairs command.""" 
    conn = 0#get_database_connection()
    cur = conn.cursor()
    option = input("Count (0)\nCount By Country (1)\n")

    if option == "0":
        cur.execute("SELECT COUNT(DISTINCT pair) FROM pairs;")
        count = cur.fetchone()[0]
        """Count all pairs."""
        print(f"Total pairs: {count}")

    elif option == "1":
        cur.execute("SELECT country, COUNT(DISTINCT pair) FROM pairs GROUP BY companies.country;")
        pairs_by_countries = cur.fetchall()
        """Count pairs by country."""
        print(pairs_by_countries)
    
    cur.close()
    conn.close()
    
    
def handle_get_pricedata():
    
    return
    # TODO: Fix this
    
    """Handle the GET price data command."""
    conn = 0#get_database_connection()
    cur = conn.cursor()
    option = input("Count (0)\nCompany\n")

    if option == "0":
        cur.execute("SELECT COUNT(DISTINCT pair) FROM price_data;")
        count = cur.fetchone()[0]
        """Count all price data."""
        print(f"Total price data: {count}")

    else:
        
        cur.execute("SELECT * FROM price_data WHERE pair = %s ORDER BY timestamp DESC LIMIT 1;", (option,))
        price_data = cur.fetchone()
        """Show price data for a company."""
        print(price_data)
        
    cur.close()
    conn.close()
    
    
def handle_get_industry():
    
    return
    # TODO: Fix this
    
    """Handle the GET industry command."""
    conn = 0#get_database_connection()
    cur = conn.cursor()
    option = input("Count (0)\nShow All (1)\n")

    if option == "1":
        """Show all industries."""
        cur.execute("SELECT COUNT(DISTINCT industries) FROM industries;")
        industries = cur.fetchall()
        print(industries)
    elif option == "0":
        cur.execute("SELECT COUNT(DISTINCT industries) FROM industries;")
        count = cur.fetchone()[0]
        """Count all industries."""
        print(f"Total different industries: {count}")
    cur.close()
    conn.close()
    
    
def handle_get_command(db: DB_Engine):
    
    exit = False
    while not(exit):
        
        cmd = str(input( GET_COMMAND_INPUT ))

        
        if not(cmd.isdigit()):
            print(DIGIT_ONLY_WARNING)
        
        else:
            cmd = int(cmd)
            
            if cmd == 0:
                handle_get_country(db)
            elif cmd == 1:
                handle_get_company(db)
            elif cmd == 2:
                handle_get_pairs(db)
            elif cmd == 3:
                handle_get_pricedata(db)
            elif cmd == 4:
                handle_get_industry(db)
            else:
                exit = True