"""
Functions to handle the GET command
"""
from dotenv          import load_dotenv
from ..messages      import *
import pandas as pd
import psycopg2  # PostgreSQL database connector
import os
load_dotenv()

def get_database_connection():
    """Establish connection to the PostgreSQL database."""
    return psycopg2.connect(
      dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )
def test_db_connection():
    """Test the database connection."""
    conn = get_database_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1;")
    result = cur.fetchone()
    print(f"Database Test Result: {result}")
    cur.close()
    conn.close()

def handle_get_country():
    """Handle the GET country command."""
    conn = get_database_connection()
    cur = conn.cursor()
    option = input("Count (0)\n Show All (1)\n ")

    if option == "1":
        """Show all countries."""
        cur.execute("SELECT COUNT(DISTINCT countries) FROM countries;")
        countries = cur.fetchall()
        test_db_connection()
        print(countries)
        
    elif option == "0":
        cur.execute("SELECT COUNT(DISTINCT countries) FROM countries;")
        count = cur.fetchone()[0]
        """Count all countries."""
        print(f"Total different country: {count}")
    cur.close()
    conn.close()
def handle_get_company():
    """Handle the GET company command."""   
    conn = get_database_connection()
    cur = conn.cursor()
    option = input("Count (0)\nCount By Country (1)\nCount By Sector (2)\n")

    if option == "0":
        cur.execute("SELECT COUNT(DISTINCT companies) FROM companies;")
        count = cur.fetchone()[0]
        """Count all companies."""
        print(f"Total companies: {count}")

    elif option == "1":
        cur.execute("SELECT country, COUNT(DISTINCT companies) FROM companies GROUP BY companies.country;")
        comp_by_countries = cur.fetchall()
        """Count companies by sector."""
        print(comp_by_countries)
    elif option == "2":
        cur.execute("SELECT sector, COUNT(DISTINCT companies) FROM companies GROUP BY sector;")
        comp_by_sector = cur.fetchall()
        """Count companies by sector."""
        print(comp_by_sector)
    cur.close()
    conn.close()
def handle_get_pairs():
    """Handle the GET pairs command.""" 
    conn = get_database_connection()
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
    """Handle the GET price data command."""
    conn = get_database_connection()
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
    """Handle the GET industry command."""
    conn = get_database_connection()
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
def handle_get_command():
    
    exit = False
    while not(exit):
        
        cmd = str(input( GET_COMMAND_INPUT ))

        
        if not(cmd.isdigit()):
            print(DIGIT_ONLY_WARNING)
        
        else:
            cmd = int(cmd)
            
            if cmd == 0:
                handle_get_country()
            elif cmd == 1:
                handle_get_company()
            elif cmd == 2:
                handle_get_pairs()
            elif cmd == 3:
                handle_get_pricedata()
            elif cmd == 4:
                handle_get_industry()
            else:
                exit = True