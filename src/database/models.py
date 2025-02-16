"""
models.py
12 Fev 2025

This module defines the database models for the application. Each class represents a database table, 
and its attributes correspond to table columns.

Tables:
    - Asset
        - Company
    - Country
    - PriceData

Dependencies:
    - yfinance
    - investpy
    
                                                        
"""

# IMPORTS

import yfinance as yf
import investpy
import requests
from urllib.error import URLError
from datetime import datetime

###############################################################################
###############################################################################



# ASSET

class Asset:
    
    def __init__(self, ticker: str):
        self.ticker = ticker

    def get_ticker(self) -> str:
        return self.ticker

    def set_ticker(self, ticker: str):
        self.ticker = ticker

###############################################################################
###############################################################################

# COMPANY

class Company(Asset):
    def __init__(self, ticker: str, country: str):
        super().__init__(ticker)    # From mother class (Asset)
        self.country = country  # Country set manually
        self.isin = None
        self.sector = None
        self.industry = None

    def get_isin(self) -> str:
        if not self.isin:
            self.set_isin()
        return self.isin

    def set_isin(self):
        try:
            data = investpy.get_stock_information(stock=self.ticker, country=self.country)
            self.isin = data.get('isin', None)
            if not self.isin:
                raise ValueError("ISIN not available.")
        except (requests.exceptions.ConnectionError, URLError) as e:
            raise ConnectionError(f"Network error while fetching ISIN: {e}")

    def get_sector(self) -> str:
        if not self.sector:
            self.set_sector()
        return self.sector

    def set_sector(self):
        try:
            company = yf.Ticker(self.ticker)
            sector = company.info.get('sector')
            if sector:
                self.sector = sector
            else:
                raise ValueError("Sector data not available.")
        except (requests.exceptions.ConnectionError, URLError) as e:
            raise ConnectionError(f"Network error while fetching sector: {e}")

    def get_industry(self) -> str:
        if not self.industry:
            self.set_industry()
        return self.industry

    def set_industry(self):
        try:
            company = yf.Ticker(self.ticker)
            industry = company.info.get('industry')
            if industry:
                self.industry = industry
            else:
                raise ValueError("Industry data not available.")
        except (requests.exceptions.ConnectionError, URLError) as e:
            raise ConnectionError(f"Network error while fetching industry: {e}")

    def get_country(self) -> str:
        return self.country
    
###############################################################################
###############################################################################

# COUNTRY

class Country:
    def __init__(self, name: str, isin_code: str, bloomberg_code: str, yfinance_code: str, currency: str):
        self.name = name
        self.isin_code = isin_code
        self.bloomberg_code = bloomberg_code
        self.yfinance_code = yfinance_code
        self.currency = currency

    def __repr__(self):
        return (f"Country(name='{self.name}', isin_code='{self.isin_code}', bloomberg_code='{self.bloomberg_code}', "
                f"yfinance_code='{self.yfinance_code}', currency='{self.currency}')")
        
###############################################################################
###############################################################################

# PRICEDATA

class PriceData:
    def __init__(self, date: datetime, ticker: str, open_status: bool, price: float):
        self.date = date
        self.ticker = ticker
        self.open_status = open_status
        self.price = price

    def __repr__(self):
        return (f"PriceData(date='{self.date}', ticker='{self.ticker}', open_status={self.open_status}, "
                f"price={self.price})")
        
###############################################################################
###############################################################################