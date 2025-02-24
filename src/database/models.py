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
    -

Dependencies:
    - yfinance
    - investpy
    -
                                                         
"""

# IMPORTS

import yfinance as yf
import investpy
import requests
from urllib.error import URLError
from datetime import datetime

from sqlalchemy import Column, DateTime, String, Integer, ForeignKey, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base

###############################################################################

# DECLARATIONS

Base = declarative_base()
metadata = Base.metadata

###############################################################################
###############################################################################

class Asset(Base):
    
    __tablename__ = 'assets'
    
    ticker  = Column('ticker',  String(15), primary_key=True)
    atype   = Column('atype',    String(50)) 
    
    def __init__(self, ticker: str):
        self.ticker = ticker

    def get_ticker(self) -> str:
        return self.ticker

    def set_ticker(self, ticker: str):
        self.ticker = ticker

    def __repr__(self):
        return f"<Asset(ticker='{self.ticker}', atype='{self.atype}')>"

    def __eq__(self, other):
        if isinstance(other, Asset):
            return self.ticker == other.ticker
        return False

    # This will indicate to SQLAlchemy that this is the parent of the next class
    __mapper_args__ = {
        'polymorphic_identity': 'asset',   # To identify the base class
        'polymorphic_on': 'atype'           # This column will store the type of the object (Asset or Company)
    }

###############################################################################
###############################################################################

class Company(Asset):
    
    __tablename__ = 'companies'
    
    ticker  = Column('ticker',  String(15), ForeignKey('assets.ticker'), primary_key=True) 
    country = Column('country', String(50))
    isin    = Column('isin',    String(30))
    sector  = Column('sector',  String(20))
    industry= Column('industry',String(20))
    shares_outstanding = Column('shares_outstanding', Float)  # New column for shares outstanding
    floating_shares = Column('floating_shares', Float)         # New column for floating shares

    def __init__(self, ticker: str, country: str):
        super().__init__(ticker)    # From parent class (Asset)
        self.country = country      # Country set manually
        self.isin = None
        self.sector = None
        self.industry = None
        self.shares_outstanding = None
        self.floating_shares = None

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

    def get_shares_outstanding(self) -> float:
        if self.shares_outstanding is None:
            self.set_shares_outstanding()
        return self.shares_outstanding

    def set_shares_outstanding(self):
        try:
            company = yf.Ticker(self.ticker)
            shares = company.info.get('sharesOutstanding')
            if shares:
                self.shares_outstanding = float(shares)
            else:
                raise ValueError("Shares outstanding data not available.")
        except (requests.exceptions.ConnectionError, URLError) as e:
            raise ConnectionError(f"Network error while fetching shares outstanding: {e}")

    def get_floating_shares(self) -> float:
        if self.floating_shares is None:
            self.set_floating_shares()
        return self.floating_shares

    def set_floating_shares(self):
        try:
            company = yf.Ticker(self.ticker)
            floats = company.info.get('floatShares')
            if floats:
                self.floating_shares = float(floats)
            else:
                raise ValueError("Floating shares data not available.")
        except (requests.exceptions.ConnectionError, URLError) as e:
            raise ConnectionError(f"Network error while fetching floating shares: {e}")

    def __eq__(self, other):
        if isinstance(other, Company):
            return (self.ticker, self.country, self.isin, self.sector, self.industry,
                    self.shares_outstanding, self.floating_shares) == \
                   (other.ticker, other.country, other.isin, other.sector, other.industry,
                    other.shares_outstanding, other.floating_shares)
        return False

    __mapper_args__ = {
        'polymorphic_identity': 'company',  # This identifies the Company class
    }

    
###############################################################################
###############################################################################

class Country(Base):
    __tablename__ = 'countries'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)
    isin_code = Column(String(20))
    bloomberg_code = Column(String(20))
    yfinance_code = Column(String(20))
    currency = Column(String(10))
    
    def __init__(self, name: str, isin_code: str, bloomberg_code: str, yfinance_code: str, currency: str):
        self.name = name
        self.isin_code = isin_code
        self.bloomberg_code = bloomberg_code
        self.yfinance_code = yfinance_code
        self.currency = currency

    def __repr__(self):
        return (f"Country(name='{self.name}', isin_code='{self.isin_code}', bloomberg_code='{self.bloomberg_code}', "
                f"yfinance_code='{self.yfinance_code}', currency='{self.currency}')")

    def __eq__(self, other):
        if isinstance(other, Country):
            return (self.name, self.isin_code, self.bloomberg_code, self.yfinance_code, self.currency) == \
                   (other.name, other.isin_code, other.bloomberg_code, other.yfinance_code, other.currency)
        return False


        
###############################################################################
###############################################################################

class PriceData(Base):
    __tablename__ = "price_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)
    ticker = Column(String, nullable=False)
    open_status = Column(Boolean, nullable=False)
    price = Column(Float, nullable=False)
     
    def __init__(self, date: datetime, ticker: str, open_status: bool, price: float):
        self.date = date
        self.ticker = ticker
        self.open_status = open_status
        self.price = price
    
    def __repr__(self):
        return(f"PriceData(date='{self.date}', ticker='{self.ticker}', open_status={self.open_status}, price={self.price})")

    def __eq__(self, other):
        if isinstance(other, PriceData):
            return (self.date, self.ticker, self.open_status, self.price) == (other.date, other.ticker, other.open_status, other.price)
        return False

###############################################################################
###############################################################################

class Pair(Base):
    __tablename__ = "pair"

    tickerA = Column(String, primary_key=True)
    tickerB = Column(String, primary_key=True)
    ceof = Column(Float, nullable=False)

    def __eq__(self, other):
        if isinstance(other, Pair):
            return (self.tickerA, self.tickerB, self.ceof) == (other.tickerA, other.tickerB, other.ceof)
        return False