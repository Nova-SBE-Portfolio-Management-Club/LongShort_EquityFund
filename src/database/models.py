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

from sqlalchemy import Column, DateTime, String, Integer, ForeignKey, Boolean, Float, PrimaryKeyConstraint, ForeignKeyConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

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
    last_update_date = Column(DateTime, nullable=True, default=None) 
    valid = Column(Boolean, nullable=False, default=True)  
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.valid  = True

    def get_ticker(self) -> str:
        return self.ticker

    def set_ticker(self, ticker: str):
        self.ticker = ticker
        
    def to_dict(self):
        """
        Returns a dictionary where keys are the attributes, with their respective values
        This is very handy for saving Python Objects to the database (SQL)
        Try to find "to_dict" in the db_engine.py file to see some use-cases
        """
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}

    def __repr__(self):
        return f"<Asset(ticker='{self.ticker}', atype='{self.atype}')>"

    def __eq__(self, other):
        if isinstance(other, Asset):
            return self.ticker == other.ticker
        return False
    
    # To create cascading behaviour -> If an asset gets deleted, the company will also be deleted (vice-versa)
    asset_child_company = relationship("Company", back_populates="company_parent", cascade="all, delete")
    asset_child_future = relationship("Futures", back_populates="future_parent", cascade="all, delete")

    # This will indicate to SQLAlchemy that this is the parent of the next class
    __mapper_args__ = {
        'polymorphic_identity': 'asset',   # To identify the base class
        'polymorphic_on': 'atype'           # This column will store the type of the object (Asset or Company)
    }

###############################################################################
###############################################################################

class Company(Asset):
    
    __tablename__ = 'companies'
    
    ticker  = Column('ticker',  String(15), ForeignKey('assets.ticker', name='companies_ticker_fkey', ondelete="CASCADE"), primary_key=True) 
    name    = Column('name',    String(50))
    country = Column('country', String(50), ForeignKey('countries.name', name='companies_cntry_fkey', ondelete="CASCADE"))
    isin    = Column('isin',    String(30))
    sector  = Column('sector',  String(20))
    industry= Column('industry',String(20))
    shares_outstanding = Column('shares_outstanding', Float)  # New column for shares outstanding
    floating_shares = Column('floating_shares', Float)         # New column for floating shares

    def __init__(self, ticker: str, name: str, country: str, isin: str = None):
        super().__init__(ticker)            # From parent class (Asset)
        self.name               = name      # Name set manually 
        self.country            = country   # Country set manually
        self.isin               = isin
        self.sector             = None
        self.industry           = None
        self.shares_outstanding = None
        self.floating_shares    = None

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
        
        
    def get_yfin_ticker(self, db):
        """
        Returns the Ticker to be used with YFinance. Given a DB_Engine (db)
        """
        country:Country = db.get_company_country(self)
        if not country:
            raise ValueError()
        
        yfin_sufx = country.yfinance_code
        if yfin_sufx is None:
            return self.ticker
        else:
            return self.ticker + '.' + yfin_sufx
    
    def to_dict(self):
        """
        Returns a dictionary where keys are the attributes, with their respective values
        This is very handy for saving Python Objects to the database (SQL)
        Try to find "to_dict" in the db_engine.py file to see some use-cases
        """
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}

    def __eq__(self, other):
        if isinstance(other, Company):
            return (self.ticker, self.country, self.isin, self.sector, self.industry,
                    self.shares_outstanding, self.floating_shares) == \
                   (other.ticker, other.country, other.isin, other.sector, other.industry,
                    other.shares_outstanding, other.floating_shares)
        return False
    
    
    # Foreign-Key Relationships
    company_parent      = relationship("Asset", back_populates="asset_child_company", cascade="all, delete")
    company_country     = relationship("Country", back_populates="country_companies", cascade="all, delete")
    company_pricedatas  = relationship("PriceData",back_populates='pricedata_company')

    # Table Args - Needed to handle conflicts 
    #__table_args__ = (UniqueConstraint("ticker", name="unique_company_ticker"),)

    __mapper_args__ = {
        'polymorphic_identity': 'company',  # This identifies the Company class
    }

    
###############################################################################
###############################################################################

class Country(Base):
    
    __tablename__ = 'countries'

    name = Column(String(50), primary_key=True, nullable=False)
    isin_code = Column(String(20))
    bloomberg_code = Column(String(20))
    yfinance_code = Column(String(20))
    currency = Column(String(10))
    yc_code = Column(String(20))
    
    def __init__(self, name: str, isin_code: str, bloomberg_code: str, yfinance_code: str, currency: str, yc_code: str):
        self.name = name
        self.isin_code = isin_code
        self.bloomberg_code = bloomberg_code
        self.yfinance_code = yfinance_code
        self.currency = currency
        self.yc_code = yc_code

    def __repr__(self):
        return (f"Country(name='{self.name}', isin_code='{self.isin_code}', bloomberg_code='{self.bloomberg_code}', "
                f"yfinance_code='{self.yfinance_code}', currency='{self.currency}', yc_code='{self.yc_code}')")

    def __eq__(self, other):
        if isinstance(other, Country):
            return (self.name, self.isin_code, self.bloomberg_code, self.yfinance_code, self.currency, self.yc_code) == \
                   (other.name, other.isin_code, other.bloomberg_code, other.yfinance_code, other.currency, other.yc_code)
        return False
    
    
    # Foreign Key Relationships
    country_companies = relationship("Company", back_populates="company_country")
    country_macrodata = relationship("MacroData", back_populates="macrodata_country")

        
###############################################################################
###############################################################################

class PriceData(Base):
    __tablename__ = "prices_data"

    date = Column(DateTime, nullable=False)
    ticker = Column(String, ForeignKey("assets.ticker", name='pricedata_cmpny_fkey', ondelete="CASCADE"), nullable=False)
    open_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    
    def to_dict(self):
        """
        Returns a dictionary where keys are the attributes, with their respective values
        This is very handy for saving Python Objects to the database (SQL)
        Try to find "to_dict" in the db_engine.py file to see some use-cases
        """
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}
    
    def to_dict_update(self):
        """
        Returns a dictionary of the model's fields excluding the conflict keys.
        Useful for Updating the database in CONFLICT situations.
        
        This function is not being used right now, but it can be useful in the future.
        """
        primary_keys = {"date", "ticker"} # Update this if you change the primary key
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
            if column.name not in primary_keys
        }
     
    def __init__(self, date: datetime, ticker: str, open_price: float, close_price: float):
        self.date = date
        self.ticker = ticker
        self.open_price = open_price
        self.close_price = close_price
    
    def __repr__(self):
        return(f"PriceData(date='{self.date}', ticker='{self.ticker}', open_price={self.open_price}, close_price={self.close_price})")

    def __eq__(self, other):
        if isinstance(other, PriceData):
            return (self.date, self.ticker, self.open_price, self.close_price) == (other.date, other.ticker, other.open_price, other.close_price)
        return False
    
    # Foreign Key Relationship
    pricedata_company = relationship("Company",back_populates='company_pricedatas', cascade="all, delete")
    
    # Define Composite Primary Key
    __table_args__ = (
        PrimaryKeyConstraint('date', 'ticker', name="pk_prices_data"),
    )

###############################################################################
###############################################################################

class Pair(Base):
    __tablename__ = "pairs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tickerA = Column(String, ForeignKey("assets.ticker"))
    tickerB = Column(String, ForeignKey("assets.ticker"))
    ceof = Column(Float, nullable=False)

    def __eq__(self, other):
        if isinstance(other, Pair):
            return (self.tickerA, self.tickerB, self.ceof) == (other.tickerA, other.tickerB, other.ceof)
        return False

###############################################################################
###############################################################################

class Futures(Asset):
    
    __tablename__ = 'futures'
    
    ticker = Column('ticker', String(15), ForeignKey('assets.ticker', name='futures_ticker_fkey', ondelete="CASCADE"), primary_key=True)
    bloomberg_ticker = Column('bloomberg_ticker', String(20))
    
    def __init__(self, ticker: str, bloomberg_ticker: str):
        super().__init__(ticker)
        self.atype = 'future'
        self.bloomberg_ticker = bloomberg_ticker

    def get_bloomberg_ticker(self) -> str:
        return self.bloomberg_ticker

    def set_bloomberg_ticker(self, bloomberg_ticker: str):
        self.bloomberg_ticker = bloomberg_ticker
        
    def to_dict(self):
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}

    def __repr__(self):
        return f"<Futures(ticker='{self.ticker}', bloomberg_ticker='{self.bloomberg_ticker}')>"

    def __eq__(self, other):
        if isinstance(other, Futures):
            return (self.ticker, self.bloomberg_ticker) == (other.ticker, other.bloomberg_ticker)
        return False
    
    # Relationships
    future_parent = relationship("Asset", back_populates="asset_child_future", cascade="all, delete")
    
    __mapper_args__ = {
        'polymorphic_identity': 'future',
    }

###############################################################################
###############################################################################

class MacroData(Base):
    __tablename__ = "macro_data"

    country = Column(String(50), ForeignKey("countries.name", name='macrodata_cntry_fkey', ondelete="CASCADE"), nullable=False)
    data_name = Column(String(20), nullable=False)  # YC_2Y, CPI, GDP, etc.
    date = Column(DateTime, nullable=False)
    value = Column(Float, nullable=False)
    
    def __init__(self, country: str, data_name: str, date: datetime, value: float):
        self.country = country
        self.data_name = data_name
        self.date = date
        self.value = value

    def to_dict(self):
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}

    def __repr__(self):
        return (f"MacroData(country='{self.country}', data_name='{self.data_name}', "
                f"date='{self.date}', value={self.value})")

    def __eq__(self, other):
        if isinstance(other, MacroData):
            return (self.country, self.data_name, self.date, self.value) == \
                   (other.country, other.data_name, other.date, other.value)
        return False
    
    # Relationships
    macrodata_country = relationship("Country", back_populates="country_macrodata")
    macrodata_info = relationship(
        "MacroData_Info",
        primaryjoin="and_(MacroData.data_name==MacroData_Info.data_name, MacroData.country==MacroData_Info.country)",
        back_populates="macrodata_entries",
        overlaps="macrodata_country,country_macrodata"
    )

    __table_args__ = (
        PrimaryKeyConstraint('country', 'data_name', 'date', name="pk_macro_data"),
        ForeignKeyConstraint(
            ['data_name', 'country'],
            ['macro_data_info.data_name', 'macro_data_info.country'],
            name="macrodata_info_fk",
            ondelete="CASCADE"
        ),
    )

###############################################################################
###############################################################################

class MiscellData(Base):
    __tablename__ = "miscell_data"

    data_name = Column(String(50), ForeignKey("misc_data_info.data_name", name='miscdata_name_fkey'), nullable=False)  # MOON_CYCLE, DAYS_UNTIL_XMAS, etc.
    date = Column(DateTime, nullable=False)
    value = Column(Float, nullable=False)
    
    def __init__(self, data_name: str, date: datetime, value: float):
        self.data_name = data_name
        self.date = date
        self.value = value

    def to_dict(self):
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}

    def __repr__(self):
        return f"MiscellData(data_name='{self.data_name}', date='{self.date}', value={self.value})"

    def __eq__(self, other):
        if isinstance(other, MiscellData):
            return (self.data_name, self.date, self.value) == (other.data_name, other.date, other.value)
        return False
    
    # Composite Primary Key
    __table_args__ = (
        PrimaryKeyConstraint('data_name', 'date', name="pk_miscell_data"),
    )

    # Relationships
    miscdata_parent = relationship("MiscData_Info", foreign_keys=[data_name])

###############################################################################
###############################################################################

class MacroData_Info(Base):
    __tablename__ = "macro_data_info"

    data_name = Column(String(20), nullable=False)
    country = Column(String(50), ForeignKey("countries.name", name='macrodatainfo_cntry_fkey', ondelete="CASCADE"), nullable=False)
    last_update_date = Column(DateTime, nullable=True, default=None)
    description = Column(String(200), nullable=True)
    frequency = Column(String(20), nullable=True)  
    source = Column(String(50), nullable=True)
    
    def __init__(self, data_name: str, last_update_date: datetime, country: str, description: str = None, frequency: str = None, source: str = None):
        self.data_name = data_name
        self.last_update_date = last_update_date
        self.country = country
        self.description = description
        self.frequency = frequency
        self.source = source

    def to_dict(self):
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}

    def __repr__(self):
        return (f"MacroData_Info(data_name='{self.data_name}', country='{self.country}', "
                f"last_update='{self.last_update_date}', description='{self.description}')")

    def __eq__(self, other):
        if isinstance(other, MacroData_Info):
            return (self.data_name, self.country) == (other.data_name, other.country)
        return False
    
    # Relationships
    macrodata_info_country = relationship("Country", foreign_keys=[country])
    macrodata_entries = relationship(
        "MacroData",
        primaryjoin="and_(MacroData_Info.data_name==MacroData.data_name, MacroData_Info.country==MacroData.country)",
        back_populates="macrodata_info",
        overlaps="macrodata_country,country_macrodata"
    )

    __table_args__ = (
        PrimaryKeyConstraint('data_name', 'country', name="pk_macro_data_info"),
    )
 
###############################################################################
###############################################################################

class MiscData_Info(Base):
    __tablename__ = "misc_data_info"

    data_name = Column(String(50), primary_key=True, nullable=False)
    last_update_date = Column(DateTime, nullable=True, default=None)
    description = Column(String(200), nullable=True)
    frequency = Column(String(20), nullable=True)  
    source = Column(String(50), nullable=True)     
    
    def __init__(self, data_name: str, description: str = None, 
                 frequency: str = None, source: str = None):
        self.data_name = data_name
        self.description = description
        self.frequency = frequency
        self.source = source

    def to_dict(self):
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}

    def __repr__(self):
        return (f"MiscData_Info(data_name='{self.data_name}', "
                f"last_update='{self.last_update_date}', description='{self.description}')")

    def __eq__(self, other):
        if isinstance(other, MiscData_Info):
            return self.data_name == other.data_name
        return False

###############################################################################
###############################################################################