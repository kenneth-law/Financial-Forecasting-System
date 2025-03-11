# Copyright 2025 Kenneth Law
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 


"""
ASX Financial Database Schema Creation Script 


This script establishes the database schema for an ASX financial data application.
It creates tables for stocks, stock prices, financial reports, stock trades, users, 
and portfolios. The script handles connections to PostgreSQL and ensures proper 
relationships between tables.

Author: Kenneth
Date: 2025-03-11


DISCLAIMER:
This software is provided for educational and informational purposes only.
The author(s) are not registered investment advisors and do not provide financial advice.
This software does not guarantee accuracy of data and should not be the sole basis for any investment decision.
Users run and use this software at their own risk. The author(s) accept no liability for any loss or damage 
resulting from its use. Always consult with a qualified financial professional before making investment decisions.
"""

import psycopg2


def connect_to_database(dbname, user, password, host="localhost"):
    """
    Establishes a connection to the PostgreSQL database.
    
    Args:
        dbname (str): Name of the database to connect to
        user (str): Username for database authentication
        password (str): Password for database authentication
        host (str): Database server hostname, defaults to localhost
        
    Returns:
        tuple: (conn, cursor) PostgreSQL connection and cursor objects
    """
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host
    )
    cursor = conn.cursor()
    return conn, cursor


def create_stocks_table(cursor):
    """
    Creates the stocks table which serves as the main reference table
    for stock information including ticker, company name, and sector.
    
    Args:
        cursor (psycopg2.cursor): Active database cursor
        
    Side effects:
        Executes SQL to create the stocks table if it doesn't exist
    """
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stocks (
        id SERIAL PRIMARY KEY,
        ticker TEXT UNIQUE NOT NULL,
        company_name TEXT,
        sector TEXT
    )
    """)


def create_stock_prices_table(cursor):
    """
    Creates the stock_prices table to store time-series data for stock prices,
    including open, high, low, close prices and volume for each trading day.
    
    Args:
        cursor (psycopg2.cursor): Active database cursor
        
    Side effects:
        Executes SQL to create the stock_prices table if it doesn't exist
    """
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stock_prices (
        id SERIAL PRIMARY KEY,
        ticker TEXT REFERENCES stocks(ticker) ON DELETE CASCADE,
        date DATE NOT NULL,
        open_price FLOAT CHECK (open_price >= 0),
        high_price FLOAT CHECK (high_price >= 0),
        low_price FLOAT CHECK (low_price >= 0),
        close_price FLOAT CHECK (close_price >= 0),
        volume BIGINT CHECK (volume >= 0),
        UNIQUE (ticker, date)  -- Ensures no duplicate records for the same stock & date
    )
    """)


def create_financial_reports_table(cursor):
    """
    Creates the financial_reports table to store quarterly and annual financial data,
    including revenue, net income, and earnings per share (EPS).
    
    Args:
        cursor (psycopg2.cursor): Active database cursor
        
    Side effects:
        Executes SQL to create the financial_reports table if it doesn't exist
    """
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS financial_reports (
        id SERIAL PRIMARY KEY,
        ticker TEXT REFERENCES stocks(ticker) ON DELETE CASCADE,
        report_date DATE NOT NULL,
        revenue FLOAT CHECK (revenue >= 0),
        net_income FLOAT CHECK (net_income >= 0),
        eps FLOAT CHECK (eps >= 0),
        UNIQUE (ticker, report_date)
    )
    """)


def create_stock_trades_table(cursor):
    """
    Creates the stock_trades table to record buy and sell transactions,
    including trade date, type (buy/sell), volume, and price.
    
    Args:
        cursor (psycopg2.cursor): Active database cursor
        
    Side effects:
        Executes SQL to create the stock_trades table if it doesn't exist
    """
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stock_trades (
        id SERIAL PRIMARY KEY,
        ticker TEXT REFERENCES stocks(ticker) ON DELETE CASCADE,
        trade_date DATE NOT NULL,
        trade_type TEXT CHECK (trade_type IN ('BUY', 'SELL')),
        trade_volume BIGINT CHECK (trade_volume > 0),
        trade_price FLOAT CHECK (trade_price >= 0)
    )
    """)


def create_users_table(cursor):
    """
    Creates the users table to store user information for portfolio tracking,
    including username and email.
    
    Args:
        cursor (psycopg2.cursor): Active database cursor
        
    Side effects:
        Executes SQL to create the users table if it doesn't exist
    """
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL
    )
    """)


def create_portfolio_table(cursor):
    """
    Creates the portfolio table to track stocks held by users,
    including quantity and average purchase price.
    
    Args:
        cursor (psycopg2.cursor): Active database cursor
        
    Side effects:
        Executes SQL to create the portfolio table if it doesn't exist
    """
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolio (
        id SERIAL PRIMARY KEY,
        user_id INT REFERENCES users(id) ON DELETE CASCADE,
        ticker TEXT REFERENCES stocks(ticker) ON DELETE CASCADE,
        quantity INT CHECK (quantity >= 0),
        avg_buy_price FLOAT CHECK (avg_buy_price >= 0),
        UNIQUE (user_id, ticker)  -- Ensures no duplicate stock holdings per user
    )
    """)


def initialize_database_schema(dbname, user, password, host="localhost"):
    """
    Main function to initialize the complete database schema by creating all required tables.
    
    Args:
        dbname (str): Name of the database to connect to
        user (str): Username for database authentication
        password (str): Password for database authentication
        host (str): Database server hostname, defaults to localhost
        
    Side effects:
        Creates all required tables in the database
        Commits changes to the database
    """
    conn, cursor = connect_to_database(dbname, user, password, host)
    
    try:
        # Create all tables
        create_stocks_table(cursor)
        create_stock_prices_table(cursor)
        create_financial_reports_table(cursor)
        create_stock_trades_table(cursor)
        create_users_table(cursor)
        create_portfolio_table(cursor)
        
        # Commit changes
        conn.commit()
        print("Database tables created successfully!")
    finally:
        # Close connection
        cursor.close()
        conn.close()


if __name__ == "__main__":
    # Example usage
    userpassword = "asx200"
    initialize_database_schema(
        dbname="asx_financials",
        user="asx_user",
        password=userpassword,
        host="localhost"
    )
