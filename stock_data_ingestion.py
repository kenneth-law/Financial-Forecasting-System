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
ASX Financial Data Ingestion Script

This script fetches ASX 200 stock market data using Yahoo Finance (yfinance)
and inserts the retrieved data into a PostgreSQL database.

Features:
- Inserts stock metadata (ticker, company name, sector)
- Retrieves and stores historical stock prices (OHLC data)
- Uses psycopg2 to interact with PostgreSQL
- Prevents duplicate data insertion using conflict resolution

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
import yfinance as yf
import pandas as pd
import time


# Database connection details
USER = "asx_user"
PASSWORD = "asx200"
DB_NAME = "asx_financials"
HOST = "localhost"

def connect_to_db():
    """
    Establishes a connection to the PostgreSQL database.
    
    Returns:
        conn (psycopg2.connection): Active database connection
        cursor (psycopg2.cursor): Database cursor for executing SQL queries
    """
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=USER,
        password=PASSWORD,
        host=HOST
    )
    cursor = conn.cursor()
    return conn, cursor

def insert_stock_metadata(cursor, conn, tickers):
    """
    Inserts stock metadata (ticker, company name, sector) into the database.

    Args:
        cursor (psycopg2.cursor): Database cursor
        conn (psycopg2.connection): Active database connection
        tickers (list): List of ASX stock tickers
    """
    for ticker in tickers:
        try:
            print(f"Fetching metadata for {ticker}...")
            stock = yf.Ticker(ticker)
            
            # Add error handling for stock.info access
            try:
                info = stock.info
                company_name = info.get("longName", "Unknown")
                sector = info.get("sector", "Unknown")
            except Exception as e:
                print(f"Error getting info for {ticker}: {e}")
                company_name = "Unknown"
                sector = "Unknown"

            cursor.execute("""
                INSERT INTO stocks (ticker, company_name, sector) 
                VALUES (%s, %s, %s) 
                ON CONFLICT (ticker) DO NOTHING
            """, (ticker, company_name, sector))
            
            # Add a small delay to avoid API rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")
            continue  # Skip to next ticker on error

    conn.commit()
    print("Stock metadata inserted successfully!")


def insert_stock_prices(cursor, conn, tickers):
    """
    Fetches historical stock price data from Yahoo Finance and inserts it into the database.

    Args:
        cursor (psycopg2.cursor): Database cursor
        conn (psycopg2.connection): Active database connection
        tickers (list): List of ASX stock tickers
    """
    for ticker in tickers:
        try:
            print(f"Downloading data for {ticker}...")
            stock_data = yf.download(ticker, period="5y")

            if stock_data.empty:
                print(f"Warning: No stock price data found for {ticker}. Skipping...")
                continue  # Skip to the next ticker if no data found
            
            # Process the data - using the index directly as it contains the dates
            for date_idx, row in stock_data.iterrows():
                try:
                    # Convert the date index to a date object
                    date_obj = pd.Timestamp(date_idx).date()
                    
                    # Use .iloc[0] to extract scalar values from Series objects
                    # This fixes the FutureWarning about converting Series to float/int
                    cursor.execute("""
                        INSERT INTO stock_prices (ticker, date, open_price, high_price, low_price, close_price, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, date) DO NOTHING
                    """, (
                        ticker, 
                        date_obj,
                        float(row['Open'].iloc[0]) if isinstance(row['Open'], pd.Series) else float(row['Open']),
                        float(row['High'].iloc[0]) if isinstance(row['High'], pd.Series) else float(row['High']),
                        float(row['Low'].iloc[0]) if isinstance(row['Low'], pd.Series) else float(row['Low']),
                        float(row['Close'].iloc[0]) if isinstance(row['Close'], pd.Series) else float(row['Close']),
                        int(row['Volume'].iloc[0]) if isinstance(row['Volume'], pd.Series) else int(row['Volume'])
                    ))
                    
                except Exception as e:
                    print(f"Error processing date {date_idx} for {ticker}: {e}")
                    # Continue with the next row instead of failing completely
                    continue

            # Commit after each ticker to save progress
            conn.commit()
            
            # Add a small delay to avoid API rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
            continue  # Skip to next ticker on error

    print("Stock prices inserted successfully!")

def main():
    """
    Main function to execute stock metadata and price data insertion.
    """
    # Display an example of the expected input format
    print("Example ticker format: AIA.AX, ALL.AX, AMC.AX, ANZ.AX, APA.AX, ASX.AX")
    
    # Get user input and properly split into a list
    ticker_input = input("Enter your tickers (comma-separated): ")
    tickers = [ticker.strip() for ticker in ticker_input.split(',') if ticker.strip()]
    
    if not tickers:
        print("No valid tickers provided. Exiting.")
        return
    
    print(f"Processing {len(tickers)} tickers: {', '.join(tickers)}")
    
    # Establish database connection
    try:
        conn, cursor = connect_to_db()
        print("Successfully connected to database.")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    try:
        # Process in smaller batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1} of {(len(tickers) + batch_size - 1)//batch_size}")
            insert_stock_metadata(cursor, conn, batch)
            insert_stock_prices(cursor, conn, batch)
    except Exception as e:
        print(f"Error during data processing: {e}")
    finally:
        cursor.close()
        conn.close()
        print("Database connection closed.")

if __name__ == "__main__":
    main()