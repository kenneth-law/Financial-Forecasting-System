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
ASX Financial Database Table Removal Script

This script drops all tables created by the database schema creation script.
Use this to reset the database to a clean state for testing or redeployment.

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


def drop_all_tables(cursor):
    """
    Drops all tables from the ASX financial database.
    
    Args:
        cursor (psycopg2.cursor): Active database cursor
        
    Side effects:
        Executes SQL to drop all tables
    """
    # Drop tables in reverse order of creation to respect foreign key constraints
    tables = [
        "portfolio",
        "users",
        "stock_trades",
        "financial_reports",
        "stock_prices",
        "stocks"
    ]
    
    for table in tables:
        print(f"Dropping table: {table}")
        cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")


def reset_database(dbname, user, password, host="localhost"):
    """
    Main function to reset the database by dropping all tables.
    
    Args:
        dbname (str): Name of the database to connect to
        user (str): Username for database authentication
        password (str): Password for database authentication
        host (str): Database server hostname, defaults to localhost
        
    Side effects:
        Drops all tables in the database
        Commits changes to the database
    """
    conn, cursor = connect_to_database(dbname, user, password, host)
    
    try:
        # Get confirmation from user
        confirm = input("This will drop all tables in the database. Are you sure? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return
        
        # Drop all tables
        drop_all_tables(cursor)
        
        # Commit changes
        conn.commit()
        print("All database tables have been dropped successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        # Close connection
        cursor.close()
        conn.close()


if __name__ == "__main__":
    # Database connection details - using the same as in the schema creation script
    userpassword = input("password please")
    reset_database(
        dbname="asx_financials",
        user="asx_user",
        password=userpassword,
        host="localhost"
    )