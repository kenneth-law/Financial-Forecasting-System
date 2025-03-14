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
ASX Financial Data Retrieval Web Interface

This script provides a simple web interface to retrieve stock data from the
PostgreSQL database. It uses Flask as a lightweight web framework to serve
HTML pages and handle queries for stock information.

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
import pandas as pd
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta

# Database connection details
USER = "asx_user"
PASSWORD = "asx200"
DB_NAME = "asx_financials"
HOST = "localhost"

app = Flask(__name__)

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

def get_stock_info(ticker):
    """
    Retrieves basic information about a stock from the database.
    
    Args:
        ticker (str): The stock ticker symbol
        
    Returns:
        dict: Basic stock information
    """
    conn, cursor = connect_to_db()
    try:
        cursor.execute("""
            SELECT ticker, company_name, sector
            FROM stocks
            WHERE ticker = %s
        """, (ticker,))
        
        result = cursor.fetchone()
        if result:
            return {
                "ticker": result[0],
                "company_name": result[1],
                "sector": result[2]
            }
        return None
    finally:
        cursor.close()
        conn.close()

def get_stock_prices(ticker, start_date=None, end_date=None):
    """
    Retrieves price history for a stock between specified dates.
    
    Args:
        ticker (str): The stock ticker symbol
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        
    Returns:
        list: List of dictionaries containing price data
    """
    conn, cursor = connect_to_db()
    
    # Set default dates if not provided
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if not start_date:
        # Default to 30 days ago if not specified
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    try:
        cursor.execute("""
            SELECT date, open_price, high_price, low_price, close_price, volume
            FROM stock_prices
            WHERE ticker = %s AND date BETWEEN %s AND %s
            ORDER BY date
        """, (ticker, start_date, end_date))
        
        columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        results = cursor.fetchall()
        
        price_data = []
        for row in results:
            price_data.append(dict(zip(columns, row)))
        
        return price_data
    finally:
        cursor.close()
        conn.close()

def get_financial_report(ticker):
    """
    Retrieves the latest financial report for a stock.
    
    Args:
        ticker (str): The stock ticker symbol
        
    Returns:
        dict: Financial report data
    """
    conn, cursor = connect_to_db()
    try:
        cursor.execute("""
            SELECT report_date, revenue, net_income, eps
            FROM financial_reports
            WHERE ticker = %s
            ORDER BY report_date DESC
            LIMIT 1
        """, (ticker,))
        
        result = cursor.fetchone()
        if result:
            return {
                "report_date": result[0],
                "revenue": result[1],
                "net_income": result[2],
                "eps": result[3]
            }
        return None
    finally:
        cursor.close()
        conn.close()

def get_available_tickers():
    """
    Gets a list of all available stock tickers in the database.
    
    Returns:
        list: List of ticker symbols
    """
    conn, cursor = connect_to_db()
    try:
        cursor.execute("SELECT ticker FROM stocks ORDER BY ticker")
        return [row[0] for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

@app.route('/')
def home():
    """Render the home page with the list of available tickers"""
    tickers = get_available_tickers()
    return render_template('index.html', tickers=tickers)

@app.route('/stock/<ticker>')
def stock_data(ticker):
    """Render the stock data page for a specific ticker"""
    stock_info = get_stock_info(ticker)
    if not stock_info:
        return "Stock not found", 404
    
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    price_data = get_stock_prices(ticker, start_date, end_date)
    financial_data = get_financial_report(ticker)
    
    return render_template(
        'stock.html',
        stock=stock_info,
        prices=price_data,
        financials=financial_data
    )

@app.route('/api/stock/<ticker>')
def api_stock_data(ticker):
    """API endpoint to get stock data in JSON format"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    stock_info = get_stock_info(ticker)
    if not stock_info:
        return jsonify({"error": "Stock not found"}), 404
    
    price_data = get_stock_prices(ticker, start_date, end_date)
    financial_data = get_financial_report(ticker)
    
    return jsonify({
        "stock_info": stock_info,
        "price_data": price_data,
        "financial_data": financial_data
    })

if __name__ == "__main__":
    # Create templates directory and HTML files if needed
    import os
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create index.html if it doesn't exist
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>ASX Financial Data</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        form {
            margin-bottom: 30px;
        }
        select, input, button {
            padding: 8px 12px;
            margin: 8px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #3498db;
            color: white;
            cursor: pointer;
            border: none;
        }
        button:hover {
            background-color: #2980b9;
        }
        .disclaimer {
            font-size: 0.8em;
            background-color: #f8f9fa;
            padding: 10px;
            border-left: 4px solid #e74c3c;
            margin-top: 40px;
        }
        .ticker-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 20px;
        }
        .ticker-item {
            padding: 10px;
            background-color: #ecf0f1;
            border-radius: 4px;
            text-align: center;
        }
        .ticker-item a {
            text-decoration: none;
            color: #2c3e50;
            font-weight: bold;
        }
        .ticker-item a:hover {
            color: #3498db;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>ASX Financial Data Retrieval</h1>
        
        <form action="/stock/search" method="get">
            <h3>Search for a Stock</h3>
            <input type="text" name="ticker" placeholder="Enter Stock Ticker (e.g., CBA.AX)" required>
            <button type="submit">Search</button>
        </form>
        
        <h3>Available Stocks</h3>
        <div class="ticker-list">
            {% for ticker in tickers %}
            <div class="ticker-item">
                <a href="/stock/{{ ticker }}">{{ ticker }}</a>
            </div>
            {% endfor %}
        </div>
        
        <div class="disclaimer">
            <p><strong>DISCLAIMER:</strong> This data is provided for educational and informational purposes only.
            This is not financial advice. Always consult with a qualified financial professional before making investment decisions.</p>
        </div>
    </div>
</body>
</html>""")
    
    # Create stock.html if it doesn't exist
    if not os.path.exists('templates/stock.html'):
        with open('templates/stock.html', 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>{{ stock.ticker }} - ASX Financial Data</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .stock-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #eee;
            padding-bottom: 20px;
            margin-bottom: 20px;
        }
        .stock-info {
            flex: 1;
        }
        .stock-sector {
            background-color: #3498db;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            display: inline-block;
            font-size: 0.9em;
        }
        .date-filter {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .date-filter input, .date-filter button {
            padding: 8px 12px;
            margin: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .date-filter button {
            background-color: #3498db;
            color: white;
            cursor: pointer;
            border: none;
        }
        .date-filter button:hover {
            background-color: #2980b9;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .section {
            margin-bottom: 30px;
        }
        .back-btn {
            display: inline-block;
            padding: 8px 16px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .back-btn:hover {
            background-color: #2980b9;
        }
        .financials {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        .financial-item {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            text-align: center;
        }
        .financial-value {
            font-size: 1.4em;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
        }
        .financial-label {
            font-size: 0.9em;
            color: #7f8c8d;
        }
        .disclaimer {
            font-size: 0.8em;
            background-color: #f8f9fa;
            padding: 10px;
            border-left: 4px solid #e74c3c;
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn"> Back to All Stocks</a>
        
        <div class="stock-header">
            <div class="stock-info">
                <h1>{{ stock.ticker }}</h1>
                <h2>{{ stock.company_name }}</h2>
                <div class="stock-sector">{{ stock.sector or 'Sector Unknown' }}</div>
            </div>
        </div>
        
        <div class="date-filter">
            <form method="get">
                <label for="start_date">Start Date:</label>
                <input type="date" id="start_date" name="start_date">
                
                <label for="end_date">End Date:</label>
                <input type="date" id="end_date" name="end_date">
                
                <button type="submit">Filter</button>
            </form>
        </div>
        
        {% if financials %}
        <div class="section">
            <h3>Latest Financial Report ({{ financials.report_date }})</h3>
            <div class="financials">
                <div class="financial-item">
                    <div class="financial-label">Revenue</div>
                    <div class="financial-value">${{ '{:,.2f}'.format(financials.revenue) }}</div>
                </div>
                <div class="financial-item">
                    <div class="financial-label">Net Income</div>
                    <div class="financial-value">${{ '{:,.2f}'.format(financials.net_income) }}</div>
                </div>
                <div class="financial-item">
                    <div class="financial-label">EPS</div>
                    <div class="financial-value">${{ '{:.2f}'.format(financials.eps) }}</div>
                </div>
            </div>
        </div>
        {% endif %}
        
        <div class="section">
            <h3>Price History</h3>
            {% if prices %}
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Open</th>
                        <th>High</th>
                        <th>Low</th>
                        <th>Close</th>
                        <th>Volume</th>
                    </tr>
                </thead>
                <tbody>
                    {% for price in prices %}
                    <tr>
                        <td>{{ price.date }}</td>
                        <td>${{ '{:.2f}'.format(price.open) }}</td>
                        <td>${{ '{:.2f}'.format(price.high) }}</td>
                        <td>${{ '{:.2f}'.format(price.low) }}</td>
                        <td>${{ '{:.2f}'.format(price.close) }}</td>
                        <td>{{ '{:,}'.format(price.volume) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <p>No price data available for this stock.</p>
            {% endif %}
        </div>
        
        <div class="disclaimer">
            <p><strong>DISCLAIMER:</strong> This data is provided for educational and informational purposes only.
            This is not financial advice. Always consult with a qualified financial professional before making investment decisions.</p>
        </div>
    </div>
</body>
</html>""")
    
    print("Starting web server on http://127.0.0.1:5000/")
    app.run(debug=True)