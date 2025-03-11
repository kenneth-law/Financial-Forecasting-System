# ASX Financial Forecasting System

## DISCLAIMER:
This software is provided for educational and informational purposes only.
The author(s) are not registered investment advisors and do not provide financial advice.
This software does not guarantee accuracy of data and should not be the sole basis for any investment decision.
Users run and use this software at their own risk. The author(s) accept no liability for any loss or damage 
resulting from its use. Always consult with a qualified financial professional before making investment decisions.

## Terms of Use

By using this software, you agree:
1. To use it at your own risk
2. Not to hold the author(s) liable for any damages
3. To comply with the terms of the [LICENSE](LICENSE)
4. That this is not financial advice

This project is not affiliated with the ASX or any financial institution.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Overview
This project is designed to build an AI-driven financial forecasting system leveraging PostgreSQL for structured data storage and fine-tuning large language models (LLMs) for market analysis. The system stores ASX 200 stock market data, integrates financial reports, and allows for portfolio tracking. Fine-tuning will be conducted either locally or using AWS cloud-based resources.

## Database Setup Guide

### Setting Up PostgreSQL Database on macOS

#### 1. Install PostgreSQL (if not already installed)

Using Homebrew:
```bash
brew install postgresql@15
brew services start postgresql@15
```

Verify PostgreSQL is running:
```bash
brew services list
# or
pg_ctl -D /opt/homebrew/var/postgresql@15 status
```

#### 2. Create the Database and User

Access the PostgreSQL shell:
```bash
psql postgres
# If that doesn't work, try:
# sudo -u postgres psql
```

Create the database:
```sql
CREATE DATABASE asx_financials;
```

Create a user:
```sql
CREATE USER asx_user WITH PASSWORD 'asx200';
```

Grant privileges:
```sql
GRANT ALL PRIVILEGES ON DATABASE asx_financials TO asx_user;
ALTER USER asx_user CREATEDB;
```

Connect to the database:
```sql
\c asx_financials
```

Grant schema privileges (must run this after connecting to the database):
```sql
GRANT ALL ON SCHEMA public TO asx_user;
```

Verify setup:
```sql
\l    -- List databases
\du   -- List users
```

Exit PostgreSQL shell:
```sql
\q
```

#### 3. Test Connection with Your New User

```bash
psql -U asx_user -d asx_financials -h localhost -W
# Enter password when prompted: asx200
```

You should see `asx_financials=>` which indicates you're connected successfully.

### Setting Up PostgreSQL Database on Windows

#### 1. Install PostgreSQL
- Download and install PostgreSQL from [postgresql.org](https://www.postgresql.org/download/windows/)
- During installation, set a password for the default 'postgres' user
- Keep note of the port number (default is 5432)

#### 2. Create the Database and User

Open the SQL Shell (psql) from the Start menu, or command prompt:
```
psql -U postgres
```

Follow the same SQL commands as in the macOS section:
```sql
CREATE DATABASE asx_financials;
CREATE USER asx_user WITH PASSWORD 'asx200';
GRANT ALL PRIVILEGES ON DATABASE asx_financials TO asx_user;
ALTER USER asx_user CREATEDB;
\c asx_financials
GRANT ALL ON SCHEMA public TO asx_user;
```

### Setting Up PostgreSQL Database on Linux

#### 1. Install PostgreSQL

For Ubuntu/Debian:
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

For Red Hat/Fedora:
```bash
sudo dnf install postgresql-server postgresql-contrib
sudo postgresql-setup --initdb
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

#### 2. Create the Database and User

Switch to the postgres user:
```bash
sudo -i -u postgres
psql
```

Then follow the same SQL commands as in the macOS section.

### 4. Initialize the Database Schema

After setting up the database and user, you can initialize the schema:

```bash
python DBSchemaCreation.py
```

This will create all the necessary tables in your database.

### 5. Reset the Database (if needed)

To drop all tables and reset the database:

```bash
python drop_tables.py
```

## Database Design
The database is structured in PostgreSQL and optimized for time-series financial data. The schema follows a relational model that supports efficient queries for stock prices, financial reports, transactions, and user portfolios.

### Schema Design
#### 1. **Stocks Table** (Master reference table)
- `id`: Primary key, auto-incremented
- `ticker`: Unique stock symbol
- `company_name`: Full company name
- `sector`: Industry sector

#### 2. **Stock Prices Table** (Time-series stock data)
- `id`: Primary key
- `ticker`: Foreign key referencing `stocks.ticker`
- `date`: Trade date
- `open_price`, `high_price`, `low_price`, `close_price`: Daily OHLC prices
- `volume`: Trading volume

#### 3. **Financial Reports Table** (Earnings data)
- `id`: Primary key
- `ticker`: Foreign key referencing `stocks.ticker`
- `report_date`: Date of financial report
- `revenue`, `net_income`, `eps`: Key financial metrics

#### 4. **Stock Trades Table** (User transactions)
- `id`: Primary key
- `ticker`: Foreign key referencing `stocks.ticker`
- `trade_date`: Date of transaction
- `trade_type`: 'BUY' or 'SELL'
- `trade_volume`: Number of shares traded
- `trade_price`: Price per share

#### 5. **Users Table** (Portfolio tracking users)
- `id`: Primary key
- `username`: Unique identifier
- `email`: User email

#### 6. **Portfolio Table** (User stock holdings)
- `id`: Primary key
- `user_id`: Foreign key referencing `users.id`
- `ticker`: Foreign key referencing `stocks.ticker`
- `quantity`: Number of shares held
- `avg_buy_price`: Average purchase price per share

## Model Fine-Tuning Strategy
To enhance forecasting accuracy, a fine-tuning pipeline will be implemented for training an LLM on financial data. The model will be based on DeepSeek B14 or DeepSeek R1-32B and fine-tuned on ASX 200 stock trends, earnings reports, and financial news sentiment analysis.

### Local Fine-Tuning (TODO)
**Hardware Requirements:**
- NVIDIA RTX 3080 (10GB/12GB VRAM) or better
- 32GB RAM
- SSD storage (at least 1TB recommended)

**Optimization Strategies:**
- Use 4-bit quantization with QLoRA to reduce VRAM usage
- Gradient accumulation to manage memory limitations
- Offload computations to CPU when necessary

### AWS Fine-Tuning
For larger-scale training, AWS will be used to leverage GPU-based cloud computing.

**AWS Instances for Fine-Tuning:**
- **g5.2xlarge** (NVIDIA A10G, 24GB VRAM) - Cost-effective for small-scale fine-tuning
- **p4d.24xlarge** (NVIDIA A100, 320GB VRAM) - High-performance full fine-tuning
- **p5.48xlarge** (NVIDIA H100, 640GB VRAM) - Optimal for training DeepSeek R1-32B at scale

**Storage & Data Pipeline:**
- PostgreSQL stores structured training data
- Data is extracted using Python (`psycopg2`, `pandas`)
- AWS S3 used for large dataset storage and retrieval

## Deployment Plan (TODO)
### Local Deployment
- Model runs via FastAPI for inference
- PostgreSQL database locally managed
- Frontend (optional) for user queries

### AWS Cloud Deployment
- Model served through an AWS EC2 instance
- PostgreSQL managed via AWS RDS
- API endpoints exposed for financial analysis and forecasting

## Next Steps
1. Automate daily stock and financial report updates into PostgreSQL
2. Implement data preprocessing for model fine-tuning
3. Deploy an inference API for real-time financial predictions
4. Optimize training pipelines for cost efficiency

This project aims to build an end-to-end financial analysis system that leverages AI for predictive insights while ensuring efficient data management and scalability.