from airflow import DAG
from airflow.decorators import task
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import os

# Define constants
STOCKS = ["AAPL", "NVDA"]
DAYS_HISTORY = 180
SNOWFLAKE_TABLE = "dev.raw.market_data"
TEMP_DIR = "/tmp/"
SNOWFLAKE_STAGE = "dev.raw.stock_stage"

def return_snowflake_conn():
    """Initialize Snowflake connection using Airflow Hook"""
    hook = SnowflakeHook(snowflake_conn_id='snowflake_conn')
    return hook.get_conn().cursor()

@task
def fetch_stock_data():
    """Fetch historical stock data for AAPL & NVDA from yFinance and save as CSV files."""
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=DAYS_HISTORY)).strftime('%Y-%m-%d')

    file_paths = []

    for stock in STOCKS:
        data = yf.download(stock, start=start_date, end=end_date)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)  # Drop ticker level if MultiIndex

        data["Symbol"] = stock  # Add symbol column
        data.reset_index(inplace=True)  # Ensure 'Date' is a column

        # Reorder to match Snowflake schema
        data = data[['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        file_path = os.path.join(TEMP_DIR, f"{stock}_data.csv")

        data.to_csv(file_path, index=False)
        file_paths.append(file_path)

    return file_paths

@task
def load_to_snowflake(file_paths):
    """Upload CSV files to Snowflake stage and run COPY INTO with full refresh (idempotency)."""
    cur = return_snowflake_conn()
    try:
        cur.execute("BEGIN;")

        # Ensure the session is using the correct database and schema
        cur.execute("USE SCHEMA dev.raw;")

        # Create Snowflake stage if it doesn't exist
        create_stage_sql = f"""
        CREATE OR REPLACE STAGE {SNOWFLAKE_STAGE}
        FILE_FORMAT = (TYPE = 'CSV', FIELD_OPTIONALLY_ENCLOSED_BY='"', SKIP_HEADER=1);
        """
        cur.execute(create_stage_sql)

        # Initialize Snowflake Hook for file upload
        hook = SnowflakeHook(snowflake_conn_id="snowflake_conn")

        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            symbol = file_name.split("_")[0]  # Extract stock symbol from filename (e.g., AAPL_data.csv -> AAPL)

            # Upload file using Snowflake Hook
            hook.run(f"PUT 'file://{file_path}' @{SNOWFLAKE_STAGE}/{file_name} AUTO_COMPRESS=TRUE OVERWRITE=TRUE")

            # Define the fully qualified temp table name
            temp_table = "dev.raw.temp_delete_dates"

            # Create a temporary table to store distinct dates
            cur.execute(f"CREATE OR REPLACE TEMP TABLE {temp_table} (DATE DATE);")

            # Load unique dates from the staged file into the temporary table
            load_dates_sql = f"""
            COPY INTO {temp_table} (DATE)
            FROM (SELECT DISTINCT $2 FROM @{SNOWFLAKE_STAGE}/{file_name})
            FILE_FORMAT = (TYPE = 'CSV', FIELD_OPTIONALLY_ENCLOSED_BY='"', SKIP_HEADER=1);
            """
            cur.execute(load_dates_sql)

            # Delete only rows matching the same symbol and the extracted dates
            delete_sql = f"""
            DELETE FROM {SNOWFLAKE_TABLE} 
            WHERE SYMBOL = '{symbol}' 
            AND DATE IN (SELECT DATE FROM {temp_table});
            """
            cur.execute(delete_sql)

            # Copy data into Snowflake
            copy_sql = f"""
            COPY INTO {SNOWFLAKE_TABLE}
            FROM @{SNOWFLAKE_STAGE}/{file_name}
            FILE_FORMAT = (TYPE = 'CSV', FIELD_OPTIONALLY_ENCLOSED_BY='"', SKIP_HEADER=1)
            ON_ERROR = 'CONTINUE';
            """
            cur.execute(copy_sql)

            os.remove(file_path)  # Cleanup temporary file

        cur.execute("COMMIT;")
    except Exception as e:
        cur.execute("ROLLBACK;")
        print(f"Error loading data to Snowflake: {e}")
        raise




with DAG(
    dag_id='extract_stock_data',
    start_date=datetime(2025, 2, 21),
    schedule_interval='0 9 * * *',  # Daily at 9 AM
    catchup=False,
    tags=['ETL', 'Stock'],
) as dag:
    stock_files = fetch_stock_data()
    load_to_snowflake(stock_files)
