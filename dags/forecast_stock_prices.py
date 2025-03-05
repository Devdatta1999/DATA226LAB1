from airflow import DAG
from airflow.decorators import task
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Constants
SNOWFLAKE_TABLE = "dev.raw.market_data"
FORECAST_TABLE = "dev.raw.forecasted_prices"
FINAL_TABLE = "dev.raw.final_market_data"
STOCKS = ["AAPL", "NVDA"]
FORECAST_DAYS = 7

def return_snowflake_conn():
    """Initialize Snowflake connection using Airflow Hook"""
    hook = SnowflakeHook(snowflake_conn_id='snowflake_conn')
    return hook.get_conn().cursor()

@task
def fetch_data_from_snowflake():
    """Fetch historical closing prices from Snowflake for model training."""
    cur = return_snowflake_conn()
    stock_data = {}

    for stock in STOCKS:
        query = f"""
        SELECT date, close FROM {SNOWFLAKE_TABLE} 
        WHERE symbol = '{stock}' 
        ORDER BY date;
        """
        cur.execute(query)
        df = pd.DataFrame(cur.fetchall(), columns=["date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        stock_data[stock] = df  # Store each stock's data separately

    return stock_data

@task
def train_and_forecast(stock_data):
    """Train a forecasting model and predict the next 7 days of closing prices."""
    predictions = []

    for stock, df in stock_data.items():
        df.set_index("date", inplace=True)  # Set date as index

        # Train ARIMA model (basic time series forecasting)
        model = sm.tsa.ARIMA(df["close"], order=(5, 1, 0))  # ARIMA(5,1,0)
        model_fit = model.fit()

        # Generate predictions for the next 7 days
        forecast_dates = [df.index[-1] + timedelta(days=i) for i in range(1, FORECAST_DAYS + 1)]
        forecast_values = model_fit.forecast(steps=FORECAST_DAYS)

        # Store predictions
        for date, value in zip(forecast_dates, forecast_values):
            predictions.append((stock, date.strftime("%Y-%m-%d"), float(value)))

    return predictions

@task
def store_forecast_in_snowflake(predictions):
    """Store the forecasted values in Snowflake with idempotency (delete before insert)."""
    cur = return_snowflake_conn()

    try:
        cur.execute("BEGIN;")

        # Delete existing forecasts for the next 7 days to ensure fresh data
        delete_sql = f"""
        DELETE FROM {FORECAST_TABLE} 
        WHERE DATE >= CURRENT_DATE;
        """
        cur.execute(delete_sql)

        # Insert new forecasted values
        insert_sql = f"""
        INSERT INTO {FORECAST_TABLE} (symbol, date, predicted_close) 
        VALUES (%s, %s, %s);
        """
        cur.executemany(insert_sql, predictions)

        cur.execute("COMMIT;")
    except Exception as e:
        cur.execute("ROLLBACK;")
        print(f"Error inserting forecast data into Snowflake: {e}")
        raise


@task
def merge_forecast_into_final_table():
    """Merge historical and forecasted data into the final table."""
    cur = return_snowflake_conn()

    try:
        cur.execute("BEGIN;")

        merge_sql = f"""
        CREATE OR REPLACE TABLE {FINAL_TABLE} AS 
        SELECT symbol, date, close AS final_price FROM {SNOWFLAKE_TABLE}
        UNION ALL
        SELECT symbol, date, predicted_close AS final_price FROM {FORECAST_TABLE};
        """
        cur.execute(merge_sql)

        cur.execute("COMMIT;")
    except Exception as e:
        cur.execute("ROLLBACK;")
        print(f"Error merging data into final table: {e}")
        raise

with DAG(
    dag_id='forecast_stock_prices',
    start_date=datetime(2025, 2, 21),
    schedule_interval='0 10 * * *',  # Runs daily at 10 AM after ETL DAG
    catchup=False,
    tags=['Forecasting', 'Stock'],
) as dag:
    
    stock_data = fetch_data_from_snowflake()
    predictions = train_and_forecast(stock_data)
    store_forecast_in_snowflake(predictions) >> merge_forecast_into_final_table()
