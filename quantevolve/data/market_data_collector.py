import os
import sys
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for direct script execution
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import configuration utilities
from quantevolve.config import load_config


def fetch_and_save_market_data(
    symbol: str, interval: str, start_str: str, end_str: str, filepath: str, config_path: str = None
):
    """
    Fetches OHLCV market data from Binance, converts it to a Pandas DataFrame,
    and saves it to a CSV file. Uses API keys from config or environment variables.

    Args:
        symbol (str): The trading symbol (e.g., "BTCUSDT").
        interval (str): The kline interval (e.g., Client.KLINE_INTERVAL_1HOUR).
        start_str (str): The start date string (e.g., "1 Jan, 2023").
        end_str (str): The end date string (e.g., "1 Jan, 2024").
        filepath (str): The path to save the CSV file (e.g., "data/market_data.csv").
        config_path (str, optional): Path to the config file. If None, uses default config loading logic.
    """
    try:
        # Load configuration
        config = load_config(config_path)

        # Try to get API keys from config first, then fall back to environment variables
        api_key = None
        api_secret = None

        if hasattr(config, "data_collection") and config.data_collection:
            # Access data_collection attributes safely
            if hasattr(config.data_collection, "binance_api_key"):
                api_key = config.data_collection.binance_api_key
            if hasattr(config.data_collection, "binance_api_secret"):
                api_secret = config.data_collection.binance_api_secret

        # Fall back to environment variables if not in config or empty
        if not api_key or api_key.strip() == "":
            api_key = os.environ.get("BINANCE_API_KEY")
        if not api_secret or api_secret.strip() == "":
            api_secret = os.environ.get("BINANCE_API_SECRET")

        if not api_key or not api_secret:
            print(
                "Error: Binance API key and/or secret not found in config or environment variables."
            )
            print(
                "Please add them to configs/quantevolve_config.yaml under data_collection section"
            )
            print("or set the BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")
            # Attempt to install python-binance if not available
            print("Attempting to install python-binance if not already installed...")
            os.system("pip install python-binance")
            return

        client = Client(api_key, api_secret)

        # Fetch klines (OHLCV) data
        klines = client.get_historical_klines(symbol, interval, start_str, end_str)

        if not klines:
            print(f"No data found for symbol {symbol} in the specified period.")
            return

        # Convert data to Pandas DataFrame
        df = pd.DataFrame(
            klines,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )

        # Select and convert necessary columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])

        # Create directory if it doesn't exist
        data_directory = os.path.dirname(filepath)
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
            print(f"Created directory: {data_directory}")

        # Save DataFrame to CSV
        df.to_csv(filepath, index=False)
        print(f"Market data saved to {filepath}")

    except Exception as e:
        print(f"An error occurred: {e}")
        # If the error is due to missing python-binance, try to install it.
        if "No module named 'binance'" in str(e):
            print("Error: python-binance library not found.")
            print("Attempting to install python-binance...")
            os.system("pip install python-binance")
            print("python-binance installation attempt complete. Please retry.")
        elif "APIError(code=-2014)" in str(e):  # Invalid API Key
            print("Error: Binance API key is invalid or permissions are missing.")
            print(
                "Please ensure your API key is correctly set and has trading/market data permissions."
            )
        elif "APIError(code=-2015)" in str(e):  # Invalid API Secret
            print("Error: Binance API secret is invalid.")
            print("Please ensure your API secret is correctly set.")


if __name__ == "__main__":
    # Load configuration
    config = load_config()

    # Get default parameters from config if available, otherwise use hardcoded defaults
    default_symbol = "BTCUSDT"
    default_interval = Client.KLINE_INTERVAL_1HOUR
    default_lookback_days = 365
    default_filepath = "examples/quant_evolve/data/btc_usdt_1h_2023.csv"

    # Try to get values from config
    if hasattr(config, "data_collection") and config.data_collection:
        if (
            hasattr(config.data_collection, "default_symbol")
            and config.data_collection.default_symbol
        ):
            default_symbol = config.data_collection.default_symbol

        if (
            hasattr(config.data_collection, "default_interval")
            and config.data_collection.default_interval
        ):
            # Map string interval representation to Client constants
            interval_map = {
                "1m": Client.KLINE_INTERVAL_1MINUTE,
                "3m": Client.KLINE_INTERVAL_3MINUTE,
                "5m": Client.KLINE_INTERVAL_5MINUTE,
                "15m": Client.KLINE_INTERVAL_15MINUTE,
                "30m": Client.KLINE_INTERVAL_30MINUTE,
                "1h": Client.KLINE_INTERVAL_1HOUR,
                "2h": Client.KLINE_INTERVAL_2HOUR,
                "4h": Client.KLINE_INTERVAL_4HOUR,
                "6h": Client.KLINE_INTERVAL_6HOUR,
                "8h": Client.KLINE_INTERVAL_8HOUR,
                "12h": Client.KLINE_INTERVAL_12HOUR,
                "1d": Client.KLINE_INTERVAL_1DAY,
                "3d": Client.KLINE_INTERVAL_3DAY,
                "1w": Client.KLINE_INTERVAL_1WEEK,
                "1M": Client.KLINE_INTERVAL_1MONTH,
            }
            if config.data_collection.default_interval in interval_map:
                default_interval = interval_map[config.data_collection.default_interval]

        if (
            hasattr(config.data_collection, "default_lookback_days")
            and config.data_collection.default_lookback_days
        ):
            default_lookback_days = config.data_collection.default_lookback_days

    # Calculate date range based on lookback days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=default_lookback_days)

    default_start_str = start_date.strftime("%d %b, %Y")
    default_end_str = end_date.strftime("%d %b, %Y")

    # Create a more descriptive default filepath based on the symbol and interval
    interval_str = (
        config.data_collection.default_interval
        if hasattr(config, "data_collection")
        and hasattr(config.data_collection, "default_interval")
        else "1h"
    )
    year_str = str(datetime.now().year)
    default_filepath = f"data/{default_symbol.lower()}_{interval_str}_{year_str}.csv"

    print(
        f"Fetching data for {default_symbol} from {default_start_str} to {default_end_str} ({interval_str})."
    )
    fetch_and_save_market_data(
        symbol=default_symbol,
        interval=default_interval,
        start_str=default_start_str,
        end_str=default_end_str,
        filepath=default_filepath,
    )
    print(f"Data saved to {default_filepath}")
    print("Script execution finished.")
