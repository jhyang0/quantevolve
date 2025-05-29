"""
Technical indicators module for QuantEvolve.

This module provides functions for calculating various technical indicators
used in trading strategies. This is a wrapper around TA-Lib to make it easier
to use in QuantEvolve strategies.

Reference: https://ta-lib.github.io/ta-lib-python/
"""

import numpy as np
import pandas as pd
import talib
from talib import abstract
from enum import Enum


# Define MA types for easier access
class MAType(Enum):
    SMA = 0  # Simple Moving Average
    EMA = 1  # Exponential Moving Average
    WMA = 2  # Weighted Moving Average
    DEMA = 3  # Double Exponential Moving Average
    TEMA = 4  # Triple Exponential Moving Average
    TRIMA = 5  # Triangular Moving Average
    KAMA = 6  # Kaufman Adaptive Moving Average
    MAMA = 7  # MESA Adaptive Moving Average
    T3 = 8  # Triple Exponential Moving Average T3


# Helper function to convert pandas DataFrame to format expected by TA-Lib abstract API
def _prepare_inputs(df):
    """
    Convert DataFrame to format expected by TA-Lib abstract API

    Args:
        df (pd.DataFrame): DataFrame with market data

    Returns:
        dict: Dictionary with inputs for TA-Lib
    """
    inputs = {}

    # Map column names to lowercase for talib
    column_mapping = {
        "open": ["open", "Open", "OPEN"],
        "high": ["high", "High", "HIGH"],
        "low": ["low", "Low", "LOW"],
        "close": ["close", "Close", "CLOSE"],
        "volume": ["volume", "Volume", "VOLUME"],
    }

    # Find matching columns in the DataFrame
    for talib_name, possible_names in column_mapping.items():
        for col_name in possible_names:
            if col_name in df.columns:
                inputs[talib_name] = df[col_name].values
                break

    return inputs


def add_moving_averages(df, periods=None, column="close", ma_type=MAType.SMA):
    """
    Add Moving Average columns to the dataframe using TA-Lib.

    Args:
        df (pd.DataFrame): DataFrame with market data
        periods (list): List of periods for MAs. Default: [5, 10, 20, 50, 200]
        column (str): Column to calculate moving averages for. Default: 'close'
        ma_type (MAType): Type of moving average. Default: MAType.SMA

    Returns:
        pd.DataFrame: DataFrame with added MA columns
    """
    result_df = df.copy()

    if periods is None:
        periods = [5, 10, 20, 50, 200]

    # Get the correct column data
    if column in result_df.columns:
        price_data = result_df[column].values
    else:
        # Default to close if specified column doesn't exist
        price_data = result_df["close"].values if "close" in result_df.columns else None

    if price_data is None:
        raise ValueError(
            f"Column '{column}' not found in DataFrame and no 'close' column available"
        )

    # Get the MA function based on ma_type
    if ma_type == MAType.SMA:
        ma_func = talib.SMA
        ma_name = "sma"
    elif ma_type == MAType.EMA:
        ma_func = talib.EMA
        ma_name = "ema"
    elif ma_type == MAType.WMA:
        ma_func = talib.WMA
        ma_name = "wma"
    elif ma_type == MAType.DEMA:
        ma_func = talib.DEMA
        ma_name = "dema"
    elif ma_type == MAType.TEMA:
        ma_func = talib.TEMA
        ma_name = "tema"
    elif ma_type == MAType.TRIMA:
        ma_func = talib.TRIMA
        ma_name = "trima"
    elif ma_type == MAType.KAMA:
        ma_func = talib.KAMA
        ma_name = "kama"
    elif ma_type == MAType.T3:
        ma_func = talib.T3
        ma_name = "t3"
    else:
        ma_func = talib.SMA
        ma_name = "sma"

    # Calculate MA for each period
    for period in periods:
        result_df[f"{ma_name}_{period}"] = ma_func(price_data, timeperiod=period)

    return result_df


def add_rsi(df, periods=None, column="close"):
    """
    Add Relative Strength Index (RSI) to the dataframe using TA-Lib.

    Args:
        df (pd.DataFrame): DataFrame with market data
        periods (list): List of periods for RSI. Default: [14]
        column (str): Column to calculate RSI for. Default: 'close'

    Returns:
        pd.DataFrame: DataFrame with added RSI columns
    """
    result_df = df.copy()

    if periods is None:
        periods = [14]

    # Get the correct column data
    if column in result_df.columns:
        price_data = result_df[column].values
    else:
        # Default to close if specified column doesn't exist
        price_data = result_df["close"].values if "close" in result_df.columns else None

    if price_data is None:
        raise ValueError(
            f"Column '{column}' not found in DataFrame and no 'close' column available"
        )

    # Calculate RSI for each period
    for period in periods:
        result_df[f"rsi_{period}"] = talib.RSI(price_data, timeperiod=period)

    return result_df


def add_bollinger_bands(df, periods=None, deviations=2.0, column="close"):
    """
    Add Bollinger Bands to the dataframe using TA-Lib.

    Args:
        df (pd.DataFrame): DataFrame with market data
        periods (list): List of periods for Bollinger Bands. Default: [20]
        deviations (float): Number of standard deviations. Default: 2.0
        column (str): Column to calculate Bollinger Bands for. Default: 'close'

    Returns:
        pd.DataFrame: DataFrame with added Bollinger Bands columns
    """
    result_df = df.copy()

    if periods is None:
        periods = [20]

    # Get the correct column data
    if column in result_df.columns:
        price_data = result_df[column].values
    else:
        # Default to close if specified column doesn't exist
        price_data = result_df["close"].values if "close" in result_df.columns else None

    if price_data is None:
        raise ValueError(
            f"Column '{column}' not found in DataFrame and no 'close' column available"
        )

    # Calculate Bollinger Bands for each period
    for period in periods:
        upper, middle, lower = talib.BBANDS(
            price_data,
            timeperiod=period,
            nbdevup=deviations,
            nbdevdn=deviations,
            matype=0,  # Simple Moving Average
        )
        result_df[f"bb_upper_{period}"] = upper
        result_df[f"bb_middle_{period}"] = middle
        result_df[f"bb_lower_{period}"] = lower

    return result_df


def add_macd(df, column="close", fastperiod=12, slowperiod=26, signalperiod=9):
    """
    Add Moving Average Convergence Divergence (MACD) to the dataframe using TA-Lib.

    Args:
        df (pd.DataFrame): DataFrame with market data
        column (str): Column to calculate MACD for. Default: 'close'
        fastperiod (int): Fast period. Default: 12
        slowperiod (int): Slow period. Default: 26
        signalperiod (int): Signal period. Default: 9

    Returns:
        pd.DataFrame: DataFrame with added MACD columns
    """
    result_df = df.copy()

    # Get the correct column data
    if column in result_df.columns:
        price_data = result_df[column].values
    else:
        # Default to close if specified column doesn't exist
        price_data = result_df["close"].values if "close" in result_df.columns else None

    if price_data is None:
        raise ValueError(
            f"Column '{column}' not found in DataFrame and no 'close' column available"
        )

    # Calculate MACD
    macd, macdsignal, macdhist = talib.MACD(
        price_data, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod
    )

    result_df["macd_line"] = macd
    result_df["macd_signal"] = macdsignal
    result_df["macd_histogram"] = macdhist

    return result_df


def add_atr(df, period=14):
    """
    Add Average True Range (ATR) to the dataframe using TA-Lib.

    Args:
        df (pd.DataFrame): DataFrame with market data (must include high, low, close)
        period (int): Period for ATR calculation. Default: 14

    Returns:
        pd.DataFrame: DataFrame with added ATR column
    """
    result_df = df.copy()

    # Check if required columns exist
    required_cols = ["high", "low", "close"]
    for col in required_cols:
        if col not in result_df.columns and col.capitalize() not in result_df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    # Get high, low, close data
    high = result_df["high"].values if "high" in result_df.columns else result_df["High"].values
    low = result_df["low"].values if "low" in result_df.columns else result_df["Low"].values
    close = result_df["close"].values if "close" in result_df.columns else result_df["Close"].values

    # Calculate ATR
    result_df["atr"] = talib.ATR(high, low, close, timeperiod=period)

    # Add ATR percentage of price (volatility measure)
    result_df["atr_pct"] = result_df["atr"] / close * 100

    return result_df


def add_stochastic(df, fastk_period=14, slowk_period=3, slowd_period=3):
    """
    Add Stochastic Oscillator to the dataframe using TA-Lib.

    Args:
        df (pd.DataFrame): DataFrame with market data (must include high, low, close)
        fastk_period (int): Fast %K period. Default: 14
        slowk_period (int): Slow %K period. Default: 3
        slowd_period (int): Slow %D period. Default: 3

    Returns:
        pd.DataFrame: DataFrame with added Stochastic Oscillator columns
    """
    result_df = df.copy()

    # Check if required columns exist
    required_cols = ["high", "low", "close"]
    for col in required_cols:
        if col not in result_df.columns and col.capitalize() not in result_df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    # Get high, low, close data
    high = result_df["high"].values if "high" in result_df.columns else result_df["High"].values
    low = result_df["low"].values if "low" in result_df.columns else result_df["Low"].values
    close = result_df["close"].values if "close" in result_df.columns else result_df["Close"].values

    # Calculate Stochastic
    slowk, slowd = talib.STOCH(
        high,
        low,
        close,
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowk_matype=0,
        slowd_period=slowd_period,
        slowd_matype=0,
    )

    result_df["stoch_k"] = slowk
    result_df["stoch_d"] = slowd

    return result_df


def add_ichimoku(df, tenkan_period=9, kijun_period=26, senkou_span_b_period=52):
    """
    Add Ichimoku Cloud components to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with market data (must include high, low, close)
        tenkan_period (int): Tenkan-sen (Conversion Line) period. Default: 9
        kijun_period (int): Kijun-sen (Base Line) period. Default: 26
        senkou_span_b_period (int): Senkou Span B period. Default: 52

    Returns:
        pd.DataFrame: DataFrame with added Ichimoku Cloud columns
    """
    result_df = df.copy()

    # Check if required columns exist
    required_cols = ["high", "low", "close"]
    for col in required_cols:
        if col not in result_df.columns and col.capitalize() not in result_df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    # Get high, low data
    high = result_df["high"].values if "high" in result_df.columns else result_df["High"].values
    low = result_df["low"].values if "low" in result_df.columns else result_df["Low"].values
    close = result_df["close"].values if "close" in result_df.columns else result_df["Close"].values

    # Calculate Tenkan-sen (Conversion Line)
    tenkan_high = talib.MAX(high, timeperiod=tenkan_period)
    tenkan_low = talib.MIN(low, timeperiod=tenkan_period)
    result_df["ichimoku_tenkan"] = (tenkan_high + tenkan_low) / 2

    # Calculate Kijun-sen (Base Line)
    kijun_high = talib.MAX(high, timeperiod=kijun_period)
    kijun_low = talib.MIN(low, timeperiod=kijun_period)
    result_df["ichimoku_kijun"] = (kijun_high + kijun_low) / 2

    # Calculate Senkou Span A (Leading Span A)
    result_df["ichimoku_senkou_a"] = (
        result_df["ichimoku_tenkan"] + result_df["ichimoku_kijun"]
    ) / 2

    # Calculate Senkou Span B (Leading Span B)
    senkou_b_high = talib.MAX(high, timeperiod=senkou_span_b_period)
    senkou_b_low = talib.MIN(low, timeperiod=senkou_span_b_period)
    result_df["ichimoku_senkou_b"] = (senkou_b_high + senkou_b_low) / 2

    # Calculate Chikou Span (Lagging Span)
    result_df["ichimoku_chikou"] = pd.Series(close).shift(-kijun_period)

    return result_df


# Signal utility functions


def crossover(df, series1, series2):
    """
    Detect when series1 crosses above series2.

    Args:
        df (pd.DataFrame): DataFrame containing the series
        series1 (str): Column name of first series
        series2 (str): Column name of second series

    Returns:
        pd.Series: Boolean series, True when crossover occurs
    """
    if series1 not in df.columns or series2 not in df.columns:
        raise ValueError(f"One or both series '{series1}', '{series2}' not found in DataFrame")

    # Current period: series1 > series2
    # Previous period: series1 <= series2
    s1 = df[series1]
    s2 = df[series2]

    result = pd.Series(False, index=df.index)
    result[1:] = (s1[1:] > s2[1:]) & (s1[:-1].values <= s2[:-1].values)

    return result


def crossunder(df, series1, series2):
    """
    Detect when series1 crosses below series2.

    Args:
        df (pd.DataFrame): DataFrame containing the series
        series1 (str): Column name of first series
        series2 (str): Column name of second series

    Returns:
        pd.Series: Boolean series, True when crossunder occurs
    """
    if series1 not in df.columns or series2 not in df.columns:
        raise ValueError(f"One or both series '{series1}', '{series2}' not found in DataFrame")

    # Current period: series1 < series2
    # Previous period: series1 >= series2
    s1 = df[series1]
    s2 = df[series2]

    result = pd.Series(False, index=df.index)
    result[1:] = (s1[1:] < s2[1:]) & (s1[:-1].values >= s2[:-1].values)

    return result


def add_all_indicators(df, indicators=None):
    """
    Add multiple technical indicators to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with market data
        indicators (dict): Dictionary specifying which indicators to add and their parameters.
                          Example: {
                              'sma': {'periods': [10, 20, 50]},
                              'rsi': {'periods': [14]},
                              'bollinger': {'periods': [20], 'deviations': 2.0},
                              'macd': {},  # Use default parameters
                              'atr': {'period': 14},
                              'stochastic': {},  # Use default parameters
                              'ichimoku': {}  # Use default parameters
                          }

    Returns:
        pd.DataFrame: DataFrame with all requested indicators added
    """
    result_df = df.copy()

    if indicators is None:
        # Default: Add SMA, RSI, and ATR with default parameters
        indicators = {"sma": {}, "rsi": {}, "atr": {}}

    # Process each requested indicator
    for indicator, params in indicators.items():
        if indicator == "sma" or indicator == "ma":
            periods = params.get("periods", None)
            column = params.get("column", "close")
            ma_type = params.get("ma_type", MAType.SMA)
            result_df = add_moving_averages(result_df, periods, column, ma_type)

        elif indicator == "ema":
            periods = params.get("periods", None)
            column = params.get("column", "close")
            result_df = add_moving_averages(result_df, periods, column, MAType.EMA)

        elif indicator == "rsi":
            periods = params.get("periods", None)
            column = params.get("column", "close")
            result_df = add_rsi(result_df, periods, column)

        elif indicator == "bollinger" or indicator == "bbands":
            periods = params.get("periods", None)
            deviations = params.get("deviations", 2.0)
            column = params.get("column", "close")
            result_df = add_bollinger_bands(result_df, periods, deviations, column)

        elif indicator == "macd":
            column = params.get("column", "close")
            fastperiod = params.get("fastperiod", 12)
            slowperiod = params.get("slowperiod", 26)
            signalperiod = params.get("signalperiod", 9)
            result_df = add_macd(result_df, column, fastperiod, slowperiod, signalperiod)

        elif indicator == "atr":
            period = params.get("period", 14)
            result_df = add_atr(result_df, period)

        elif indicator == "stochastic":
            fastk_period = params.get("fastk_period", 14)
            slowk_period = params.get("slowk_period", 3)
            slowd_period = params.get("slowd_period", 3)
            result_df = add_stochastic(result_df, fastk_period, slowk_period, slowd_period)

        elif indicator == "ichimoku":
            tenkan_period = params.get("tenkan_period", 9)
            kijun_period = params.get("kijun_period", 26)
            senkou_span_b_period = params.get("senkou_span_b_period", 52)
            result_df = add_ichimoku(result_df, tenkan_period, kijun_period, senkou_span_b_period)

        else:
            raise ValueError(f"Unknown indicator type: {indicator}")

    return result_df


# Alias for backward compatibility
def add_exponential_moving_averages(df, periods=None, column="close"):
    """
    Add Exponential Moving Average (EMA) columns to the dataframe.
    Alias for add_moving_averages with ma_type=MAType.EMA for backward compatibility.

    Args:
        df (pd.DataFrame): DataFrame with market data
        periods (list): List of periods for EMAs. Default: [5, 10, 20, 50, 200]
        column (str): Column to calculate moving averages for. Default: 'close'

    Returns:
        pd.DataFrame: DataFrame with added EMA columns
    """
    return add_moving_averages(df, periods, column, ma_type=MAType.EMA)


def add_rsi(df, period=14, column="close"):
    """
    Add Relative Strength Index (RSI) to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with market data
        period (int): Period for RSI calculation. Default: 14
        column (str): Column to calculate RSI for. Default: 'close'

    Returns:
        pd.DataFrame: DataFrame with added RSI column
    """
    result_df = df.copy()
    delta = result_df[column].diff()

    gain = delta.copy()
    gain[gain < 0] = 0

    loss = delta.copy()
    loss[loss > 0] = 0
    loss = abs(loss)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
    result_df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    return result_df


def add_bollinger_bands(df, period=20, std_dev=2, column="close"):
    """
    Add Bollinger Bands to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with market data
        period (int): Period for moving average. Default: 20
        std_dev (int/float): Number of standard deviations. Default: 2
        column (str): Column to calculate bands for. Default: 'close'

    Returns:
        pd.DataFrame: DataFrame with added Bollinger Bands columns
    """
    result_df = df.copy()

    # Calculate middle band (SMA)
    result_df[f"bb_middle_{period}"] = (
        result_df[column].rolling(window=period, min_periods=1).mean()
    )

    # Calculate standard deviation
    rolling_std = result_df[column].rolling(window=period, min_periods=1).std()

    # Calculate upper and lower bands
    result_df[f"bb_upper_{period}"] = result_df[f"bb_middle_{period}"] + (rolling_std * std_dev)
    result_df[f"bb_lower_{period}"] = result_df[f"bb_middle_{period}"] - (rolling_std * std_dev)

    return result_df


def add_macd(df, fast_period=12, slow_period=26, signal_period=9, column="close"):
    """
    Add Moving Average Convergence Divergence (MACD) to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with market data
        fast_period (int): Period for fast EMA. Default: 12
        slow_period (int): Period for slow EMA. Default: 26
        signal_period (int): Period for signal line. Default: 9
        column (str): Column to calculate MACD for. Default: 'close'

    Returns:
        pd.DataFrame: DataFrame with added MACD columns
    """
    result_df = df.copy()

    # Calculate fast and slow EMAs
    fast_ema = result_df[column].ewm(span=fast_period, adjust=False).mean()
    slow_ema = result_df[column].ewm(span=slow_period, adjust=False).mean()

    # Calculate MACD line and signal line
    result_df["macd_line"] = fast_ema - slow_ema
    result_df["macd_signal"] = result_df["macd_line"].ewm(span=signal_period, adjust=False).mean()

    # Calculate MACD histogram
    result_df["macd_histogram"] = result_df["macd_line"] - result_df["macd_signal"]

    return result_df


def add_atr(df, period=14):
    """
    Add Average True Range (ATR) to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with market data (must include 'high', 'low', 'close')
        period (int): Period for ATR calculation. Default: 14

    Returns:
        pd.DataFrame: DataFrame with added ATR column
    """
    result_df = df.copy()

    # Calculate True Range
    result_df["tr0"] = abs(result_df["high"] - result_df["low"])
    result_df["tr1"] = abs(result_df["high"] - result_df["close"].shift(1))
    result_df["tr2"] = abs(result_df["low"] - result_df["close"].shift(1))
    result_df["tr"] = result_df[["tr0", "tr1", "tr2"]].max(axis=1)

    # Calculate ATR
    result_df["atr"] = result_df["tr"].rolling(window=period, min_periods=1).mean()

    # Clean up temporary columns
    result_df = result_df.drop(["tr0", "tr1", "tr2", "tr"], axis=1)

    return result_df


def add_stochastic(df, k_period=14, d_period=3, smooth_k=3):
    """
    Add Stochastic Oscillator to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with market data
        k_period (int): Period for %K line. Default: 14
        d_period (int): Period for %D line. Default: 3
        smooth_k (int): Period for smoothing %K. Default: 3

    Returns:
        pd.DataFrame: DataFrame with added Stochastic Oscillator columns
    """
    result_df = df.copy()

    # Calculate %K
    low_min = result_df["low"].rolling(window=k_period, min_periods=1).min()
    high_max = result_df["high"].rolling(window=k_period, min_periods=1).max()

    # Handle division by zero
    denom = high_max - low_min
    denom = denom.replace(0, np.finfo(float).eps)

    result_df["stoch_%k_raw"] = 100 * ((result_df["close"] - low_min) / denom)

    # Apply smoothing to %K if requested
    if smooth_k > 1:
        result_df["stoch_%k"] = (
            result_df["stoch_%k_raw"].rolling(window=smooth_k, min_periods=1).mean()
        )
    else:
        result_df["stoch_%k"] = result_df["stoch_%k_raw"]

    # Calculate %D (SMA of %K)
    result_df["stoch_%d"] = result_df["stoch_%k"].rolling(window=d_period, min_periods=1).mean()

    # Clean up temporary columns
    result_df = result_df.drop(["stoch_%k_raw"], axis=1)

    return result_df


def add_ichimoku(df, tenkan_period=9, kijun_period=26, senkou_b_period=52, chikou_period=26):
    """
    Add Ichimoku Cloud indicators to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with market data
        tenkan_period (int): Tenkan-sen (Conversion Line) period. Default: 9
        kijun_period (int): Kijun-sen (Base Line) period. Default: 26
        senkou_b_period (int): Senkou Span B period. Default: 52
        chikou_period (int): Chikou Span (Lagging Span) period. Default: 26

    Returns:
        pd.DataFrame: DataFrame with added Ichimoku Cloud columns
    """
    result_df = df.copy()

    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for the past 9 periods
    high_tenkan = result_df["high"].rolling(window=tenkan_period, min_periods=1).max()
    low_tenkan = result_df["low"].rolling(window=tenkan_period, min_periods=1).min()
    result_df["ichimoku_tenkan"] = (high_tenkan + low_tenkan) / 2

    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for the past 26 periods
    high_kijun = result_df["high"].rolling(window=kijun_period, min_periods=1).max()
    low_kijun = result_df["low"].rolling(window=kijun_period, min_periods=1).min()
    result_df["ichimoku_kijun"] = (high_kijun + low_kijun) / 2

    # Calculate Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2 shifted forward 26 periods
    result_df["ichimoku_senkou_a"] = (
        (result_df["ichimoku_tenkan"] + result_df["ichimoku_kijun"]) / 2
    ).shift(kijun_period)

    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for the past 52 periods, shifted forward 26 periods
    high_senkou = result_df["high"].rolling(window=senkou_b_period, min_periods=1).max()
    low_senkou = result_df["low"].rolling(window=senkou_b_period, min_periods=1).min()
    result_df["ichimoku_senkou_b"] = ((high_senkou + low_senkou) / 2).shift(kijun_period)

    # Calculate Chikou Span (Lagging Span): Current closing price shifted backwards 26 periods
    result_df["ichimoku_chikou"] = result_df["close"].shift(-chikou_period)

    return result_df


def add_volume_indicators(df, period=20):
    """
    Add volume-based indicators to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with market data
        period (int): Period for volume indicators. Default: 20

    Returns:
        pd.DataFrame: DataFrame with added volume indicators
    """
    result_df = df.copy()

    # Volume Moving Average
    result_df[f"volume_sma_{period}"] = (
        result_df["volume"].rolling(window=period, min_periods=1).mean()
    )

    # Volume Ratio (current volume / average volume)
    result_df["volume_ratio"] = result_df["volume"] / result_df[f"volume_sma_{period}"].replace(
        0, np.finfo(float).eps
    )

    # On-Balance Volume (OBV)
    result_df["obv"] = 0
    for i in range(1, len(result_df)):
        if result_df["close"].iloc[i] > result_df["close"].iloc[i - 1]:
            result_df["obv"].iloc[i] = result_df["obv"].iloc[i - 1] + result_df["volume"].iloc[i]
        elif result_df["close"].iloc[i] < result_df["close"].iloc[i - 1]:
            result_df["obv"].iloc[i] = result_df["obv"].iloc[i - 1] - result_df["volume"].iloc[i]
        else:
            result_df["obv"].iloc[i] = result_df["obv"].iloc[i - 1]

    return result_df


def add_all_indicators(df, include=None, exclude=None):
    """
    Add all or selected technical indicators to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with market data
        include (list): List of indicator functions to include. Default: all
        exclude (list): List of indicator functions to exclude. Default: none

    Returns:
        pd.DataFrame: DataFrame with all requested indicators
    """
    result_df = df.copy()

    # Define all available indicator functions
    all_indicators = {
        "sma": add_moving_averages,
        "ema": add_exponential_moving_averages,
        "rsi": add_rsi,
        "bollinger": add_bollinger_bands,
        "macd": add_macd,
        "atr": add_atr,
        "stochastic": add_stochastic,
        "ichimoku": add_ichimoku,
        "volume": add_volume_indicators,
    }

    # Determine which indicators to calculate
    indicators_to_add = set(all_indicators.keys())

    if include is not None:
        indicators_to_add = indicators_to_add.intersection(include)

    if exclude is not None:
        indicators_to_add = indicators_to_add - set(exclude)

    # Calculate each requested indicator
    for indicator_name in indicators_to_add:
        result_df = all_indicators[indicator_name](result_df)

    return result_df


# Utility functions for trading strategy development


def crossover(series1, series2):
    """
    Detect when series1 crosses above series2.

    Args:
        series1 (pd.Series): First series
        series2 (pd.Series): Second series

    Returns:
        pd.Series: Boolean series, True when crossover occurs
    """
    # Current period: series1 > series2
    # Previous period: series1 <= series2
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))


def crossunder(series1, series2):
    """
    Detect when series1 crosses below series2.

    Args:
        series1 (pd.Series): First series
        series2 (pd.Series): Second series

    Returns:
        pd.Series: Boolean series, True when crossunder occurs
    """
    # Current period: series1 < series2
    # Previous period: series1 >= series2
    return (series1 < series2) & (series1.shift(1) >= series2.shift(1))


def percentile_rank(series, period=20):
    """
    Calculate the percentile rank of each value in the series over the past N periods.

    Args:
        series (pd.Series): Input data series
        period (int): Lookback period

    Returns:
        pd.Series: Percentile rank (0-100) of each value
    """

    def rolling_percentile(window):
        if len(window) == 0:
            return np.nan
        current = window[-1]
        rank = sum(1 for x in window if x < current)
        return 100 * rank / (len(window) - 1) if len(window) > 1 else 50

    result = pd.Series(index=series.index)
    for i in range(len(series)):
        window_start = max(0, i - period + 1)
        window = series.iloc[window_start : i + 1].values
        result.iloc[i] = rolling_percentile(window)

    return result


def highest(series, period):
    """
    Get the highest value in the series over the past N periods.

    Args:
        series (pd.Series): Input data series
        period (int): Lookback period

    Returns:
        pd.Series: Series of highest values
    """
    return series.rolling(window=period, min_periods=1).max()


def lowest(series, period):
    """
    Get the lowest value in the series over the past N periods.

    Args:
        series (pd.Series): Input data series
        period (int): Lookback period

    Returns:
        pd.Series: Series of lowest values
    """
    return series.rolling(window=period, min_periods=1).min()
