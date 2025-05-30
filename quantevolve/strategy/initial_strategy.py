import numpy as np
import pandas as pd


def run_strategy(market_data_df: pd.DataFrame, params: list) -> pd.Series:
    """
    Runs a trading strategy based on the provided market data and parameters.
    
    IMPORTANT: This function should only use data up to the current point
    to prevent look-ahead bias in backtesting.

    Args:
        market_data_df (pd.DataFrame): DataFrame with 'close' prices.
        params (list): Strategy parameters, e.g., [short_window, long_window] for SMA crossover.

    Returns:
        pd.Series: A Series of trading signals (1 for buy, -1 for sell, 0 for hold),
                   aligned with the input DataFrame's index.
    """
    # Initialize signals with zeros
    signals = pd.Series(index=market_data_df.index, data=0.0)
    
    # Validate parameters
    if len(params) < 2:
        # Return all zeros for invalid parameters
        return signals

    # EVOLVE-BLOCK-START
    # Simple Moving Average (SMA) Crossover Strategy
    try:
        short_window = max(2, int(abs(params[0])))  # Minimum 2 for meaningful SMA
        long_window = max(5, int(abs(params[1])))   # Minimum 5 for meaningful SMA
        
        # Ensure we have enough data
        if len(market_data_df) < long_window:
            return signals
            
        # Swap if short > long to maintain logical relationship
        if short_window >= long_window:
            short_window = max(2, long_window - 1)  # Keep short < long
            
        # Calculate SMAs with proper minimum periods to prevent look-ahead bias
        close_prices = market_data_df['close']
        short_sma = close_prices.rolling(
            window=short_window, 
            min_periods=short_window  # FIXED: Require full window
        ).mean()
        
        long_sma = close_prices.rolling(
            window=long_window, 
            min_periods=long_window   # FIXED: Require full window
        ).mean()
        
        # Simple crossover logic - only where we have valid SMAs
        valid_indices = (~short_sma.isna()) & (~long_sma.isna())
        
        if valid_indices.sum() > 1:  # Need at least 2 valid points for crossover
            # Get previous values for crossover detection
            short_prev = short_sma.shift(1)
            long_prev = long_sma.shift(1)
            
            # Buy signal: short crosses above long
            buy_condition = (
                valid_indices & 
                (~short_prev.isna()) & 
                (~long_prev.isna()) &
                (short_prev <= long_prev) & 
                (short_sma > long_sma)
            )
            
            # Sell signal: short crosses below long  
            sell_condition = (
                valid_indices & 
                (~short_prev.isna()) & 
                (~long_prev.isna()) &
                (short_prev >= long_prev) & 
                (short_sma < long_sma)
            )
            
            # Apply signals
            signals.loc[buy_condition] = 1.0
            signals.loc[sell_condition] = -1.0
            
    except Exception as e:
        # If any error occurs, return zero signals
        pass
    
    # EVOLVE-BLOCK-END
    
    return signals


def get_parameters() -> list:
    """
    Returns default parameters for the strategy.

    Returns:
        list: A list of parameters for the strategy.
    """
    # Default parameters for SMA crossover: [short_window, long_window]
    # UPDATED: Use more reasonable defaults for signal generation
    return [10, 30]  # 10-day and 30-day moving averages


def get_trading_signals_func():
    """
    Returns the `run_strategy` function.
    This allows the evaluator to get a handle to the strategy logic.
    """
    return run_strategy


if __name__ == "__main__":
    # Create a sample DataFrame for basic testing
    data_size = 50
    sample_data = {
        "timestamp": pd.to_datetime(pd.date_range(start="2023-01-01", periods=data_size, freq="D")),
        "open": np.random.rand(data_size) * 100 + 100,
        "high": np.random.rand(data_size) * 10 + 200,
        "low": np.random.rand(data_size) * 10 + 90,
        # 'close': np.random.rand(data_size) * 100 + 100,
        "volume": np.random.rand(data_size) * 1000 + 100,
    }
    # Create a close price series that shows some crossovers for testing
    close_prices = np.zeros(data_size)
    close_prices[0] = 100
    for i in range(1, data_size):
        # For NumPy arrays, direct indexing is still supported
        close_prices[i] = (
            close_prices[i - 1] + np.random.randn() * 2 + (np.sin(i / 5) * 2)
        )  # Add some trend and noise
    sample_data["close"] = close_prices

    market_df = pd.DataFrame(sample_data)
    market_df.set_index("timestamp", inplace=True)

    print("Sample Market Data:")
    print(market_df.head())
    print("\n")

    # Get default parameters
    default_params = get_parameters()
    # Always ensure format specifiers are properly matched to data types
    # For strings, use str() or regular string concatenation if needed
    print(f"Default Parameters (short_window, long_window): {default_params}")
    print("\n")

    # Get the strategy function
    strategy_func = get_trading_signals_func()

    # Run the strategy
    # Create a copy to avoid modifying the original df if run_strategy adds columns
    signals_output = strategy_func(market_df.copy(), default_params)

    print("Generated Signals (1=Buy, -1=Sell, 0=Hold):")
    # Print signals alongside close prices and SMAs for context
    result_df = market_df[["close"]].copy()
    # Recalculate SMAs for printing, as run_strategy might not return them
    # or might have operated on a copy.
    short_w, long_w = default_params[0], default_params[1]
    result_df["short_sma"] = result_df["close"].rolling(window=short_w, min_periods=1).mean()
    result_df["long_sma"] = result_df["close"].rolling(window=long_w, min_periods=1).mean()
    result_df["signal"] = signals_output

    print(result_df)

    # Verify signal counts
    print("\nSignal Counts:")
    print(signals_output.value_counts())

    # Test with edge case params
    print("\nTesting with short_window > long_window (e.g., [20, 10]):")
    params_edge = [20, 10]
    signals_edge = strategy_func(market_df.copy(), params_edge)
    print(signals_edge.value_counts())

    print("\nTesting with params where short_window == long_window (e.g., [10,10]):")
    params_equal = [10, 10]
    signals_equal = strategy_func(market_df.copy(), params_equal)
    print(signals_equal.value_counts())

    try:
        print("\nTesting with invalid params (e.g., [0, 10]):")
        strategy_func(market_df.copy(), [0, 10])
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        print("\nTesting with insufficient params (e.g., [10]):")
        strategy_func(market_df.copy(), [10])
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nInitial strategy script created and basic tests run.")
