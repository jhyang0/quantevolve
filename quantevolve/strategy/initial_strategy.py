import numpy as np
import pandas as pd


def run_strategy(market_data_df: pd.DataFrame, params: list) -> pd.Series:
    """
    Runs a trading strategy based on the provided market data and parameters.

    Args:
        market_data_df (pd.DataFrame): DataFrame with 'close' prices.
        params (list): Strategy parameters, e.g., [short_window, long_window] for SMA crossover.

    Returns:
        pd.Series: A Series of trading signals (1 for buy, -1 for sell, 0 for hold),
                   aligned with the input DataFrame's index.
    """
    signals = pd.Series(index=market_data_df.index, data=0.0)

    # Important: When working with Series indexed by integers, always use .iloc[] for positional indexing
    # Example: signals.iloc[i] = 1.0  # NOT signals[i] = 1.0

    # EVOLVE-BLOCK-START
    # Simple Moving Average (SMA) Crossover Strategy
    if len(params) < 2:
        raise ValueError(
            "SMA Crossover strategy requires at least two parameters: short_window and long_window."
        )

    short_window = int(params[0])
    long_window = int(params[1])

    if short_window <= 0 or long_window <= 0:
        raise ValueError("Window sizes must be positive.")
    if short_window >= long_window:
        # This is a common convention, though some strategies might invert this.
        # For a typical crossover, short should be less than long.
        # Depending on evolution, this constraint might be relaxed or handled differently.
        # For now, let's assume short_window < long_window for a clear crossover.
        # Alternatively, return all zeros or handle as an invalid parameter set.
        # print("Warning: Short window should ideally be less than long window for a typical crossover. Adjusting params or returning no signals.")
        # For now, we allow it but it might result in no signals or inverse signals depending on interpretation.
        pass

    # Calculate short-term SMA
    market_data_df["short_sma"] = (
        market_data_df["close"].rolling(window=short_window, min_periods=1).mean()
    )

    # Calculate long-term SMA
    market_data_df["long_sma"] = (
        market_data_df["close"].rolling(window=long_window, min_periods=1).mean()
    )

    # Generate signals
    # Initial state: no signal (already 0.0)
    # Buy signal (1): short SMA crosses above long SMA
    signals[market_data_df["short_sma"] > market_data_df["long_sma"]] = 1.0
    # Sell signal (-1): short SMA crosses below long SMA
    signals[market_data_df["short_sma"] < market_data_df["long_sma"]] = -1.0

    # More precise crossover detection:
    # A buy signal is generated when the short SMA was below or equal to the long SMA in the previous period,
    # AND the short SMA is above the long SMA in the current period.
    # A sell signal is generated when the short SMA was above or equal to the long SMA in the previous period,
    # AND the short SMA is below the long SMA in the current period.

    # Create a 'position' Series: 1 if short > long, -1 if short < long, 0 if equal (or NaN initially)
    position = pd.Series(index=market_data_df.index, data=0.0)
    position[market_data_df["short_sma"] > market_data_df["long_sma"]] = 1.0
    position[market_data_df["short_sma"] < market_data_df["long_sma"]] = -1.0

    # Find actual crossover points
    # The signal is the change in position from the previous period
    # .diff() will result in NaN for the first row, which is fine (no signal)
    # A change from -1 to 1 is a buy (diff = 2). A change from 1 to -1 is a sell (diff = -2).
    # We care about the *direction* of the cross.
    # If position was -1 (short < long) and is now 1 (short > long) -> Buy
    # If position was 1 (short > long) and is now -1 (short < long) -> Sell

    # Shift position to compare current with previous
    prev_position = position.shift(1)

    # Generate buy signals
    buy_signals = (prev_position < 1) & (
        position == 1
    )  # short SMA crossed above long SMA (was <=, now >)

    # Generate sell signals
    sell_signals = (prev_position > -1) & (
        position == -1
    )  # short SMA crossed below long SMA (was >=, now <)

    signals[buy_signals] = 1.0
    signals[sell_signals] = -1.0

    # Remove SMAs from original df if added temporarily for calculation, to avoid side effects
    # market_data_df.drop(columns=['short_sma', 'long_sma'], inplace=True, errors='ignore')
    # For this structure, it's better if run_strategy does not modify the input df.
    # So, create copies for SMA calculation.

    # EVOLVE-BLOCK-END

    # Example of proper pandas Series indexing using .iloc (positional indexer)
    # Instead of signals[i] = value, use signals.iloc[i] = value
    # Instead of signals[i], use signals.iloc[i]
    #
    # For example, if you want to set specific position values:
    # for i in range(len(signals)):
    #     if some_condition:
    #         signals.iloc[i] = 1.0  # Buy signal
    #     elif another_condition:
    #         signals.iloc[i] = -1.0  # Sell signal
    #     else:
    #         signals.iloc[i] = 0.0  # Hold
    #
    # Ensure signals are same length as input, padding with 0s where SMAs are not defined.
    # The `signals` Series is already initialized with zeros and has the same index.
    # Rolling SMAs with min_periods=1 handle initial periods, but true crossover signals
    # will only appear after both windows have enough data for their first "real" SMA value
    # and a crossover can be detected by comparing current and previous states.
    # The .shift(1) in crossover logic naturally handles the first period by making it NaN, which won't trigger a signal.

    return signals.fillna(0.0)  # Ensure any NaNs from shift are 0


def get_parameters() -> list:
    """
    Returns a default list of parameters for the initial strategy.
    For SMA Crossover: [short_window, long_window]
    """
    return [10, 20]  # Example: 10-period SMA and 20-period SMA


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
