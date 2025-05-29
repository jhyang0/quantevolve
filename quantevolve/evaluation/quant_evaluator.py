import numpy as np
import pandas as pd
import importlib.util
import os
import sys
from pathlib import Path

def evaluate(program_path: str) -> dict:
    """
    Evaluates a trading strategy program.

    Args:
        program_path (str): Path to the Python file containing the strategy.

    Returns:
        dict: A dictionary of performance metrics.
    """
    # --- Default error/failure metrics ---
    default_error_metrics = {
        'pnl': 0.0,
        'sharpe_ratio': -100.0, # Very low for minimization problems or if fitness is maximized
        'negative_max_drawdown': -1.0, # Max drawdown is negative, so -1 is worst
        'num_trades': 0,
        'error': 1.0, # Indicates an error occurred
        'can_run': 0.0, # Indicates the program could not be run/evaluated
        'combined_score': -100.0, # Fitness score QuantEvolve tries to maximize
        'error_message': 'Evaluation not started'
    }

    try:
        # --- 1. Load Market Data ---
        # Determine the project root based on this file's location.
        # Assumes this script is at examples/quant_evolve/quant_evaluator.py
        # and data is at examples/quant_evolve/data/
        current_script_path = Path(__file__).resolve()
        project_root = current_script_path.parent.parent.parent # examples/quant_evolve -> examples -> project_root
        
        # Try a few common locations for the data file relative to project root or script dir
        # This handles running from project root (`python examples/quant_evolve/quant_evaluator.py`)
        # or from within the `examples/quant_evolve` directory.
        market_data_paths_to_try = [
            project_root / "examples" / "quant_evolve" / "data" / "btc_usdt_1h_2023.csv",
            Path("examples") / "quant_evolve" / "data" / "btc_usdt_1h_2023.csv", # If CWD is project root
            Path("data") / "btc_usdt_1h_2023.csv", # If CWD is examples/quant_evolve
            Path("../data/btc_usdt_1h_2023.csv") # If CWD is inside a subfolder of quant_evolve (less likely)
        ]

        market_data_path = None
        for p in market_data_paths_to_try:
            if p.exists():
                market_data_path = p
                break
        
        if not market_data_path:
            print(f"Error: Market data file not found. Tried: {[str(p) for p in market_data_paths_to_try]}")
            default_error_metrics['error_message'] = 'Market data file not found.'
            return default_error_metrics

        market_data_df = pd.read_csv(market_data_path)
        
        # Preprocess data
        market_data_df['timestamp'] = pd.to_datetime(market_data_df['timestamp'])
        market_data_df.set_index('timestamp', inplace=True)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            market_data_df[col] = pd.to_numeric(market_data_df[col], errors='coerce')
        
        market_data_df.dropna(subset=numeric_cols, inplace=True) # Drop rows where essential data is NaN

        if market_data_df.empty:
            print("Error: Market data is empty after processing.")
            default_error_metrics['error_message'] = 'Market data empty after processing.'
            return default_error_metrics

    except FileNotFoundError:
        print(f"Error: Market data file not found at {str(market_data_path)}")
        default_error_metrics['error_message'] = f'Market data file not found at {str(market_data_path)}'
        return default_error_metrics
    except Exception as e:
        print(f"Error loading market data: {str(e)}")
        default_error_metrics['error_message'] = f'Error loading market data: {str(e)}'
        return default_error_metrics

    try:
        # --- 2. Load Evolved Strategy ---
        # Ensure program_path is absolute or relative to a known location (e.g., CWD)
        # For QuantEvolve, program_path will likely be relative to where it runs.
        # If this evaluator is called from the project root, relative paths like "examples/quant_evolve/some_strategy.py" are fine.
        
        strategy_path = Path(program_path).resolve()
        if not strategy_path.exists():
            print(f"Error: Strategy program file not found at {str(strategy_path)}")
            default_error_metrics['error_message'] = f'Strategy program file not found: {str(program_path)}'
            return default_error_metrics

        module_name = strategy_path.stem 
        spec = importlib.util.spec_from_file_location(module_name, strategy_path)
        if spec is None or spec.loader is None:
            print(f"Error: Could not create module spec for {str(strategy_path)}")
            default_error_metrics['error_message'] = f'Could not create module spec for {str(program_path)}'
            return default_error_metrics
            
        strategy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy_module)

        if not hasattr(strategy_module, 'get_parameters'):
            print(f"Error: 'get_parameters' function not found in {str(program_path)}")
            default_error_metrics['error_message'] = f"'get_parameters' missing in {str(program_path)}"
            return default_error_metrics
        params = strategy_module.get_parameters()

        if not hasattr(strategy_module, 'run_strategy'):
            print(f"Error: 'run_strategy' function not found in {str(program_path)}")
            default_error_metrics['error_message'] = f"'run_strategy' missing in {str(program_path)}"
            return default_error_metrics
        run_strategy_func = strategy_module.run_strategy

    except ImportError as e:
        print(f"Error importing strategy module from {str(program_path)}: {str(e)}")
        default_error_metrics['error_message'] = f'ImportError for {str(program_path)}: {str(e)}'
        return default_error_metrics
    except Exception as e:
        print(f"Error loading strategy from {str(program_path)}: {str(e)}")
        default_error_metrics['error_message'] = f'Error loading strategy {str(program_path)}: {str(e)}'
        return default_error_metrics

    try:
        # --- 3. Get Trading Signals ---
        # Pass a copy of the market data to avoid modification by the strategy
        signals = run_strategy_func(market_data_df.copy(), params)
        if not isinstance(signals, pd.Series):
            print("Error: Strategy did not return a Pandas Series.")
            default_error_metrics['error_message'] = 'Strategy did not return a Pandas Series.'
            return default_error_metrics
        if not signals.index.equals(market_data_df.index):
            print("Error: Signals index does not match market data index. Realigning...")
            signals = signals.reindex(market_data_df.index, fill_value=0.0)


    except Exception as e:
        print(f"Error running strategy from {str(program_path)}: {str(e)}")
        # Potentially log the traceback here for more detailed debugging
        import traceback
        tb_str = traceback.format_exc()
        print(tb_str)
        default_error_metrics['error_message'] = f'Error running strategy {str(program_path)}: {str(e)}'
        return default_error_metrics

    try:
        # --- 4. Implement a Simple Vectorized Backtest ---
        # Ensure signals are numeric and fill NaNs
        signals = pd.to_numeric(signals, errors='coerce').fillna(0.0)
        
        # Ensure signals are only 1, -1, or 0.
        signals = np.sign(signals).astype(int)


        # Shift signals to trade on the next bar's open/close.
        # A signal generated at time t based on data up to t, means we trade at t+1.
        positions = signals.shift(1).fillna(0.0)
        
        # Calculate daily/bar returns
        market_returns = market_data_df['close'].pct_change().fillna(0.0)
        
        # Calculate strategy returns
        strategy_returns = positions * market_returns
        if not isinstance(strategy_returns, pd.Series): # Should be a Series
             strategy_returns = pd.Series(strategy_returns, index=market_data_df.index).fillna(0.0)


        # --- 5. Calculate Performance Metrics ---
        pnl = strategy_returns.sum()

        # Sharpe Ratio
        # Assuming 1-hour data, so 24 data points per day.
        # For a full year (approx 365 days):
        annualization_factor = 24 * 365 
        
        mean_strategy_return = strategy_returns.mean()
        std_dev_strategy_returns = strategy_returns.std()

        if std_dev_strategy_returns == 0 or np.isnan(std_dev_strategy_returns) or np.isinf(std_dev_strategy_returns):
            sharpe_ratio = -100.0 if mean_strategy_return <= 0 else 0 # Penalize if no variance but negative/zero mean
        else:
            sharpe_ratio = (mean_strategy_return * np.sqrt(annualization_factor)) / std_dev_strategy_returns
            if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio): # check again after calculation
                sharpe_ratio = -100.0


        # Maximum Drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        # Max drawdown is the minimum value of the drawdown series (most negative)
        negative_max_drawdown = drawdown.min() 
        if np.isnan(negative_max_drawdown) or np.isinf(negative_max_drawdown):
            negative_max_drawdown = -1.0 # Worst possible drawdown if calculation fails

        # Number of Trades
        # A trade is counted when position changes (0 to 1, 0 to -1, 1 to -1, -1 to 1, 1 to 0, -1 to 0)
        num_trades = (positions.diff().fillna(0) != 0).sum()


        # --- 6. Return Metrics ---
        # Ensure combined_score is robust
        # If sharpe_ratio is very low (e.g. -100 due to error or bad performance), combined_score should also be very low.
        # If sharpe_ratio is a valid negative number, combined_score should reflect that.
        # If sharpe_ratio is positive, that's good.
        combined_score = sharpe_ratio if sharpe_ratio > -100.0 else -100.0
        
        # If pnl is negative and sharpe is also bad, further penalize combined_score
        # This is an example, can be tuned.
        if pnl < 0 and combined_score > -5: # if sharpe was positive or slightly negative despite overall loss
            combined_score = max(-5.0, combined_score * 0.5) # Reduce, but not to -100 unless sharpe was already bad

        # If num_trades is very low (e.g. < 2), it might not be a meaningful strategy.
        # Could penalize combined_score here, e.g. if num_trades < 2, combined_score = min(combined_score, -50)
        if num_trades < 2 :
            combined_score = min(combined_score, -50.0) # Penalize if very few trades


        metrics = {
            'pnl': float(pnl),
            'sharpe_ratio': float(sharpe_ratio),
            'negative_max_drawdown': float(negative_max_drawdown),
            'num_trades': int(num_trades),
            'error': 0.0, # No error in this path
            'can_run': 1.0, # Program ran successfully
            'combined_score': float(combined_score),
            'error_message': ''
        }
        return metrics

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        print(f"Error during backtesting/metrics calculation for {str(program_path)}: {str(e)}\n{tb_str}")
        default_error_metrics['error_message'] = f'Backtesting error for {str(program_path)}: {str(e)}'
        return default_error_metrics

if __name__ == "__main__":
    # This assumes you are running from the project root directory,
    # or that examples/quant_evolve/initial_strategy.py is in the Python path.
    
    # Construct path to initial_strategy.py relative to this script
    # This makes it runnable from any CWD as long as the project structure is intact.
    current_script_dir = Path(__file__).parent.resolve()
    dummy_program_path = str(current_script_dir / "initial_strategy.py")
    
    print(f"Evaluating initial strategy: {str(dummy_program_path)}")

    # Check if market data exists, if not, guide user.
    # The evaluate function itself now has more robust path checking.
    # For __main__ testing, we rely on `evaluate` finding the data.
    # Example: check one of the paths `evaluate` would try.
    expected_data_path = current_script_dir / "data" / "btc_usdt_1h_2023.csv"
    if not expected_data_path.exists():
         alt_data_path = Path(__file__).parent.parent.parent / "examples/quant_evolve/data/btc_usdt_1h_2023.csv"
         if not alt_data_path.exists():
            print(f"Market data not found at expected locations like: {str(expected_data_path)} or {str(alt_data_path)}")
            print("Please ensure 'btc_usdt_1h_2023.csv' is in the 'examples/quant_evolve/data/' directory.")
            print("You might need to run 'market_data_collector.py' first if it hasn't been run.")
            # Create a dummy file for the test to proceed with some data, though results will be meaningless.
            # This is just to allow the evaluator script itself to be tested without real data if needed.
            print("Creating a minimal dummy market data CSV for testing purposes...")
            dummy_data_dir = current_script_dir / "data"
            dummy_data_dir.mkdir(parents=True, exist_ok=True)
            dummy_csv_path = dummy_data_dir / "btc_usdt_1h_2023.csv"
            dummy_df_content = "timestamp,open,high,low,close,volume\n"
            for i in range(100): # ~4 days of hourly data
                dt = pd.Timestamp('2023-01-01T00:00:00') + pd.Timedelta(hours=i)
                dummy_df_content += f"{dt.isoformat()},{100+i*0.1},{101+i*0.1},{99+i*0.1},{100.5+i*0.1},{1000}\n"
            with open(dummy_csv_path, "w") as f:
                f.write(dummy_df_content)
            print(f"Dummy data created at {str(dummy_csv_path)}")


    results = evaluate(dummy_program_path)

    print("\nEvaluation Metrics:")
    for key, value in results.items():
        print(f"{key}: {value}")

    # Example of how to check if an error occurred:
    if results.get('error', 0.0) > 0.0 or results.get('can_run', 1.0) == 0.0:
        print(f"\nEvaluation of {str(dummy_program_path)} encountered an error or could not run.")
        print(f"Error Message: {results.get('error_message', 'N/A')}")
    else:
        print(f"\nEvaluation of {str(dummy_program_path)} completed successfully.")
        print(f"Combined Score: {results['combined_score']}")

    # Test with a non-existent strategy to see error handling
    print("\nEvaluating non-existent strategy:")
    non_existent_program_path = "examples/quant_evolve/non_existent_strategy.py"
    results_non_existent = evaluate(non_existent_program_path)
    print("\nEvaluation Metrics (non-existent strategy):")
    for key, value in results_non_existent.items():
        print(f"{key}: {value}")
    if results_non_existent.get('error', 0.0) > 0.0:
        print(f"Error correctly handled for non-existent strategy: {str(results_non_existent.get('error_message'))}")

    # Test with a strategy that might have an error (e.g. missing functions)
    # Create a dummy bad strategy file
    bad_strategy_path = current_script_dir / "temp_bad_strategy.py"
    with open(bad_strategy_path, "w") as f:
        f.write("import pandas as pd\n\n# Missing get_parameters or run_strategy\n")
    
    print(f"\nEvaluating bad strategy (missing functions): {bad_strategy_path}")
    results_bad_strategy = evaluate(str(bad_strategy_path))
    print("\nEvaluation Metrics (bad strategy):")
    for key, value in results_bad_strategy.items():
        print(f"{key}: {value}")
    if results_bad_strategy.get('error', 0.0) > 0.0:
        print(f"Error correctly handled for bad strategy: {results_bad_strategy.get('error_message')}")
    
    # Clean up dummy bad strategy file
    if bad_strategy_path.exists():
        os.remove(bad_strategy_path)
