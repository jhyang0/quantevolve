# QuantEvolve

**Evolutionary Quantitative Trading Strategy Development System**

> This project is a fork of [OpenEvolve](https://github.com/codelion/openevolve) with modifications and enhancements focused on quantitative trading strategy development.

## Overview

QuantEvolve is a system that leverages evolutionary algorithms and Large Language Models (LLMs) to automatically develop and optimize quantitative trading strategies. The primary goal is to continuously evolve trading strategy code, written in Python, to achieve optimal risk-adjusted returns when backtested on historical market data. By iteratively refining strategies based on performance metrics, QuantEvolve aims to discover novel and effective approaches to trading.

## Architecture

The system is composed of several key components that work together in a cycle:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Market Data    │────▶│  Strategy        │────▶│  Backtester     │
│  Collector      │     │  Generator       │     │  & Evaluator    │
│  (Binance API)  │     │  (LLM Engine)    │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       ▲                         │
         │                       │                         │
         │              ┌────────┴────────┐                │
         └─────────────▶│  Strategy       │◀───────────────┘
                        │  Database       │
                        └─────────────────┘
```

-   **Market Data Collector:** Fetches historical market data (OHLCV) from sources like the Binance API. This data forms the basis for backtesting.
-   **Strategy Generator (LLM Engine):** Utilizes Google Gemini models to generate new trading strategies or modify existing ones. It takes promising strategies and performance feedback to suggest code changes.
-   **Backtester & Evaluator:** Executes the generated trading strategies against historical market data. It calculates various performance metrics (e.g., PnL, Sharpe Ratio, Max Drawdown) to assess the effectiveness of each strategy.
-   **Strategy Database:** Stores all evaluated strategies, their code, and their performance metrics. This database serves as a pool of "genetic material" for the evolutionary process, allowing the system to learn from past successes and failures.

## Core Features

-   **Automated Strategy Development:** Automates the workflow from market data collection to trading strategy generation, backtesting, and iterative refinement.
-   **Evolutionary Optimization:** Employs LLMs to analyze existing strategies and their performance, then intelligently modifies their code or creates entirely new strategies to improve upon them.
-   **Market Adaptation (Goal):** While the current implementation relies on historical backtesting, the long-term vision is for strategies to adapt to changing market conditions, potentially through continuous retraining or online learning mechanisms.

## Getting Started / Setup

### Prerequisites

-   Python (3.9+ recommended)
-   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your_repository_url_here> quantevolve
    ```
    (Replace `<your_repository_url_here>` with the actual URL of this repository, or assume it's already cloned if running these commands locally.)

2.  **Navigate to the directory:**
    ```bash
    cd quantevolve
    ```

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

4.  **Install dependencies:**
    QuantEvolve is designed to be installed as a package. This command installs the `quantevolve` package itself along with its core dependencies listed in `setup.py`.
    ```bash
    pip install -e .
    ```
    Additionally, for the specific QuantEvolve functionalities, you need to install:
    ```bash
    pip install google-generativeai python-binance pandas numpy
    ```
    The `google-generativeai` package is required for the Gemini integration.

### API Keys

-   **Google Gemini API Key:** QuantEvolve uses Google Gemini models for strategy generation. You need to provide a Google API key. You can either:

    1. Set the `GOOGLE_API_KEY` environment variable:
    ```bash
    export GOOGLE_API_KEY="your_google_api_key_here"
    ```
    
    2. Add it directly to the `quantevolve_config.yaml` file:
    ```yaml
    llm:
      api_key: "your_google_api_key_here"
    ```

-   **Binance API for Data Collection:**
    The `quantevolve/data/market_data_collector.py` script fetches historical kline data from Binance. You can provide your Binance API credentials in two ways:

    1. Set environment variables:
    ```bash
    export BINANCE_API_KEY="your_binance_api_key_here"
    export BINANCE_API_SECRET="your_binance_api_secret_here"
    ```
    
    2. Add them to the `quantevolve_config.yaml` file (recommended):
    ```yaml
    data_collection:
      binance_api_key: "your_binance_api_key_here"
      binance_api_secret: "your_binance_api_secret_here"
      # You can also customize these default parameters:
      default_symbol: "BTCUSDT"
      default_interval: "1h"
      default_lookback_days: 365
    ```

## How to Run QuantEvolve

### Step 1: Fetch Market Data

Before running the evolution, you need historical market data.
```bash
python -m quantevolve.data.market_data_collector
```
This script downloads historical OHLCV data based on your configuration settings in `quantevolve_config.yaml` (or defaults to BTC/USDT 1-hour interval for the past year). The data is saved to the specified path and is used by the `quant_evaluator.py` for backtesting.

### Step 2: Run the Evolution Process

Once the market data is available, start the QuantEvolve process:
```bash
python -m quantevolve.cli
```
This command will:
-   Use `quantevolve/strategy/initial_strategy.py` as the starting point for evolution.
-   Use `quantevolve/evaluation/quant_evaluator.py` to backtest and evaluate strategies.
-   Load its configuration from `configs/quantevolve_config.yaml`.
-   Save outputs to `quantevolve_output/` (the default output directory).

You can customize the evolution parameters, LLM settings, and other aspects by modifying `configs/quantevolve_config.yaml`. For additional command line options, run:
```bash
python -m quantevolve.cli --help
```

## Output and Results

QuantEvolve saves all its outputs in the `quantevolve_output/` directory (or the directory specified by the `--output_dir` argument if used, or configured in the YAML file).

-   **`best/`**:
    -   `best_strategy.py`: The Python script of the best overall trading strategy found during the entire run.
    -   `best_strategy_info.json`: A JSON file containing metadata and detailed performance metrics for the `best_strategy.py`.
-   **`checkpoints/`**: Contains periodic checkpoints of the evolution state. Each checkpoint includes the current population of strategies and the best strategy found up to that point, allowing you to resume runs or analyze intermediate results.
-   **`logs/`**: Contains detailed logs of the evolutionary process (e.g., `quantevolve_YYYYMMDD_HHMMSS.log`), including LLM interactions, evaluation scores, and errors. This is crucial for monitoring and debugging.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or find any bugs, please feel free to:
-   Report issues on the GitHub issue tracker.
-   Suggest features or enhancements.
-   Submit pull requests with your contributions.

Please see `CONTRIBUTING.md` for more details on how to contribute (if the file exists).

## License

QuantEvolve is released under the Apache 2.0 License. See the `LICENSE` file for more details.
