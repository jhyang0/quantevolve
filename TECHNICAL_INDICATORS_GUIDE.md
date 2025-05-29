# QuantEvolve Technical Indicators Guide

## Overview

QuantEvolve has been enhanced with comprehensive TA-Lib technical indicator support, providing access to 150+ professional-grade technical indicators for advanced quantitative trading strategy development. This guide explains how to leverage these powerful tools in your trading strategies.

## Key Enhancements

### 1. Comprehensive TA-Lib Integration
- **150+ Technical Indicators** across 8 categories
- **Automated indicator calculation** with error handling
- **Category-based indicator grouping** for strategic development
- **Enhanced LLM prompts** with indicator knowledge

### 2. Enhanced Strategy Generation
- **Multi-indicator strategies** with confirmation systems
- **Adaptive strategies** that change based on market regime
- **Risk-aware position sizing** using volatility indicators
- **Pattern recognition** integration for precise timing

### 3. Advanced Evaluation System
- **Strategy complexity scoring** based on indicator usage
- **Innovation metrics** for advanced technique recognition
- **Risk-adjusted performance** evaluation
- **Code quality assessment** with best practices scoring

## Available Indicator Categories

### Trend Following Indicators
Perfect for identifying and following market trends:
- **Moving Averages**: SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3
- **MACD Family**: MACD, MACDEXT, MACDFIX, PPO, APO
- **Directional Movement**: ADX, ADXR, DX, PLUS_DI, MINUS_DI
- **Trend Lines**: LINEARREG, TSF, HT_TRENDLINE, SAR, SAREXT

### Momentum Indicators
For detecting price momentum and reversals:
- **Oscillators**: RSI, STOCH, STOCHF, STOCHRSI, WILLR, CCI, CMO
- **Rate of Change**: MOM, ROC, ROCP, ROCR, ROCR100, TRIX
- **Specialized**: AROON, AROONOSC, BOP, MFI, ULTOSC

### Volatility Indicators
For measuring market volatility and risk:
- **Range Indicators**: ATR, NATR, TRANGE
- **Bands**: BBANDS (Bollinger Bands)
- **Statistical**: STDDEV, VAR

### Volume Indicators
For analyzing trading volume patterns:
- **Volume Flow**: OBV, AD, ADOSC
- **Volume-Price**: MFI (Money Flow Index)

### Pattern Recognition
60+ candlestick patterns for precise entry/exit timing:
- **Reversal Patterns**: CDLDOJI, CDLHAMMER, CDLENGULFING
- **Continuation Patterns**: CDLMORNINGSTAR, CDLEVENINGSTAR
- **Complex Patterns**: CDLABANDONEDBABY, CDLTHREEWHITESOLDIERS

### Cycle Analysis
Advanced cycle detection using Hilbert Transform:
- **HT_DCPERIOD**: Dominant cycle period detection
- **HT_SINE**: Sine wave cycle analysis
- **HT_TRENDMODE**: Trend vs cycle mode detection

### Statistical Functions
Mathematical analysis tools:
- **Regression**: LINEARREG, LINEARREG_ANGLE, LINEARREG_SLOPE
- **Correlation**: CORREL, BETA
- **Forecasting**: TSF (Time Series Forecast)

### Price Transform
Price data transformations:
- **AVGPRICE**: Average of OHLC
- **MEDPRICE**: Median price (H+L)/2
- **TYPPRICE**: Typical price (H+L+C)/3
- **WCLPRICE**: Weighted close price

## Quick Start Guide

### Basic Usage

```python
from quantevolve.utils.indicators import (
    calculate_indicator,
    add_comprehensive_indicators,
    generate_multi_indicator_signals
)

# Add single indicator
df = calculate_indicator(df, 'RSI', timeperiod=14)
df = calculate_indicator(df, 'MACD', fastperiod=12, slowperiod=26, signalperiod=9)

# Add comprehensive indicators by category
df = add_comprehensive_indicators(
    df, 
    categories=['momentum', 'trend', 'volatility']
)

# Generate signals using multiple indicators
signals = generate_multi_indicator_signals(df, {
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'use_macd': True,
    'use_bollinger': True
})
```

### Enhanced Strategy Generation

```python
from quantevolve.strategy.enhanced_strategy_generator import (
    generate_enhanced_strategy,
    EnhancedStrategyGenerator
)

# Generate different types of strategies
generator = EnhancedStrategyGenerator()

# Multi-indicator strategy
strategy_code = generator.generate_strategy("multi_indicator", {
    "categories": ["momentum", "trend", "volatility"],
    "confirmation_count": 2,
    "use_volume": True
})

# Trend following strategy
trend_strategy = generator.generate_strategy("trend_following", {
    "fast_period": 10,
    "slow_period": 30,
    "use_adx": True,
    "adx_threshold": 25
})

# Mean reversion strategy
mean_reversion = generator.generate_strategy("mean_reversion", {
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "use_stoch": True
})
```

## Advanced Strategy Examples

### 1. Multi-Confirmation Trend Following

```python
def run_strategy(market_data_df: pd.DataFrame, params: list) -> pd.Series:
    signals = pd.Series(index=market_data_df.index, data=0.0)
    df = market_data_df.copy()
    
    # Add comprehensive trend indicators
    df = calculate_indicator(df, 'SMA', timeperiod=20)
    df = calculate_indicator(df, 'EMA', timeperiod=50)
    df = calculate_indicator(df, 'MACD', fastperiod=12, slowperiod=26, signalperiod=9)
    df = calculate_indicator(df, 'ADX', timeperiod=14)
    
    # Multi-confirmation system
    trend_up = (df['sma'] > df['ema']) & (df['macd'] > df['macdsignal']) & (df['adx'] > 25)
    trend_down = (df['sma'] < df['ema']) & (df['macd'] < df['macdsignal']) & (df['adx'] > 25)
    
    signals[trend_up] = 1.0
    signals[trend_down] = -1.0
    
    return signals
```

### 2. Volatility-Adaptive Position Sizing

```python
def run_strategy(market_data_df: pd.DataFrame, params: list) -> pd.Series:
    signals = pd.Series(index=market_data_df.index, data=0.0)
    df = market_data_df.copy()
    
    # Add volatility indicators
    df = calculate_indicator(df, 'ATR', timeperiod=14)
    df = calculate_indicator(df, 'BBANDS', timeperiod=20)
    df = calculate_indicator(df, 'RSI', timeperiod=14)
    
    # Base signals
    oversold = df['rsi'] < 30
    overbought = df['rsi'] > 70
    
    # Volatility adjustment
    high_volatility = df['atr'] > df['atr'].rolling(50).mean() * 1.5
    position_multiplier = np.where(high_volatility, 0.5, 1.0)  # Reduce position in high volatility
    
    signals[oversold] = 1.0 * position_multiplier[oversold]
    signals[overbought] = -1.0 * position_multiplier[overbought]
    
    return signals
```

### 3. Pattern Recognition with Confirmation

```python
def run_strategy(market_data_df: pd.DataFrame, params: list) -> pd.Series:
    signals = pd.Series(index=market_data_df.index, data=0.0)
    df = market_data_df.copy()
    
    # Add pattern recognition
    df = calculate_indicator(df, 'CDLHAMMER')
    df = calculate_indicator(df, 'CDLENGULFING')
    df = calculate_indicator(df, 'CDLDOJI')
    
    # Add confirmation indicators
    df = calculate_indicator(df, 'RSI', timeperiod=14)
    df = calculate_indicator(df, 'STOCH', fastk_period=14)
    
    # Pattern-based signals with confirmation
    bullish_patterns = (df['cdlhammer'] > 0) | (df['cdlengulfing'] > 0)
    bearish_patterns = (df['cdlhammer'] < 0) | (df['cdlengulfing'] < 0)
    
    momentum_oversold = (df['rsi'] < 35) & (df['slowk'] < 30)
    momentum_overbought = (df['rsi'] > 65) & (df['slowk'] > 70)
    
    signals[bullish_patterns & momentum_oversold] = 1.0
    signals[bearish_patterns & momentum_overbought] = -1.0
    
    return signals
```

## Enhanced Evaluation Features

The enhanced evaluation system now provides comprehensive analysis:

```python
from quantevolve.evaluation.enhanced_evaluator import evaluate_enhanced

# Get comprehensive evaluation
results = evaluate_enhanced("path/to/strategy.py")

# Enhanced metrics available:
print(f"Indicator Score: {results['indicator_score']}")
print(f"Complexity Score: {results['complexity_score']}")
print(f"Innovation Score: {results['innovation_score']}")
print(f"Risk-Adjusted Score: {results['risk_adjusted_score']}")
print(f"Comprehensive Score: {results['comprehensive_score']}")

# Strategy analysis details
analysis = results['strategy_analysis']
print(f"Indicators Used: {analysis['indicators_used']}")
print(f"Categories: {analysis['indicator_categories']}")
print(f"Has Risk Management: {analysis['has_risk_management']}")
```

## Best Practices

### 1. Multi-Indicator Confirmation
Always use multiple indicators from different categories to confirm signals:
- **Trend** + **Momentum** indicators for trend following
- **Momentum** + **Volatility** indicators for mean reversion
- **Volume** indicators for signal strength confirmation

### 2. Risk Management Integration
Use volatility indicators for dynamic risk management:
```python
# ATR-based position sizing
position_size = base_capital / (df['atr'] * risk_multiplier)

# Volatility-based stop losses
stop_distance = df['atr'] * 2.0
```

### 3. Market Regime Adaptation
Use regime detection for strategy switching:
```python
# ADX for trend strength
trending_market = df['adx'] > 25
ranging_market = df['adx'] <= 25

# Different strategies for different regimes
signals = np.where(trending_market, trend_signals, mean_reversion_signals)
```

### 4. Pattern Recognition Enhancement
Combine candlestick patterns with technical indicators:
```python
# Pattern detection with momentum confirmation
bullish_reversal = (df['cdlhammer'] > 0) & (df['rsi'] < 30)
bearish_reversal = (df['cdlshootingstar'] < 0) & (df['rsi'] > 70)
```

## Performance Optimization Tips

### 1. Efficient Indicator Calculation
Use batch calculation for multiple indicators:
```python
# Efficient: Calculate multiple indicators at once
df = add_comprehensive_indicators(df, categories=['momentum', 'trend'])

# Less efficient: Calculate indicators one by one
df = calculate_indicator(df, 'RSI')
df = calculate_indicator(df, 'MACD')
df = calculate_indicator(df, 'SMA')
```

### 2. Memory Management
For large datasets, consider indicator subsets:
```python
# Use only necessary indicators
essential_indicators = ['RSI', 'MACD', 'ATR', 'SMA', 'EMA']
for indicator in essential_indicators:
    df = calculate_indicator(df, indicator)
```

### 3. Signal Generation Optimization
Use vectorized operations for signal generation:
```python
# Vectorized signal generation
buy_signals = (df['rsi'] < 30) & (df['macd'] > df['macdsignal'])
sell_signals = (df['rsi'] > 70) & (df['macd'] < df['macdsignal'])

signals[buy_signals] = 1.0
signals[sell_signals] = -1.0
```

## Integration with LLM Strategy Evolution

The enhanced TA-Lib integration is fully compatible with QuantEvolve's LLM-driven strategy evolution:

### 1. Enhanced Prompts
The LLM now has access to comprehensive indicator information and can suggest sophisticated combinations.

### 2. Strategy Templates
Pre-built strategy templates demonstrate best practices for indicator usage.

### 3. Automatic Strategy Enhancement
The LLM can automatically suggest indicator improvements based on performance analysis.

## Troubleshooting

### Common Issues and Solutions

1. **TA-Lib Import Error**
   ```bash
   pip install TA-Lib
   # On macOS with homebrew:
   brew install ta-lib
   pip install TA-Lib
   ```

2. **Insufficient Data for Indicator Calculation**
   ```python
   # Ensure sufficient data points
   if len(df) < max_period:
       print(f"Warning: Insufficient data for period {max_period}")
   ```

3. **NaN Values in Indicators**
   ```python
   # Handle NaN values properly
   df['indicator'].fillna(method='forward', inplace=True)
   ```

4. **Performance Issues with Many Indicators**
   ```python
   # Use selective indicator calculation
   available_indicators = get_available_indicators_for_data(df)
   optimized_indicators = select_best_indicators(available_indicators)
   ```

## Future Enhancements

### Planned Features
- **Custom indicator development** framework
- **Multi-timeframe analysis** capabilities
- **Real-time indicator updates** for live trading
- **Machine learning-enhanced** indicator selection
- **Portfolio-level** indicator analysis

### Contributing
To contribute new indicators or improvements:
1. Add indicator definitions to `talib_indicators_registry.py`
2. Implement calculation functions in `indicators.py`
3. Update prompt templates with new indicator information
4. Add evaluation criteria for new indicators
5. Create example strategies demonstrating usage

This comprehensive technical indicators system transforms QuantEvolve into a professional-grade quantitative trading platform capable of generating sophisticated, multi-indicator strategies with advanced risk management and performance evaluation capabilities.