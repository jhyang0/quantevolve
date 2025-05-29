"""
Enhanced Strategy Generator for QuantEvolve

This module provides advanced strategy generation capabilities that leverage
the comprehensive TA-Lib indicator library for creating sophisticated trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from ..utils.indicators import (
    add_comprehensive_indicators,
    calculate_indicator,
    generate_multi_indicator_signals,
    get_available_indicators_for_data,
    crossover,
    crossunder,
)
from ..utils.talib_indicators_registry import get_strategy_relevant_indicators, get_indicator_info


class EnhancedStrategyGenerator:
    """Enhanced strategy generator with comprehensive TA-Lib indicator support"""

    def __init__(self):
        self.strategy_templates = {
            "trend_following": self._generate_trend_following_strategy,
            "mean_reversion": self._generate_mean_reversion_strategy,
            "momentum": self._generate_momentum_strategy,
            "multi_indicator": self._generate_multi_indicator_strategy,
            "adaptive": self._generate_adaptive_strategy,
            "breakout": self._generate_breakout_strategy,
            "volatility_based": self._generate_volatility_based_strategy,
        }

    def generate_strategy(
        self, strategy_type: str = "multi_indicator", config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a trading strategy of the specified type.

        Args:
            strategy_type (str): Type of strategy to generate
            config (Dict): Configuration parameters for strategy generation

        Returns:
            str: Generated strategy code
        """
        if strategy_type not in self.strategy_templates:
            available_types = ", ".join(self.strategy_templates.keys())
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. Available: {available_types}"
            )

        if config is None:
            config = self._get_default_config(strategy_type)

        return self.strategy_templates[strategy_type](config)

    def _get_default_config(self, strategy_type: str) -> Dict[str, Any]:
        """Get default configuration for strategy type"""
        default_configs = {
            "trend_following": {
                "fast_period": 10,
                "slow_period": 30,
                "use_adx": True,
                "adx_threshold": 25,
                "use_volume": False,
            },
            "mean_reversion": {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "bb_period": 20,
                "bb_std": 2.0,
                "use_stoch": True,
            },
            "momentum": {
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "roc_period": 10,
                "mom_period": 14,
            },
            "multi_indicator": {
                "categories": ["momentum", "trend", "volatility"],
                "confirmation_count": 2,
                "use_volume": True,
            },
            "adaptive": {
                "regime_indicator": "ADX",
                "regime_threshold": 25,
                "trending_strategy": "trend_following",
                "ranging_strategy": "mean_reversion",
            },
            "breakout": {
                "bb_period": 20,
                "bb_std": 2.0,
                "volume_multiplier": 1.5,
                "atr_period": 14,
            },
            "volatility_based": {
                "atr_period": 14,
                "volatility_threshold": 1.5,
                "position_sizing": "atr_based",
            },
        }
        return default_configs.get(strategy_type, {})

    def _generate_trend_following_strategy(self, config: Dict[str, Any]) -> str:
        """Generate a trend-following strategy using multiple indicators"""
        fast_period = config.get("fast_period", 10)
        slow_period = config.get("slow_period", 30)
        use_adx = config.get("use_adx", True)
        adx_threshold = config.get("adx_threshold", 25)
        use_volume = config.get("use_volume", False)

        strategy_code = f'''import numpy as np
import pandas as pd
from quantevolve.utils.indicators import calculate_indicator, add_trend_indicators, crossover, crossunder


def run_strategy(market_data_df: pd.DataFrame, params: list) -> pd.Series:
    """
    Enhanced Trend Following Strategy with Multiple Technical Indicators
    
    This strategy combines moving averages, MACD, and ADX for robust trend detection.
    Optional volume confirmation can be enabled for additional signal filtering.
    """
    signals = pd.Series(index=market_data_df.index, data=0.0)
    
    # Extract parameters
    if len(params) >= 2:
        fast_period = int(params[0]) if params[0] > 0 else {fast_period}
        slow_period = int(params[1]) if params[1] > 0 else {slow_period}
    else:
        fast_period = {fast_period}
        slow_period = {slow_period}
    
    # EVOLVE-BLOCK-START
    try:
        # Add comprehensive trend indicators
        df = market_data_df.copy()
        
        # Moving Averages
        df = calculate_indicator(df, 'SMA', timeperiod=fast_period)
        df = calculate_indicator(df, 'SMA', timeperiod=slow_period)
        df.rename(columns={{'sma': f'sma_{{fast_period}}'}}, inplace=True)
        df = calculate_indicator(df, 'SMA', timeperiod=slow_period)
        df.rename(columns={{'sma': f'sma_{{slow_period}}'}}, inplace=True)
        
        # MACD for momentum confirmation
        df = calculate_indicator(df, 'MACD', fastperiod=12, slowperiod=26, signalperiod=9)
        
        # ADX for trend strength
        {"df = calculate_indicator(df, 'ADX', timeperiod=14)" if use_adx else "# ADX disabled"}
        
        # Volume indicators
        {"df = calculate_indicator(df, 'OBV')" if use_volume else "# Volume indicators disabled"}
        
        # Generate primary trend signals
        ma_bullish = df[f'sma_{fast_period}'] > df[f'sma_{slow_period}']
        ma_bearish = df[f'sma_{fast_period}'] < df[f'sma_{slow_period}']
        
        # MACD confirmation
        macd_bullish = df['macd'] > df['macdsignal']
        macd_bearish = df['macd'] < df['macdsignal']
        
        # Trend strength filter
        {"strong_trend = df['adx'] > " + str(adx_threshold) if use_adx else "strong_trend = pd.Series(True, index=df.index)"}
        
        # Volume confirmation
        {"volume_confirmation = df['obv'] > df['obv'].shift(1)" if use_volume else "volume_confirmation = pd.Series(True, index=df.index)"}
        
        # Generate buy signals
        buy_signals = ma_bullish & macd_bullish & strong_trend & volume_confirmation
        
        # Generate sell signals  
        sell_signals = ma_bearish & macd_bearish & strong_trend
        
        # Crossover detection for precise entry/exit
        ma_cross_up = crossover(df, f'sma_{fast_period}', f'sma_{slow_period}')
        ma_cross_down = crossunder(df, f'sma_{fast_period}', f'sma_{slow_period}')
        
        # Final signal generation
        signals[ma_cross_up & macd_bullish & strong_trend] = 1.0
        signals[ma_cross_down & macd_bearish] = -1.0
        
        # Additional signal from sustained trend
        signals[(buy_signals & ~buy_signals.shift(1).fillna(False))] = 1.0
        signals[(sell_signals & ~sell_signals.shift(1).fillna(False))] = -1.0
        
    except Exception as e:
        print(f"Error in trend following strategy: {{e}}")
        # Fallback to simple MA crossover
        short_ma = market_data_df['close'].rolling(window=fast_period).mean()
        long_ma = market_data_df['close'].rolling(window=slow_period).mean()
        signals[short_ma > long_ma] = 1.0
        signals[short_ma < long_ma] = -1.0
    
    # EVOLVE-BLOCK-END
    
    return signals.fillna(0.0)


def get_parameters() -> list:
    """Returns default parameters for the trend following strategy."""
    return [{fast_period}, {slow_period}]  # [fast_period, slow_period]


def get_trading_signals_func():
    """Returns the run_strategy function."""
    return run_strategy
'''
        return strategy_code

    def _generate_mean_reversion_strategy(self, config: Dict[str, Any]) -> str:
        """Generate a mean reversion strategy using RSI, Bollinger Bands, and Stochastic"""
        rsi_period = config.get("rsi_period", 14)
        rsi_oversold = config.get("rsi_oversold", 30)
        rsi_overbought = config.get("rsi_overbought", 70)
        bb_period = config.get("bb_period", 20)
        bb_std = config.get("bb_std", 2.0)
        use_stoch = config.get("use_stoch", True)

        strategy_code = f'''import numpy as np
import pandas as pd
from quantevolve.utils.indicators import calculate_indicator, crossover, crossunder


def run_strategy(market_data_df: pd.DataFrame, params: list) -> pd.Series:
    """
    Enhanced Mean Reversion Strategy using RSI, Bollinger Bands, and Stochastic
    
    This strategy identifies oversold/overbought conditions using multiple oscillators
    and confirms signals with Bollinger Band touches.
    """
    signals = pd.Series(index=market_data_df.index, data=0.0)
    
    # Extract parameters
    if len(params) >= 2:
        rsi_oversold = max(10, min(40, float(params[0]))) if params[0] > 0 else {rsi_oversold}
        rsi_overbought = max(60, min(90, float(params[1]))) if params[1] > 0 else {rsi_overbought}
    else:
        rsi_oversold = {rsi_oversold}
        rsi_overbought = {rsi_overbought}
    
    # EVOLVE-BLOCK-START
    try:
        df = market_data_df.copy()
        
        # RSI for momentum
        df = calculate_indicator(df, 'RSI', timeperiod={rsi_period})
        
        # Bollinger Bands for price extremes
        df = calculate_indicator(df, 'BBANDS', timeperiod={bb_period}, nbdevup={bb_std}, nbdevdn={bb_std})
        
        # Stochastic oscillator for additional confirmation
        {"df = calculate_indicator(df, 'STOCH', fastk_period=14, slowk_period=3, slowd_period=3)" if use_stoch else "# Stochastic disabled"}
        
        # Williams %R for momentum confirmation
        df = calculate_indicator(df, 'WILLR', timeperiod=14)
        
        # Mean reversion signals
        rsi_oversold_signal = df['rsi'] < rsi_oversold
        rsi_overbought_signal = df['rsi'] > rsi_overbought
        
        # Bollinger Band touches
        bb_lower_touch = df['close'] <= df['lowerband']
        bb_upper_touch = df['close'] >= df['upperband']
        
        # Stochastic confirmation
        {"stoch_oversold = df['slowk'] < 20" if use_stoch else "stoch_oversold = pd.Series(True, index=df.index)"}
        {"stoch_overbought = df['slowk'] > 80" if use_stoch else "stoch_overbought = pd.Series(True, index=df.index)"}
        
        # Williams %R confirmation
        willr_oversold = df['willr'] < -80
        willr_overbought = df['willr'] > -20
        
        # Combined buy signals (multiple confirmations)
        buy_signals = (
            (rsi_oversold_signal | bb_lower_touch) & 
            stoch_oversold & 
            willr_oversold
        )
        
        # Combined sell signals
        sell_signals = (
            (rsi_overbought_signal | bb_upper_touch) & 
            stoch_overbought & 
            willr_overbought
        )
        
        # Generate signals on first occurrence
        signals[buy_signals & ~buy_signals.shift(1).fillna(False)] = 1.0
        signals[sell_signals & ~sell_signals.shift(1).fillna(False)] = -1.0
        
        # Exit signals when RSI moves back toward neutral
        rsi_exit_long = (df['rsi'] > 50) & (df['rsi'].shift(1) <= 50)
        rsi_exit_short = (df['rsi'] < 50) & (df['rsi'].shift(1) >= 50)
        
        # Add exit signals (but don't override entry signals)
        signals[rsi_exit_long & (signals.shift(1) != 1.0)] = 0.0
        signals[rsi_exit_short & (signals.shift(1) != -1.0)] = 0.0
        
    except Exception as e:
        print(f"Error in mean reversion strategy: {{e}}")
        # Fallback to simple RSI strategy
        df = market_data_df.copy()
        df = calculate_indicator(df, 'RSI', timeperiod={rsi_period})
        signals[df['rsi'] < rsi_oversold] = 1.0
        signals[df['rsi'] > rsi_overbought] = -1.0
    
    # EVOLVE-BLOCK-END
    
    return signals.fillna(0.0)


def get_parameters() -> list:
    """Returns default parameters for the mean reversion strategy."""
    return [{rsi_oversold}, {rsi_overbought}]  # [rsi_oversold, rsi_overbought]


def get_trading_signals_func():
    """Returns the run_strategy function."""
    return run_strategy
'''
        return strategy_code

    def _generate_multi_indicator_strategy(self, config: Dict[str, Any]) -> str:
        """Generate a strategy that uses multiple indicator categories"""
        categories = config.get("categories", ["momentum", "trend", "volatility"])
        confirmation_count = config.get("confirmation_count", 2)
        use_volume = config.get("use_volume", True)

        strategy_code = f'''import numpy as np
import pandas as pd
from quantevolve.utils.indicators import (
    add_comprehensive_indicators, 
    generate_multi_indicator_signals,
    calculate_indicator,
    crossover, crossunder
)


def run_strategy(market_data_df: pd.DataFrame, params: list) -> pd.Series:
    """
    Multi-Indicator Strategy using comprehensive TA-Lib indicators
    
    This strategy combines indicators from multiple categories for robust signal generation.
    Requires multiple confirmations before generating trading signals.
    """
    signals = pd.Series(index=market_data_df.index, data=0.0)
    
    # Extract parameters for customization
    confirmation_threshold = int(params[0]) if len(params) > 0 and params[0] > 0 else {confirmation_count}
    
    # EVOLVE-BLOCK-START
    try:
        df = market_data_df.copy()
        
        # Add comprehensive indicators from multiple categories
        df = add_comprehensive_indicators(
            df, 
            categories={categories},
            custom_indicators={{
                'RSI': {{'timeperiod': 14}},
                'MACD': {{'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}},
                'BBANDS': {{'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2}},
                'ADX': {{'timeperiod': 14}},
                'STOCH': {{'fastk_period': 14, 'slowk_period': 3, 'slowd_period': 3}},
                'ATR': {{'timeperiod': 14}},
                {'\'OBV\': {},' if use_volume else ''}
            }}
        )
        
        # Initialize signal counters
        buy_score = pd.Series(0, index=df.index)
        sell_score = pd.Series(0, index=df.index)
        
        # Momentum signals
        if 'rsi' in df.columns:
            buy_score += (df['rsi'] < 30).astype(int)
            sell_score += (df['rsi'] > 70).astype(int)
        
        # Trend signals
        if 'macd' in df.columns and 'macdsignal' in df.columns:
            buy_score += (df['macd'] > df['macdsignal']).astype(int)
            sell_score += (df['macd'] < df['macdsignal']).astype(int)
        
        # Volatility signals (Bollinger Bands)
        if 'lowerband' in df.columns and 'upperband' in df.columns:
            buy_score += (df['close'] <= df['lowerband']).astype(int)
            sell_score += (df['close'] >= df['upperband']).astype(int)
        
        # Stochastic signals
        if 'slowk' in df.columns:
            buy_score += (df['slowk'] < 20).astype(int)
            sell_score += (df['slowk'] > 80).astype(int)
        
        # ADX trend strength filter
        trend_strength = pd.Series(True, index=df.index)
        if 'adx' in df.columns:
            trend_strength = df['adx'] > 20
        
        # Volume confirmation
        volume_confirm = pd.Series(True, index=df.index)
        {"if 'obv' in df.columns:" if use_volume else "# Volume confirmation disabled"}
        {"    volume_confirm = df['obv'] > df['obv'].shift(1)" if use_volume else ""}
        
        # Generate signals based on confirmation count
        buy_signals = (buy_score >= confirmation_threshold) & trend_strength & volume_confirm
        sell_signals = (sell_score >= confirmation_threshold) & trend_strength
        
        # Only generate signals on first occurrence
        signals[buy_signals & ~buy_signals.shift(1).fillna(False)] = 1.0
        signals[sell_signals & ~sell_signals.shift(1).fillna(False)] = -1.0
        
        # Exit signals when momentum reverses
        if 'rsi' in df.columns:
            momentum_reversal_up = (df['rsi'] > 50) & (df['rsi'].shift(1) <= 50)
            momentum_reversal_down = (df['rsi'] < 50) & (df['rsi'].shift(1) >= 50)
            
            # Clear position on momentum reversal
            signals[momentum_reversal_up & (buy_score < confirmation_threshold)] = 0.0
            signals[momentum_reversal_down & (sell_score < confirmation_threshold)] = 0.0
        
        # Risk management: ATR-based stop loss signals
        if 'atr' in df.columns:
            # This could be enhanced with actual stop-loss logic
            pass
        
    except Exception as e:
        print(f"Error in multi-indicator strategy: {{e}}")
        # Fallback to simple dual-indicator strategy
        df = market_data_df.copy()
        df = calculate_indicator(df, 'RSI', timeperiod=14)
        df = calculate_indicator(df, 'MACD', fastperiod=12, slowperiod=26, signalperiod=9)
        
        rsi_buy = df['rsi'] < 30
        macd_buy = df['macd'] > df['macdsignal']
        signals[rsi_buy & macd_buy] = 1.0
        
        rsi_sell = df['rsi'] > 70
        macd_sell = df['macd'] < df['macdsignal']
        signals[rsi_sell & macd_sell] = -1.0
    
    # EVOLVE-BLOCK-END
    
    return signals.fillna(0.0)


def get_parameters() -> list:
    """Returns default parameters for the multi-indicator strategy."""
    return [{confirmation_count}]  # [confirmation_threshold]


def get_trading_signals_func():
    """Returns the run_strategy function."""
    return run_strategy
'''
        return strategy_code

    def _generate_momentum_strategy(self, config: Dict[str, Any]) -> str:
        """Generate a momentum-based strategy"""
        return self._generate_trend_following_strategy(config)  # Placeholder

    def _generate_adaptive_strategy(self, config: Dict[str, Any]) -> str:
        """Generate an adaptive strategy that switches between different approaches"""
        return self._generate_multi_indicator_strategy(config)  # Placeholder

    def _generate_breakout_strategy(self, config: Dict[str, Any]) -> str:
        """Generate a breakout strategy using Bollinger Bands and volume"""
        return self._generate_multi_indicator_strategy(config)  # Placeholder

    def _generate_volatility_based_strategy(self, config: Dict[str, Any]) -> str:
        """Generate a volatility-based strategy using ATR"""
        return self._generate_multi_indicator_strategy(config)  # Placeholder

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy types"""
        return list(self.strategy_templates.keys())

    def get_strategy_description(self, strategy_type: str) -> str:
        """Get description of a specific strategy type"""
        descriptions = {
            "trend_following": "Uses moving averages, MACD, and ADX to identify and follow trends",
            "mean_reversion": "Uses RSI, Bollinger Bands, and Stochastic to identify oversold/overbought conditions",
            "momentum": "Uses momentum indicators like MACD, ROC, and CMO to capitalize on price momentum",
            "multi_indicator": "Combines multiple indicator categories with confirmation requirements",
            "adaptive": "Switches between strategies based on market regime detection",
            "breakout": "Identifies breakouts using volatility and volume indicators",
            "volatility_based": "Uses volatility measures for position sizing and risk management",
        }
        return descriptions.get(strategy_type, "No description available")


# Convenience functions for direct use
def generate_enhanced_strategy(
    strategy_type: str = "multi_indicator", config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate an enhanced trading strategy using comprehensive TA-Lib indicators.

    Args:
        strategy_type (str): Type of strategy to generate
        config (Dict): Configuration parameters

    Returns:
        str: Generated strategy code
    """
    generator = EnhancedStrategyGenerator()
    return generator.generate_strategy(strategy_type, config)


def get_strategy_suggestions(data_columns: List[str]) -> Dict[str, List[str]]:
    """
    Get strategy suggestions based on available data columns.

    Args:
        data_columns (List[str]): Available data columns

    Returns:
        Dict[str, List[str]]: Suggested strategies by category
    """
    suggestions = {"basic": [], "advanced": [], "volume_based": []}

    has_ohlc = all(
        col in [c.lower() for c in data_columns] for col in ["open", "high", "low", "close"]
    )
    has_volume = any(col.lower() == "volume" for col in data_columns)

    if has_ohlc:
        suggestions["basic"].extend(["trend_following", "mean_reversion", "momentum"])
        suggestions["advanced"].extend(["multi_indicator", "adaptive", "breakout"])

        if has_volume:
            suggestions["volume_based"].extend(["multi_indicator", "breakout", "volatility_based"])

    return suggestions
