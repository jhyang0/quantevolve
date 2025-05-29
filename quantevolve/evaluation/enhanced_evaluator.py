"""
Enhanced Evaluator for QuantEvolve

This module provides enhanced evaluation capabilities that consider technical indicator
usage, strategy complexity, and comprehensive performance metrics for strategies
that leverage the full TA-Lib indicator library.
"""

import numpy as np
import pandas as pd
import importlib.util
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from ..utils.indicators import (
    get_available_indicators_for_data,
    create_indicator_summary
)
from ..utils.talib_indicators_registry import (
    TALIB_INDICATORS_REGISTRY,
    get_indicators_by_category,
    get_strategy_relevant_indicators
)


class EnhancedQuantEvaluator:
    """Enhanced evaluator with comprehensive indicator analysis"""
    
    def __init__(self, market_data_path: Optional[str] = None):
        self.market_data_path = market_data_path
        self.market_data_df = None
        self.available_indicators = None
        
    def load_market_data(self, market_data_path: Optional[str] = None) -> bool:
        """Load and preprocess market data"""
        if market_data_path:
            self.market_data_path = market_data_path
        
        if not self.market_data_path:
            # Try to find market data in common locations
            current_script_path = Path(__file__).resolve()
            project_root = current_script_path.parent.parent.parent
            
            market_data_paths_to_try = [
                project_root / "examples" / "quant_evolve" / "data" / "btc_usdt_1h_2023.csv",
                Path("examples") / "quant_evolve" / "data" / "btc_usdt_1h_2023.csv",
                Path("data") / "btc_usdt_1h_2023.csv",
                Path("../data/btc_usdt_1h_2023.csv"),
            ]
            
            for p in market_data_paths_to_try:
                if p.exists():
                    self.market_data_path = str(p)
                    break
            
            if not self.market_data_path:
                print(f"Error: Market data file not found. Tried: {[str(p) for p in market_data_paths_to_try]}")
                return False
        
        try:
            self.market_data_df = pd.read_csv(self.market_data_path)
            
            # Preprocess data
            self.market_data_df["timestamp"] = pd.to_datetime(self.market_data_df["timestamp"])
            self.market_data_df.set_index("timestamp", inplace=True)
            
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                if col in self.market_data_df.columns:
                    self.market_data_df[col] = pd.to_numeric(self.market_data_df[col], errors="coerce")
            
            self.market_data_df.dropna(subset=numeric_cols, inplace=True)
            
            if self.market_data_df.empty:
                print("Error: Market data is empty after processing.")
                return False
            
            # Analyze available indicators for this data
            self.available_indicators = get_available_indicators_for_data(self.market_data_df)
            
            return True
            
        except Exception as e:
            print(f"Error loading market data: {str(e)}")
            return False
    
    def analyze_strategy_code(self, strategy_path: str) -> Dict[str, Any]:
        """Analyze strategy code for indicator usage and complexity"""
        analysis = {
            "indicators_used": [],
            "indicator_categories": set(),
            "complexity_score": 0,
            "has_multi_timeframe": False,
            "has_risk_management": False,
            "has_position_sizing": False,
            "code_quality_score": 0,
            "total_lines": 0,
            "function_count": 0
        }
        
        try:
            with open(strategy_path, 'r') as f:
                code_content = f.read()
            
            # Count total lines and functions
            lines = code_content.split('\n')
            analysis["total_lines"] = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            analysis["function_count"] = len(re.findall(r'def\s+\w+\s*\(', code_content))
            
            # Analyze indicator usage
            all_indicators = []
            for category_data in TALIB_INDICATORS_REGISTRY.values():
                all_indicators.extend(category_data.get('indicators', {}).keys())
            
            for indicator in all_indicators:
                if indicator in code_content:
                    analysis["indicators_used"].append(indicator)
                    # Find which category this indicator belongs to
                    for category, category_data in TALIB_INDICATORS_REGISTRY.items():
                        if indicator in category_data.get('indicators', {}):
                            analysis["indicator_categories"].add(category)
                            break
            
            # Analyze strategy sophistication
            analysis["complexity_score"] = self._calculate_complexity_score(code_content, analysis)
            
            # Check for advanced features
            analysis["has_multi_timeframe"] = any(term in code_content.lower() for term in 
                                                ['timeframe', 'period.*period', 'fast.*slow'])
            
            analysis["has_risk_management"] = any(term in code_content.lower() for term in 
                                                ['atr', 'stop', 'risk', 'drawdown', 'volatility'])
            
            analysis["has_position_sizing"] = any(term in code_content.lower() for term in 
                                                ['position_size', 'sizing', 'capital', 'allocation'])
            
            # Code quality indicators
            analysis["code_quality_score"] = self._calculate_code_quality_score(code_content)
            
            # Convert set to list for JSON serialization
            analysis["indicator_categories"] = list(analysis["indicator_categories"])
            
        except Exception as e:
            print(f"Error analyzing strategy code: {str(e)}")
        
        return analysis
    
    def _calculate_complexity_score(self, code_content: str, analysis: Dict[str, Any]) -> float:
        """Calculate strategy complexity score based on various factors"""
        score = 0.0
        
        # Base score from number of indicators
        score += len(analysis["indicators_used"]) * 2
        
        # Bonus for using multiple categories
        score += len(analysis["indicator_categories"]) * 5
        
        # Bonus for advanced patterns
        if 'crossover' in code_content or 'crossunder' in code_content:
            score += 10
        
        if 'confirmation' in code_content.lower():
            score += 15
        
        if 'regime' in code_content.lower() or 'adaptive' in code_content.lower():
            score += 20
        
        # Bonus for error handling
        if 'try:' in code_content and 'except' in code_content:
            score += 10
        
        # Bonus for parameter validation
        if 'params' in code_content and ('len(params)' in code_content or 'if params' in code_content):
            score += 5
        
        return score
    
    def _calculate_code_quality_score(self, code_content: str) -> float:
        """Calculate code quality score"""
        score = 0.0
        
        # Documentation
        if '"""' in code_content:
            score += 20
        if 'Args:' in code_content and 'Returns:' in code_content:
            score += 10
        
        # Error handling
        if 'try:' in code_content and 'except' in code_content:
            score += 15
        
        # Proper imports
        if 'from quantevolve.utils.indicators import' in code_content:
            score += 10
        
        # Comments
        comment_lines = len([line for line in code_content.split('\n') if line.strip().startswith('#')])
        score += min(comment_lines * 2, 20)  # Max 20 points for comments
        
        return min(score, 100)  # Cap at 100
    
    def evaluate_enhanced(self, program_path: str) -> Dict[str, Any]:
        """Enhanced evaluation including indicator analysis and strategy sophistication"""
        
        # Default error metrics
        default_error_metrics = {
            "pnl": 0.0,
            "sharpe_ratio": -100.0,
            "negative_max_drawdown": -1.0,
            "num_trades": 0,
            "error": 1.0,
            "can_run": 0.0,
            "combined_score": -100.0,
            "error_message": "Evaluation not started",
            # Enhanced metrics
            "strategy_analysis": {},
            "indicator_score": 0.0,
            "complexity_score": 0.0,
            "innovation_score": 0.0,
            "risk_adjusted_score": 0.0,
            "comprehensive_score": -100.0
        }
        
        # Load market data if not already loaded
        if self.market_data_df is None:
            if not self.load_market_data():
                default_error_metrics["error_message"] = "Failed to load market data"
                return default_error_metrics
        
        # Analyze strategy code
        try:
            strategy_analysis = self.analyze_strategy_code(program_path)
            default_error_metrics["strategy_analysis"] = strategy_analysis
        except Exception as e:
            print(f"Error analyzing strategy code: {str(e)}")
            strategy_analysis = {}
        
        # Load and run strategy (using existing logic)
        try:
            strategy_path = Path(program_path).resolve()
            if not strategy_path.exists():
                default_error_metrics["error_message"] = f"Strategy program file not found: {str(program_path)}"
                return default_error_metrics
            
            # Import strategy module
            module_name = strategy_path.stem
            spec = importlib.util.spec_from_file_location(module_name, strategy_path)
            if spec is None or spec.loader is None:
                default_error_metrics["error_message"] = f"Could not create module spec for {str(program_path)}"
                return default_error_metrics
            
            strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_module)
            
            if not hasattr(strategy_module, "get_parameters"):
                default_error_metrics["error_message"] = f"'get_parameters' missing in {str(program_path)}"
                return default_error_metrics
            params = strategy_module.get_parameters()
            
            if not hasattr(strategy_module, "run_strategy"):
                default_error_metrics["error_message"] = f"'run_strategy' missing in {str(program_path)}"
                return default_error_metrics
            run_strategy_func = strategy_module.run_strategy
            
        except Exception as e:
            default_error_metrics["error_message"] = f"Error loading strategy {str(program_path)}: {str(e)}"
            return default_error_metrics
        
        # Get trading signals
        try:
            signals = run_strategy_func(self.market_data_df.copy(), params)
            if not isinstance(signals, pd.Series):
                default_error_metrics["error_message"] = "Strategy did not return a Pandas Series."
                return default_error_metrics
            if not signals.index.equals(self.market_data_df.index):
                signals = signals.reindex(self.market_data_df.index, fill_value=0.0)
                
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            print(f"Error running strategy: {str(e)}\\n{tb_str}")
            default_error_metrics["error_message"] = f"Error running strategy {str(program_path)}: {str(e)}"
            return default_error_metrics
        
        # Calculate enhanced metrics
        try:
            # Basic performance metrics (from original evaluator)
            basic_metrics = self._calculate_basic_metrics(signals)
            
            # Enhanced metrics
            enhanced_metrics = self._calculate_enhanced_metrics(signals, strategy_analysis)
            
            # Combine all metrics
            final_metrics = {**basic_metrics, **enhanced_metrics}
            final_metrics["strategy_analysis"] = strategy_analysis
            
            return final_metrics
            
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            print(f"Error during enhanced evaluation: {str(e)}\\n{tb_str}")
            default_error_metrics["error_message"] = f"Enhanced evaluation error: {str(e)}"
            return default_error_metrics
    
    def _calculate_basic_metrics(self, signals: pd.Series) -> Dict[str, Any]:
        """Calculate basic performance metrics"""
        # Ensure signals are numeric and fill NaNs
        signals = pd.to_numeric(signals, errors="coerce").fillna(0.0)
        signals = np.sign(signals).astype(int)
        
        # Shift signals to trade on the next bar
        positions = signals.shift(1).fillna(0.0)
        
        # Calculate returns
        market_returns = self.market_data_df["close"].pct_change().fillna(0.0)
        strategy_returns = positions * market_returns
        
        if not isinstance(strategy_returns, pd.Series):
            strategy_returns = pd.Series(strategy_returns, index=self.market_data_df.index).fillna(0.0)
        
        # Basic metrics
        pnl = strategy_returns.sum()
        
        # Sharpe Ratio (assuming hourly data)
        annualization_factor = 24 * 365
        mean_strategy_return = strategy_returns.mean()
        std_dev_strategy_returns = strategy_returns.std()
        
        if std_dev_strategy_returns == 0 or np.isnan(std_dev_strategy_returns) or np.isinf(std_dev_strategy_returns):
            sharpe_ratio = -100.0 if mean_strategy_return <= 0 else 0
        else:
            sharpe_ratio = (mean_strategy_return * np.sqrt(annualization_factor)) / std_dev_strategy_returns
            if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
                sharpe_ratio = -100.0
        
        # Maximum Drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        negative_max_drawdown = drawdown.min()
        if np.isnan(negative_max_drawdown) or np.isinf(negative_max_drawdown):
            negative_max_drawdown = -1.0
        
        # Number of Trades
        num_trades = (positions.diff().fillna(0) != 0).sum()
        
        # Combined score
        combined_score = sharpe_ratio if sharpe_ratio > -100.0 else -100.0
        if pnl < 0 and combined_score > -5:
            combined_score = max(-5.0, combined_score * 0.5)
        if num_trades < 2:
            combined_score = min(combined_score, -50.0)
        
        return {
            "pnl": float(pnl),
            "sharpe_ratio": float(sharpe_ratio),
            "negative_max_drawdown": float(negative_max_drawdown),
            "num_trades": int(num_trades),
            "error": 0.0,
            "can_run": 1.0,
            "combined_score": float(combined_score),
            "error_message": "",
        }
    
    def _calculate_enhanced_metrics(self, signals: pd.Series, strategy_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate enhanced performance metrics"""
        
        # Indicator sophistication score
        indicator_score = self._calculate_indicator_score(strategy_analysis)
        
        # Complexity score (already calculated in strategy_analysis)
        complexity_score = strategy_analysis.get("complexity_score", 0.0)
        
        # Innovation score (using less common indicators or advanced techniques)
        innovation_score = self._calculate_innovation_score(strategy_analysis)
        
        # Risk-adjusted score
        risk_adjusted_score = self._calculate_risk_adjusted_score(signals, strategy_analysis)
        
        # Comprehensive score combining all factors
        comprehensive_score = self._calculate_comprehensive_score(
            indicator_score, complexity_score, innovation_score, risk_adjusted_score, strategy_analysis
        )
        
        # Additional metrics
        signal_metrics = self._calculate_signal_metrics(signals)
        
        enhanced_metrics = {
            "indicator_score": float(indicator_score),
            "complexity_score": float(complexity_score),
            "innovation_score": float(innovation_score),
            "risk_adjusted_score": float(risk_adjusted_score),
            "comprehensive_score": float(comprehensive_score),
            **signal_metrics
        }
        
        return enhanced_metrics
    
    def _calculate_indicator_score(self, strategy_analysis: Dict[str, Any]) -> float:
        """Calculate score based on indicator usage"""
        score = 0.0
        
        # Base score for number of indicators
        num_indicators = len(strategy_analysis.get("indicators_used", []))
        score += min(num_indicators * 5, 50)  # Max 50 points, 5 per indicator
        
        # Bonus for using multiple categories
        num_categories = len(strategy_analysis.get("indicator_categories", []))
        score += num_categories * 10  # 10 points per category
        
        # Bonus for specific valuable indicators
        valuable_indicators = ['MACD', 'RSI', 'BBANDS', 'ADX', 'ATR', 'STOCH', 'OBV']
        used_valuable = [ind for ind in strategy_analysis.get("indicators_used", []) if ind in valuable_indicators]
        score += len(used_valuable) * 3
        
        return min(score, 100)  # Cap at 100
    
    def _calculate_innovation_score(self, strategy_analysis: Dict[str, Any]) -> float:
        """Calculate innovation score based on advanced techniques"""
        score = 0.0
        
        # Bonus for using less common indicators
        advanced_indicators = ['HT_SINE', 'HT_TRENDMODE', 'LINEARREG', 'BETA', 'CORREL', 'TSF']
        used_advanced = [ind for ind in strategy_analysis.get("indicators_used", []) if ind in advanced_indicators]
        score += len(used_advanced) * 10
        
        # Bonus for pattern recognition
        pattern_indicators = [ind for ind in strategy_analysis.get("indicators_used", []) if ind.startswith('CDL')]
        score += len(pattern_indicators) * 5
        
        # Bonus for advanced features
        if strategy_analysis.get("has_multi_timeframe", False):
            score += 20
        if strategy_analysis.get("has_risk_management", False):
            score += 15
        if strategy_analysis.get("has_position_sizing", False):
            score += 15
        
        # Code quality bonus
        score += strategy_analysis.get("code_quality_score", 0) * 0.2
        
        return min(score, 100)  # Cap at 100
    
    def _calculate_risk_adjusted_score(self, signals: pd.Series, strategy_analysis: Dict[str, Any]) -> float:
        """Calculate risk-adjusted performance score"""
        score = 50.0  # Base score
        
        # Penalty for too many signals (overtrading)
        signal_frequency = (signals != 0).sum() / len(signals)
        if signal_frequency > 0.1:  # More than 10% of time in signal
            score -= (signal_frequency - 0.1) * 100
        
        # Bonus for risk management indicators
        if strategy_analysis.get("has_risk_management", False):
            score += 20
        
        # Bonus for using volatility indicators
        volatility_indicators = ['ATR', 'NATR', 'BBANDS', 'STDDEV']
        used_volatility = [ind for ind in strategy_analysis.get("indicators_used", []) if ind in volatility_indicators]
        score += len(used_volatility) * 5
        
        return max(0, min(score, 100))  # Cap between 0 and 100
    
    def _calculate_comprehensive_score(self, indicator_score: float, complexity_score: float, 
                                     innovation_score: float, risk_adjusted_score: float,
                                     strategy_analysis: Dict[str, Any]) -> float:
        """Calculate comprehensive score combining all factors"""
        
        # Weighted combination
        weights = {
            "indicator": 0.3,
            "complexity": 0.2,
            "innovation": 0.3,
            "risk_adjusted": 0.2
        }
        
        comprehensive_score = (
            indicator_score * weights["indicator"] +
            complexity_score * weights["complexity"] +
            innovation_score * weights["innovation"] +
            risk_adjusted_score * weights["risk_adjusted"]
        )
        
        # Bonus for well-rounded strategies (using multiple aspects)
        if (indicator_score > 30 and complexity_score > 20 and 
            innovation_score > 20 and risk_adjusted_score > 40):
            comprehensive_score *= 1.1  # 10% bonus
        
        return min(comprehensive_score, 100)  # Cap at 100
    
    def _calculate_signal_metrics(self, signals: pd.Series) -> Dict[str, Any]:
        """Calculate signal-specific metrics"""
        signal_changes = (signals.diff() != 0).sum()
        signal_frequency = (signals != 0).sum() / len(signals)
        
        # Signal consistency (how often signals change)
        consistency_score = max(0, 100 - (signal_changes / len(signals)) * 1000)
        
        return {
            "signal_frequency": float(signal_frequency),
            "signal_changes": int(signal_changes),
            "signal_consistency_score": float(consistency_score)
        }


def evaluate_enhanced(program_path: str, market_data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Enhanced evaluation function with comprehensive indicator analysis.
    
    Args:
        program_path (str): Path to the strategy program
        market_data_path (Optional[str]): Path to market data CSV
        
    Returns:
        Dict[str, Any]: Comprehensive evaluation metrics
    """
    evaluator = EnhancedQuantEvaluator(market_data_path)
    return evaluator.evaluate_enhanced(program_path)


# Backward compatibility - use enhanced evaluator as default
def evaluate(program_path: str) -> Dict[str, Any]:
    """Backward compatible evaluate function using enhanced evaluator"""
    return evaluate_enhanced(program_path)