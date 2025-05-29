"""
TA-Lib Indicators Registry for QuantEvolve

This module provides a comprehensive registry of all available TA-Lib indicators
organized by category to facilitate strategy generation and LLM prompting.
"""

from typing import Dict, List, Any
import inspect

# Comprehensive TA-Lib indicators organized by category
TALIB_INDICATORS_REGISTRY = {
    "overlap_studies": {
        "description": "Overlap studies - indicators that overlay on price charts",
        "indicators": {
            "BBANDS": {
                "name": "Bollinger Bands",
                "params": {"timeperiod": 5, "nbdevup": 2.0, "nbdevdn": 2.0, "matype": 0},
                "inputs": ["close"],
                "outputs": ["upperband", "middleband", "lowerband"],
                "description": "Bollinger Bands - volatility indicator with upper/lower bands"
            },
            "DEMA": {
                "name": "Double Exponential Moving Average",
                "params": {"timeperiod": 30},
                "inputs": ["close"],
                "outputs": ["dema"],
                "description": "Double Exponential Moving Average - reduced lag EMA"
            },
            "EMA": {
                "name": "Exponential Moving Average",
                "params": {"timeperiod": 30},
                "inputs": ["close"],
                "outputs": ["ema"],
                "description": "Exponential Moving Average - trend following indicator"
            },
            "HT_TRENDLINE": {
                "name": "Hilbert Transform - Instantaneous Trendline",
                "params": {},
                "inputs": ["close"],
                "outputs": ["ht_trendline"],
                "description": "Hilbert Transform Instantaneous Trendline - adaptive trend"
            },
            "KAMA": {
                "name": "Kaufman Adaptive Moving Average",
                "params": {"timeperiod": 30},
                "inputs": ["close"],
                "outputs": ["kama"],
                "description": "Kaufman Adaptive Moving Average - efficiency ratio based MA"
            },
            "MA": {
                "name": "Moving Average",
                "params": {"timeperiod": 30, "matype": 0},
                "inputs": ["close"],
                "outputs": ["ma"],
                "description": "Moving Average - configurable MA type"
            },
            "MAMA": {
                "name": "MESA Adaptive Moving Average",
                "params": {"fastlimit": 0.5, "slowlimit": 0.05},
                "inputs": ["close"],
                "outputs": ["mama", "fama"],
                "description": "MESA Adaptive Moving Average - cycle adaptive"
            },
            "MAVP": {
                "name": "Moving Average with Variable Period",
                "params": {"minperiod": 2, "maxperiod": 30, "matype": 0},
                "inputs": ["close", "periods"],
                "outputs": ["mavp"],
                "description": "Moving Average with Variable Period"
            },
            "MIDPOINT": {
                "name": "MidPoint over period",
                "params": {"timeperiod": 14},
                "inputs": ["close"],
                "outputs": ["midpoint"],
                "description": "MidPoint over period - (high+low)/2"
            },
            "MIDPRICE": {
                "name": "Midpoint Price over period",
                "params": {"timeperiod": 14},
                "inputs": ["high", "low"],
                "outputs": ["midprice"],
                "description": "Midpoint Price over period"
            },
            "SAR": {
                "name": "Parabolic SAR",
                "params": {"acceleration": 0.02, "maximum": 0.2},
                "inputs": ["high", "low"],
                "outputs": ["sar"],
                "description": "Parabolic SAR - trend reversal indicator"
            },
            "SAREXT": {
                "name": "Parabolic SAR - Extended",
                "params": {"startvalue": 0, "offsetonreverse": 0, "accelerationinitlong": 0.02, 
                          "accelerationlong": 0.02, "accelerationmaxlong": 0.2,
                          "accelerationinitshort": 0.02, "accelerationshort": 0.02, 
                          "accelerationmaxshort": 0.2},
                "inputs": ["high", "low"],
                "outputs": ["sarext"],
                "description": "Parabolic SAR Extended - enhanced SAR with more parameters"
            },
            "SMA": {
                "name": "Simple Moving Average",
                "params": {"timeperiod": 30},
                "inputs": ["close"],
                "outputs": ["sma"],
                "description": "Simple Moving Average - basic trend indicator"
            },
            "T3": {
                "name": "Triple Exponential Moving Average (T3)",
                "params": {"timeperiod": 5, "vfactor": 0.7},
                "inputs": ["close"],
                "outputs": ["t3"],
                "description": "Triple Exponential Moving Average - smoother EMA"
            },
            "TEMA": {
                "name": "Triple Exponential Moving Average",
                "params": {"timeperiod": 30},
                "inputs": ["close"],
                "outputs": ["tema"],
                "description": "Triple Exponential Moving Average - reduced lag"
            },
            "TRIMA": {
                "name": "Triangular Moving Average",
                "params": {"timeperiod": 30},
                "inputs": ["close"],
                "outputs": ["trima"],
                "description": "Triangular Moving Average - double smoothed"
            },
            "WMA": {
                "name": "Weighted Moving Average",
                "params": {"timeperiod": 30},
                "inputs": ["close"],
                "outputs": ["wma"],
                "description": "Weighted Moving Average - linearly weighted"
            }
        }
    },
    "momentum_indicators": {
        "description": "Momentum indicators - measure rate of price change",
        "indicators": {
            "ADX": {
                "name": "Average Directional Movement Index",
                "params": {"timeperiod": 14},
                "inputs": ["high", "low", "close"],
                "outputs": ["adx"],
                "description": "Average Directional Movement Index - trend strength"
            },
            "ADXR": {
                "name": "Average Directional Movement Index Rating",
                "params": {"timeperiod": 14},
                "inputs": ["high", "low", "close"],
                "outputs": ["adxr"],
                "description": "Average Directional Movement Index Rating"
            },
            "APO": {
                "name": "Absolute Price Oscillator",
                "params": {"fastperiod": 12, "slowperiod": 26, "matype": 0},
                "inputs": ["close"],
                "outputs": ["apo"],
                "description": "Absolute Price Oscillator - price momentum"
            },
            "AROON": {
                "name": "Aroon",
                "params": {"timeperiod": 14},
                "inputs": ["high", "low"],
                "outputs": ["aroondown", "aroonup"],
                "description": "Aroon - trend direction and strength"
            },
            "AROONOSC": {
                "name": "Aroon Oscillator",
                "params": {"timeperiod": 14},
                "inputs": ["high", "low"],
                "outputs": ["aroonosc"],
                "description": "Aroon Oscillator - Aroon Up minus Aroon Down"
            },
            "BOP": {
                "name": "Balance Of Power",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["bop"],
                "description": "Balance Of Power - buying vs selling pressure"
            },
            "CCI": {
                "name": "Commodity Channel Index",
                "params": {"timeperiod": 14},
                "inputs": ["high", "low", "close"],
                "outputs": ["cci"],
                "description": "Commodity Channel Index - cyclical trends"
            },
            "CMO": {
                "name": "Chande Momentum Oscillator",
                "params": {"timeperiod": 14},
                "inputs": ["close"],
                "outputs": ["cmo"],
                "description": "Chande Momentum Oscillator - momentum with volume"
            },
            "DX": {
                "name": "Directional Movement Index",
                "params": {"timeperiod": 14},
                "inputs": ["high", "low", "close"],
                "outputs": ["dx"],
                "description": "Directional Movement Index - trend direction"
            },
            "MACD": {
                "name": "Moving Average Convergence/Divergence",
                "params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
                "inputs": ["close"],
                "outputs": ["macd", "macdsignal", "macdhist"],
                "description": "MACD - trend following momentum indicator"
            },
            "MACDEXT": {
                "name": "MACD with controllable MA type",
                "params": {"fastperiod": 12, "fastmatype": 0, "slowperiod": 26, 
                          "slowmatype": 0, "signalperiod": 9, "signalmatype": 0},
                "inputs": ["close"],
                "outputs": ["macd", "macdsignal", "macdhist"],
                "description": "MACD Extended - configurable MA types"
            },
            "MACDFIX": {
                "name": "Moving Average Convergence/Divergence Fix 12/26",
                "params": {"signalperiod": 9},
                "inputs": ["close"],
                "outputs": ["macd", "macdsignal", "macdhist"],
                "description": "MACD Fixed - standard 12/26 periods"
            },
            "MFI": {
                "name": "Money Flow Index",
                "params": {"timeperiod": 14},
                "inputs": ["high", "low", "close", "volume"],
                "outputs": ["mfi"],
                "description": "Money Flow Index - volume-weighted RSI"
            },
            "MINUS_DI": {
                "name": "Minus Directional Indicator",
                "params": {"timeperiod": 14},
                "inputs": ["high", "low", "close"],
                "outputs": ["minus_di"],
                "description": "Minus Directional Indicator - negative trend"
            },
            "MINUS_DM": {
                "name": "Minus Directional Movement",
                "params": {"timeperiod": 14},
                "inputs": ["high", "low"],
                "outputs": ["minus_dm"],
                "description": "Minus Directional Movement"
            },
            "MOM": {
                "name": "Momentum",
                "params": {"timeperiod": 10},
                "inputs": ["close"],
                "outputs": ["mom"],
                "description": "Momentum - rate of change"
            },
            "PLUS_DI": {
                "name": "Plus Directional Indicator",
                "params": {"timeperiod": 14},
                "inputs": ["high", "low", "close"],
                "outputs": ["plus_di"],
                "description": "Plus Directional Indicator - positive trend"
            },
            "PLUS_DM": {
                "name": "Plus Directional Movement",
                "params": {"timeperiod": 14},
                "inputs": ["high", "low"],
                "outputs": ["plus_dm"],
                "description": "Plus Directional Movement"
            },
            "PPO": {
                "name": "Percentage Price Oscillator",
                "params": {"fastperiod": 12, "slowperiod": 26, "matype": 0},
                "inputs": ["close"],
                "outputs": ["ppo"],
                "description": "Percentage Price Oscillator - MACD as percentage"
            },
            "ROC": {
                "name": "Rate of change",
                "params": {"timeperiod": 10},
                "inputs": ["close"],
                "outputs": ["roc"],
                "description": "Rate of Change - momentum indicator"
            },
            "ROCP": {
                "name": "Rate of change Percentage",
                "params": {"timeperiod": 10},
                "inputs": ["close"],
                "outputs": ["rocp"],
                "description": "Rate of Change Percentage"
            },
            "ROCR": {
                "name": "Rate of change ratio",
                "params": {"timeperiod": 10},
                "inputs": ["close"],
                "outputs": ["rocr"],
                "description": "Rate of Change Ratio"
            },
            "ROCR100": {
                "name": "Rate of change ratio 100 scale",
                "params": {"timeperiod": 10},
                "inputs": ["close"],
                "outputs": ["rocr100"],
                "description": "Rate of Change Ratio 100 scale"
            },
            "RSI": {
                "name": "Relative Strength Index",
                "params": {"timeperiod": 14},
                "inputs": ["close"],
                "outputs": ["rsi"],
                "description": "Relative Strength Index - overbought/oversold"
            },
            "STOCH": {
                "name": "Stochastic",
                "params": {"fastk_period": 5, "slowk_period": 3, "slowk_matype": 0,
                          "slowd_period": 3, "slowd_matype": 0},
                "inputs": ["high", "low", "close"],
                "outputs": ["slowk", "slowd"],
                "description": "Stochastic Oscillator - momentum indicator"
            },
            "STOCHF": {
                "name": "Stochastic Fast",
                "params": {"fastk_period": 5, "fastd_period": 3, "fastd_matype": 0},
                "inputs": ["high", "low", "close"],
                "outputs": ["fastk", "fastd"],
                "description": "Stochastic Fast - faster version"
            },
            "STOCHRSI": {
                "name": "Stochastic Relative Strength Index",
                "params": {"timeperiod": 14, "fastk_period": 5, "fastd_period": 3, "fastd_matype": 0},
                "inputs": ["close"],
                "outputs": ["fastk", "fastd"],
                "description": "Stochastic RSI - RSI with stochastic calculation"
            },
            "TRIX": {
                "name": "1-day Rate-Of-Change (ROC) of a Triple Smooth EMA",
                "params": {"timeperiod": 30},
                "inputs": ["close"],
                "outputs": ["trix"],
                "description": "TRIX - triple smooth EMA rate of change"
            },
            "ULTOSC": {
                "name": "Ultimate Oscillator",
                "params": {"timeperiod1": 7, "timeperiod2": 14, "timeperiod3": 28},
                "inputs": ["high", "low", "close"],
                "outputs": ["ultosc"],
                "description": "Ultimate Oscillator - multi-timeframe momentum"
            },
            "WILLR": {
                "name": "Williams' %R",
                "params": {"timeperiod": 14},
                "inputs": ["high", "low", "close"],
                "outputs": ["willr"],
                "description": "Williams %R - overbought/oversold indicator"
            }
        }
    },
    "volume_indicators": {
        "description": "Volume indicators - analyze trading volume patterns",
        "indicators": {
            "AD": {
                "name": "Chaikin A/D Line",
                "params": {},
                "inputs": ["high", "low", "close", "volume"],
                "outputs": ["ad"],
                "description": "Chaikin A/D Line - accumulation/distribution"
            },
            "ADOSC": {
                "name": "Chaikin A/D Oscillator",
                "params": {"fastperiod": 3, "slowperiod": 10},
                "inputs": ["high", "low", "close", "volume"],
                "outputs": ["adosc"],
                "description": "Chaikin A/D Oscillator - A/D line momentum"
            },
            "OBV": {
                "name": "On Balance Volume",
                "params": {},
                "inputs": ["close", "volume"],
                "outputs": ["obv"],
                "description": "On Balance Volume - volume momentum indicator"
            }
        }
    },
    "volatility_indicators": {
        "description": "Volatility indicators - measure price volatility",
        "indicators": {
            "ATR": {
                "name": "Average True Range",
                "params": {"timeperiod": 14},
                "inputs": ["high", "low", "close"],
                "outputs": ["atr"],
                "description": "Average True Range - volatility measure"
            },
            "NATR": {
                "name": "Normalized Average True Range",
                "params": {"timeperiod": 14},
                "inputs": ["high", "low", "close"],
                "outputs": ["natr"],
                "description": "Normalized Average True Range - ATR as percentage"
            },
            "TRANGE": {
                "name": "True Range",
                "params": {},
                "inputs": ["high", "low", "close"],
                "outputs": ["trange"],
                "description": "True Range - single period volatility"
            }
        }
    },
    "price_transform": {
        "description": "Price transform functions - price data transformations",
        "indicators": {
            "AVGPRICE": {
                "name": "Average Price",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["avgprice"],
                "description": "Average Price - (O+H+L+C)/4"
            },
            "MEDPRICE": {
                "name": "Median Price",
                "params": {},
                "inputs": ["high", "low"],
                "outputs": ["medprice"],
                "description": "Median Price - (H+L)/2"
            },
            "TYPPRICE": {
                "name": "Typical Price",
                "params": {},
                "inputs": ["high", "low", "close"],
                "outputs": ["typprice"],
                "description": "Typical Price - (H+L+C)/3"
            },
            "WCLPRICE": {
                "name": "Weighted Close Price",
                "params": {},
                "inputs": ["high", "low", "close"],
                "outputs": ["wclprice"],
                "description": "Weighted Close Price - (H+L+2*C)/4"
            }
        }
    },
    "cycle_indicators": {
        "description": "Cycle indicators - analyze market cycles",
        "indicators": {
            "HT_DCPERIOD": {
                "name": "Hilbert Transform - Dominant Cycle Period",
                "params": {},
                "inputs": ["close"],
                "outputs": ["ht_dcperiod"],
                "description": "Hilbert Transform Dominant Cycle Period"
            },
            "HT_DCPHASE": {
                "name": "Hilbert Transform - Dominant Cycle Phase",
                "params": {},
                "inputs": ["close"],
                "outputs": ["ht_dcphase"],
                "description": "Hilbert Transform Dominant Cycle Phase"
            },
            "HT_PHASOR": {
                "name": "Hilbert Transform - Phasor Components",
                "params": {},
                "inputs": ["close"],
                "outputs": ["inphase", "quadrature"],
                "description": "Hilbert Transform Phasor Components"
            },
            "HT_SINE": {
                "name": "Hilbert Transform - SineWave",
                "params": {},
                "inputs": ["close"],
                "outputs": ["sine", "leadsine"],
                "description": "Hilbert Transform SineWave - cycle turning points"
            },
            "HT_TRENDMODE": {
                "name": "Hilbert Transform - Trend vs Cycle Mode",
                "params": {},
                "inputs": ["close"],
                "outputs": ["ht_trendmode"],
                "description": "Hilbert Transform Trend vs Cycle Mode"
            }
        }
    },
    "pattern_recognition": {
        "description": "Pattern recognition - candlestick patterns",
        "indicators": {
            "CDL2CROWS": {
                "name": "Two Crows",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdl2crows"],
                "description": "Two Crows candlestick pattern"
            },
            "CDL3BLACKCROWS": {
                "name": "Three Black Crows",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdl3blackcrows"],
                "description": "Three Black Crows candlestick pattern"
            },
            "CDL3INSIDE": {
                "name": "Three Inside Up/Down",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdl3inside"],
                "description": "Three Inside Up/Down candlestick pattern"
            },
            "CDL3LINESTRIKE": {
                "name": "Three-Line Strike",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdl3linestrike"],
                "description": "Three-Line Strike candlestick pattern"
            },
            "CDL3OUTSIDE": {
                "name": "Three Outside Up/Down",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdl3outside"],
                "description": "Three Outside Up/Down candlestick pattern"
            },
            "CDL3STARSINSOUTH": {
                "name": "Three Stars In The South",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdl3starsinsouth"],
                "description": "Three Stars In The South candlestick pattern"
            },
            "CDL3WHITESOLDIERS": {
                "name": "Three Advancing White Soldiers",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdl3whitesoldiers"],
                "description": "Three Advancing White Soldiers candlestick pattern"
            },
            "CDLABANDONEDBABY": {
                "name": "Abandoned Baby",
                "params": {"penetration": 0.3},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlabandonedbaby"],
                "description": "Abandoned Baby candlestick pattern"
            },
            "CDLADVANCEBLOCK": {
                "name": "Advance Block",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdladvanceblock"],
                "description": "Advance Block candlestick pattern"
            },
            "CDLBELTHOLD": {
                "name": "Belt-hold",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlbelthold"],
                "description": "Belt-hold candlestick pattern"
            },
            "CDLBREAKAWAY": {
                "name": "Breakaway",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlbreakaway"],
                "description": "Breakaway candlestick pattern"
            },
            "CDLCLOSINGMARUBOZU": {
                "name": "Closing Marubozu",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlclosingmarubozu"],
                "description": "Closing Marubozu candlestick pattern"
            },
            "CDLCONCEALBABYSWALL": {
                "name": "Concealing Baby Swallow",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlconcealbabyswall"],
                "description": "Concealing Baby Swallow candlestick pattern"
            },
            "CDLCOUNTERATTACK": {
                "name": "Counterattack",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlcounterattack"],
                "description": "Counterattack candlestick pattern"
            },
            "CDLDARKCLOUDCOVER": {
                "name": "Dark Cloud Cover",
                "params": {"penetration": 0.5},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdldarkcloudcover"],
                "description": "Dark Cloud Cover candlestick pattern"
            },
            "CDLDOJI": {
                "name": "Doji",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdldoji"],
                "description": "Doji candlestick pattern"
            },
            "CDLDOJISTAR": {
                "name": "Doji Star",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdldojistar"],
                "description": "Doji Star candlestick pattern"
            },
            "CDLDRAGONFLYDOJI": {
                "name": "Dragonfly Doji",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdldragonflydoji"],
                "description": "Dragonfly Doji candlestick pattern"
            },
            "CDLENGULFING": {
                "name": "Engulfing Pattern",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlengulfing"],
                "description": "Engulfing Pattern candlestick pattern"
            },
            "CDLEVENINGDOJISTAR": {
                "name": "Evening Doji Star",
                "params": {"penetration": 0.3},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdleveningdojistar"],
                "description": "Evening Doji Star candlestick pattern"
            },
            "CDLEVENINGSTAR": {
                "name": "Evening Star",
                "params": {"penetration": 0.3},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdleveningstar"],
                "description": "Evening Star candlestick pattern"
            },
            "CDLGAPSIDESIDEWHITE": {
                "name": "Up/Down-gap side-by-side white lines",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlgapsidesidewhite"],
                "description": "Up/Down-gap side-by-side white lines"
            },
            "CDLGRAVESTONEDOJI": {
                "name": "Gravestone Doji",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlgravestonedoji"],
                "description": "Gravestone Doji candlestick pattern"
            },
            "CDLHAMMER": {
                "name": "Hammer",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlhammer"],
                "description": "Hammer candlestick pattern"
            },
            "CDLHANGINGMAN": {
                "name": "Hanging Man",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlhangingman"],
                "description": "Hanging Man candlestick pattern"
            },
            "CDLHARAMI": {
                "name": "Harami Pattern",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlharami"],
                "description": "Harami Pattern candlestick pattern"
            },
            "CDLHARAMICROSS": {
                "name": "Harami Cross Pattern",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlharamicross"],
                "description": "Harami Cross Pattern candlestick pattern"
            },
            "CDLHIGHWAVE": {
                "name": "High-Wave Candle",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlhighwave"],
                "description": "High-Wave Candle candlestick pattern"
            },
            "CDLHIKKAKE": {
                "name": "Hikkake Pattern",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlhikkake"],
                "description": "Hikkake Pattern candlestick pattern"
            },
            "CDLHIKKAKEMOD": {
                "name": "Modified Hikkake Pattern",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlhikkakemod"],
                "description": "Modified Hikkake Pattern candlestick pattern"
            },
            "CDLHOMINGPIGEON": {
                "name": "Homing Pigeon",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlhomingpigeon"],
                "description": "Homing Pigeon candlestick pattern"
            },
            "CDLIDENTICAL3CROWS": {
                "name": "Identical Three Crows",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlidentical3crows"],
                "description": "Identical Three Crows candlestick pattern"
            },
            "CDLINNECK": {
                "name": "In-Neck Pattern",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlinneck"],
                "description": "In-Neck Pattern candlestick pattern"
            },
            "CDLINVERTEDHAMMER": {
                "name": "Inverted Hammer",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlinvertedhammer"],
                "description": "Inverted Hammer candlestick pattern"
            },
            "CDLKICKING": {
                "name": "Kicking",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlkicking"],
                "description": "Kicking candlestick pattern"
            },
            "CDLKICKINGBYLENGTH": {
                "name": "Kicking - bull/bear determined by the longer marubozu",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlkickingbylength"],
                "description": "Kicking by length candlestick pattern"
            },
            "CDLLADDERBOTTOM": {
                "name": "Ladder Bottom",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlladderbottom"],
                "description": "Ladder Bottom candlestick pattern"
            },
            "CDLLONGLEGGEDDOJI": {
                "name": "Long Legged Doji",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdllongleggeddoji"],
                "description": "Long Legged Doji candlestick pattern"
            },
            "CDLLONGLINE": {
                "name": "Long Line Candle",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdllongline"],
                "description": "Long Line Candle candlestick pattern"
            },
            "CDLMARUBOZU": {
                "name": "Marubozu",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlmarubozu"],
                "description": "Marubozu candlestick pattern"
            },
            "CDLMATCHINGLOW": {
                "name": "Matching Low",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlmatchinglow"],
                "description": "Matching Low candlestick pattern"
            },
            "CDLMATHOLD": {
                "name": "Mat Hold",
                "params": {"penetration": 0.5},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlmathold"],
                "description": "Mat Hold candlestick pattern"
            },
            "CDLMORNINGDOJISTAR": {
                "name": "Morning Doji Star",
                "params": {"penetration": 0.3},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlmorningdojistar"],
                "description": "Morning Doji Star candlestick pattern"
            },
            "CDLMORNINGSTAR": {
                "name": "Morning Star",
                "params": {"penetration": 0.3},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlmorningstar"],
                "description": "Morning Star candlestick pattern"
            },
            "CDLONNECK": {
                "name": "On-Neck Pattern",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlonneck"],
                "description": "On-Neck Pattern candlestick pattern"
            },
            "CDLPIERCING": {
                "name": "Piercing Pattern",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlpiercing"],
                "description": "Piercing Pattern candlestick pattern"
            },
            "CDLRICKSHAWMAN": {
                "name": "Rickshaw Man",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlrickshawman"],
                "description": "Rickshaw Man candlestick pattern"
            },
            "CDLRISEFALL3METHODS": {
                "name": "Rising/Falling Three Methods",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlrisefall3methods"],
                "description": "Rising/Falling Three Methods candlestick pattern"
            },
            "CDLSEPARATINGLINES": {
                "name": "Separating Lines",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlseparatinglines"],
                "description": "Separating Lines candlestick pattern"
            },
            "CDLSHOOTINGSTAR": {
                "name": "Shooting Star",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlshootingstar"],
                "description": "Shooting Star candlestick pattern"
            },
            "CDLSHORTLINE": {
                "name": "Short Line Candle",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlshortline"],
                "description": "Short Line Candle candlestick pattern"
            },
            "CDLSPINNINGTOP": {
                "name": "Spinning Top",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlspinningtop"],
                "description": "Spinning Top candlestick pattern"
            },
            "CDLSTALLEDPATTERN": {
                "name": "Stalled Pattern",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlstalledpattern"],
                "description": "Stalled Pattern candlestick pattern"
            },
            "CDLSTICKSANDWICH": {
                "name": "Stick Sandwich",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlsticksandwich"],
                "description": "Stick Sandwich candlestick pattern"
            },
            "CDLTAKURI": {
                "name": "Takuri (Dragonfly Doji with very long lower shadow)",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdltakuri"],
                "description": "Takuri candlestick pattern"
            },
            "CDLTASUKIGAP": {
                "name": "Tasuki Gap",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdltasukigap"],
                "description": "Tasuki Gap candlestick pattern"
            },
            "CDLTHRUSTING": {
                "name": "Thrusting Pattern",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlthrusting"],
                "description": "Thrusting Pattern candlestick pattern"
            },
            "CDLTRISTAR": {
                "name": "Tristar Pattern",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdltristar"],
                "description": "Tristar Pattern candlestick pattern"
            },
            "CDLUNIQUE3RIVER": {
                "name": "Unique 3 River",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlunique3river"],
                "description": "Unique 3 River candlestick pattern"
            },
            "CDLUPSIDEGAP2CROWS": {
                "name": "Upside Gap Two Crows",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlupsidegap2crows"],
                "description": "Upside Gap Two Crows candlestick pattern"
            },
            "CDLXSIDEGAP3METHODS": {
                "name": "Upside/Downside Gap Three Methods",
                "params": {},
                "inputs": ["open", "high", "low", "close"],
                "outputs": ["cdlxsidegap3methods"],
                "description": "Upside/Downside Gap Three Methods candlestick pattern"
            }
        }
    },
    "statistic_functions": {
        "description": "Statistical functions - mathematical transformations",
        "indicators": {
            "BETA": {
                "name": "Beta",
                "params": {"timeperiod": 5},
                "inputs": ["high", "low"],
                "outputs": ["beta"],
                "description": "Beta - correlation coefficient"
            },
            "CORREL": {
                "name": "Pearson's Correlation Coefficient (r)",
                "params": {"timeperiod": 30},
                "inputs": ["high", "low"],
                "outputs": ["correl"],
                "description": "Pearson's Correlation Coefficient"
            },
            "LINEARREG": {
                "name": "Linear Regression",
                "params": {"timeperiod": 14},
                "inputs": ["close"],
                "outputs": ["linearreg"],
                "description": "Linear Regression - trend line"
            },
            "LINEARREG_ANGLE": {
                "name": "Linear Regression Angle",
                "params": {"timeperiod": 14},
                "inputs": ["close"],
                "outputs": ["linearreg_angle"],
                "description": "Linear Regression Angle - trend slope"
            },
            "LINEARREG_INTERCEPT": {
                "name": "Linear Regression Intercept",
                "params": {"timeperiod": 14},
                "inputs": ["close"],
                "outputs": ["linearreg_intercept"],
                "description": "Linear Regression Intercept"
            },
            "LINEARREG_SLOPE": {
                "name": "Linear Regression Slope",
                "params": {"timeperiod": 14},
                "inputs": ["close"],
                "outputs": ["linearreg_slope"],
                "description": "Linear Regression Slope - rate of change"
            },
            "STDDEV": {
                "name": "Standard Deviation",
                "params": {"timeperiod": 5, "nbdev": 1.0},
                "inputs": ["close"],
                "outputs": ["stddev"],
                "description": "Standard Deviation - volatility measure"
            },
            "TSF": {
                "name": "Time Series Forecast",
                "params": {"timeperiod": 14},
                "inputs": ["close"],
                "outputs": ["tsf"],
                "description": "Time Series Forecast - predicted value"
            },
            "VAR": {
                "name": "Variance",
                "params": {"timeperiod": 5, "nbdev": 1.0},
                "inputs": ["close"],
                "outputs": ["var"],
                "description": "Variance - volatility measure"
            }
        }
    },
    "math_transform": {
        "description": "Math transform functions - mathematical operations on price data",
        "indicators": {
            "ACOS": {
                "name": "Vector Trigonometric ACos",
                "params": {},
                "inputs": ["close"],
                "outputs": ["acos"],
                "description": "Arc Cosine transformation"
            },
            "ASIN": {
                "name": "Vector Trigonometric ASin",
                "params": {},
                "inputs": ["close"],
                "outputs": ["asin"],
                "description": "Arc Sine transformation"
            },
            "ATAN": {
                "name": "Vector Trigonometric ATan",
                "params": {},
                "inputs": ["close"],
                "outputs": ["atan"],
                "description": "Arc Tangent transformation"
            },
            "CEIL": {
                "name": "Vector Ceil",
                "params": {},
                "inputs": ["close"],
                "outputs": ["ceil"],
                "description": "Ceiling function - smallest integer >= value"
            },
            "COS": {
                "name": "Vector Trigonometric Cos",
                "params": {},
                "inputs": ["close"],
                "outputs": ["cos"],
                "description": "Cosine transformation"
            },
            "COSH": {
                "name": "Vector Trigonometric Cosh",
                "params": {},
                "inputs": ["close"],
                "outputs": ["cosh"],
                "description": "Hyperbolic Cosine transformation"
            },
            "EXP": {
                "name": "Vector Arithmetic Exp",
                "params": {},
                "inputs": ["close"],
                "outputs": ["exp"],
                "description": "Exponential function"
            },
            "FLOOR": {
                "name": "Vector Floor",
                "params": {},
                "inputs": ["close"],
                "outputs": ["floor"],
                "description": "Floor function - largest integer <= value"
            },
            "LN": {
                "name": "Vector Log Natural",
                "params": {},
                "inputs": ["close"],
                "outputs": ["ln"],
                "description": "Natural logarithm"
            },
            "LOG10": {
                "name": "Vector Log10",
                "params": {},
                "inputs": ["close"],
                "outputs": ["log10"],
                "description": "Base 10 logarithm"
            },
            "SIN": {
                "name": "Vector Trigonometric Sin",
                "params": {},
                "inputs": ["close"],
                "outputs": ["sin"],
                "description": "Sine transformation"
            },
            "SINH": {
                "name": "Vector Trigonometric Sinh",
                "params": {},
                "inputs": ["close"],
                "outputs": ["sinh"],
                "description": "Hyperbolic Sine transformation"
            },
            "SQRT": {
                "name": "Vector Square Root",
                "params": {},
                "inputs": ["close"],
                "outputs": ["sqrt"],
                "description": "Square root transformation"
            },
            "TAN": {
                "name": "Vector Trigonometric Tan",
                "params": {},
                "inputs": ["close"],
                "outputs": ["tan"],
                "description": "Tangent transformation"
            },
            "TANH": {
                "name": "Vector Trigonometric Tanh",
                "params": {},
                "inputs": ["close"],
                "outputs": ["tanh"],
                "description": "Hyperbolic Tangent transformation"
            }
        }
    },
    "math_operators": {
        "description": "Math operator functions - arithmetic operations",
        "indicators": {
            "ADD": {
                "name": "Vector Arithmetic Add",
                "params": {},
                "inputs": ["high", "low"],
                "outputs": ["add"],
                "description": "Addition of two series"
            },
            "DIV": {
                "name": "Vector Arithmetic Div",
                "params": {},
                "inputs": ["high", "low"],
                "outputs": ["div"],
                "description": "Division of two series"
            },
            "MAX": {
                "name": "Highest value over a specified period",
                "params": {"timeperiod": 30},
                "inputs": ["close"],
                "outputs": ["max"],
                "description": "Maximum value over period"
            },
            "MAXINDEX": {
                "name": "Index of highest value over a specified period",
                "params": {"timeperiod": 30},
                "inputs": ["close"],
                "outputs": ["maxindex"],
                "description": "Index of maximum value over period"
            },
            "MIN": {
                "name": "Lowest value over a specified period",
                "params": {"timeperiod": 30},
                "inputs": ["close"],
                "outputs": ["min"],
                "description": "Minimum value over period"
            },
            "MININDEX": {
                "name": "Index of lowest value over a specified period",
                "params": {"timeperiod": 30},
                "inputs": ["close"],
                "outputs": ["minindex"],
                "description": "Index of minimum value over period"
            },
            "MINMAX": {
                "name": "Lowest and highest values over a specified period",
                "params": {"timeperiod": 30},
                "inputs": ["close"],
                "outputs": ["min", "max"],
                "description": "Minimum and maximum values over period"
            },
            "MINMAXINDEX": {
                "name": "Indexes of lowest and highest values over a specified period",
                "params": {"timeperiod": 30},
                "inputs": ["close"],
                "outputs": ["minidx", "maxidx"],
                "description": "Indexes of minimum and maximum values"
            },
            "MULT": {
                "name": "Vector Arithmetic Mult",
                "params": {},
                "inputs": ["high", "low"],
                "outputs": ["mult"],
                "description": "Multiplication of two series"
            },
            "SUB": {
                "name": "Vector Arithmetic Subtraction",
                "params": {},
                "inputs": ["high", "low"],
                "outputs": ["sub"],
                "description": "Subtraction of two series"
            },
            "SUM": {
                "name": "Summation",
                "params": {"timeperiod": 30},
                "inputs": ["close"],
                "outputs": ["sum"],
                "description": "Sum over period"
            }
        }
    }
}


def get_indicators_by_category(category: str) -> Dict[str, Any]:
    """Get all indicators for a specific category"""
    return TALIB_INDICATORS_REGISTRY.get(category, {})


def get_all_categories() -> List[str]:
    """Get all available indicator categories"""
    return list(TALIB_INDICATORS_REGISTRY.keys())


def get_indicator_info(indicator_name: str) -> Dict[str, Any]:
    """Get information about a specific indicator"""
    for category_data in TALIB_INDICATORS_REGISTRY.values():
        if indicator_name in category_data.get("indicators", {}):
            return category_data["indicators"][indicator_name]
    return {}


def get_indicators_requiring_volume() -> List[str]:
    """Get list of indicators that require volume data"""
    volume_indicators = []
    for category_data in TALIB_INDICATORS_REGISTRY.values():
        for indicator_name, indicator_info in category_data.get("indicators", {}).items():
            if "volume" in indicator_info.get("inputs", []):
                volume_indicators.append(indicator_name)
    return volume_indicators


def get_indicators_by_input_requirements(required_inputs: List[str]) -> List[str]:
    """Get indicators that can be calculated with the given input data"""
    compatible_indicators = []
    for category_data in TALIB_INDICATORS_REGISTRY.values():
        for indicator_name, indicator_info in category_data.get("indicators", {}).items():
            indicator_inputs = set(indicator_info.get("inputs", []))
            if indicator_inputs.issubset(set(required_inputs)):
                compatible_indicators.append(indicator_name)
    return compatible_indicators


def get_strategy_relevant_indicators() -> Dict[str, List[str]]:
    """Get indicators organized by their strategic relevance"""
    return {
        "trend_following": [
            "SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "T3",
            "MACD", "MACDEXT", "MACDFIX", "PPO", "APO", "LINEARREG",
            "LINEARREG_SLOPE", "LINEARREG_ANGLE", "TSF", "HT_TRENDLINE"
        ],
        "mean_reversion": [
            "RSI", "STOCH", "STOCHF", "STOCHRSI", "WILLR", "CCI", "MFI",
            "ULTOSC", "CMO", "BBANDS", "NATR"
        ],
        "momentum": [
            "MOM", "ROC", "ROCP", "ROCR", "ROCR100", "TRIX", "ADX", "ADXR",
            "DX", "PLUS_DI", "MINUS_DI", "AROON", "AROONOSC", "BOP"
        ],
        "volatility": [
            "ATR", "NATR", "TRANGE", "BBANDS", "STDDEV", "VAR"
        ],
        "volume": [
            "OBV", "AD", "ADOSC", "MFI"
        ],
        "support_resistance": [
            "SAR", "SAREXT", "MIDPOINT", "MIDPRICE", "AVGPRICE", "MEDPRICE",
            "TYPPRICE", "WCLPRICE"
        ],
        "cycle_analysis": [
            "HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR", "HT_SINE", "HT_TRENDMODE"
        ],
        "pattern_recognition": [
            name for name in TALIB_INDICATORS_REGISTRY["pattern_recognition"]["indicators"].keys()
        ]
    }


def generate_indicator_description_for_llm() -> str:
    """Generate a comprehensive description of available indicators for LLM context"""
    description = "# TA-Lib Technical Indicators Available for Strategy Generation\n\n"
    
    for category, category_data in TALIB_INDICATORS_REGISTRY.items():
        description += f"## {category.replace('_', ' ').title()}\n"
        description += f"{category_data['description']}\n\n"
        
        for indicator_name, indicator_info in category_data["indicators"].items():
            description += f"### {indicator_name}\n"
            description += f"**Name**: {indicator_info['name']}\n"
            description += f"**Description**: {indicator_info['description']}\n"
            description += f"**Inputs**: {', '.join(indicator_info['inputs'])}\n"
            description += f"**Outputs**: {', '.join(indicator_info['outputs'])}\n"
            if indicator_info['params']:
                params_str = ', '.join([f"{k}={v}" for k, v in indicator_info['params'].items()])
                description += f"**Default Parameters**: {params_str}\n"
            description += "\n"
        
        description += "\n"
    
    return description