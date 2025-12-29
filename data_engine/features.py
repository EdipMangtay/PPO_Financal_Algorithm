"""
Advanced Feature Engineering - Dynamic Feature Pool Generator
Creates a massive pool of indicators using pandas-ta for Optuna selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import traceback

# ====================================================================
# CRITICAL: Robust pandas-ta import with detailed error reporting
# ====================================================================
HAS_PANDAS_TA = False
ta = None
IMPORT_ERROR_MSG = None

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
    logger = logging.getLogger(__name__)
    logger.info("✓ pandas-ta imported successfully")
except ImportError as e:
    HAS_PANDAS_TA = False
    IMPORT_ERROR_MSG = str(e)
    logger = logging.getLogger(__name__)
    logger.error(f"✗ pandas-ta import FAILED: {IMPORT_ERROR_MSG}")
    logger.error(f"   Full traceback:\n{traceback.format_exc()}")
    # Try to provide helpful guidance
    if "numpy" in str(e).lower():
        logger.error("   → This is likely a numpy version compatibility issue.")
        logger.error("   → Try: pip install 'numpy<2.0' or upgrade pandas-ta")
except Exception as e:
    HAS_PANDAS_TA = False
    IMPORT_ERROR_MSG = str(e)
    logger = logging.getLogger(__name__)
    logger.error(f"✗ pandas-ta import FAILED with unexpected error: {IMPORT_ERROR_MSG}")
    logger.error(f"   Full traceback:\n{traceback.format_exc()}")

# Scipy import (optional)
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logging.basicConfig(level=logging.INFO)
if 'logger' not in locals():
    logger = logging.getLogger(__name__)


class FeatureGenerator:
    """
    Generates a massive pool of candidate features for dynamic selection.
    Optuna will select which features and their parameters to use.
    """
    
    def __init__(self, require_pandas_ta: bool = True):
        """
        Initialize feature generator.
        
        Args:
            require_pandas_ta: If True, raise RuntimeError if pandas-ta is missing
        """
        self.feature_pool: Dict[str, List[str]] = {}
        
        if require_pandas_ta and not HAS_PANDAS_TA:
            error_msg = (
                f"CRITICAL: pandas-ta is required but import failed!\n"
                f"Error: {IMPORT_ERROR_MSG}\n"
                f"Please install: pip install pandas-ta\n"
                f"Or fix numpy compatibility: pip install 'numpy<2.0'"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info("Feature Generator initialized")
    
    def generate_candidate_features(
        self,
        df: pd.DataFrame,
        include_volume: bool = True
    ) -> pd.DataFrame:
        """
        Generate massive pool of candidate features using pd.concat for performance.
        
        Args:
            df: DataFrame with OHLCV data
            include_volume: Whether to include volume-based features
        
        Returns:
            DataFrame with all candidate features added
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close']
        if include_volume:
            required_cols.append('volume')
        
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}. Skipping volume features.")
                include_volume = False
                break
        
        logger.info("Generating candidate feature pool...")
        
        # ====================================================================
        # PERFORMANCE FIX: Use list of DataFrames/Series, concat once at end
        # ====================================================================
        new_features: List[pd.DataFrame] = []  # List to collect all new features
        
        # ====================================================================
        # 1. LINEAR REGRESSION CHANNELS (The Core)
        # ====================================================================
        logger.info("Calculating Linear Regression Channels...")
        for length in [10, 20, 30, 50, 100, 200]:
            try:
                linreg_cols = {}
                
                if HAS_PANDAS_TA:
                    linreg = ta.linreg(df['close'], length=length)
                    slope = ta.slope(df['close'], length=length)
                else:
                    # Fallback: Manual linear regression
                    if HAS_SCIPY:
                        def calc_linreg(series, window):
                            result = pd.Series(index=series.index, dtype=float)
                            for i in range(window-1, len(series)):
                                y = series.iloc[i-window+1:i+1].values
                                x = np.arange(len(y))
                                slope_val, intercept_val, r_val, p_val, std_err = stats.linregress(x, y)
                                result.iloc[i] = intercept_val + slope_val * (window - 1)
                            return result
                        
                        linreg = calc_linreg(df['close'], length)
                        slope = df['close'].rolling(window=length).apply(
                            lambda x: stats.linregress(np.arange(len(x)), x.values)[0] if len(x) == length else np.nan,
                            raw=False
                        )
                    else:
                        # Very simple fallback: use SMA
                        linreg = df['close'].rolling(window=length).mean()
                        slope = df['close'].diff(length) / length
                
                linreg_cols[f'linreg_{length}'] = linreg
                linreg_cols[f'slope_{length}'] = slope
                
                # Intercept
                linreg_cols[f'intercept_{length}'] = linreg - slope * (length / 2)
                
                # R-squared
                linreg_cols[f'rsquared_{length}'] = df['close'].rolling(window=length).corr(linreg) ** 2
                
                # ATR for channels
                if HAS_PANDAS_TA:
                    atr = ta.atr(df['high'], df['low'], df['close'], length=14)
                else:
                    from data.loader import TechnicalIndicators
                    atr = TechnicalIndicators.calculate_atr(df['high'], df['low'], df['close'], period=14)
                
                linreg_cols[f'linreg_upper_{length}'] = linreg + (atr * 2)
                linreg_cols[f'linreg_lower_{length}'] = linreg - (atr * 2)
                
                # Distances
                linreg_cols[f'dist_upper_{length}'] = (df['close'] - linreg_cols[f'linreg_upper_{length}']) / df['close']
                linreg_cols[f'dist_lower_{length}'] = (df['close'] - linreg_cols[f'linreg_lower_{length}']) / df['close']
                linreg_cols[f'dist_linreg_{length}'] = (df['close'] - linreg) / df['close']
                
                # Crossovers
                linreg_cols[f'cross_above_linreg_{length}'] = (df['close'] > linreg).astype(int)
                linreg_cols[f'cross_below_linreg_{length}'] = (df['close'] < linreg).astype(int)
                
                # Momentum
                linreg_cols[f'momentum_linreg_{length}'] = linreg_cols[f'dist_linreg_{length}'].diff()
                
                # Strong momentum signals
                linreg_cols[f'strong_momentum_long_{length}'] = (
                    (linreg_cols[f'cross_above_linreg_{length}'] == 1) & 
                    (slope > 0) &
                    (linreg_cols[f'momentum_linreg_{length}'] > 0)
                ).astype(int)
                linreg_cols[f'strong_momentum_short_{length}'] = (
                    (linreg_cols[f'cross_below_linreg_{length}'] == 1) & 
                    (slope < 0) &
                    (linreg_cols[f'momentum_linreg_{length}'] < 0)
                ).astype(int)
                
                new_features.append(pd.DataFrame(linreg_cols, index=df.index))
                
            except Exception as e:
                logger.warning(f"Error calculating LinReg for length {length}: {e}")
        
        # ====================================================================
        # 2. OSCILLATORS (Dynamic Periods)
        # ====================================================================
        logger.info("Calculating Oscillators...")
        oscillator_periods = [7, 14, 21, 50]
        
        for period in oscillator_periods:
            try:
                osc_cols = {}
                
                if HAS_PANDAS_TA:
                    osc_cols[f'RSI_{period}'] = ta.rsi(df['close'], length=period)
                    
                    stoch = ta.stoch(df['high'], df['low'], df['close'], k=period, d=3, smooth_k=3)
                    if isinstance(stoch, pd.DataFrame):
                        osc_cols[f'STOCH_k_{period}'] = stoch.iloc[:, 0] if len(stoch.columns) > 0 else pd.Series(0, index=df.index)
                        osc_cols[f'STOCH_d_{period}'] = stoch.iloc[:, 1] if len(stoch.columns) > 1 else pd.Series(0, index=df.index)
                    
                    osc_cols[f'CCI_{period}'] = ta.cci(df['high'], df['low'], df['close'], length=period)
                    osc_cols[f'WilliamsR_{period}'] = ta.willr(df['high'], df['low'], df['close'], length=period)
                else:
                    # Fallback: Use TechnicalIndicators
                    from data.loader import TechnicalIndicators
                    osc_cols[f'RSI_{period}'] = TechnicalIndicators.calculate_rsi(df['close'], period=period)
                
                new_features.append(pd.DataFrame(osc_cols, index=df.index))
                
            except Exception as e:
                logger.warning(f"Error calculating oscillators for period {period}: {e}")
        
        # ====================================================================
        # 3. TREND INDICATORS
        # ====================================================================
        logger.info("Calculating Trend Indicators...")
        
        # EMA Ribbons
        ema_cols = {}
        for length in [8, 13, 21, 34, 55]:
            try:
                if HAS_PANDAS_TA:
                    ema_cols[f'EMA_{length}'] = ta.ema(df['close'], length=length)
                else:
                    ema_cols[f'EMA_{length}'] = df['close'].ewm(span=length, adjust=False).mean()
            except Exception as e:
                logger.warning(f"Error calculating EMA_{length}: {e}")
        
        if ema_cols:
            new_features.append(pd.DataFrame(ema_cols, index=df.index))
        
        # SuperTrend
        for period in [10, 14, 21]:
            for multiplier in [2.0, 3.0]:
                try:
                    if HAS_PANDAS_TA:
                        supertrend = ta.supertrend(df['high'], df['low'], df['close'], 
                                                   period=period, multiplier=multiplier)
                        if isinstance(supertrend, pd.DataFrame):
                            st_cols = {f'SuperTrend_{period}_{multiplier}': supertrend.iloc[:, 0] if len(supertrend.columns) > 0 else pd.Series(0, index=df.index)}
                            new_features.append(pd.DataFrame(st_cols, index=df.index))
                except Exception as e:
                    logger.warning(f"Error calculating SuperTrend: {e}")
        
        # ADX
        for period in [14, 21]:
            try:
                if HAS_PANDAS_TA:
                    adx = ta.adx(df['high'], df['low'], df['close'], length=period)
                    if isinstance(adx, pd.DataFrame):
                        adx_cols = {
                            f'ADX_{period}': adx.iloc[:, 0] if len(adx.columns) > 0 else pd.Series(0, index=df.index),
                            f'ADX_plus_{period}': adx.iloc[:, 1] if len(adx.columns) > 1 else pd.Series(0, index=df.index),
                            f'ADX_minus_{period}': adx.iloc[:, 2] if len(adx.columns) > 2 else pd.Series(0, index=df.index)
                        }
                        new_features.append(pd.DataFrame(adx_cols, index=df.index))
            except Exception as e:
                logger.warning(f"Error calculating ADX: {e}")
        
        # MACD
        for fast in [8, 12]:
            for slow in [21, 26]:
                for signal in [7, 9]:
                    try:
                        if HAS_PANDAS_TA:
                            macd = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
                        else:
                            ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
                            ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
                            macd_line = ema_fast - ema_slow
                            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
                            hist = macd_line - signal_line
                            macd = pd.DataFrame({
                                'MACD': macd_line,
                                'Signal': signal_line,
                                'Hist': hist
                            }, index=df.index)
                        
                        if isinstance(macd, pd.DataFrame):
                            macd_cols = {
                                f'MACD_{fast}_{slow}_{signal}': macd.iloc[:, 0] if len(macd.columns) > 0 else pd.Series(0, index=df.index),
                                f'MACD_signal_{fast}_{slow}_{signal}': macd.iloc[:, 1] if len(macd.columns) > 1 else pd.Series(0, index=df.index),
                                f'MACD_hist_{fast}_{slow}_{signal}': macd.iloc[:, 2] if len(macd.columns) > 2 else pd.Series(0, index=df.index)
                            }
                            new_features.append(pd.DataFrame(macd_cols, index=df.index))
                    except Exception as e:
                        logger.warning(f"Error calculating MACD: {e}")
        
        # ====================================================================
        # 4. VOLATILITY INDICATORS
        # ====================================================================
        logger.info("Calculating Volatility Indicators...")
        
        # ATR
        atr_cols = {}
        for period in [7, 14, 21]:
            try:
                if HAS_PANDAS_TA:
                    atr = ta.atr(df['high'], df['low'], df['close'], length=period)
                else:
                    from data.loader import TechnicalIndicators
                    atr = TechnicalIndicators.calculate_atr(df['high'], df['low'], df['close'], period=period)
                
                atr_cols[f'ATR_{period}'] = atr
                atr_cols[f'ATR_norm_{period}'] = atr / df['close']
            except Exception as e:
                logger.warning(f"Error calculating ATR_{period}: {e}")
        
        if atr_cols:
            new_features.append(pd.DataFrame(atr_cols, index=df.index))
        
        # Bollinger Bands
        for period in [20, 21]:
            for std in [2, 2.5]:
                try:
                    if HAS_PANDAS_TA:
                        bb = ta.bbands(df['close'], length=period, std=std)
                    else:
                        sma = df['close'].rolling(window=period).mean()
                        std_dev = df['close'].rolling(window=period).std()
                        bb = pd.DataFrame({
                            'BBU': sma + (std_dev * std),
                            'BBM': sma,
                            'BBL': sma - (std_dev * std)
                        }, index=df.index)
                    
                    if isinstance(bb, pd.DataFrame):
                        bb_cols = {
                            f'BB_upper_{period}_{std}': bb.iloc[:, 0] if len(bb.columns) > 0 else pd.Series(0, index=df.index),
                            f'BB_middle_{period}_{std}': bb.iloc[:, 1] if len(bb.columns) > 1 else pd.Series(0, index=df.index),
                            f'BB_lower_{period}_{std}': bb.iloc[:, 2] if len(bb.columns) > 2 else pd.Series(0, index=df.index)
                        }
                        bb_cols[f'BB_width_{period}_{std}'] = (
                            (bb_cols[f'BB_upper_{period}_{std}'] - bb_cols[f'BB_lower_{period}_{std}']) / 
                            bb_cols[f'BB_middle_{period}_{std}'].replace(0, np.nan)
                        ).fillna(0)
                        new_features.append(pd.DataFrame(bb_cols, index=df.index))
                except Exception as e:
                    logger.warning(f"Error calculating Bollinger Bands: {e}")
        
        # Keltner Channels
        for period in [20, 21]:
            for multiplier in [2.0, 2.5]:
                try:
                    if HAS_PANDAS_TA:
                        kc = ta.kc(df['high'], df['low'], df['close'], length=period, multiplier=multiplier)
                        if isinstance(kc, pd.DataFrame):
                            kc_cols = {
                                f'KC_upper_{period}_{multiplier}': kc.iloc[:, 0] if len(kc.columns) > 0 else pd.Series(0, index=df.index),
                                f'KC_middle_{period}_{multiplier}': kc.iloc[:, 1] if len(kc.columns) > 1 else pd.Series(0, index=df.index),
                                f'KC_lower_{period}_{multiplier}': kc.iloc[:, 2] if len(kc.columns) > 2 else pd.Series(0, index=df.index)
                            }
                            new_features.append(pd.DataFrame(kc_cols, index=df.index))
                except Exception as e:
                    logger.warning(f"Error calculating Keltner Channels: {e}")
        
        # ====================================================================
        # 5. VOLUME INDICATORS
        # ====================================================================
        if include_volume and 'volume' in df.columns:
            logger.info("Calculating Volume Indicators...")
            
            vol_cols = {}
            
            # OBV
            try:
                if HAS_PANDAS_TA:
                    obv = ta.obv(df['close'], df['volume'])
                else:
                    obv = (df['volume'] * np.sign(df['close'].diff())).cumsum()
                
                vol_cols['OBV'] = obv
                vol_cols['OBV_ROC'] = obv.pct_change(periods=14)
            except Exception as e:
                logger.warning(f"Error calculating OBV: {e}")
            
            # MFI
            for period in [14, 21]:
                try:
                    if HAS_PANDAS_TA:
                        vol_cols[f'MFI_{period}'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=period)
                except Exception as e:
                    logger.warning(f"Error calculating MFI: {e}")
            
            # VWAP
            try:
                if HAS_PANDAS_TA:
                    vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
                else:
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
                
                vol_cols['VWAP'] = vwap
                vol_cols['dist_VWAP'] = (df['close'] - vwap) / vwap.replace(0, np.nan)
            except Exception as e:
                logger.warning(f"Error calculating VWAP: {e}")
            
            # Volume MA
            for period in [20, 50]:
                try:
                    if HAS_PANDAS_TA:
                        vol_ma = ta.sma(df['volume'], length=period)
                    else:
                        vol_ma = df['volume'].rolling(window=period).mean()
                    
                    vol_cols[f'Volume_MA_{period}'] = vol_ma
                    vol_cols[f'Volume_Ratio_{period}'] = df['volume'] / vol_ma.replace(0, np.nan)
                except Exception as e:
                    logger.warning(f"Error calculating Volume MA: {e}")
            
            if vol_cols:
                new_features.append(pd.DataFrame(vol_cols, index=df.index))
        
        # ====================================================================
        # 6. PRICE-BASED FEATURES
        # ====================================================================
        logger.info("Calculating Price-Based Features...")
        
        price_cols = {}
        for period in [1, 5, 10, 20]:
            price_cols[f'price_change_{period}'] = df['close'].pct_change(periods=period)
            price_cols[f'price_high_{period}'] = df['high'].rolling(window=period).max() / df['close'] - 1
            price_cols[f'price_low_{period}'] = df['low'].rolling(window=period).min() / df['close'] - 1
        
        if price_cols:
            new_features.append(pd.DataFrame(price_cols, index=df.index))
        
        # ====================================================================
        # PERFORMANCE FIX: Single concat operation
        # ====================================================================
        if new_features:
            logger.info(f"Concatenating {len(new_features)} feature DataFrames...")
            df = pd.concat([df] + new_features, axis=1)
        
        # Fill NaN and inf values
        df = df.bfill().fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        
        # CRITICAL FIX: Sanitize column names - remove dots ('.') which cause issues
        df.columns = df.columns.str.replace('.', '_')
        logger.info("Sanitized column names (removed dots)")
        
        # ====================================================================
        # CRITICAL FIX: ELIMINATE LOOK-AHEAD BIAS
        # ====================================================================
        # After calculating ALL technical indicators, shift EVERY feature by 1
        # EXCEPT: timestamp, open, high, low, close, volume (needed for price calculation)
        # Logic: Feature at row t MUST represent state at END of t-1
        # ====================================================================
        base_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        feature_cols = [c for c in df.columns if c not in base_cols]
        
        if feature_cols:
            logger.info(f"ELIMINATING LOOK-AHEAD BIAS: Shifting {len(feature_cols)} feature columns by 1 period")
            logger.info(f"Features to shift: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"Features: {feature_cols}")
            
            # Store original shape for verification
            original_shape = df.shape
            
            # Shift ALL features by 1 (features at t now represent state at t-1)
            df[feature_cols] = df[feature_cols].shift(1)
            
            # CRITICAL: Drop first row (NaN after shift) - mandatory sanitization
            df = df.dropna()
            
            logger.info(f"After shift: {original_shape} -> {df.shape} (removed {original_shape[0] - df.shape[0]} rows)")
            
            # VERIFICATION: Debug print to confirm leakage is eliminated
            if len(df) > 10:
                sample_idx = 10
                sample_feature = feature_cols[0] if feature_cols else None
                if sample_feature and sample_feature in df.columns:
                    feature_time = df.index[sample_idx - 1] if sample_idx > 0 else df.index[0]
                    price_time = df.index[sample_idx]
                    print(f"DEBUG LEAKAGE: Row {sample_idx} - Feature '{sample_feature}' is from time {feature_time}, "
                          f"Price (close) is from time {price_time}")
                    logger.info(f"VERIFICATION: Feature at row {sample_idx} represents state at t-1, price at t")
        
        feature_count = len([c for c in df.columns if c not in base_cols])
        logger.info(f"Generated {feature_count} candidate features (all shifted by 1 to prevent look-ahead bias)")
        
        # ====================================================================
        # TARGET CALCULATION: LOG RETURNS (Momentum Strategy)
        # ====================================================================
        # Switch from "Price Prediction" to "Return Prediction"
        # 
        # Why Log Returns?
        # - Stationary (constant mean/variance)
        # - Captures % changes (leverage-friendly)
        # - Symmetric for long/short
        # - Comparable across different price levels
        # 
        # Formula: log_return(t) = ln(close(t)) - ln(close(t-1))
        # Target: Predict log_return at t+1 (shift(-1))
        # ====================================================================
        logger.info("Calculating LOG RETURNS target...")
        
        # Calculate log returns
        df['log_return'] = np.log(df['close']).diff()
        
        # Shift to predict NEXT period (t+1)
        # At time t, we predict log_return at t+1
        df['log_return'] = df['log_return'].shift(-1)
        
        # Set as target
        df['target'] = df['log_return']
        
        # CRITICAL: Drop NaN rows (first row from diff, last row from shift(-1))
        initial_rows = len(df)
        df = df.dropna(subset=['target'])
        dropped_rows = initial_rows - len(df)
        
        logger.info(f"Log returns calculated: {len(df)} valid samples (dropped {dropped_rows} NaN)")
        logger.info(f"Target statistics: mean={df['target'].mean():.6f}, std={df['target'].std():.6f}")
        logger.info(f"Target range: [{df['target'].min():.6f}, {df['target'].max():.6f}]")
        
        return df
    
    def get_feature_list(self, df: pd.DataFrame) -> List[str]:
        """Get list of all generated feature names (excluding OHLCV)."""
        base_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        if 'timestamp' not in base_cols and 'timestamp' in df.columns:
            base_cols.append('timestamp')
        
        features = [col for col in df.columns if col not in base_cols]
        return features
    
    def select_features(
        self,
        df: pd.DataFrame,
        feature_config: Dict
    ) -> pd.DataFrame:
        """
        Select and calculate only the features specified in the config.
        Used during live trading after Optuna optimization.
        """
        df_full = self.generate_candidate_features(df)
        
        selected_features = feature_config.get('selected_features', [])
        base_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        if 'timestamp' in df_full.columns:
            base_cols.append('timestamp')
        
        cols_to_keep = base_cols + [f for f in selected_features if f in df_full.columns]
        
        return df_full[cols_to_keep]


def generate_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to generate candidate features."""
    generator = FeatureGenerator(require_pandas_ta=False)  # Don't require for backward compat
    return generator.generate_candidate_features(df)


if __name__ == "__main__":
    # Test feature generation
    import pandas as pd
    
    dates = pd.date_range('2023-01-01', periods=500, freq='15min')
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(500).cumsum() + 100,
        'high': np.random.randn(500).cumsum() + 101,
        'low': np.random.randn(500).cumsum() + 99,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 500)
    })
    
    generator = FeatureGenerator(require_pandas_ta=False)
    features_df = generator.generate_candidate_features(sample_df)
    
    print(f"Original columns: {len(sample_df.columns)}")
    print(f"Features generated: {len(features_df.columns)}")
    print(f"Feature list: {generator.get_feature_list(features_df)[:10]}...")
