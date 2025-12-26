"""
Advanced Feature Engineering - Dynamic Feature Pool Generator
Creates a massive pool of indicators using pandas-ta for Optuna selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    logger = logging.getLogger(__name__)
    logger.warning("pandas-ta not available. Some features will be limited. Install with: pip install pandas-ta (requires Python 3.12+)")

logging.basicConfig(level=logging.INFO)
if 'logger' not in locals():
    logger = logging.getLogger(__name__)


class FeatureGenerator:
    """
    Generates a massive pool of candidate features for dynamic selection.
    Optuna will select which features and their parameters to use.
    """
    
    def __init__(self):
        """Initialize feature generator."""
        self.feature_pool: Dict[str, List[str]] = {}
        logger.info("Feature Generator initialized")
    
    def generate_candidate_features(
        self,
        df: pd.DataFrame,
        include_volume: bool = True
    ) -> pd.DataFrame:
        """
        Generate massive pool of candidate features.
        
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
        # 1. LINEAR REGRESSION CHANNELS (The Core)
        # Includes shorter lengths for high-frequency trading (10, 20, 30)
        # ====================================================================
        logger.info("Calculating Linear Regression Channels...")
        for length in [10, 20, 30, 50, 100, 200]:  # Added shorter lengths for frequent signals
            try:
                # Linear Regression
                linreg = ta.linreg(df['close'], length=length)
                df[f'linreg_{length}'] = linreg
                
                # Slope
                slope = ta.slope(df['close'], length=length)
                df[f'slope_{length}'] = slope
                
                # Intercept (approximate from linreg and slope)
                if f'linreg_{length}' in df.columns and f'slope_{length}' in df.columns:
                    # Intercept = linreg - slope * (length/2)
                    df[f'intercept_{length}'] = df[f'linreg_{length}'] - df[f'slope_{length}'] * (length / 2)
                
                # R-squared (coefficient of determination)
                # Approximate using correlation squared
                df[f'rsquared_{length}'] = df['close'].rolling(window=length).corr(
                    df[f'linreg_{length}']
                ) ** 2
                
                # Upper and Lower Channels (using ATR for channel width)
                atr = ta.atr(df['high'], df['low'], df['close'], length=14)
                df[f'linreg_upper_{length}'] = df[f'linreg_{length}'] + (atr * 2)
                df[f'linreg_lower_{length}'] = df[f'linreg_{length}'] - (atr * 2)
                
                # Distance from channels (normalized)
                df[f'dist_upper_{length}'] = (df['close'] - df[f'linreg_upper_{length}']) / df['close']
                df[f'dist_lower_{length}'] = (df['close'] - df[f'linreg_lower_{length}']) / df['close']
                df[f'dist_linreg_{length}'] = (df['close'] - df[f'linreg_{length}']) / df['close']
                
                # Mid-Line Crossovers (for inner channel trading)
                # Price crossing above/below the linear regression line
                df[f'cross_above_linreg_{length}'] = (df['close'] > df[f'linreg_{length}']).astype(int)
                df[f'cross_below_linreg_{length}'] = (df['close'] < df[f'linreg_{length}']).astype(int)
                
                # Momentum indicator: Rate of change of distance from mid-line
                df[f'momentum_linreg_{length}'] = df[f'dist_linreg_{length}'].diff()
                
                # Strong momentum signal: Price moving away from mid-line with positive slope
                if f'slope_{length}' in df.columns:
                    df[f'strong_momentum_long_{length}'] = (
                        (df[f'cross_above_linreg_{length}'] == 1) & 
                        (df[f'slope_{length}'] > 0) &
                        (df[f'momentum_linreg_{length}'] > 0)
                    ).astype(int)
                    df[f'strong_momentum_short_{length}'] = (
                        (df[f'cross_below_linreg_{length}'] == 1) & 
                        (df[f'slope_{length}'] < 0) &
                        (df[f'momentum_linreg_{length}'] < 0)
                    ).astype(int)
                
            except Exception as e:
                logger.warning(f"Error calculating LinReg for length {length}: {e}")
        
        # ====================================================================
        # 2. OSCILLATORS (Dynamic Periods)
        # ====================================================================
        logger.info("Calculating Oscillators...")
        oscillator_periods = [7, 14, 21, 50]
        
        for period in oscillator_periods:
            try:
                # RSI
                rsi = ta.rsi(df['close'], length=period)
                df[f'RSI_{period}'] = rsi
                
                # Stochastic
                stoch = ta.stoch(df['high'], df['low'], df['close'], k=period, d=3, smooth_k=3)
                if isinstance(stoch, pd.DataFrame):
                    df[f'STOCH_k_{period}'] = stoch.iloc[:, 0] if len(stoch.columns) > 0 else 0
                    df[f'STOCH_d_{period}'] = stoch.iloc[:, 1] if len(stoch.columns) > 1 else 0
                
                # CCI (Commodity Channel Index)
                cci = ta.cci(df['high'], df['low'], df['close'], length=period)
                df[f'CCI_{period}'] = cci
                
                # Williams %R
                willr = ta.willr(df['high'], df['low'], df['close'], length=period)
                df[f'WilliamsR_{period}'] = willr
                
            except Exception as e:
                logger.warning(f"Error calculating oscillators for period {period}: {e}")
        
        # ====================================================================
        # 3. TREND INDICATORS
        # ====================================================================
        logger.info("Calculating Trend Indicators...")
        
        # EMA Ribbons (multiple EMAs)
        for length in [8, 13, 21, 34, 55]:
            try:
                ema = ta.ema(df['close'], length=length)
                df[f'EMA_{length}'] = ema
            except Exception as e:
                logger.warning(f"Error calculating EMA_{length}: {e}")
        
        # SuperTrend
        for period in [10, 14, 21]:
            for multiplier in [2.0, 3.0]:
                try:
                    supertrend = ta.supertrend(df['high'], df['low'], df['close'], 
                                             period=period, multiplier=multiplier)
                    if isinstance(supertrend, pd.DataFrame):
                        df[f'SuperTrend_{period}_{multiplier}'] = supertrend.iloc[:, 0] if len(supertrend.columns) > 0 else 0
                except Exception as e:
                    logger.warning(f"Error calculating SuperTrend: {e}")
        
        # ADX (Average Directional Index)
        for period in [14, 21]:
            try:
                adx = ta.adx(df['high'], df['low'], df['close'], length=period)
                if isinstance(adx, pd.DataFrame):
                    df[f'ADX_{period}'] = adx.iloc[:, 0] if len(adx.columns) > 0 else 0
                    df[f'ADX_plus_{period}'] = adx.iloc[:, 1] if len(adx.columns) > 1 else 0
                    df[f'ADX_minus_{period}'] = adx.iloc[:, 2] if len(adx.columns) > 2 else 0
            except Exception as e:
                logger.warning(f"Error calculating ADX: {e}")
        
        # MACD (multiple periods)
        for fast in [8, 12]:
            for slow in [21, 26]:
                for signal in [7, 9]:
                    try:
                        macd = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
                        if isinstance(macd, pd.DataFrame):
                            df[f'MACD_{fast}_{slow}_{signal}'] = macd.iloc[:, 0] if len(macd.columns) > 0 else 0
                            df[f'MACD_signal_{fast}_{slow}_{signal}'] = macd.iloc[:, 1] if len(macd.columns) > 1 else 0
                            df[f'MACD_hist_{fast}_{slow}_{signal}'] = macd.iloc[:, 2] if len(macd.columns) > 2 else 0
                    except Exception as e:
                        logger.warning(f"Error calculating MACD: {e}")
        
        # ====================================================================
        # 4. VOLATILITY INDICATORS
        # ====================================================================
        logger.info("Calculating Volatility Indicators...")
        
        # ATR (multiple periods)
        for period in [7, 14, 21]:
            try:
                atr = ta.atr(df['high'], df['low'], df['close'], length=period)
                df[f'ATR_{period}'] = atr
                # Normalized ATR
                df[f'ATR_norm_{period}'] = atr / df['close']
            except Exception as e:
                logger.warning(f"Error calculating ATR_{period}: {e}")
        
        # Bollinger Bands (multiple periods and std devs)
        for period in [20, 21]:
            for std in [2, 2.5]:
                try:
                    bb = ta.bbands(df['close'], length=period, std=std)
                    if isinstance(bb, pd.DataFrame):
                        df[f'BB_upper_{period}_{std}'] = bb.iloc[:, 0] if len(bb.columns) > 0 else 0
                        df[f'BB_middle_{period}_{std}'] = bb.iloc[:, 1] if len(bb.columns) > 1 else 0
                        df[f'BB_lower_{period}_{std}'] = bb.iloc[:, 2] if len(bb.columns) > 2 else 0
                        df[f'BB_width_{period}_{std}'] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / bb.iloc[:, 1] if len(bb.columns) > 1 else 0
                except Exception as e:
                    logger.warning(f"Error calculating Bollinger Bands: {e}")
        
        # Keltner Channels
        for period in [20, 21]:
            for multiplier in [2.0, 2.5]:
                try:
                    kc = ta.kc(df['high'], df['low'], df['close'], length=period, multiplier=multiplier)
                    if isinstance(kc, pd.DataFrame):
                        df[f'KC_upper_{period}_{multiplier}'] = kc.iloc[:, 0] if len(kc.columns) > 0 else 0
                        df[f'KC_middle_{period}_{multiplier}'] = kc.iloc[:, 1] if len(kc.columns) > 1 else 0
                        df[f'KC_lower_{period}_{multiplier}'] = kc.iloc[:, 2] if len(kc.columns) > 2 else 0
                except Exception as e:
                    logger.warning(f"Error calculating Keltner Channels: {e}")
        
        # ====================================================================
        # 5. VOLUME INDICATORS (if volume available)
        # ====================================================================
        if include_volume and 'volume' in df.columns:
            logger.info("Calculating Volume Indicators...")
            
            # OBV (On-Balance Volume)
            try:
                obv = ta.obv(df['close'], df['volume'])
                df['OBV'] = obv
                # OBV Rate of Change
                df['OBV_ROC'] = obv.pct_change(periods=14)
            except Exception as e:
                logger.warning(f"Error calculating OBV: {e}")
            
            # MFI (Money Flow Index)
            for period in [14, 21]:
                try:
                    mfi = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=period)
                    df[f'MFI_{period}'] = mfi
                except Exception as e:
                    logger.warning(f"Error calculating MFI: {e}")
            
            # VWAP (Volume Weighted Average Price)
            try:
                vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
                df['VWAP'] = vwap
                # Distance from VWAP
                df['dist_VWAP'] = (df['close'] - vwap) / vwap
            except Exception as e:
                logger.warning(f"Error calculating VWAP: {e}")
            
            # Volume MA and ratios
            for period in [20, 50]:
                try:
                    vol_ma = ta.sma(df['volume'], length=period)
                    df[f'Volume_MA_{period}'] = vol_ma
                    df[f'Volume_Ratio_{period}'] = df['volume'] / vol_ma
                except Exception as e:
                    logger.warning(f"Error calculating Volume MA: {e}")
        
        # ====================================================================
        # 6. PRICE-BASED FEATURES
        # ====================================================================
        logger.info("Calculating Price-Based Features...")
        
        # Price changes
        for period in [1, 5, 10, 20]:
            df[f'price_change_{period}'] = df['close'].pct_change(periods=period)
            df[f'price_high_{period}'] = df['high'].rolling(window=period).max() / df['close'] - 1
            df[f'price_low_{period}'] = df['low'].rolling(window=period).min() / df['close'] - 1
        
        # Fill NaN values
        df = df.bfill().fillna(0)
        
        # Replace inf values
        df = df.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Generated {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']])} candidate features")
        
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
        
        Args:
            df: Raw OHLCV DataFrame
            feature_config: Dictionary with feature names and their parameters
        
        Returns:
            DataFrame with only selected features
        """
        # This would be called during live trading
        # For now, we'll generate all features and then filter
        df_full = self.generate_candidate_features(df)
        
        # Select only features in config
        selected_features = feature_config.get('selected_features', [])
        base_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        if 'timestamp' in df_full.columns:
            base_cols.append('timestamp')
        
        # Keep base columns + selected features
        cols_to_keep = base_cols + [f for f in selected_features if f in df_full.columns]
        
        return df_full[cols_to_keep]


def generate_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to generate candidate features."""
    generator = FeatureGenerator()
    return generator.generate_candidate_features(df)


if __name__ == "__main__":
    # Test feature generation
    import pandas as pd
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=500, freq='15min')
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(500).cumsum() + 100,
        'high': np.random.randn(500).cumsum() + 101,
        'low': np.random.randn(500).cumsum() + 99,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 500)
    })
    
    generator = FeatureGenerator()
    features_df = generator.generate_candidate_features(sample_df)
    
    print(f"Original columns: {len(sample_df.columns)}")
    print(f"Features generated: {len(features_df.columns)}")
    print(f"Feature list: {generator.get_feature_list(features_df)[:10]}...")  # Show first 10

