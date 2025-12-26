"""
Data Loader with Context-Aware BTC/USDT Dominance Injection
Uses ccxt with asyncio and handles API rate limits.
"""

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (if exists)
load_dotenv()

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TOP_20_COINS,
    SCALP_TIMEFRAME,
    SWING_TIMEFRAME,
    API_RATE_LIMIT_DELAY,
    EXCHANGE_NAME,
    EXCHANGE_API_KEY,
    EXCHANGE_SECRET,
    EXCHANGE_SANDBOX,
    CONTEXT_FEATURES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for trading."""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal, and Histogram."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd.fillna(0), macd_signal.fillna(0), macd_hist.fillna(0)
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.bfill().fillna(0)
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper.fillna(prices), sma.fillna(prices), lower.fillna(prices)
    
    @staticmethod
    def calculate_volume_ma(volume: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Volume Moving Average."""
        return volume.rolling(window=period).mean().fillna(volume)


class DataLoader:
    """Async data loader with context-aware BTC/USDT dominance injection."""
    
    def __init__(self, exchange_name: str = EXCHANGE_NAME):
        """Initialize the data loader."""
        self.exchange_name = exchange_name
        self.exchange: Optional[ccxt.Exchange] = None
        self.data_dir = Path(__file__).parent.parent / "data" / "raw"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    async def _get_exchange(self) -> ccxt.Exchange:
        """Get or create exchange instance."""
        if self.exchange is None:
            exchange_class = getattr(ccxt, self.exchange_name)
            self.exchange = exchange_class({
                'apiKey': os.getenv('BINANCE_API_KEY', EXCHANGE_API_KEY),
                'secret': os.getenv('BINANCE_SECRET', EXCHANGE_SECRET),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # Use futures
                },
                'sandbox': EXCHANGE_SANDBOX,
            })
        return self.exchange
    
    async def _fetch_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        since: Optional[int] = None,
        limit: Optional[int] = 500
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.
        If limit is None, fetches all available data using pagination.
        """
        exchange = await self._get_exchange()
        
        try:
            # CRITICAL FIX: If limit is None, fetch all data using pagination
            if limit is None:
                all_ohlcv = []
                current_since = since
                max_limit = 1000  # Binance API maximum per request
                
                while True:
                    await asyncio.sleep(API_RATE_LIMIT_DELAY)
                    
                    batch = await exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=current_since,
                        limit=max_limit
                    )
                    
                    if not batch or len(batch) == 0:
                        break
                    
                    all_ohlcv.extend(batch)
                    
                    # If we got less than max_limit, we've reached the end
                    if len(batch) < max_limit:
                        break
                    
                    # Update since to the last timestamp + 1ms
                    current_since = batch[-1][0] + 1
                
                ohlcv = all_ohlcv
                logger.info(f"Fetched {len(ohlcv)} candles for {symbol} {timeframe} (full dataset)")
            else:
                # Rate limit delay
                await asyncio.sleep(API_RATE_LIMIT_DELAY)
                
                ohlcv = await exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )
            
            if not ohlcv:
                return pd.DataFrame()
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    async def _fetch_usdt_dominance(self) -> pd.Series:
        """Fetch USDT Dominance (simplified - using BTC dominance as proxy)."""
        # Note: Real USDT dominance requires external API (e.g., CoinGecko)
        # For now, we'll use a simplified calculation based on BTC market cap
        # In production, integrate with CoinGecko API
        exchange = await self._get_exchange()
        
        try:
            await asyncio.sleep(API_RATE_LIMIT_DELAY)
            ticker = await exchange.fetch_ticker('BTC/USDT')
            # Simplified: Use inverse of BTC price movement as proxy
            # In production, fetch from CoinGecko: https://api.coingecko.com/api/v3/global
            btc_price = ticker['last']
            
            # Mock USDT dominance (in production, fetch real data)
            # For now, return a constant series that will be replaced with real data
            logger.warning("USDT Dominance: Using mock data. Integrate CoinGecko API for production.")
            return pd.Series([50.0] * 500, name='USDT_Dominance')  # Placeholder
            
        except Exception as e:
            logger.error(f"Error fetching USDT dominance: {e}")
            return pd.Series([50.0] * 500, name='USDT_Dominance')
    
    async def _fetch_btc_data(self, timeframe: str, limit: Optional[int] = 500) -> pd.DataFrame:
        """Fetch BTC data for context injection."""
        btc_df = await self._fetch_ohlcv('BTC/USDT', timeframe, limit=limit)
        if btc_df.empty:
            return btc_df
        
        # Calculate BTC RSI
        btc_df['BTC_RSI'] = TechnicalIndicators.calculate_rsi(btc_df['close'])
        btc_df['BTC_Close'] = btc_df['close']
        
        return btc_df[['BTC_Close', 'BTC_RSI']]
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe."""
        if df.empty:
            return df
        
        df = df.copy()
        
        # RSI
        df['RSI'] = TechnicalIndicators.calculate_rsi(df['close'])
        
        # MACD
        macd, signal, hist = TechnicalIndicators.calculate_macd(df['close'])
        df['MACD'] = macd
        df['MACD_signal'] = signal
        df['MACD_hist'] = hist
        
        # ATR
        df['ATR'] = TechnicalIndicators.calculate_atr(
            df['high'], df['low'], df['close']
        )
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(df['close'])
        df['BB_upper'] = bb_upper
        df['BB_middle'] = bb_middle
        df['BB_lower'] = bb_lower
        
        # Volume MA
        df['Volume_MA'] = TechnicalIndicators.calculate_volume_ma(df['volume'])
        
        return df
    
    def _inject_context_features(
        self, 
        coin_df: pd.DataFrame, 
        btc_df: pd.DataFrame, 
        usdt_dom: pd.Series
    ) -> pd.DataFrame:
        """
        CRITICAL: Inject BTC and USDT dominance into every coin's dataframe.
        This makes the model context-aware of market regime.
        """
        if coin_df.empty or btc_df.empty:
            return coin_df
        
        df = coin_df.copy()
        
        # Align indices (forward fill for missing values)
        btc_aligned = btc_df.reindex(df.index, method='ffill')
        df['BTC_Close'] = btc_aligned['BTC_Close'].bfill().fillna(df['close'].iloc[0] if len(df) > 0 else 0)
        df['BTC_RSI'] = btc_aligned['BTC_RSI'].fillna(50.0)
        
        # Inject USDT Dominance (align with dataframe length)
        if len(usdt_dom) >= len(df):
            df['USDT_Dominance'] = usdt_dom.iloc[:len(df)].values
        else:
            # Pad or interpolate
            usdt_padded = pd.Series([usdt_dom.iloc[-1]] * len(df))
            usdt_padded.iloc[:len(usdt_dom)] = usdt_dom.values
            df['USDT_Dominance'] = usdt_padded.values
        
        return df
    
    async def fetch_recent(
        self, 
        days: int = 7, 
        timeframe: str = SCALP_TIMEFRAME,
        coins: Optional[List[str]] = None,
        fetch_all_timeframes: bool = True,
        use_cache: bool = True  # YENİ: Cache kullan
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch recent data for all coins with BTC/USDT context injection.
        Önce disk cache'den kontrol eder, yoksa indirir.
        
        Args:
            days: Number of days to fetch
            timeframe: Primary candle timeframe
            coins: List of coins to fetch (defaults to TOP_20_COINS)
            fetch_all_timeframes: If True, fetch 15m, 1h, and 4h data before optimization
            use_cache: If True, check disk cache first before downloading
        
        Returns:
            Dictionary mapping coin symbols to dataframes with context features
        """
        if coins is None:
            coins = TOP_20_COINS
        
        # ====================================================================
        # CACHE KONTROLÜ: Önce disk'ten kontrol et
        # ====================================================================
        if use_cache and fetch_all_timeframes:
            logger.info("=" * 60)
            logger.info("CHECKING DISK CACHE...")
            logger.info("=" * 60)
            
            timeframes_to_fetch = ['15m', '1h', '4h']
            all_cached = True
            cached_data = {}
            
            for tf in timeframes_to_fetch:
                cached_data[tf] = await self.load_from_disk(timeframe=tf, coins=coins)
                for coin in coins:
                    if coin not in cached_data[tf]:
                        all_cached = False
                        logger.info(f"⚠ {coin} {tf} not in cache, will download")
                        break
                if not all_cached:
                    break
            
            if all_cached:
                logger.info("=" * 60)
                logger.info("✓ ALL DATA LOADED FROM CACHE! Skipping download.")
                logger.info("=" * 60)
                # Primary timeframe için return et
                return cached_data.get(timeframe, {})
        
        # ====================================================================
        # CRITICAL FIX: Download all timeframes BEFORE optimization
        # ====================================================================
        if fetch_all_timeframes:
            logger.info("=" * 60)
            logger.info("STEP 1: Downloading ALL timeframes (15m, 1h, 4h)")
            logger.info("=" * 60)
            
            timeframes_to_fetch = ['15m', '1h', '4h']
            for tf in timeframes_to_fetch:
                logger.info(f"\nFetching {days} days of {tf} data for {len(coins)} coins...")
                
                # Calculate since timestamp
                since_dt = datetime.utcnow() - timedelta(days=days)
                since_ms = int(since_dt.timestamp() * 1000)
                
                # Fetch BTC context data for this timeframe
                logger.info(f"Fetching BTC context data for {tf}...")
                # CRITICAL FIX: Remove limit to fetch full dataset
                btc_df = await self._fetch_btc_data(tf, limit=None)
                
                # Fetch USDT dominance
                if tf == '15m':  # Only fetch once
                    logger.info("Fetching USDT dominance...")
                    usdt_dom = await self._fetch_usdt_dominance()
                
                # Fetch all coin data for this timeframe
                for coin in coins:
                    logger.info(f"Fetching {coin} {tf}...")
                    # CRITICAL FIX: Remove limit to fetch full dataset
                    coin_df = await self._fetch_ohlcv(coin, tf, since=since_ms, limit=None)
                    
                    if not coin_df.empty:
                        # Add technical indicators
                        coin_df = self._add_technical_indicators(coin_df)
                        
                        # CRITICAL: Inject BTC and USDT dominance
                        coin_df = self._inject_context_features(coin_df, btc_df, usdt_dom)
                        
                        # Save to disk
                        save_path = self.data_dir / f"{coin.replace('/', '_')}_{tf}.parquet"
                        coin_df.to_parquet(save_path)
                        logger.info(f"✓ Saved {coin} {tf} data to {save_path}")
            
            logger.info("=" * 60)
            logger.info("✓ All timeframes downloaded successfully!")
            logger.info("=" * 60)
        
        # Now fetch the primary timeframe data for return
        logger.info(f"\nFetching {days} days of {timeframe} data for {len(coins)} coins...")
        
        # Calculate since timestamp
        since_dt = datetime.utcnow() - timedelta(days=days)
        since_ms = int(since_dt.timestamp() * 1000)
        
        # Fetch BTC context data first
        logger.info("Fetching BTC context data...")
        # CRITICAL FIX: Remove limit to fetch full dataset
        btc_df = await self._fetch_btc_data(timeframe, limit=None)
        
        # Fetch USDT dominance
        logger.info("Fetching USDT dominance...")
        usdt_dom = await self._fetch_usdt_dominance()
        
        # Fetch all coin data
        coin_data = {}
        for coin in coins:
            logger.info(f"Fetching {coin}...")
            # CRITICAL FIX: Remove limit to fetch full dataset
            coin_df = await self._fetch_ohlcv(coin, timeframe, since=since_ms, limit=None)
            
            if not coin_df.empty:
                # Add technical indicators
                coin_df = self._add_technical_indicators(coin_df)
                
                # CRITICAL: Inject BTC and USDT dominance
                coin_df = self._inject_context_features(coin_df, btc_df, usdt_dom)
                
                coin_data[coin] = coin_df
                
                # Save to disk
                save_path = self.data_dir / f"{coin.replace('/', '_')}_{timeframe}.parquet"
                coin_df.to_parquet(save_path)
                logger.info(f"Saved {coin} data to {save_path}")
        
        await self.close()
        return coin_data
    
    async def load_from_disk(
        self, 
        timeframe: str = SCALP_TIMEFRAME,
        coins: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load data from disk if available."""
        if coins is None:
            coins = TOP_20_COINS
        
        coin_data = {}
        for coin in coins:
            save_path = self.data_dir / f"{coin.replace('/', '_')}_{timeframe}.parquet"
            if save_path.exists():
                df = pd.read_parquet(save_path)
                coin_data[coin] = df
                logger.info(f"Loaded {coin} from disk")
        
        return coin_data
    
    async def close(self):
        """Close exchange connection."""
        if self.exchange:
            await self.exchange.close()
            self.exchange = None


# Convenience function
async def fetch_recent_data(days: int = 7, timeframe: str = SCALP_TIMEFRAME) -> Dict[str, pd.DataFrame]:
    """Convenience function to fetch recent data."""
    loader = DataLoader()
    try:
        return await loader.fetch_recent(days=days, timeframe=timeframe)
    finally:
        await loader.close()


if __name__ == "__main__":
    # Test data loading
    async def test():
        loader = DataLoader()
        data = await loader.fetch_recent(days=7, timeframe='15m')
        print(f"Loaded {len(data)} coins")
        if data:
            first_coin = list(data.keys())[0]
            print(f"\nSample data for {first_coin}:")
            print(data[first_coin].head())
            print(f"\nColumns: {data[first_coin].columns.tolist()}")
    
    asyncio.run(test())

