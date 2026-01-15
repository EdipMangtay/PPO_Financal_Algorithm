"""
Download BTC/USDT data for training - Standalone version
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import ccxt.async_support as ccxt
except ImportError:
    print("ERROR: ccxt not installed. Run: pip install ccxt")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_ohlcv_data(exchange, symbol: str, timeframe: str, days: int):
    """Fetch OHLCV data from exchange."""
    since_dt = datetime.utcnow() - timedelta(days=days)
    since_ms = int(since_dt.timestamp() * 1000)
    
    all_data = []
    current_since = since_ms
    
    logger.info(f"Fetching {symbol} {timeframe} data (last {days} days)...")
    
    while True:
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            if not ohlcv:
                break
            
            all_data.extend(ohlcv)
            
            # Update since for next batch
            current_since = ohlcv[-1][0] + 1
            
            # Check if we got less than limit (end of data)
            if len(ohlcv) < 1000:
                break
                
            logger.info(f"  Fetched {len(all_data)} candles so far...")
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            break
    
    if not all_data:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    logger.info(f"✓ Fetched {len(df)} candles for {symbol} {timeframe}")
    return df

async def download_btc_data(days: int = 730):
    """
    Download BTC/USDT data for all timeframes (15m, 1h, 4h).
    
    Args:
        days: Number of days of historical data to fetch (default: 730 = 2 years)
    """
    logger.info("=" * 60)
    logger.info("DOWNLOADING BTC/USDT DATA")
    logger.info("=" * 60)
    logger.info(f"Fetching {days} days of data for BTC/USDT...")
    logger.info("Timeframes: 15m, 1h, 4h")
    logger.info("")
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize exchange
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',  # Use spot market for public data
        }
    })
    
    try:
        symbol = 'BTC/USDT'
        timeframes = ['15m', '1h', '4h']
        
        for tf in timeframes:
            try:
                # Fetch data
                df = await fetch_ohlcv_data(exchange, symbol, tf, days)
                
                if df is not None and len(df) > 0:
                    # Save to parquet
                    filename = f"BTC_USDT_{tf}.parquet"
                    filepath = data_dir / filename
                    df.to_parquet(filepath, index=False)
                    logger.info(f"✓ Saved {len(df)} rows to {filepath}")
                else:
                    logger.warning(f"⚠ No data fetched for {tf}")
                    
            except Exception as e:
                logger.error(f"✗ Error downloading {tf}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("DATA DOWNLOAD COMPLETE")
        logger.info("=" * 60)
        
        # Verify files
        for tf in timeframes:
            filepath = data_dir / f"BTC_USDT_{tf}.parquet"
            if filepath.exists():
                df = pd.read_parquet(filepath)
                logger.info(f"✓ {tf}: {len(df)} rows in {filepath}")
            else:
                logger.warning(f"⚠ {tf}: File not found")
                
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        await exchange.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download BTC/USDT data")
    parser.add_argument('--days', type=int, default=730,
                       help='Number of days of historical data (default: 730 = 2 years)')
    
    args = parser.parse_args()
    
    asyncio.run(download_btc_data(days=args.days))

