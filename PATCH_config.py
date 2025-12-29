# PATCH FOR config.py
# Replace lines 40-44 with:

# Trading Fees & Execution Costs (Binance USDT-M Futures Default)
TAKER_FEE: float = 0.0004  # 0.04% taker fee (Binance USDT-M default)
MAKER_FEE: float = 0.0002  # 0.02% maker fee (Binance USDT-M default)
TRANSACTION_FEE: float = 0.0004  # Default to taker
SLIPPAGE_PCT: float = 0.0005  # 0.05% slippage per trade
SPREAD_PCT: float = 0.0002  # 0.02% bid-ask spread

# Funding Rate (default 0, only charge if explicitly enabled)
FUNDING_RATE: float = 0.0  # Default: no funding charges




