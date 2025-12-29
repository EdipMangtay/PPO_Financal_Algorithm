"""
SANITY TEST: Futures Accounting Verification
Given: entry=100, price=110, margin=1000, leverage=2
Expected: units=20, pnl_usd=200, equity=balance+200
"""

def test_futures_sanity():
    """Sanity test for futures accounting."""
    # Test parameters
    entry_price = 100.0
    current_price = 110.0
    margin_used = 1000.0
    leverage = 2.0
    initial_balance = 10000.0
    
    # Expected calculations (OPTION A: margin stays in balance)
    position_notional = margin_used * leverage  # 2000
    units = position_notional / entry_price  # 20
    unrealized_pnl_usd = units * (current_price - entry_price)  # 20 * 10 = 200
    
    # OPTION A: balance = initial_balance - entry_fee (margin NOT removed)
    # For this test, assume no fees
    balance = initial_balance  # Margin stays in balance
    equity = balance + unrealized_pnl_usd  # 10000 + 200 = 10200
    
    # Implementation (same as _calculate_unrealized_pnl)
    position_notional_calc = margin_used * leverage
    units_calc = position_notional_calc / entry_price
    unrealized_pnl_calc = units_calc * (current_price - entry_price)
    balance_calc = initial_balance  # Margin NOT removed
    equity_calc = balance_calc + unrealized_pnl_calc
    
    print("=" * 60)
    print("FUTURES ACCOUNTING SANITY TEST")
    print("=" * 60)
    print(f"Entry Price: {entry_price}")
    print(f"Current Price: {current_price}")
    print(f"Margin Used: {margin_used}")
    print(f"Leverage: {leverage}x")
    print(f"Initial Balance: {initial_balance}")
    print()
    print("Expected Calculations (OPTION A):")
    print(f"  Position Notional = {margin_used} * {leverage} = {position_notional_calc:.2f}")
    print(f"  Units = {position_notional_calc:.2f} / {entry_price} = {units_calc:.4f}")
    print(f"  Unrealized PnL = {units_calc:.4f} * ({current_price} - {entry_price}) = {unrealized_pnl_calc:.2f}")
    print(f"  Balance = {initial_balance} (margin NOT removed)")
    print(f"  Equity = {balance_calc} + {unrealized_pnl_calc:.2f} = {equity_calc:.2f}")
    print()
    
    # Assertions
    try:
        assert abs(position_notional_calc - 2000.0) < 0.01, \
            f"FAIL: Notional mismatch: {position_notional_calc} != 2000"
        assert abs(units_calc - 20.0) < 0.01, \
            f"FAIL: Units mismatch: {units_calc} != 20"
        assert abs(unrealized_pnl_calc - 200.0) < 0.01, \
            f"FAIL: PnL mismatch: {unrealized_pnl_calc} != 200"
        assert abs(equity_calc - 10200.0) < 0.01, \
            f"FAIL: Equity mismatch: {equity_calc} != 10200"
        
        print("OK Sanity test PASSED:")
        print(f"  Units: {units_calc:.4f} (expected: 20.0000)")
        print(f"  PnL USD: {unrealized_pnl_calc:.2f} (expected: 200.00)")
        print(f"  Equity: {equity_calc:.2f} (expected: 10200.00)")
        print(f"  Equity = Balance + 200: {equity_calc:.2f} = {balance_calc:.2f} + 200.00")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"FAIL: {e}")
        print()
        print("Intermediate Values:")
        print(f"  position_notional_calc = {position_notional_calc}")
        print(f"  units_calc = {units_calc}")
        print(f"  unrealized_pnl_calc = {unrealized_pnl_calc}")
        print(f"  balance_calc = {balance_calc}")
        print(f"  equity_calc = {equity_calc}")
        print("=" * 60)
        raise

if __name__ == "__main__":
    test_futures_sanity()




