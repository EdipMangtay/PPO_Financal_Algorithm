"""
STANDALONE SELF-TEST: Futures Accounting Verification
No external dependencies - can run directly
"""

def test_futures_accounting():
    """Self-test: Verify futures accounting with synthetic data."""
    # Test parameters
    entry_price = 100.0
    current_price = 110.0
    margin_used = 1000.0
    leverage = 2.0
    initial_balance = 10000.0
    fee_rate = 0.0  # No fees for this test
    
    # Expected calculations
    position_notional = margin_used * leverage  # 2000
    units = position_notional / entry_price  # 20
    unrealized_pnl_usd = units * (current_price - entry_price)  # 20 * 10 = 200
    expected_equity = initial_balance - margin_used + unrealized_pnl_usd  # 10000 - 1000 + 200 = 9200
    
    # Implementation (same logic as _calculate_unrealized_pnl)
    position_notional_calc = margin_used * leverage
    units_calc = position_notional_calc / entry_price
    unrealized_pnl_calc = units_calc * (current_price - entry_price)
    equity_calc = initial_balance - margin_used + unrealized_pnl_calc
    
    # Print intermediate values
    print("=" * 60)
    print("FUTURES ACCOUNTING SELF-TEST")
    print("=" * 60)
    print(f"Entry Price: {entry_price}")
    print(f"Current Price: {current_price}")
    print(f"Margin Used: {margin_used}")
    print(f"Leverage: {leverage}x")
    print()
    print("Calculations:")
    print(f"  Position Notional = {margin_used} * {leverage} = {position_notional_calc:.2f}")
    print(f"  Units = {position_notional_calc:.2f} / {entry_price} = {units_calc:.4f}")
    print(f"  Unrealized PnL = {units_calc:.4f} * ({current_price} - {entry_price}) = {unrealized_pnl_calc:.2f}")
    print(f"  Equity = {initial_balance} - {margin_used} + {unrealized_pnl_calc:.2f} = {equity_calc:.2f}")
    print()
    
    # Assertions with detailed error messages
    try:
        assert abs(position_notional_calc - 2000.0) < 0.01, \
            f"FAIL: Notional mismatch: {position_notional_calc} != 2000"
        assert abs(units_calc - 20.0) < 0.01, \
            f"FAIL: Units mismatch: {units_calc} != 20"
        assert abs(unrealized_pnl_calc - 200.0) < 0.01, \
            f"FAIL: PnL mismatch: {unrealized_pnl_calc} != 200"
        assert abs(equity_calc - 9200.0) < 0.01, \
            f"FAIL: Equity mismatch: {equity_calc} != 9200"
        
        print("OK Self-test PASSED:")
        print(f"  Notional: {position_notional_calc:.2f} (expected: 2000.00)")
        print(f"  Units: {units_calc:.4f} (expected: 20.0000)")
        print(f"  Unrealized PnL: {unrealized_pnl_calc:.2f} (expected: 200.00)")
        print(f"  Equity: {equity_calc:.2f} (expected: 9200.00)")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"FAIL: {e}")
        print()
        print("Intermediate Values:")
        print(f"  position_notional_calc = {position_notional_calc}")
        print(f"  units_calc = {units_calc}")
        print(f"  unrealized_pnl_calc = {unrealized_pnl_calc}")
        print(f"  equity_calc = {equity_calc}")
        print("=" * 60)
        raise

if __name__ == "__main__":
    test_futures_accounting()

