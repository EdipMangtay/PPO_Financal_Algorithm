# PATCH FOR tuning/optimizer.py
# Replace the Buy&Hold baseline section (lines 167-230) with:

        # ====================================================================
        # BASELINE 2: Buy&Hold Agent (Long from start, hold to end)
        # ====================================================================
        logger.info("Baseline 2: Buy&Hold Agent (should match asset return)...")
        env_bh = TradingEnv(data=data_dict, initial_balance=INITIAL_BALANCE)
        obs, info = env_bh.reset(options={'ignore_stops': True, 'debug_mode': True})
        
        # Open long position immediately
        initial_price = test_data.iloc[0]['close']
        final_price = test_data.iloc[min(steps, len(test_data)-1)]['close']
        expected_return = (final_price - initial_price) / initial_price
        
        # CRITICAL: Open position manually at step 0
        action = np.array([1.0])  # Long action
        obs, reward, terminated, truncated, info = env_bh.step(
            action,
            tft_confidence_15m=1.0,
            tft_confidence_1h=1.0,
            tft_confidence_4h=1.0
        )
        
        # Verify position opened
        if len(env_bh.positions) == 0:
            logger.error("CRITICAL: Buy&Hold position did not open!")
            raise RuntimeError("Buy&Hold position failed to open")
        
        # Log step 0 (after opening)
        env_bh._log_baseline_debug(0, is_last=False)
        
        # Continue holding (no action changes, position stays open)
        for step in range(1, steps):
            action = np.array([1.0])  # Keep long action
            obs, reward, terminated, truncated, info = env_bh.step(
                action,
                tft_confidence_15m=1.0,
                tft_confidence_1h=1.0,
                tft_confidence_4h=1.0
            )
            
            # Log at specific steps
            if step == 1 or step == 5:
                env_bh._log_baseline_debug(step, is_last=False)
            
            if terminated or truncated:
                logger.warning(f"Buy&Hold terminated early at step {step}")
                break
        
        # CRITICAL: Log last step before closing
        env_bh._log_baseline_debug(steps - 1, is_last=True)
        
        # CRITICAL: Close position at end to realize PnL
        if len(env_bh.positions) > 0:
            for coin in list(env_bh.positions.keys()):
                final_price_for_close = env_bh._get_current_price(coin)
                env_bh._close_position(coin, "Baseline End", final_price_for_close)
        
        final_value_bh = env_bh.portfolio_value
        return_bh = (final_value_bh - INITIAL_BALANCE) / INITIAL_BALANCE
        
        # Calculate expected return with leverage and fees
        leverage_used = BASE_LEVERAGE  # Default
        if len(env_bh.trades) > 0:
            leverage_used = env_bh.trades[0].leverage
        
        # Expected return with leverage
        expected_with_leverage = expected_return * leverage_used
        
        # Account for fees: entry + exit = 2 * TAKER_FEE on notional
        # Notional = margin * leverage, so fee impact = 2 * TAKER_FEE * leverage
        from config import TAKER_FEE
        fee_impact = 2 * TAKER_FEE * leverage_used
        expected_after_fees = expected_with_leverage - fee_impact
        
        logger.info(f"  Buy&Hold: Return={return_bh*100:.4f}%, Expected Asset Return={expected_return*100:.4f}%, "
                   f"Expected with Leverage={expected_with_leverage*100:.4f}%, "
                   f"Expected after Fees={expected_after_fees*100:.4f}%, Leverage={leverage_used:.1f}x")
        
        # VALIDATION: Should be close to leveraged asset return minus fees
        # Allow tolerance for slippage and rounding
        tolerance = 0.005  # 0.5% tolerance
        expected_min = expected_after_fees - tolerance
        expected_max = expected_after_fees + tolerance
        
        if return_bh < expected_min or return_bh > expected_max:
            error_msg = (
                f"BASELINE VALIDATION FAILED: Buy&Hold agent returned {return_bh*100:.4f}% "
                f"(expected ~{expected_after_fees*100:.4f}% ± {tolerance*100:.2f}%). "
                f"Check debug logs above for details."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        env_bh.close()





