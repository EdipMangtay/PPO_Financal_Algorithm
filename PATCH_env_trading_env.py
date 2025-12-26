# PATCH FOR env/trading_env.py
# Apply these changes:

# 1. In __init__ method, add after self.debug_mode:
        self.funding_rate: float = 0.0  # Default: no funding
        self.cumulative_fees: float = 0.0  # Track total fees paid

# 2. In reset() method, add after self.debug_mode = False:
        self.cumulative_fees = 0.0

# 3. Add this new method after _log_trade():
    def _log_baseline_debug(self, step: int, is_last: bool = False):
        """Log baseline debug info at specific steps."""
        if not self.debug_mode:
            return
        
        for coin, position in self.positions.items():
            current_price = self._get_current_price(coin)
            if current_price is None:
                continue
            
            try:
                timestamp = self.data[coin].index[self.current_step] if self.current_step < len(self.data[coin]) else None
            except:
                timestamp = f"step_{self.current_step}"
            
            position_notional = position.margin_used * position.leverage
            units = position_notional / position.entry_price
            unrealized_pnl_usd = self._calculate_unrealized_pnl(position, current_price)
            
            # Calculate cumulative fees
            entry_fee = position_notional * TAKER_FEE
            exit_fee = position_notional * TAKER_FEE if is_last else 0.0
            cumulative_fees = entry_fee + exit_fee if is_last else entry_fee
            
            equity = self.portfolio_value
            realized_pnl = 0.0 if not is_last else unrealized_pnl_usd - exit_fee - (position_notional * SLIPPAGE_PCT)
            
            logger.info(
                f"BASELINE_DEBUG step={step} | "
                f"timestamp={timestamp} | "
                f"close_price={current_price:.4f} | "
                f"entry_price={position.entry_price:.4f} | "
                f"side={position.position_type.name} | "
                f"margin_used={position.margin_used:.2f} | "
                f"leverage={position.leverage:.1f}x | "
                f"position_notional={position_notional:.2f} | "
                f"units={units:.4f} | "
                f"unrealized_pnl_usd={unrealized_pnl_usd:.2f} | "
                f"realized_pnl_usd={realized_pnl:.2f} | "
                f"fee_open={entry_fee:.4f} | "
                f"fee_close={exit_fee:.4f} | "
                f"cumulative_fees={cumulative_fees:.4f} | "
                f"balance={self.balance:.2f} | "
                f"equity={equity:.2f} | "
                f"is_position_open={True} | "
                f"stop_triggered_reason=None"
            )

# 4. Update _close_position() method - replace the fee calculation section:
        # FEES: Exit fee only (entry fee was already charged on open)
        # Fee is charged on NOTIONAL value
        exit_fee = position_notional * TAKER_FEE
        
        # SLIPPAGE (applied on exit only)
        slippage_cost = position_notional * SLIPPAGE_PCT
        
        # NO FUNDING (default funding_rate = 0.0)
        funding_cost = 0.0
        
        # Net realized PnL (after exit fees and slippage)
        net_pnl = unrealized_pnl_usd - exit_fee - slippage_cost - funding_cost
        
        # CRITICAL: Return margin + add realized PnL
        self.balance += position.margin_used  # Return margin
        self.balance += net_pnl  # Add realized PnL
        
        # Update cumulative fees
        self.cumulative_fees += exit_fee

# 5. In step() method, after portfolio_value calculation (around line 761), add:
        # Debug logging for baselines
        if self.debug_mode:
            coin_data = self.data.get(current_coin)
            if coin_data is not None:
                is_last = (self.current_step >= len(coin_data) - 1)
                if self.current_step == 0 or self.current_step == 1 or self.current_step == 5 or is_last:
                    self._log_baseline_debug(self.current_step, is_last=is_last)

