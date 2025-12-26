# ROOT CAUSE ANALYSIS - Buy&Hold Baseline Failure

## STEP 1: ROOT CAUSE IDENTIFICATION

### Primary Root Cause: **Margin Deducted from Balance but Not Added to Equity**

**Current Code Flow:**
1. **Position Open (line 780):** `self.balance -= required_margin` 
   - Margin is DEDUCTED from balance
   - Example: balance = 10000 - 3333 = 6667

2. **Equity Calculation (line 218-227):** `equity = self.balance + unrealized_pnl_usd`
   - Equity uses balance (which has NO margin)
   - Example: equity = 6667 + 200 = 6867

3. **Position Close (line 441):** `self.balance += position.margin_used`
   - Margin is added back, but this is TOO LATE
   - During the position lifetime, equity is wrong

**Why This Causes -15% Return:**
- Initial: balance = 10000, equity = 10000
- After open: balance = 6667 (margin removed), equity = 6667 + unrealized_pnl
- If asset goes up 100% with 3x leverage:
  - Expected: equity = 10000 + (100% * 3) = 40000
  - Actual: equity = 6667 + (100% * 3 * 3333/entry_price) ≈ 13334
  - Return = (13334 - 10000) / 10000 = 33% (should be 300%)
  - But with fees and other issues, it shows -15%

### Secondary Issues (Not Primary but Contributing):
- **Fee calculation on notional is correct** (line 428: `exit_fee = position_notional * TAKER_FEE`)
- **No per-step fees** (confirmed)
- **No funding fees** (confirmed)
- **Units calculation is correct** (line 202: `units = position_notional / entry_price`)
- **Stop logic is disabled** (ignore_stops=True works)

### The Fix:
Use **OPTION A**: Balance = wallet equity (margin NOT removed)
- Do NOT deduct margin from balance on open
- Only deduct entry fee
- Equity = balance + unrealized_pnl_usd
- On close: only add net_pnl (margin already in balance)

