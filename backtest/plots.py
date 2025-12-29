"""
Backtest Visualization - Generate equity curve plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

def plot_equity_curve(
    equity_curve: List[float],
    output_path: Path,
    title: str = "Equity Curve"
):
    """Plot equity curve."""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        equity_array = np.array(equity_curve)
        ax.plot(equity_array, linewidth=2)
        ax.set_xlabel('Bar')
        ax.set_ylabel('Equity')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add final equity annotation
        final_equity = equity_array[-1]
        initial_equity = equity_array[0]
        return_pct = (final_equity - initial_equity) / initial_equity * 100
        
        ax.text(
            0.02, 0.98,
            f'Initial: ${initial_equity:,.2f}\nFinal: ${final_equity:,.2f}\nReturn: {return_pct:.2f}%',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved equity curve plot to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to create plot: {e}")





