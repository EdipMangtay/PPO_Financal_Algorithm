"""
Logging Utilities - Structured logging
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

def setup_logging(
    log_dir: Path,
    timeframe: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging to both file and console.
    
    Args:
        log_dir: Directory for log files
        timeframe: Optional timeframe identifier
        level: Logging level
    
    Returns:
        Logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger_name = f"trading_{timeframe}" if timeframe else "trading"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    log_file = log_dir / f"run_{timeframe}.log" if timeframe else log_dir / "run.log"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger



