"""Signal generators for different timeframes."""

from signals.signal_base import SignalBase
from signals.signal_15m import Signal15m
from signals.signal_1h import Signal1h
from signals.signal_4h import Signal4h

__all__ = ['SignalBase', 'Signal15m', 'Signal1h', 'Signal4h']



