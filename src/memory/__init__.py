"""
Memory reading, scanning, and BMS chart infrastructure for iDate Revival.

All detection is done via direct process memory reading + BMS chart scheduling.
No CV or template matching is used.
"""

from .process import ProcessAttacher, ProcessInfo
from .scanner import MemoryScanner, ScanResult, ValueType
from .reader import MemoryReader, MemoryPattern
from .memory_detector import MemoryDetector
from .bms_parser import BmsParser, BmsChart, BmsNote

__all__ = [
    "ProcessAttacher",
    "ProcessInfo",
    "MemoryScanner",
    "ScanResult",
    "ValueType",
    "MemoryReader",
    "MemoryPattern",
    "MemoryDetector",
    "BmsParser",
    "BmsChart",
    "BmsNote",
]
