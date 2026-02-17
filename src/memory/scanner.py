"""
Memory value scanner – Cheat Engine-style value search and filtering.

Workflow:
    1. First scan: search all readable memory for a known value.
    2. Next scan: filter candidates by changed / unchanged / new value.
    3. Repeat until one or a few addresses remain.
    4. Export surviving addresses for use in MemoryReader.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .process import ProcessAttacher, MemoryRegion


class ValueType(str, Enum):
    """Supported value types for scanning."""
    INT8 = "int8"
    UINT8 = "uint8"
    INT16 = "int16"
    UINT16 = "uint16"
    INT32 = "int32"
    UINT32 = "uint32"
    INT64 = "int64"
    UINT64 = "uint64"
    FLOAT = "float"
    DOUBLE = "double"
    BYTE_ARRAY = "bytes"


# Value type metadata: (struct_fmt, byte_size)
_TYPE_INFO: dict[ValueType, tuple[str, int]] = {
    ValueType.INT8:   ("<b", 1),
    ValueType.UINT8:  ("<B", 1),
    ValueType.INT16:  ("<h", 2),
    ValueType.UINT16: ("<H", 2),
    ValueType.INT32:  ("<i", 4),
    ValueType.UINT32: ("<I", 4),
    ValueType.INT64:  ("<q", 8),
    ValueType.UINT64: ("<Q", 8),
    ValueType.FLOAT:  ("<f", 4),
    ValueType.DOUBLE: ("<d", 8),
}


class ScanMode(str, Enum):
    """How to filter candidates."""
    EXACT = "exact"           # Value == target
    NOT_EQUAL = "not_equal"   # Value != target
    GREATER = "greater"       # Value > target
    LESS = "less"             # Value < target
    BETWEEN = "between"       # min <= value <= max
    CHANGED = "changed"       # Value changed since last scan
    UNCHANGED = "unchanged"   # Value unchanged since last scan
    INCREASED = "increased"   # Value > previous
    DECREASED = "decreased"   # Value < previous
    UNKNOWN = "unknown"       # Store all (first scan with unknown value)


@dataclass
class ScanResult:
    """Result of a memory scan."""
    address: int
    value: int | float
    previous_value: int | float | None = None

    def __repr__(self) -> str:
        hex_addr = f"0x{self.address:016X}"
        return f"ScanResult({hex_addr}, value={self.value}, prev={self.previous_value})"


class MemoryScanner:
    """
    Cheat Engine-style memory scanner.

    Attach to a process, run a first scan, then filter with subsequent scans
    until the target addresses are isolated.
    """

    # Maximum chunk size for reading memory regions (4 MB)
    CHUNK_SIZE = 4 * 1024 * 1024

    def __init__(self, attacher: ProcessAttacher):
        """
        Args:
            attacher: An attached ProcessAttacher instance.
        """
        if not attacher.is_attached:
            raise ValueError("ProcessAttacher must be attached before scanning")
        self.attacher = attacher
        self.candidates: list[ScanResult] = []
        self._value_type: ValueType = ValueType.INT32
        self._scan_count: int = 0
        self._progress_callback: Callable[[float, str], None] | None = None
        self._regions: list[MemoryRegion] = []

    @property
    def scan_count(self) -> int:
        return self._scan_count

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set a callback(progress_0_to_1, message) for UI updates."""
        self._progress_callback = callback

    def _report(self, progress: float, msg: str):
        if self._progress_callback:
            self._progress_callback(progress, msg)

    # ------------------------------------------------------------------ #
    #  First scan
    # ------------------------------------------------------------------ #

    def first_scan(
        self,
        value: int | float | None,
        value_type: ValueType = ValueType.INT32,
        mode: ScanMode = ScanMode.EXACT,
        value_max: int | float | None = None,
    ) -> int:
        """
        Scan all readable memory for a value.

        Args:
            value: Target value (None for UNKNOWN scan).
            value_type: Data type to search for.
            mode: Search mode.
            value_max: Upper bound for BETWEEN scans.

        Returns:
            Number of candidates found.
        """
        self._value_type = value_type
        self._scan_count = 1
        self.candidates = []

        if value_type == ValueType.BYTE_ARRAY:
            raise NotImplementedError("Byte array scanning not implemented yet")

        fmt, size = _TYPE_INFO[value_type]

        self._report(0.0, "Enumerating memory regions...")
        self._regions = self.attacher.get_readable_regions()
        total_bytes = sum(r.size for r in self._regions)

        self._report(0.0, f"Scanning {len(self._regions)} regions ({total_bytes / 1024 / 1024:.1f} MB)...")

        scanned = 0
        t0 = time.perf_counter()

        for region in self._regions:
            region_offset = 0
            while region_offset < region.size:
                chunk_size = min(self.CHUNK_SIZE, region.size - region_offset)
                data = self.attacher.read(region.base_address + region_offset, chunk_size)
                if data is None:
                    region_offset += chunk_size
                    scanned += chunk_size
                    continue

                # Scan through the chunk
                end = len(data) - size + 1
                for i in range(0, end, size):  # aligned scan
                    raw = data[i: i + size]
                    if len(raw) < size:
                        break
                    try:
                        val = struct.unpack(fmt, raw)[0]
                    except struct.error:
                        continue

                    match = self._check_value(val, value, mode, value_max)
                    if match:
                        addr = region.base_address + region_offset + i
                        self.candidates.append(ScanResult(address=addr, value=val))

                region_offset += chunk_size
                scanned += chunk_size
                if total_bytes > 0:
                    self._report(scanned / total_bytes, f"Found {len(self.candidates)} candidates...")

        elapsed = time.perf_counter() - t0
        self._report(1.0, f"First scan done: {len(self.candidates)} results in {elapsed:.1f}s")
        return len(self.candidates)

    # ------------------------------------------------------------------ #
    #  Next scan (filter existing candidates)
    # ------------------------------------------------------------------ #

    def next_scan(
        self,
        value: int | float | None = None,
        mode: ScanMode = ScanMode.EXACT,
        value_max: int | float | None = None,
    ) -> int:
        """
        Filter existing candidates by re-reading their values.

        Args:
            value: New target value (None for CHANGED/UNCHANGED etc.).
            mode: Filter mode.
            value_max: Upper bound for BETWEEN mode.

        Returns:
            Number of remaining candidates.
        """
        if not self.candidates:
            return 0

        self._scan_count += 1
        fmt, size = _TYPE_INFO[self._value_type]

        surviving: list[ScanResult] = []
        total = len(self.candidates)

        self._report(0.0, f"Filtering {total} candidates...")

        for idx, cand in enumerate(self.candidates):
            data = self.attacher.read(cand.address, size)
            if data is None or len(data) < size:
                continue

            try:
                current = struct.unpack(fmt, data)[0]
            except struct.error:
                continue

            keep = False

            if mode == ScanMode.EXACT:
                keep = self._check_value(current, value, ScanMode.EXACT, value_max)
            elif mode == ScanMode.NOT_EQUAL:
                keep = current != value
            elif mode == ScanMode.GREATER:
                keep = current > value
            elif mode == ScanMode.LESS:
                keep = current < value
            elif mode == ScanMode.BETWEEN:
                keep = value is not None and value_max is not None and value <= current <= value_max
            elif mode == ScanMode.CHANGED:
                keep = current != cand.value
            elif mode == ScanMode.UNCHANGED:
                keep = current == cand.value
            elif mode == ScanMode.INCREASED:
                keep = current > cand.value
            elif mode == ScanMode.DECREASED:
                keep = current < cand.value
            elif mode == ScanMode.UNKNOWN:
                keep = True

            if keep:
                surviving.append(ScanResult(
                    address=cand.address,
                    value=current,
                    previous_value=cand.value,
                ))

            if idx % 10000 == 0 and total > 0:
                self._report(idx / total, f"{len(surviving)} surviving...")

        self.candidates = surviving
        self._report(1.0, f"Scan #{self._scan_count}: {len(self.candidates)} remaining")
        return len(self.candidates)

    # ------------------------------------------------------------------ #
    #  Refresh values (re-read without filtering)
    # ------------------------------------------------------------------ #

    def refresh(self) -> int:
        """Re-read current values for all candidates without filtering."""
        fmt, size = _TYPE_INFO[self._value_type]
        refreshed: list[ScanResult] = []

        for cand in self.candidates:
            data = self.attacher.read(cand.address, size)
            if data is None or len(data) < size:
                continue
            try:
                current = struct.unpack(fmt, data)[0]
            except struct.error:
                continue
            refreshed.append(ScanResult(
                address=cand.address,
                value=current,
                previous_value=cand.value,
            ))

        self.candidates = refreshed
        return len(self.candidates)

    # ------------------------------------------------------------------ #
    #  Monitor selected addresses (live watch)
    # ------------------------------------------------------------------ #

    def monitor(
        self,
        addresses: list[int] | None = None,
        count: int = 20,
    ) -> list[ScanResult]:
        """
        Read current values for specific addresses or top N candidates.

        Args:
            addresses: Specific addresses to read. If None, uses top candidates.
            count: How many candidates to read if addresses is None.

        Returns:
            List of current ScanResults.
        """
        fmt, size = _TYPE_INFO[self._value_type]

        targets = addresses if addresses else [c.address for c in self.candidates[:count]]
        results: list[ScanResult] = []

        for addr in targets:
            data = self.attacher.read(addr, size)
            if data is None or len(data) < size:
                continue
            try:
                current = struct.unpack(fmt, data)[0]
            except struct.error:
                continue
            # Find previous
            prev = None
            for c in self.candidates:
                if c.address == addr:
                    prev = c.value
                    break
            results.append(ScanResult(address=addr, value=current, previous_value=prev))

        return results

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    def reset(self):
        """Clear all scan state."""
        self.candidates = []
        self._scan_count = 0
        self._regions = []

    @staticmethod
    def _check_value(
        current: int | float,
        target: int | float | None,
        mode: ScanMode,
        target_max: int | float | None,
    ) -> bool:
        """Check if a value matches the scan criteria."""
        if mode == ScanMode.EXACT:
            if target is None:
                return False
            if isinstance(current, float):
                return abs(current - target) < 0.001
            return current == target
        elif mode == ScanMode.NOT_EQUAL:
            return current != target
        elif mode == ScanMode.GREATER:
            return target is not None and current > target
        elif mode == ScanMode.LESS:
            return target is not None and current < target
        elif mode == ScanMode.BETWEEN:
            return target is not None and target_max is not None and target <= current <= target_max
        elif mode == ScanMode.UNKNOWN:
            return True
        return False

    def export_candidates(self, max_count: int = 100) -> list[dict]:
        """Export top candidates as serializable dicts."""
        return [
            {
                "address": f"0x{c.address:016X}",
                "address_int": c.address,
                "value": c.value,
                "previous_value": c.previous_value,
            }
            for c in self.candidates[:max_count]
        ]
