"""
Memory Reader – Continuous high-speed reader for confirmed memory patterns.

Once memory addresses / pointer chains have been discovered via the scanner,
this module reads them in a tight loop and exposes structured game state.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .process import ProcessAttacher
from .scanner import ValueType

PATTERNS_FILE = Path("memory_patterns.json")


@dataclass
class MemoryPattern:
    """
    A confirmed memory address or pointer chain.

    For direct addresses:
        address = 0x12345678, offsets = []
    For pointer chains:
        address = base_address, offsets = [0x10, 0x48, 0x0C]
        Reads: [[base + 0x10] + 0x48] + 0x0C
    """
    name: str                         # Human label (e.g., "note_0_x", "score")
    address: int                      # Base address or direct address
    offsets: list[int] = field(default_factory=list)  # Pointer chain offsets
    value_type: ValueType = ValueType.INT32
    module_name: str = ""            # Module-relative addressing
    module_offset: int = 0           # Offset from module base

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "address": f"0x{self.address:016X}",
            "address_int": self.address,
            "offsets": [f"0x{o:X}" for o in self.offsets],
            "offsets_int": self.offsets,
            "value_type": self.value_type.value,
            "module_name": self.module_name,
            "module_offset": f"0x{self.module_offset:X}" if self.module_offset else "0x0",
        }

    @classmethod
    def from_dict(cls, data: dict) -> MemoryPattern:
        addr = data.get("address_int", 0)
        if addr == 0:
            addr_str = data.get("address", "0x0")
            addr = int(addr_str, 16) if isinstance(addr_str, str) else int(addr_str)

        offsets = data.get("offsets_int", [])
        if not offsets:
            offsets_str = data.get("offsets", [])
            offsets = [int(o, 16) if isinstance(o, str) else int(o) for o in offsets_str]

        mod_off = data.get("module_offset", "0x0")
        if isinstance(mod_off, str):
            mod_off = int(mod_off, 16)

        return cls(
            name=data.get("name", "unnamed"),
            address=addr,
            offsets=offsets,
            value_type=ValueType(data.get("value_type", "int32")),
            module_name=data.get("module_name", ""),
            module_offset=mod_off,
        )


@dataclass
class GameState:
    """
    Structured game state read from memory.

    The fields below are generic – fill in what you discover during scanning.
    The MemoryDetector maps these to key presses.
    """
    # Timing / position
    timestamp: float = 0.0          # perf_counter when this snapshot was taken

    # Notes / arrows currently active
    # Each entry: {"type": "up"/"down"/..., "x": int, "y": int, "state": int}
    notes: list[dict[str, Any]] = field(default_factory=list)

    # Game state flags
    is_playing: bool = False        # Is the player in an active gameplay phase?
    score: int = 0
    combo: int = 0
    health: float = 1.0

    # Track timing
    song_position_ms: float = 0.0   # Current song position
    bpm: float = 0.0

    # Custom values (anything not covered above)
    custom: dict[str, int | float] = field(default_factory=dict)

    # Raw reads (address -> value) for debugging
    raw: dict[str, int | float] = field(default_factory=dict)


class MemoryReader:
    """
    High-speed memory reader for confirmed patterns.

    After discovering addresses with the scanner, register them here
    and call read_state() in the main loop for structured game state.
    """

    def __init__(self, attacher: ProcessAttacher):
        if not attacher.is_attached:
            raise ValueError("ProcessAttacher must be attached")
        self.attacher = attacher
        self.patterns: list[MemoryPattern] = []
        self._resolved_addresses: dict[str, int] = {}  # name -> resolved address
        self._log_callback: Callable[[str], None] | None = None

        # Map builder – callbacks that transform raw values into GameState fields
        self._mappers: dict[str, Callable[[int | float, GameState], None]] = {}

    def set_log_callback(self, callback: Callable[[str], None]):
        self._log_callback = callback

    def _log(self, msg: str):
        if self._log_callback:
            self._log_callback(msg)

    # ─── Pattern registration ──────────────────────────────────

    def add_pattern(self, pattern: MemoryPattern):
        """Register a memory pattern to read."""
        self.patterns.append(pattern)
        self._log(f"Added pattern: {pattern.name} @ 0x{pattern.address:X}")

    def add_patterns(self, patterns: list[MemoryPattern]):
        for p in patterns:
            self.add_pattern(p)

    def set_mapper(self, pattern_name: str, mapper: Callable[[int | float, GameState], None]):
        """
        Set a function that writes a raw memory value into GameState.

        Example:
            reader.set_mapper("score", lambda val, state: setattr(state, 'score', val))
        """
        self._mappers[pattern_name] = mapper

    # ─── Resolve addresses ─────────────────────────────────────

    def resolve_addresses(self) -> dict[str, int]:
        """
        Resolve all patterns to final addresses (follow pointer chains, apply module offsets).
        Call once after attaching, or again after game restart.
        """
        resolved: dict[str, int] = {}

        for pat in self.patterns:
            addr = pat.address

            # Module-relative addressing
            if pat.module_name:
                base = self.attacher.get_base_address(pat.module_name)
                if base is None:
                    self._log(f"Module '{pat.module_name}' not found for {pat.name}")
                    continue
                addr = base + pat.module_offset

            # Pointer chain
            if pat.offsets:
                final = self.attacher.follow_pointer_chain(addr, pat.offsets)
                if final is None:
                    self._log(f"Pointer chain failed for {pat.name}")
                    continue
                addr = final

            resolved[pat.name] = addr

        self._resolved_addresses = resolved
        self._log(f"Resolved {len(resolved)}/{len(self.patterns)} addresses")
        return resolved

    # ─── Read state ────────────────────────────────────────────

    def read_value(self, pattern: MemoryPattern, address: int) -> int | float | None:
        """Read a single value by type."""
        vt = pattern.value_type
        if vt == ValueType.INT8:
            data = self.attacher.read(address, 1)
            return int.from_bytes(data, "little", signed=True) if data else None
        elif vt == ValueType.UINT8:
            data = self.attacher.read(address, 1)
            return int.from_bytes(data, "little", signed=False) if data else None
        elif vt == ValueType.INT16:
            data = self.attacher.read(address, 2)
            return int.from_bytes(data, "little", signed=True) if data and len(data) >= 2 else None
        elif vt == ValueType.UINT16:
            data = self.attacher.read(address, 2)
            return int.from_bytes(data, "little", signed=False) if data and len(data) >= 2 else None
        elif vt == ValueType.INT32:
            return self.attacher.read_int32(address)
        elif vt == ValueType.UINT32:
            return self.attacher.read_uint32(address)
        elif vt == ValueType.INT64:
            return self.attacher.read_int64(address)
        elif vt == ValueType.UINT64:
            data = self.attacher.read(address, 8)
            return int.from_bytes(data, "little", signed=False) if data and len(data) >= 8 else None
        elif vt == ValueType.FLOAT:
            return self.attacher.read_float(address)
        elif vt == ValueType.DOUBLE:
            return self.attacher.read_double(address)
        return None

    def read_all_raw(self) -> dict[str, int | float | None]:
        """Read all resolved addresses and return raw values."""
        values: dict[str, int | float | None] = {}

        for pat in self.patterns:
            addr = self._resolved_addresses.get(pat.name)
            if addr is None:
                values[pat.name] = None
                continue
            values[pat.name] = self.read_value(pat, addr)

        return values

    def read_state(self) -> GameState:
        """
        Read all patterns and build a GameState.

        This is the main function called from the hot loop.
        Typically takes < 0.5ms for ~20 addresses.
        """
        state = GameState(timestamp=time.perf_counter())

        raw = self.read_all_raw()
        state.raw = {k: v for k, v in raw.items() if v is not None}

        # Apply mappers
        for name, value in raw.items():
            if value is None:
                continue
            mapper = self._mappers.get(name)
            if mapper:
                try:
                    mapper(value, state)
                except Exception:
                    pass

        return state

    # ─── Persistence ───────────────────────────────────────────

    def save_patterns(self, path: Path | None = None):
        """Save confirmed patterns to JSON."""
        path = path or PATTERNS_FILE

        data: dict = {}
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        data["confirmed"] = {
            p.name: p.to_dict() for p in self.patterns
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        self._log(f"Saved {len(self.patterns)} patterns to {path}")

    def load_patterns(self, path: Path | None = None) -> int:
        """
        Load confirmed patterns from JSON.

        Returns number of patterns loaded.
        """
        path = path or PATTERNS_FILE

        if not path.exists():
            self._log(f"Pattern file not found: {path}")
            return 0

        with open(path, "r") as f:
            data = json.load(f)

        confirmed = data.get("confirmed", {})
        loaded = 0

        for name, info in confirmed.items():
            if not isinstance(info, dict):
                continue
            try:
                pat = MemoryPattern.from_dict(info)
                pat.name = name
                self.add_pattern(pat)
                loaded += 1
            except Exception as e:
                self._log(f"Failed to load pattern '{name}': {e}")

        self._log(f"Loaded {loaded} patterns from {path}")
        return loaded

    # ─── Auto-discovery helpers ────────────────────────────────

    def promote_candidates(
        self,
        label: str,
        name_prefix: str = "addr",
        value_type: ValueType = ValueType.INT32,
        path: Path | None = None,
    ) -> int:
        """
        Promote discovered candidates from scanner to confirmed patterns.

        Args:
            label: The scanner save label to promote from
            name_prefix: Prefix for auto-generated names
            value_type: Value type for the candidates
            path: Pattern file path

        Returns:
            Number of candidates promoted
        """
        path = path or PATTERNS_FILE
        if not path.exists():
            return 0

        with open(path, "r") as f:
            data = json.load(f)

        discovered = data.get("discovered", {}).get(label, {})
        candidates = discovered.get("candidates", [])
        vtype = ValueType(discovered.get("value_type", value_type.value))

        count = 0
        for idx, cand in enumerate(candidates):
            addr = cand.get("address_int", 0)
            if addr == 0:
                continue
            pat = MemoryPattern(
                name=f"{name_prefix}_{idx}",
                address=addr,
                value_type=vtype,
            )
            self.add_pattern(pat)
            count += 1

        if count > 0:
            self.save_patterns(path)

        self._log(f"Promoted {count} candidates from '{label}'")
        return count
