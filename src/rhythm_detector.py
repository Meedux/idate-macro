"""
rhythm_detector.py -- STUB-ONLY module.

The original CV template-based detection has been removed. This file
retains only the data-classes and factory functions for backward
compatibility.  All actual detection is now done via
``src.memory.memory_detector.MemoryDetector``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ----------------------------------------------------------------- stubs
@dataclass
class CalibrationData:
    """Minimal stub -- calibration is handled by the memory path now."""
    track_x: int = 0
    track_y: int = 0
    track_width: int = 0
    track_height: int = 0
    indicator_x: int = 0
    hit_zone_width: int = 45
    is_calibrated: bool = False

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class DetectedIcon:
    """Minimal stub retained for overlay/result compatibility."""
    icon_type: str = ""
    x: int = 0
    y: int = 0
    confidence: float = 0.0
    distance_to_indicator: int = 0


@dataclass
class RhythmDetectionResult:
    """Minimal stub retained for overlay/result compatibility."""
    keyboard_active: bool = False
    track_region: Any = None
    indicator_pos: tuple[int, int] | None = None
    indicator_confidence: float = 0.0
    icons: list[DetectedIcon] = field(default_factory=list)
    icons_in_hit_zone: list[str] = field(default_factory=list)
    aligned_icons: list[str] = field(default_factory=list)
    inference_time_ms: float = 0.0
    frame_id: int = 0


# ------------------------------------------------------ key lookup
ARROW_KEYS = {
    "up":    "up",
    "down":  "down",
    "left":  "left",
    "right": "right",
    "space": "space",
    "hand":  "space",
}


def create_detector(*_args, **_kwargs):
    """
    Factory originally used by the GUI for CV detection.
    Now raises an error to surface any stale call-sites early.
    """
    raise RuntimeError(
        "CV template-based detection has been removed. "
        "Use src.memory.memory_detector.MemoryDetector instead."
    )
