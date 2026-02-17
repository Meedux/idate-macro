"""
Configuration models and loaders for idate.
Uses Pydantic for validation and TOML for file format.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from pydantic import BaseModel, Field, field_validator


class CaptureConfig(BaseModel):
    """Screen capture settings."""
    monitor: int = 0
    roi: list[int] = Field(default_factory=lambda: [0, 0, 0, 0])
    target_fps: int = 120
    dpi_scale: float = 1.0

    @field_validator("roi")
    @classmethod
    def validate_roi(cls, v: list[int]) -> list[int]:
        if len(v) != 4:
            raise ValueError("ROI must have exactly 4 values: [x, y, width, height]")
        return v


class LanesConfig(BaseModel):
    """Lane detection configuration."""
    count: int = 8
    centers: list[int] = Field(default_factory=list)
    width_tolerance: int = 40
    hit_line_y: int = 0


class DetectionConfig(BaseModel):
    """Note detection settings.  DEPRECATED -- kept for CLI compat only."""
    method: str = "color"
    note_colors: list[list[int]] = Field(default_factory=list)
    min_note_area: int = 200
    max_note_area: int = 10000
    template_threshold: float = 0.7

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        valid = {"template", "color", "contour"}
        if v not in valid:
            raise ValueError(f"Detection method must be one of {valid}")
        return v


class TimingConfig(BaseModel):
    """Timing and synchronization settings."""
    pixels_per_ms: float = 0.5
    hit_window_ms: int = 50
    reaction_offset_ms: int = 15
    use_metadata: bool = False
    beatmap_path: str = ""


class SafetyConfig(BaseModel):
    """Safety and rate limiting settings."""
    max_actions_per_second: int = 60
    stop_hotkey: str = "f12"
    require_focus: bool = True
    window_title_pattern: str = ".*"


class DebugConfig(BaseModel):
    """Debug and diagnostic settings."""
    show_overlay: bool = False
    save_frames: bool = False
    frame_output_dir: str = "debug_frames"
    log_detections: bool = True


class GameProfile(BaseModel):
    """Complete game profile configuration."""
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    lanes: LanesConfig = Field(default_factory=LanesConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> GameProfile:
        """Load game profile from TOML file."""
        path = Path(path)
        with path.open("rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        """Export profile as dictionary."""
        return self.model_dump()


class DirectionsConfig(BaseModel):
    """Single and combo direction mappings."""
    up: str = "up"
    down: str = "down"
    left: str = "left"
    right: str = "right"
    up_left: list[str] = Field(default_factory=lambda: ["up", "left"])
    down_left: list[str] = Field(default_factory=lambda: ["down", "left"])
    up_right: list[str] = Field(default_factory=lambda: ["up", "right"])
    down_right: list[str] = Field(default_factory=lambda: ["down", "right"])


class KeyTimingConfig(BaseModel):
    """Key press timing settings."""
    press_duration_ms: int = 30
    min_gap_ms: int = 10
    chord_stagger_ms: int = 0


class KeyMap(BaseModel):
    """Complete keymap configuration."""
    directions: DirectionsConfig = Field(default_factory=DirectionsConfig)
    lanes: dict[str, str | list[str]] = Field(default_factory=dict)
    timing: KeyTimingConfig = Field(default_factory=KeyTimingConfig)
    aliases: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: str | Path) -> KeyMap:
        """Load keymap from TOML file."""
        path = Path(path)
        with path.open("rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data)

    def get_keys_for_lane(self, lane_index: int) -> list[str]:
        """Get the key(s) to press for a given lane index."""
        key_or_keys = self.lanes.get(str(lane_index))
        if key_or_keys is None:
            raise KeyError(f"No key mapping for lane {lane_index}")
        if isinstance(key_or_keys, str):
            return [key_or_keys]
        return list(key_or_keys)

    def resolve_alias(self, name: str) -> str:
        """Resolve an alias to its actual direction name."""
        return self.aliases.get(name, name)


class RuntimeConfig(BaseModel):
    """Runtime configuration combining profile and keymap."""
    profile: GameProfile
    keymap: KeyMap

    @classmethod
    def load(
        cls,
        profile_path: str | Path,
        keymap_path: str | Path,
    ) -> RuntimeConfig:
        """Load both configs and create runtime configuration."""
        return cls(
            profile=GameProfile.from_toml(profile_path),
            keymap=KeyMap.from_toml(keymap_path),
        )
