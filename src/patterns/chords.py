"""
Chord and pattern mapping for rhythm game inputs.

Handles mapping between lane indices, direction names, and actual
keyboard keys, including chord (multi-key) inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import KeyMap


class Direction(Enum):
    """Standard rhythm game directions."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    UP_LEFT = "up_left"
    DOWN_LEFT = "down_left"
    UP_RIGHT = "up_right"
    DOWN_RIGHT = "down_right"


# Aliases for direction names (normalized)
DIRECTION_ALIASES: dict[str, Direction] = {
    "up": Direction.UP,
    "down": Direction.DOWN,
    "left": Direction.LEFT,
    "right": Direction.RIGHT,
    "up_left": Direction.UP_LEFT,
    "upleft": Direction.UP_LEFT,
    "up-left": Direction.UP_LEFT,
    "down_left": Direction.DOWN_LEFT,
    "downleft": Direction.DOWN_LEFT,
    "down-left": Direction.DOWN_LEFT,
    "low_left": Direction.DOWN_LEFT,  # Alias from user's spec
    "lowleft": Direction.DOWN_LEFT,
    "up_right": Direction.UP_RIGHT,
    "upright": Direction.UP_RIGHT,
    "up-right": Direction.UP_RIGHT,
    "down_right": Direction.DOWN_RIGHT,
    "downright": Direction.DOWN_RIGHT,
    "down-right": Direction.DOWN_RIGHT,
    "low_right": Direction.DOWN_RIGHT,  # Alias
    "lowright": Direction.DOWN_RIGHT,
}


def normalize_direction(name: str) -> Direction | None:
    """
    Normalize a direction name to standard Direction enum.
    
    Args:
        name: Direction name (various formats accepted).
    
    Returns:
        Direction enum value, or None if not recognized.
    """
    normalized = name.lower().strip()
    return DIRECTION_ALIASES.get(normalized)


@dataclass
class ChordPattern:
    """Represents a chord (possibly multi-key) input pattern."""
    name: str
    keys: list[str]
    direction: Direction | None = None

    @property
    def is_chord(self) -> bool:
        """Check if this is a multi-key chord."""
        return len(self.keys) > 1

    @property
    def is_single(self) -> bool:
        """Check if this is a single key press."""
        return len(self.keys) == 1


@dataclass
class PatternMapper:
    """
    Maps lane indices and directions to key patterns.
    
    Handles the translation between:
    - Lane index (0-7) -> Direction -> Keys
    - Direction name -> Keys
    - Keymap configuration -> ChordPattern
    """
    keymap: KeyMap | None = None
    _lane_patterns: dict[int, ChordPattern] = field(default_factory=dict, init=False)
    _direction_patterns: dict[Direction, ChordPattern] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if self.keymap:
            self._build_patterns()

    def _build_patterns(self) -> None:
        """Build pattern mappings from keymap configuration."""
        if not self.keymap:
            return
        
        # Build direction patterns
        directions = self.keymap.directions
        
        self._direction_patterns[Direction.UP] = ChordPattern(
            name="up", keys=[directions.up], direction=Direction.UP
        )
        self._direction_patterns[Direction.DOWN] = ChordPattern(
            name="down", keys=[directions.down], direction=Direction.DOWN
        )
        self._direction_patterns[Direction.LEFT] = ChordPattern(
            name="left", keys=[directions.left], direction=Direction.LEFT
        )
        self._direction_patterns[Direction.RIGHT] = ChordPattern(
            name="right", keys=[directions.right], direction=Direction.RIGHT
        )
        self._direction_patterns[Direction.UP_LEFT] = ChordPattern(
            name="up_left", keys=list(directions.up_left), direction=Direction.UP_LEFT
        )
        self._direction_patterns[Direction.DOWN_LEFT] = ChordPattern(
            name="down_left", keys=list(directions.down_left), direction=Direction.DOWN_LEFT
        )
        self._direction_patterns[Direction.UP_RIGHT] = ChordPattern(
            name="up_right", keys=list(directions.up_right), direction=Direction.UP_RIGHT
        )
        self._direction_patterns[Direction.DOWN_RIGHT] = ChordPattern(
            name="down_right", keys=list(directions.down_right), direction=Direction.DOWN_RIGHT
        )
        
        # Build lane patterns
        for lane_str, key_or_keys in self.keymap.lanes.items():
            try:
                lane_idx = int(lane_str)
            except ValueError:
                continue
            
            if isinstance(key_or_keys, str):
                keys = [key_or_keys]
            else:
                keys = list(key_or_keys)
            
            # Try to determine direction from keys
            direction = self._infer_direction(keys)
            
            self._lane_patterns[lane_idx] = ChordPattern(
                name=f"lane_{lane_idx}",
                keys=keys,
                direction=direction,
            )

    def _infer_direction(self, keys: list[str]) -> Direction | None:
        """Infer direction from key combination."""
        if len(keys) == 1:
            key = keys[0].lower()
            if key == "up":
                return Direction.UP
            elif key == "down":
                return Direction.DOWN
            elif key == "left":
                return Direction.LEFT
            elif key == "right":
                return Direction.RIGHT
        elif len(keys) == 2:
            key_set = {k.lower() for k in keys}
            if key_set == {"up", "left"}:
                return Direction.UP_LEFT
            elif key_set == {"down", "left"}:
                return Direction.DOWN_LEFT
            elif key_set == {"up", "right"}:
                return Direction.UP_RIGHT
            elif key_set == {"down", "right"}:
                return Direction.DOWN_RIGHT
        return None

    def get_pattern_for_lane(self, lane_index: int) -> ChordPattern | None:
        """
        Get the key pattern for a lane index.
        
        Args:
            lane_index: 0-based lane index.
        
        Returns:
            ChordPattern for the lane, or None if not mapped.
        """
        return self._lane_patterns.get(lane_index)

    def get_pattern_for_direction(self, direction: Direction | str) -> ChordPattern | None:
        """
        Get the key pattern for a direction.
        
        Args:
            direction: Direction enum or string name.
        
        Returns:
            ChordPattern for the direction, or None if not mapped.
        """
        if isinstance(direction, str):
            direction = normalize_direction(direction)
            if direction is None:
                return None
        return self._direction_patterns.get(direction)

    def get_keys_for_lane(self, lane_index: int) -> list[str]:
        """
        Get the key(s) to press for a lane.
        
        Args:
            lane_index: 0-based lane index.
        
        Returns:
            List of key names to press.
        
        Raises:
            KeyError: If lane is not mapped.
        """
        pattern = self.get_pattern_for_lane(lane_index)
        if pattern is None:
            raise KeyError(f"No pattern for lane {lane_index}")
        return pattern.keys

    def get_keys_for_direction(self, direction: Direction | str) -> list[str]:
        """
        Get the key(s) for a direction.
        
        Args:
            direction: Direction enum or string name.
        
        Returns:
            List of key names to press.
        
        Raises:
            KeyError: If direction is not mapped.
        """
        pattern = self.get_pattern_for_direction(direction)
        if pattern is None:
            raise KeyError(f"No pattern for direction {direction}")
        return pattern.keys

    def set_keymap(self, keymap: KeyMap) -> None:
        """Set or update the keymap and rebuild patterns."""
        self.keymap = keymap
        self._lane_patterns.clear()
        self._direction_patterns.clear()
        self._build_patterns()


def create_default_4lane_mapper() -> PatternMapper:
    """Create a mapper for standard 4-lane arrow key layout."""
    mapper = PatternMapper()
    
    # Direct mapping for 4 lanes
    mapper._lane_patterns = {
        0: ChordPattern("left", ["left"], Direction.LEFT),
        1: ChordPattern("down", ["down"], Direction.DOWN),
        2: ChordPattern("up", ["up"], Direction.UP),
        3: ChordPattern("right", ["right"], Direction.RIGHT),
    }
    
    # Standard direction patterns
    mapper._direction_patterns = {
        Direction.LEFT: ChordPattern("left", ["left"], Direction.LEFT),
        Direction.DOWN: ChordPattern("down", ["down"], Direction.DOWN),
        Direction.UP: ChordPattern("up", ["up"], Direction.UP),
        Direction.RIGHT: ChordPattern("right", ["right"], Direction.RIGHT),
    }
    
    return mapper


def create_default_8lane_mapper() -> PatternMapper:
    """Create a mapper for 8-lane with arrow key chords."""
    mapper = PatternMapper()
    
    # 8-lane mapping: 4 singles + 4 chords
    mapper._lane_patterns = {
        0: ChordPattern("left", ["left"], Direction.LEFT),
        1: ChordPattern("down", ["down"], Direction.DOWN),
        2: ChordPattern("up", ["up"], Direction.UP),
        3: ChordPattern("right", ["right"], Direction.RIGHT),
        4: ChordPattern("up_left", ["up", "left"], Direction.UP_LEFT),
        5: ChordPattern("down_left", ["down", "left"], Direction.DOWN_LEFT),
        6: ChordPattern("up_right", ["up", "right"], Direction.UP_RIGHT),
        7: ChordPattern("down_right", ["down", "right"], Direction.DOWN_RIGHT),
    }
    
    # All direction patterns
    mapper._direction_patterns = {
        Direction.LEFT: ChordPattern("left", ["left"], Direction.LEFT),
        Direction.DOWN: ChordPattern("down", ["down"], Direction.DOWN),
        Direction.UP: ChordPattern("up", ["up"], Direction.UP),
        Direction.RIGHT: ChordPattern("right", ["right"], Direction.RIGHT),
        Direction.UP_LEFT: ChordPattern("up_left", ["up", "left"], Direction.UP_LEFT),
        Direction.DOWN_LEFT: ChordPattern("down_left", ["down", "left"], Direction.DOWN_LEFT),
        Direction.UP_RIGHT: ChordPattern("up_right", ["up", "right"], Direction.UP_RIGHT),
        Direction.DOWN_RIGHT: ChordPattern("down_right", ["down", "right"], Direction.DOWN_RIGHT),
    }
    
    return mapper
