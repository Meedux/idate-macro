"""
Windows SendInput driver for low-latency keyboard input.

Uses ctypes to call Windows SendInput API directly, which is more
reliable for games than higher-level libraries.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable


# Windows API constants
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

# Scan codes for DirectInput compatibility (hardware-level, works in games)
SCAN_CODES: dict[str, int] = {
    # Arrow keys (extended - add 0xE000)
    "up": 0x48,
    "down": 0x50,
    "left": 0x4B,
    "right": 0x4D,
    # Common keys
    "space": 0x39,
    "enter": 0x1C,
    "escape": 0x01,
    "tab": 0x0F,
    "backspace": 0x0E,
    # Letters (QWERTY layout)
    "a": 0x1E, "b": 0x30, "c": 0x2E, "d": 0x20,
    "e": 0x12, "f": 0x21, "g": 0x22, "h": 0x23,
    "i": 0x17, "j": 0x24, "k": 0x25, "l": 0x26,
    "m": 0x32, "n": 0x31, "o": 0x18, "p": 0x19,
    "q": 0x10, "r": 0x13, "s": 0x1F, "t": 0x14,
    "u": 0x16, "v": 0x2F, "w": 0x11, "x": 0x2D,
    "y": 0x15, "z": 0x2C,
    # Numbers
    "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05,
    "5": 0x06, "6": 0x07, "7": 0x08, "8": 0x09,
    "9": 0x0A, "0": 0x0B,
}

# Extended scan codes (need KEYEVENTF_EXTENDEDKEY flag)
EXTENDED_SCAN_CODES = {"up", "down", "left", "right"}


class VK(IntEnum):
    """Virtual key codes for common keys."""
    # Arrow keys
    LEFT = 0x25
    UP = 0x26
    RIGHT = 0x27
    DOWN = 0x28
    
    # Letters
    A = 0x41
    B = 0x42
    C = 0x43
    D = 0x44
    E = 0x45
    F = 0x46
    G = 0x47
    H = 0x48
    I = 0x49
    J = 0x4A
    K = 0x4B
    L = 0x4C
    M = 0x4D
    N = 0x4E
    O = 0x4F
    P = 0x50
    Q = 0x51
    R = 0x52
    S = 0x53
    T = 0x54
    U = 0x55
    V = 0x56
    W = 0x57
    X = 0x58
    Y = 0x59
    Z = 0x5A
    
    # Numbers
    NUM_0 = 0x30
    NUM_1 = 0x31
    NUM_2 = 0x32
    NUM_3 = 0x33
    NUM_4 = 0x34
    NUM_5 = 0x35
    NUM_6 = 0x36
    NUM_7 = 0x37
    NUM_8 = 0x38
    NUM_9 = 0x39
    
    # Function keys
    F1 = 0x70
    F2 = 0x71
    F3 = 0x72
    F4 = 0x73
    F5 = 0x74
    F6 = 0x75
    F7 = 0x76
    F8 = 0x77
    F9 = 0x78
    F10 = 0x79
    F11 = 0x7A
    F12 = 0x7B
    
    # Modifiers
    SHIFT = 0x10
    CONTROL = 0x11
    ALT = 0x12
    
    # Special
    SPACE = 0x20
    ENTER = 0x0D
    ESCAPE = 0x1B
    TAB = 0x09
    BACKSPACE = 0x08


# Key name to VK code mapping
KEY_MAP: dict[str, int] = {
    # Arrow keys
    "up": VK.UP,
    "down": VK.DOWN,
    "left": VK.LEFT,
    "right": VK.RIGHT,
    
    # Letters
    "a": VK.A, "b": VK.B, "c": VK.C, "d": VK.D,
    "e": VK.E, "f": VK.F, "g": VK.G, "h": VK.H,
    "i": VK.I, "j": VK.J, "k": VK.K, "l": VK.L,
    "m": VK.M, "n": VK.N, "o": VK.O, "p": VK.P,
    "q": VK.Q, "r": VK.R, "s": VK.S, "t": VK.T,
    "u": VK.U, "v": VK.V, "w": VK.W, "x": VK.X,
    "y": VK.Y, "z": VK.Z,
    
    # Numbers
    "0": VK.NUM_0, "1": VK.NUM_1, "2": VK.NUM_2,
    "3": VK.NUM_3, "4": VK.NUM_4, "5": VK.NUM_5,
    "6": VK.NUM_6, "7": VK.NUM_7, "8": VK.NUM_8,
    "9": VK.NUM_9,
    
    # Function keys
    "f1": VK.F1, "f2": VK.F2, "f3": VK.F3,
    "f4": VK.F4, "f5": VK.F5, "f6": VK.F6,
    "f7": VK.F7, "f8": VK.F8, "f9": VK.F9,
    "f10": VK.F10, "f11": VK.F11, "f12": VK.F12,
    
    # Special
    "space": VK.SPACE,
    "enter": VK.ENTER,
    "return": VK.ENTER,
    "escape": VK.ESCAPE,
    "esc": VK.ESCAPE,
    "tab": VK.TAB,
    "backspace": VK.BACKSPACE,
    "shift": VK.SHIFT,
    "ctrl": VK.CONTROL,
    "control": VK.CONTROL,
    "alt": VK.ALT,
}

# Extended keys that need the KEYEVENTF_EXTENDEDKEY flag
EXTENDED_KEYS = {VK.UP, VK.DOWN, VK.LEFT, VK.RIGHT}


# Windows structures
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", INPUT_UNION),
    ]


# Load Windows API
try:
    user32 = ctypes.windll.user32
    SendInput = user32.SendInput
    SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    SendInput.restype = wintypes.UINT
except Exception:
    # Not on Windows
    SendInput = None
    user32 = None


def get_vk_code(key: str) -> int:
    """Get virtual key code for a key name."""
    return KEY_MAP.get(key.lower(), 0)


def _create_key_input(vk_code: int, key_up: bool = False) -> INPUT:
    """Create an INPUT structure for a key event."""
    flags = 0
    if key_up:
        flags |= KEYEVENTF_KEYUP
    if vk_code in EXTENDED_KEYS:
        flags |= KEYEVENTF_EXTENDEDKEY
    
    ki = KEYBDINPUT(
        wVk=vk_code,
        wScan=0,
        dwFlags=flags,
        time=0,
        dwExtraInfo=None,
    )
    
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = ki
    
    return inp


def key_down(key: str | int) -> bool:
    """
    Press a key down.
    
    Args:
        key: Key name (e.g., 'up', 'a') or VK code.
    
    Returns:
        True if successful.
    """
    if SendInput is None:
        return False
    
    vk = key if isinstance(key, int) else get_vk_code(key)
    if vk == 0:
        return False
    
    inp = _create_key_input(vk, key_up=False)
    result = SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    return result == 1


def key_up(key: str | int) -> bool:
    """
    Release a key.
    
    Args:
        key: Key name or VK code.
    
    Returns:
        True if successful.
    """
    if SendInput is None:
        return False
    
    vk = key if isinstance(key, int) else get_vk_code(key)
    if vk == 0:
        return False
    
    inp = _create_key_input(vk, key_up=True)
    result = SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    return result == 1


# ==================== SCAN CODE FUNCTIONS (DirectInput/Game compatible) ====================

def _create_scancode_input(scan_code: int, extended: bool, key_up: bool = False) -> INPUT:
    """Create an INPUT structure for a scan code event (hardware-level)."""
    flags = KEYEVENTF_SCANCODE
    if key_up:
        flags |= KEYEVENTF_KEYUP
    if extended:
        flags |= KEYEVENTF_EXTENDEDKEY
    
    ki = KEYBDINPUT(
        wVk=0,          # VK code = 0 when using scan codes
        wScan=scan_code,
        dwFlags=flags,
        time=0,
        dwExtraInfo=None,
    )
    
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = ki
    
    return inp


def scan_key_down(key: str) -> bool:
    """
    Press a key down using hardware scan codes (DirectInput compatible).
    
    Args:
        key: Key name (e.g., 'up', 'space').
    
    Returns:
        True if successful.
    """
    if SendInput is None:
        return False
    
    key_lower = key.lower()
    scan_code = SCAN_CODES.get(key_lower)
    if scan_code is None:
        return False
    
    extended = key_lower in EXTENDED_SCAN_CODES
    inp = _create_scancode_input(scan_code, extended, key_up=False)
    result = SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    return result == 1


def scan_key_up(key: str) -> bool:
    """
    Release a key using hardware scan codes (DirectInput compatible).
    
    Args:
        key: Key name.
    
    Returns:
        True if successful.
    """
    if SendInput is None:
        return False
    
    key_lower = key.lower()
    scan_code = SCAN_CODES.get(key_lower)
    if scan_code is None:
        return False
    
    extended = key_lower in EXTENDED_SCAN_CODES
    inp = _create_scancode_input(scan_code, extended, key_up=True)
    result = SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    return result == 1


def scan_key_press(key: str, duration_ms: float = 20) -> bool:
    """
    Press and release a key using hardware scan codes (DirectInput/game compatible).
    
    This is the recommended function for games that don't respond to VK codes.
    
    Args:
        key: Key name (e.g., 'up', 'down', 'space').
        duration_ms: How long to hold the key (milliseconds). Lower = faster.
    
    Returns:
        True if successful.
    """
    if not scan_key_down(key):
        return False
    
    if duration_ms > 0:
        time.sleep(duration_ms / 1000)
    
    return scan_key_up(key)


# ==================== END SCAN CODE FUNCTIONS ====================


def key_press(key: str | int, duration_ms: float = 30) -> bool:
    """
    Press and release a key.
    
    Args:
        key: Key name or VK code.
        duration_ms: How long to hold the key (milliseconds).
    
    Returns:
        True if successful.
    """
    if not key_down(key):
        return False
    
    if duration_ms > 0:
        time.sleep(duration_ms / 1000)
    
    return key_up(key)


def chord_press(keys: list[str | int], duration_ms: float = 30, stagger_ms: float = 0) -> bool:
    """
    Press multiple keys simultaneously (chord).
    
    Args:
        keys: List of key names or VK codes.
        duration_ms: How long to hold the keys.
        stagger_ms: Delay between each key press (0 = simultaneous).
    
    Returns:
        True if all keys were pressed successfully.
    """
    if SendInput is None:
        return False
    
    # Press all keys down
    for i, key in enumerate(keys):
        if not key_down(key):
            # Release any already-pressed keys
            for j in range(i):
                key_up(keys[j])
            return False
        
        if stagger_ms > 0 and i < len(keys) - 1:
            time.sleep(stagger_ms / 1000)
    
    # Hold
    if duration_ms > 0:
        time.sleep(duration_ms / 1000)
    
    # Release all keys (reverse order)
    success = True
    for key in reversed(keys):
        if not key_up(key):
            success = False
    
    return success


@dataclass
class ScheduledAction:
    """A scheduled key action."""
    keys: list[str]
    execute_time: float  # perf_counter timestamp
    duration_ms: float
    lane: int = -1
    action_id: int = 0
    executed: bool = False
    
    def execute(self) -> bool:
        """Execute this action."""
        if self.executed:
            return True
        
        if len(self.keys) == 1:
            result = key_press(self.keys[0], self.duration_ms)
        else:
            result = chord_press(self.keys, self.duration_ms)
        
        self.executed = True
        return result


@dataclass
class InputScheduler:
    """
    Schedules and executes key actions with precise timing.
    
    Uses a tight loop with perf_counter for sub-millisecond accuracy.
    """
    press_duration_ms: float = 30
    min_gap_ms: float = 10
    chord_stagger_ms: float = 0
    max_actions_per_second: int = 60
    
    _queue: list[ScheduledAction] = field(default_factory=list, init=False)
    _next_action_id: int = field(default=1, init=False)
    _action_count: int = field(default=0, init=False)
    _last_reset_time: float = field(default=0.0, init=False)
    _last_action_times: dict[str, float] = field(default_factory=dict, init=False)
    _on_action: Callable[[ScheduledAction], None] | None = None

    def schedule(
        self,
        keys: list[str],
        execute_time: float,
        lane: int = -1,
    ) -> int:
        """
        Schedule a key action.
        
        Args:
            keys: Keys to press (single key or chord).
            execute_time: perf_counter timestamp when to execute.
            lane: Associated lane index (for tracking).
        
        Returns:
            Action ID.
        """
        action = ScheduledAction(
            keys=keys,
            execute_time=execute_time,
            duration_ms=self.press_duration_ms,
            lane=lane,
            action_id=self._next_action_id,
        )
        self._next_action_id += 1
        self._queue.append(action)
        
        # Keep queue sorted by execute time
        self._queue.sort(key=lambda a: a.execute_time)
        
        return action.action_id

    def process(self, current_time: float | None = None) -> list[ScheduledAction]:
        """
        Process scheduled actions and execute any that are due.
        
        Args:
            current_time: Current perf_counter timestamp.
        
        Returns:
            List of actions that were executed.
        """
        if current_time is None:
            current_time = time.perf_counter()
        
        # Reset rate limiting counter every second
        if current_time - self._last_reset_time >= 1.0:
            self._action_count = 0
            self._last_reset_time = current_time
        
        executed = []
        remaining = []
        
        for action in self._queue:
            if action.executed:
                continue
            
            # Check if it's time to execute
            if action.execute_time <= current_time:
                # Rate limiting check
                if self._action_count >= self.max_actions_per_second:
                    remaining.append(action)
                    continue
                
                # Min gap check for same keys
                key_id = ",".join(sorted(action.keys))
                last_time = self._last_action_times.get(key_id, 0)
                if (current_time - last_time) * 1000 < self.min_gap_ms:
                    remaining.append(action)
                    continue
                
                # Execute
                action.execute()
                self._action_count += 1
                self._last_action_times[key_id] = current_time
                executed.append(action)
                
                # Callback
                if self._on_action:
                    self._on_action(action)
            else:
                remaining.append(action)
        
        self._queue = remaining
        return executed

    def clear(self) -> None:
        """Clear all scheduled actions."""
        self._queue.clear()

    def set_callback(self, callback: Callable[[ScheduledAction], None]) -> None:
        """Set callback for when actions are executed."""
        self._on_action = callback

    @property
    def pending_count(self) -> int:
        """Get number of pending actions."""
        return len(self._queue)


def is_available() -> bool:
    """Check if SendInput is available (Windows only)."""
    return SendInput is not None
