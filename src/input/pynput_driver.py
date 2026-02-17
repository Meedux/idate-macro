"""
Pynput-based keyboard driver for cross-platform input.

Uses pynput library for keyboard automation. Less game-compatible than
SendInput on Windows, but works cross-platform and provides hotkey support.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Thread
from typing import Callable

from pynput.keyboard import Controller, Key, Listener, KeyCode


# Key name to pynput Key mapping
KEY_MAP: dict[str, Key | str] = {
    # Arrow keys
    "up": Key.up,
    "down": Key.down,
    "left": Key.left,
    "right": Key.right,
    
    # Special keys
    "space": Key.space,
    "enter": Key.enter,
    "return": Key.enter,
    "escape": Key.esc,
    "esc": Key.esc,
    "tab": Key.tab,
    "backspace": Key.backspace,
    "shift": Key.shift,
    "ctrl": Key.ctrl,
    "control": Key.ctrl,
    "alt": Key.alt,
    
    # Function keys
    "f1": Key.f1, "f2": Key.f2, "f3": Key.f3,
    "f4": Key.f4, "f5": Key.f5, "f6": Key.f6,
    "f7": Key.f7, "f8": Key.f8, "f9": Key.f9,
    "f10": Key.f10, "f11": Key.f11, "f12": Key.f12,
}


def get_key(key_name: str) -> Key | str:
    """
    Get pynput key for a key name.
    
    Args:
        key_name: Name of the key (e.g., 'up', 'a', 'space').
    
    Returns:
        pynput Key or string character.
    """
    key_lower = key_name.lower()
    if key_lower in KEY_MAP:
        return KEY_MAP[key_lower]
    # Single character
    if len(key_name) == 1:
        return key_name.lower()
    return key_name


class PynputDriver:
    """
    Keyboard driver using pynput library.
    
    Provides basic keyboard automation with cross-platform support.
    For better game compatibility on Windows, use SendInputDriver.
    """

    def __init__(self):
        self._controller = Controller()

    def key_down(self, key: str) -> None:
        """Press a key down."""
        k = get_key(key)
        self._controller.press(k)

    def key_up(self, key: str) -> None:
        """Release a key."""
        k = get_key(key)
        self._controller.release(k)

    def key_press(self, key: str, duration_ms: float = 30) -> None:
        """Press and release a key."""
        self.key_down(key)
        if duration_ms > 0:
            time.sleep(duration_ms / 1000)
        self.key_up(key)

    def chord_press(
        self,
        keys: list[str],
        duration_ms: float = 30,
        stagger_ms: float = 0,
    ) -> None:
        """Press multiple keys simultaneously."""
        # Press all keys
        for i, key in enumerate(keys):
            self.key_down(key)
            if stagger_ms > 0 and i < len(keys) - 1:
                time.sleep(stagger_ms / 1000)
        
        # Hold
        if duration_ms > 0:
            time.sleep(duration_ms / 1000)
        
        # Release (reverse order)
        for key in reversed(keys):
            self.key_up(key)

    def type_string(self, text: str, interval_ms: float = 50) -> None:
        """Type a string of text."""
        for char in text:
            self._controller.type(char)
            if interval_ms > 0:
                time.sleep(interval_ms / 1000)


@dataclass
class HotkeyManager:
    """
    Manages global hotkeys using pynput listener.
    
    Example:
        manager = HotkeyManager()
        manager.register('f12', on_stop)
        manager.register('f11', on_pause)
        manager.start()
        
        # Later...
        manager.stop()
    """
    _hotkeys: dict[str, Callable[[], None]] = field(default_factory=dict, init=False)
    _listener: Listener | None = field(default=None, init=False)
    _running: bool = field(default=False, init=False)

    def register(self, key: str, callback: Callable[[], None]) -> None:
        """
        Register a hotkey callback.
        
        Args:
            key: Key name (e.g., 'f12').
            callback: Function to call when key is pressed.
        """
        self._hotkeys[key.lower()] = callback

    def unregister(self, key: str) -> None:
        """Unregister a hotkey."""
        self._hotkeys.pop(key.lower(), None)

    def start(self) -> None:
        """Start listening for hotkeys."""
        if self._running:
            return
        
        self._running = True
        self._listener = Listener(on_press=self._on_key_press)
        self._listener.start()

    def stop(self) -> None:
        """Stop listening for hotkeys."""
        self._running = False
        if self._listener:
            self._listener.stop()
            self._listener = None

    def _on_key_press(self, key) -> None:
        """Handle key press events."""
        try:
            # Get key name
            if hasattr(key, 'char') and key.char:
                key_name = key.char.lower()
            elif hasattr(key, 'name'):
                key_name = key.name.lower()
            else:
                return
            
            # Check for registered hotkey
            callback = self._hotkeys.get(key_name)
            if callback:
                # Run callback in separate thread to avoid blocking
                Thread(target=callback, daemon=True).start()
        except Exception:
            pass  # Ignore errors in hotkey handling

    def __enter__(self) -> HotkeyManager:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


# Global instances for convenience
_driver: PynputDriver | None = None
_hotkey_manager: HotkeyManager | None = None


def get_driver() -> PynputDriver:
    """Get or create the global pynput driver."""
    global _driver
    if _driver is None:
        _driver = PynputDriver()
    return _driver


def get_hotkey_manager() -> HotkeyManager:
    """Get or create the global hotkey manager."""
    global _hotkey_manager
    if _hotkey_manager is None:
        _hotkey_manager = HotkeyManager()
    return _hotkey_manager


def key_press(key: str, duration_ms: float = 30) -> None:
    """Press and release a key using the global driver."""
    get_driver().key_press(key, duration_ms)


def chord_press(keys: list[str], duration_ms: float = 30) -> None:
    """Press multiple keys using the global driver."""
    get_driver().chord_press(keys, duration_ms)


def register_hotkey(key: str, callback: Callable[[], None]) -> None:
    """Register a global hotkey."""
    manager = get_hotkey_manager()
    manager.register(key, callback)
    if not manager._running:
        manager.start()
