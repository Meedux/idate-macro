"""
Runtime control and safety systems.

Provides emergency stop, rate limiting, focus checking, and other
safety mechanisms for the automation system.
"""

from __future__ import annotations

import ctypes
import re
import time
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Callable

from ..input import pynput_driver


@dataclass
class SafetyLimits:
    """Safety limits configuration."""
    max_actions_per_second: int = 60
    max_consecutive_actions: int = 20
    min_action_interval_ms: float = 10
    require_focus: bool = True
    window_title_pattern: str = ".*"


@dataclass
class ActionTracker:
    """Tracks actions for rate limiting."""
    _action_times: list[float] = field(default_factory=list, init=False)
    _consecutive_count: int = field(default=0, init=False)
    _last_action_time: float = field(default=0.0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)

    def record_action(self, timestamp: float | None = None) -> None:
        """Record an action occurrence."""
        if timestamp is None:
            timestamp = time.perf_counter()
        
        with self._lock:
            self._action_times.append(timestamp)
            self._consecutive_count += 1
            self._last_action_time = timestamp
            
            # Keep only last second of actions
            cutoff = timestamp - 1.0
            self._action_times = [t for t in self._action_times if t > cutoff]

    def get_actions_per_second(self) -> int:
        """Get current actions per second."""
        with self._lock:
            return len(self._action_times)

    def get_last_action_time(self) -> float:
        """Get timestamp of last action."""
        with self._lock:
            return self._last_action_time

    def reset_consecutive(self) -> None:
        """Reset consecutive action counter."""
        with self._lock:
            self._consecutive_count = 0

    def get_consecutive_count(self) -> int:
        """Get consecutive action count."""
        with self._lock:
            return self._consecutive_count

    def reset(self) -> None:
        """Reset all tracking."""
        with self._lock:
            self._action_times.clear()
            self._consecutive_count = 0
            self._last_action_time = 0.0


class FocusChecker:
    """
    Checks if the target game window is focused.
    
    Uses Windows API to get foreground window title.
    """

    def __init__(self, window_pattern: str = ".*"):
        self._pattern = re.compile(window_pattern, re.IGNORECASE)
        self._last_check_time = 0.0
        self._last_result = False
        self._cache_duration = 0.1  # Cache result for 100ms

    def set_pattern(self, pattern: str) -> None:
        """Set the window title pattern to match."""
        self._pattern = re.compile(pattern, re.IGNORECASE)

    def is_focused(self) -> bool:
        """
        Check if target window is currently focused.
        
        Returns:
            True if focused or on non-Windows platform (permissive).
        """
        current_time = time.perf_counter()
        
        # Use cached result if recent
        if current_time - self._last_check_time < self._cache_duration:
            return self._last_result
        
        self._last_check_time = current_time
        
        try:
            # Windows-specific
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            
            # Get window title
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            
            # Check pattern match
            self._last_result = bool(self._pattern.search(title))
        except Exception:
            # Not on Windows or error; be permissive
            self._last_result = True
        
        return self._last_result

    def get_foreground_title(self) -> str:
        """Get the title of the current foreground window."""
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            return buf.value
        except Exception:
            return ""


@dataclass
class SafetyController:
    """
    Central safety controller for the automation system.
    
    Manages:
    - Emergency stop (hotkey)
    - Rate limiting
    - Focus checking
    - Action tracking
    """
    limits: SafetyLimits = field(default_factory=SafetyLimits)
    
    _stop_event: Event = field(default_factory=Event, init=False)
    _pause_event: Event = field(default_factory=Event, init=False)
    _tracker: ActionTracker = field(default_factory=ActionTracker, init=False)
    _focus_checker: FocusChecker | None = field(default=None, init=False)
    _on_stop_callbacks: list[Callable[[], None]] = field(default_factory=list, init=False)
    _hotkey_registered: bool = field(default=False, init=False)

    def __post_init__(self):
        self._focus_checker = FocusChecker(self.limits.window_title_pattern)

    def start(self, stop_hotkey: str = "f12") -> None:
        """
        Start safety monitoring.
        
        Args:
            stop_hotkey: Key to trigger emergency stop.
        """
        self._stop_event.clear()
        self._pause_event.clear()
        
        # Register emergency stop hotkey
        if not self._hotkey_registered:
            pynput_driver.register_hotkey(stop_hotkey, self._on_emergency_stop)
            self._hotkey_registered = True

    def stop(self) -> None:
        """Stop safety monitoring and trigger stop event."""
        self._stop_event.set()

    def pause(self) -> None:
        """Pause automation (can be resumed)."""
        self._pause_event.set()

    def resume(self) -> None:
        """Resume automation after pause."""
        self._pause_event.clear()
        self._tracker.reset_consecutive()

    def _on_emergency_stop(self) -> None:
        """Handle emergency stop hotkey press."""
        self._stop_event.set()
        for callback in self._on_stop_callbacks:
            try:
                callback()
            except Exception:
                pass

    def add_stop_callback(self, callback: Callable[[], None]) -> None:
        """Add callback to be called on emergency stop."""
        self._on_stop_callbacks.append(callback)

    def should_stop(self) -> bool:
        """Check if automation should stop."""
        return self._stop_event.is_set()

    def is_paused(self) -> bool:
        """Check if automation is paused."""
        return self._pause_event.is_set()

    def can_execute_action(self) -> tuple[bool, str]:
        """
        Check if an action can be executed safely.
        
        Returns:
            Tuple of (allowed, reason).
        """
        # Check stop/pause
        if self._stop_event.is_set():
            return False, "stopped"
        if self._pause_event.is_set():
            return False, "paused"
        
        # Check focus
        if self.limits.require_focus and self._focus_checker:
            if not self._focus_checker.is_focused():
                return False, "window_not_focused"
        
        # Check rate limit
        if self._tracker.get_actions_per_second() >= self.limits.max_actions_per_second:
            return False, "rate_limited"
        
        # Check consecutive limit
        if self._tracker.get_consecutive_count() >= self.limits.max_consecutive_actions:
            return False, "consecutive_limit"
        
        # Check minimum interval
        last_time = self._tracker.get_last_action_time()
        if last_time > 0:
            elapsed_ms = (time.perf_counter() - last_time) * 1000
            if elapsed_ms < self.limits.min_action_interval_ms:
                return False, "min_interval"
        
        return True, ""

    def record_action(self) -> None:
        """Record that an action was executed."""
        self._tracker.record_action()

    def reset_consecutive(self) -> None:
        """Reset consecutive action counter (e.g., after idle period)."""
        self._tracker.reset_consecutive()

    def set_window_pattern(self, pattern: str) -> None:
        """Set the target window title pattern."""
        self.limits.window_title_pattern = pattern
        if self._focus_checker:
            self._focus_checker.set_pattern(pattern)

    @property
    def actions_per_second(self) -> int:
        """Get current actions per second."""
        return self._tracker.get_actions_per_second()

    @property
    def is_window_focused(self) -> bool:
        """Check if target window is focused."""
        if self._focus_checker:
            return self._focus_checker.is_focused()
        return True


class RuntimeController:
    """
    High-level runtime controller for the automation system.
    
    Coordinates capture, detection, scheduling, and execution
    with safety controls.
    """

    def __init__(
        self,
        safety: SafetyController | None = None,
        dry_run: bool = False,
    ):
        self.safety = safety or SafetyController()
        self.dry_run = dry_run
        self._running = False
        self._stats = {
            "frames_processed": 0,
            "notes_detected": 0,
            "actions_executed": 0,
            "actions_blocked": 0,
            "start_time": 0.0,
        }

    def start(self, stop_hotkey: str = "f12") -> None:
        """Start the runtime with safety monitoring."""
        self.safety.start(stop_hotkey)
        self._running = True
        self._stats["start_time"] = time.perf_counter()

    def stop(self) -> None:
        """Stop the runtime."""
        self.safety.stop()
        self._running = False

    def is_running(self) -> bool:
        """Check if runtime is active and not stopped."""
        return self._running and not self.safety.should_stop()

    def try_execute(self, execute_fn: Callable[[], bool]) -> bool:
        """
        Try to execute an action with safety checks.
        
        Args:
            execute_fn: Function to execute if allowed.
        
        Returns:
            True if action was executed.
        """
        can_exec, reason = self.safety.can_execute_action()
        
        if not can_exec:
            self._stats["actions_blocked"] += 1
            return False
        
        if self.dry_run:
            self._stats["actions_executed"] += 1
            return True
        
        result = execute_fn()
        if result:
            self.safety.record_action()
            self._stats["actions_executed"] += 1
        
        return result

    def record_frame(self) -> None:
        """Record that a frame was processed."""
        self._stats["frames_processed"] += 1

    def record_detection(self, count: int = 1) -> None:
        """Record detected notes."""
        self._stats["notes_detected"] += count

    @property
    def stats(self) -> dict:
        """Get runtime statistics."""
        stats = dict(self._stats)
        if stats["start_time"] > 0:
            stats["runtime_seconds"] = time.perf_counter() - stats["start_time"]
        return stats

    @property
    def uptime_seconds(self) -> float:
        """Get runtime duration in seconds."""
        if self._stats["start_time"] > 0:
            return time.perf_counter() - self._stats["start_time"]
        return 0.0
