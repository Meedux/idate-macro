"""
Event tracing and logging for debugging.

Provides structured logging of detection and action events
for analysis and replay.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class EventType(Enum):
    """Types of traced events."""
    FRAME_CAPTURED = "frame_captured"
    NOTE_DETECTED = "note_detected"
    NOTE_TRACKED = "note_tracked"
    NOTE_LOST = "note_lost"
    ACTION_SCHEDULED = "action_scheduled"
    ACTION_EXECUTED = "action_executed"
    ACTION_MISSED = "action_missed"
    STATE_CHANGE = "state_change"
    ERROR = "error"


@dataclass
class TraceEvent:
    """A single trace event."""
    timestamp: float
    event_type: EventType
    data: dict[str, Any]
    frame_id: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "frame_id": self.frame_id,
            "data": self.data,
        }


@dataclass
class Tracer:
    """
    Event tracer for debugging and analysis.
    
    Records events during execution for later analysis or replay.
    """
    enabled: bool = True
    max_events: int = 100000
    
    _events: list[TraceEvent] = field(default_factory=list, init=False)
    _start_time: float = field(default=0.0, init=False)
    _current_frame: int = field(default=0, init=False)

    def start(self) -> None:
        """Start tracing."""
        self._start_time = time.perf_counter()
        self._events.clear()
        self._current_frame = 0

    def set_frame(self, frame_id: int) -> None:
        """Set current frame ID for event association."""
        self._current_frame = frame_id

    def trace(
        self,
        event_type: EventType,
        data: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a trace event.
        
        Args:
            event_type: Type of event.
            data: Event data.
        """
        if not self.enabled:
            return
        
        event = TraceEvent(
            timestamp=time.perf_counter() - self._start_time,
            event_type=event_type,
            data=data or {},
            frame_id=self._current_frame,
        )
        self._events.append(event)
        
        # Limit size
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events // 2:]

    def trace_frame(self, latency_ms: float) -> None:
        """Record frame capture event."""
        self.trace(EventType.FRAME_CAPTURED, {"latency_ms": latency_ms})

    def trace_detection(
        self,
        lane: int,
        x: int,
        y: int,
        confidence: float = 1.0,
    ) -> None:
        """Record note detection event."""
        self.trace(EventType.NOTE_DETECTED, {
            "lane": lane,
            "x": x,
            "y": y,
            "confidence": confidence,
        })

    def trace_action(
        self,
        lane: int,
        keys: list[str],
        scheduled_time: float,
        executed: bool,
        latency_ms: float = 0.0,
    ) -> None:
        """Record action event."""
        event_type = EventType.ACTION_EXECUTED if executed else EventType.ACTION_MISSED
        self.trace(event_type, {
            "lane": lane,
            "keys": keys,
            "scheduled_time": scheduled_time,
            "latency_ms": latency_ms,
        })

    def trace_error(self, error: str, details: dict | None = None) -> None:
        """Record error event."""
        self.trace(EventType.ERROR, {
            "error": error,
            "details": details or {},
        })

    def get_events(
        self,
        event_type: EventType | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[TraceEvent]:
        """
        Get filtered events.
        
        Args:
            event_type: Filter by event type.
            start_time: Filter by start time.
            end_time: Filter by end time.
        
        Returns:
            Filtered list of events.
        """
        events = self._events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if start_time is not None:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time is not None:
            events = [e for e in events if e.timestamp <= end_time]
        
        return events

    def save(self, path: str | Path) -> None:
        """
        Save trace to file.
        
        Args:
            path: Output file path (JSON format).
        """
        path = Path(path)
        
        data = {
            "start_time": self._start_time,
            "event_count": len(self._events),
            "events": [e.to_dict() for e in self._events],
        }
        
        with path.open("w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Tracer:
        """
        Load trace from file.
        
        Args:
            path: Input file path.
        
        Returns:
            Tracer with loaded events.
        """
        path = Path(path)
        
        with path.open() as f:
            data = json.load(f)
        
        tracer = cls()
        tracer._start_time = data.get("start_time", 0)
        
        for event_data in data.get("events", []):
            event = TraceEvent(
                timestamp=event_data["timestamp"],
                event_type=EventType(event_data["event_type"]),
                data=event_data["data"],
                frame_id=event_data.get("frame_id", 0),
            )
            tracer._events.append(event)
        
        return tracer

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of traced events."""
        summary: dict[str, Any] = {
            "total_events": len(self._events),
            "duration": self._events[-1].timestamp if self._events else 0,
            "event_counts": {},
        }
        
        for event_type in EventType:
            count = sum(1 for e in self._events if e.event_type == event_type)
            if count > 0:
                summary["event_counts"][event_type.value] = count
        
        return summary


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> None:
    """
    Configure logging for idate.
    
    Args:
        level: Logging level.
        log_file: Optional file to write logs to.
    """
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Configure root logger
    root_logger = logging.getLogger("src")
    root_logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
