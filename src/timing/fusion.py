"""
Timing fusion module for combining CV detection with beatmap timing.

Fuses visual note detection with predetermined timing data to improve
accuracy and reduce false positives.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..vision.note_tracker import TrackedNote

from .map_parser import NoteEvent, ParsedBeatmap


@dataclass
class FusedNote:
    """A note with both visual and timing information fused."""
    lane: int
    predicted_hit_time: float  # Timestamp when note should be hit
    confidence: float          # Fusion confidence (0.0 - 1.0)
    
    # Source information
    visual_note: TrackedNote | None = None
    timing_event: NoteEvent | None = None
    
    # Flags
    from_visual: bool = False
    from_timing: bool = False
    executed: bool = False

    @property
    def is_fused(self) -> bool:
        """Check if note has both visual and timing data."""
        return self.from_visual and self.from_timing


@dataclass
class FusionConfig:
    """Configuration for timing fusion."""
    # Maximum time difference for matching visual and timing notes (ms)
    max_time_diff_ms: float = 100
    
    # Confidence weights
    visual_weight: float = 0.6
    timing_weight: float = 0.4
    
    # Whether to require visual confirmation for timing events
    require_visual: bool = False
    
    # Whether to use timing-only prediction when visual is unavailable
    allow_timing_only: bool = True
    
    # Lookahead time for timing predictions (ms)
    lookahead_ms: float = 500


@dataclass
class TimingFusion:
    """
    Fuses CV-detected notes with beatmap timing data.
    
    This improves detection reliability by:
    1. Validating visual detections against expected timing
    2. Predicting notes that may be missed visually
    3. Reducing false positives from visual noise
    """
    beatmap: ParsedBeatmap | None = None
    config: FusionConfig = field(default_factory=FusionConfig)
    
    # Playback state
    _playback_start_time: float = field(default=0.0, init=False)
    _playback_offset_ms: float = field(default=0.0, init=False)
    _is_playing: bool = field(default=False, init=False)
    
    # Tracking
    _processed_timing_indices: set[int] = field(default_factory=set, init=False)
    _fused_notes: list[FusedNote] = field(default_factory=list, init=False)

    def set_beatmap(self, beatmap: ParsedBeatmap) -> None:
        """Set the beatmap for timing data."""
        self.beatmap = beatmap
        self._processed_timing_indices.clear()

    def start_playback(self, offset_ms: float = 0) -> None:
        """
        Start timing playback.
        
        Call this when the song/game starts to sync timing.
        
        Args:
            offset_ms: Additional offset to add to beatmap timing.
        """
        self._playback_start_time = time.perf_counter()
        self._playback_offset_ms = offset_ms
        self._is_playing = True
        self._processed_timing_indices.clear()
        self._fused_notes.clear()

    def stop_playback(self) -> None:
        """Stop timing playback."""
        self._is_playing = False

    def get_current_song_time_ms(self) -> float:
        """Get current time in the song (milliseconds)."""
        if not self._is_playing:
            return 0.0
        
        elapsed = time.perf_counter() - self._playback_start_time
        return (elapsed * 1000) + self._playback_offset_ms

    def fuse(
        self,
        visual_notes: list[TrackedNote],
        current_time: float | None = None,
    ) -> list[FusedNote]:
        """
        Fuse visual detections with timing data.
        
        Args:
            visual_notes: Notes detected by CV.
            current_time: Current timestamp (uses perf_counter if None).
        
        Returns:
            List of fused notes ready for execution.
        """
        if current_time is None:
            current_time = time.perf_counter()
        
        fused = []
        
        # Get timing events in current window
        timing_events = self._get_upcoming_timing_events()
        
        # Track matched items
        matched_visual = set()
        matched_timing = set()
        
        # First pass: match visual notes to timing events
        for v_note in visual_notes:
            if v_note.hit_executed:
                continue
            
            best_match = None
            best_diff = float("inf")
            
            for i, t_event in enumerate(timing_events):
                if i in matched_timing:
                    continue
                if t_event.lane != v_note.note.lane_index:
                    continue
                
                # Compare predicted times
                v_time_ms = v_note.predicted_hit_time * 1000
                t_time_ms = t_event.time_ms + self._playback_offset_ms
                
                diff = abs(v_time_ms - t_time_ms)
                if diff < best_diff and diff <= self.config.max_time_diff_ms:
                    best_match = i
                    best_diff = diff
            
            if best_match is not None:
                # Create fused note
                t_event = timing_events[best_match]
                matched_visual.add(id(v_note))
                matched_timing.add(best_match)
                
                # Calculate fused hit time (weighted average)
                v_hit_time = v_note.predicted_hit_time
                t_hit_time = self._timing_to_timestamp(t_event.time_ms)
                
                fused_time = (
                    v_hit_time * self.config.visual_weight +
                    t_hit_time * self.config.timing_weight
                )
                
                # Higher confidence for fused notes
                confidence = min(1.0, 0.7 + (1 - best_diff / self.config.max_time_diff_ms) * 0.3)
                
                fused_note = FusedNote(
                    lane=t_event.lane,
                    predicted_hit_time=fused_time,
                    confidence=confidence,
                    visual_note=v_note,
                    timing_event=t_event,
                    from_visual=True,
                    from_timing=True,
                )
                fused.append(fused_note)
            else:
                # Visual-only note
                if not self.config.require_visual or True:  # Always include visual
                    fused_note = FusedNote(
                        lane=v_note.note.lane_index,
                        predicted_hit_time=v_note.predicted_hit_time,
                        confidence=0.6,  # Lower confidence for visual-only
                        visual_note=v_note,
                        from_visual=True,
                        from_timing=False,
                    )
                    fused.append(fused_note)
        
        # Second pass: add timing-only events if allowed
        if self.config.allow_timing_only and self.beatmap:
            for i, t_event in enumerate(timing_events):
                if i in matched_timing:
                    continue
                
                # Check if this timing event should be included
                t_hit_time = self._timing_to_timestamp(t_event.time_ms)
                time_until = (t_hit_time - current_time) * 1000
                
                if 0 <= time_until <= self.config.lookahead_ms:
                    fused_note = FusedNote(
                        lane=t_event.lane,
                        predicted_hit_time=t_hit_time,
                        confidence=0.5,  # Lower confidence for timing-only
                        timing_event=t_event,
                        from_visual=False,
                        from_timing=True,
                    )
                    fused.append(fused_note)
        
        # Sort by hit time
        fused.sort(key=lambda n: n.predicted_hit_time)
        
        return fused

    def _get_upcoming_timing_events(self) -> list[NoteEvent]:
        """Get timing events in the current lookahead window."""
        if not self.beatmap or not self._is_playing:
            return []
        
        current_ms = self.get_current_song_time_ms()
        window_end = current_ms + self.config.lookahead_ms
        
        events = []
        for i, note in enumerate(self.beatmap.notes):
            if note.time_ms < current_ms - 100:  # Allow some past events
                continue
            if note.time_ms > window_end:
                break
            events.append(note)
        
        return events

    def _timing_to_timestamp(self, song_time_ms: float) -> float:
        """Convert song time (ms) to wall-clock timestamp."""
        if not self._is_playing:
            return time.perf_counter()
        
        time_from_start = (song_time_ms - self._playback_offset_ms) / 1000
        return self._playback_start_time + time_from_start


class FusionPredictor:
    """
    High-level predictor combining visual and timing-based detection.
    
    Example usage:
        predictor = FusionPredictor(beatmap, config)
        predictor.start()
        
        while running:
            visual_notes = tracker.get_actionable_notes()
            actions = predictor.predict(visual_notes)
            
            for action in actions:
                executor.schedule(action.lane, action.predicted_hit_time)
    """

    def __init__(
        self,
        beatmap: ParsedBeatmap | None = None,
        config: FusionConfig | None = None,
    ):
        self.config = config or FusionConfig()
        self.fusion = TimingFusion(beatmap=beatmap, config=self.config)
        self._active = False

    def set_beatmap(self, beatmap: ParsedBeatmap) -> None:
        """Set beatmap for timing data."""
        self.fusion.set_beatmap(beatmap)

    def start(self, offset_ms: float = 0) -> None:
        """Start prediction with optional timing offset."""
        self.fusion.start_playback(offset_ms)
        self._active = True

    def stop(self) -> None:
        """Stop prediction."""
        self.fusion.stop_playback()
        self._active = False

    def predict(
        self,
        visual_notes: list[TrackedNote],
        current_time: float | None = None,
    ) -> list[FusedNote]:
        """
        Get predicted notes that should be acted upon.
        
        Args:
            visual_notes: Currently tracked visual notes.
            current_time: Current timestamp.
        
        Returns:
            List of notes to act on, sorted by hit time.
        """
        if not self._active:
            # Return visual-only predictions
            fused = []
            for v_note in visual_notes:
                if not v_note.hit_executed:
                    fused.append(FusedNote(
                        lane=v_note.note.lane_index,
                        predicted_hit_time=v_note.predicted_hit_time,
                        confidence=0.6,
                        visual_note=v_note,
                        from_visual=True,
                    ))
            return sorted(fused, key=lambda n: n.predicted_hit_time)
        
        return self.fusion.fuse(visual_notes, current_time)

    @property
    def current_song_time_ms(self) -> float:
        """Get current song time in milliseconds."""
        return self.fusion.get_current_song_time_ms()
