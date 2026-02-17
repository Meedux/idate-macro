"""
Note tracking and detection for rhythm games.

Detects notes in each lane and tracks them as they approach the hit line.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .lane_detector import Lane, LaneLayout


class NoteState(Enum):
    """State of a detected note."""
    APPROACHING = "approaching"  # Note is moving toward hit line
    AT_HIT_LINE = "at_hit_line"  # Note is at or near hit line
    PASSED = "passed"            # Note has passed the hit line
    UNKNOWN = "unknown"


@dataclass
class DetectedNote:
    """A note detected in the game screen."""
    lane_index: int
    center_x: int
    center_y: int
    width: int
    height: int
    timestamp: float
    confidence: float = 1.0
    note_id: int = 0
    state: NoteState = NoteState.APPROACHING

    @property
    def bottom_y(self) -> int:
        """Get Y coordinate of note's bottom edge."""
        return self.center_y + (self.height // 2)

    @property
    def top_y(self) -> int:
        """Get Y coordinate of note's top edge."""
        return self.center_y - (self.height // 2)


@dataclass
class TrackedNote:
    """A note being tracked across multiple frames."""
    note: DetectedNote
    first_seen: float
    last_seen: float
    frames_seen: int = 1
    velocity_y: float = 0.0  # pixels per second (positive = moving down)
    predicted_hit_time: float = 0.0
    hit_executed: bool = False

    def update(self, new_detection: DetectedNote, pixels_per_ms: float) -> None:
        """Update tracking with new detection."""
        dt = new_detection.timestamp - self.last_seen
        if dt > 0:
            dy = new_detection.center_y - self.note.center_y
            self.velocity_y = dy / dt  # pixels per second
        
        self.note = new_detection
        self.last_seen = new_detection.timestamp
        self.frames_seen += 1


@dataclass
class NoteTracker:
    """
    Tracks notes across frames and predicts hit times.
    
    Uses color-based or template-based detection to find notes,
    then tracks them to estimate when they'll reach the hit line.
    """
    layout: LaneLayout
    detection_method: str = "color"
    note_colors: list[list[int]] = field(default_factory=list)
    min_note_area: int = 200
    max_note_area: int = 10000
    template_threshold: float = 0.7
    pixels_per_ms: float = 0.5
    hit_window_ms: int = 50
    
    _tracked_notes: dict[int, TrackedNote] = field(default_factory=dict, init=False)
    _next_note_id: int = field(default=1, init=False)
    _templates: list[NDArray[np.uint8]] = field(default_factory=list, init=False)

    def detect_notes(
        self,
        frame: NDArray[np.uint8],
        timestamp: float | None = None,
    ) -> list[DetectedNote]:
        """
        Detect all notes in the current frame.
        
        Args:
            frame: BGR image to analyze.
            timestamp: Frame timestamp (uses current time if None).
        
        Returns:
            List of detected notes.
        """
        if timestamp is None:
            timestamp = time.perf_counter()

        if self.detection_method == "color":
            return self._detect_by_color(frame, timestamp)
        elif self.detection_method == "contour":
            return self._detect_by_contour(frame, timestamp)
        elif self.detection_method == "template":
            return self._detect_by_template(frame, timestamp)
        else:
            return self._detect_by_color(frame, timestamp)

    def _detect_by_color(
        self,
        frame: NDArray[np.uint8],
        timestamp: float,
    ) -> list[DetectedNote]:
        """Detect notes using color thresholding."""
        notes = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for color_range in self.note_colors:
            if len(color_range) != 6:
                continue
            lower = np.array(color_range[:3])
            upper = np.array(color_range[3:])
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Find contours in combined mask
        contours, _ = cv2.findContours(
            combined_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_note_area <= area <= self.max_note_area):
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Determine which lane this note belongs to
            lane = self.layout.get_lane_for_x(center_x)
            if lane is None:
                continue
            
            # Calculate state based on position relative to hit line
            state = self._calculate_note_state(center_y)
            
            note = DetectedNote(
                lane_index=lane.index,
                center_x=center_x,
                center_y=center_y,
                width=w,
                height=h,
                timestamp=timestamp,
                note_id=self._next_note_id,
                state=state,
            )
            self._next_note_id += 1
            notes.append(note)
        
        return notes

    def _detect_by_contour(
        self,
        frame: NDArray[np.uint8],
        timestamp: float,
    ) -> list[DetectedNote]:
        """Detect notes using general contour analysis."""
        notes = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Adaptive threshold for varying lighting
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_note_area <= area <= self.max_note_area):
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            lane = self.layout.get_lane_for_x(center_x)
            if lane is None:
                continue
            
            state = self._calculate_note_state(center_y)
            
            note = DetectedNote(
                lane_index=lane.index,
                center_x=center_x,
                center_y=center_y,
                width=w,
                height=h,
                timestamp=timestamp,
                note_id=self._next_note_id,
                state=state,
            )
            self._next_note_id += 1
            notes.append(note)
        
        return notes

    def _detect_by_template(
        self,
        frame: NDArray[np.uint8],
        timestamp: float,
    ) -> list[DetectedNote]:
        """Detect notes using template matching."""
        notes = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for template in self._templates:
            template_gray = (
                cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                if len(template.shape) == 3
                else template
            )
            
            result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= self.template_threshold)
            
            th, tw = template_gray.shape[:2]
            
            for pt in zip(*locations[::-1]):
                center_x = pt[0] + tw // 2
                center_y = pt[1] + th // 2
                
                lane = self.layout.get_lane_for_x(center_x)
                if lane is None:
                    continue
                
                state = self._calculate_note_state(center_y)
                
                note = DetectedNote(
                    lane_index=lane.index,
                    center_x=center_x,
                    center_y=center_y,
                    width=tw,
                    height=th,
                    timestamp=timestamp,
                    confidence=result[pt[1], pt[0]],
                    note_id=self._next_note_id,
                    state=state,
                )
                self._next_note_id += 1
                notes.append(note)
        
        return notes

    def _calculate_note_state(self, center_y: int) -> NoteState:
        """Determine note state based on Y position."""
        hit_y = self.layout.hit_line_y
        tolerance = self.hit_window_ms * self.pixels_per_ms  # Convert to pixels
        
        if center_y < hit_y - tolerance:
            return NoteState.APPROACHING
        elif center_y > hit_y + tolerance:
            return NoteState.PASSED
        else:
            return NoteState.AT_HIT_LINE

    def update_tracking(
        self,
        detections: list[DetectedNote],
    ) -> list[TrackedNote]:
        """
        Update tracked notes with new detections.
        
        Matches new detections to existing tracks and creates new tracks
        for unmatched detections.
        
        Args:
            detections: New note detections from current frame.
        
        Returns:
            List of currently tracked notes.
        """
        # Simple matching: associate by lane and proximity
        matched_tracks = set()
        matched_detections = set()
        
        for det in detections:
            best_match = None
            best_distance = float("inf")
            
            for track_id, track in self._tracked_notes.items():
                if track_id in matched_tracks:
                    continue
                if track.note.lane_index != det.lane_index:
                    continue
                
                # Calculate distance
                dy = abs(track.note.center_y - det.center_y)
                if dy < best_distance and dy < 100:  # Max matching distance
                    best_match = track_id
                    best_distance = dy
            
            if best_match is not None:
                # Update existing track
                self._tracked_notes[best_match].update(det, self.pixels_per_ms)
                matched_tracks.add(best_match)
                matched_detections.add(det.note_id)
            else:
                # Create new track
                track = TrackedNote(
                    note=det,
                    first_seen=det.timestamp,
                    last_seen=det.timestamp,
                )
                self._tracked_notes[det.note_id] = track
        
        # Remove stale tracks (not seen for too long)
        current_time = time.perf_counter()
        stale_threshold = 0.5  # 500ms
        stale_ids = [
            tid for tid, track in self._tracked_notes.items()
            if current_time - track.last_seen > stale_threshold
        ]
        for tid in stale_ids:
            del self._tracked_notes[tid]
        
        return list(self._tracked_notes.values())

    def estimate_hit_time(
        self,
        track: TrackedNote,
        reaction_offset_ms: float = 0,
    ) -> float:
        """
        Estimate when a note will reach the hit line.
        
        Args:
            track: Tracked note to estimate.
            reaction_offset_ms: Offset to compensate for reaction time.
        
        Returns:
            Estimated timestamp when note should be hit.
        """
        if track.velocity_y <= 0:
            # Note not moving down or moving up; use pixel-based estimate
            distance = self.layout.hit_line_y - track.note.center_y
            time_to_hit_ms = distance / self.pixels_per_ms
        else:
            # Use measured velocity
            distance = self.layout.hit_line_y - track.note.center_y
            time_to_hit_ms = (distance / track.velocity_y) * 1000
        
        hit_time = track.last_seen + (time_to_hit_ms / 1000) - (reaction_offset_ms / 1000)
        track.predicted_hit_time = hit_time
        return hit_time

    def get_actionable_notes(
        self,
        current_time: float | None = None,
        lookahead_ms: float = 100,
    ) -> list[TrackedNote]:
        """
        Get notes that should be acted upon soon.
        
        Args:
            current_time: Current timestamp (uses perf_counter if None).
            lookahead_ms: How far ahead to look for actionable notes.
        
        Returns:
            List of notes ready to be hit, sorted by predicted hit time.
        """
        if current_time is None:
            current_time = time.perf_counter()
        
        actionable = []
        for track in self._tracked_notes.values():
            if track.hit_executed:
                continue
            if track.note.state == NoteState.PASSED:
                continue
            
            hit_time = self.estimate_hit_time(track)
            time_until_hit = (hit_time - current_time) * 1000  # to ms
            
            if -self.hit_window_ms <= time_until_hit <= lookahead_ms:
                actionable.append(track)
        
        return sorted(actionable, key=lambda t: t.predicted_hit_time)

    def add_template(self, template: NDArray[np.uint8]) -> None:
        """Add a template for template-based detection."""
        self._templates.append(template)

    def clear_templates(self) -> None:
        """Clear all templates."""
        self._templates.clear()

    def reset_tracking(self) -> None:
        """Clear all tracked notes."""
        self._tracked_notes.clear()


def draw_detections(
    frame: NDArray[np.uint8],
    notes: list[DetectedNote],
    color_map: dict[NoteState, tuple[int, int, int]] | None = None,
) -> NDArray[np.uint8]:
    """
    Draw detected notes on a frame for visualization.
    
    Args:
        frame: BGR image to draw on.
        notes: List of detected notes.
        color_map: Map of note states to BGR colors.
    
    Returns:
        Frame with detections drawn.
    """
    if color_map is None:
        color_map = {
            NoteState.APPROACHING: (0, 255, 0),      # Green
            NoteState.AT_HIT_LINE: (0, 255, 255),    # Yellow
            NoteState.PASSED: (0, 0, 255),           # Red
            NoteState.UNKNOWN: (128, 128, 128),      # Gray
        }
    
    output = frame.copy()
    
    for note in notes:
        color = color_map.get(note.state, (255, 255, 255))
        
        # Draw bounding box
        x1 = note.center_x - note.width // 2
        y1 = note.center_y - note.height // 2
        x2 = note.center_x + note.width // 2
        y2 = note.center_y + note.height // 2
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Draw lane label
        label = f"L{note.lane_index}"
        cv2.putText(
            output, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )
    
    return output
