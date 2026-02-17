"""
Debug overlay for visualizing detection and timing.

Provides OpenCV-based visualization of lane detection, note tracking,
and timing information for debugging and calibration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ..vision.lane_detector import LaneLayout
    from ..vision.note_tracker import DetectedNote, TrackedNote


@dataclass
class OverlayConfig:
    """Configuration for debug overlay."""
    show_lanes: bool = True
    show_notes: bool = True
    show_timing: bool = True
    show_stats: bool = True
    show_hit_line: bool = True
    
    lane_color: tuple[int, int, int] = (0, 255, 0)      # Green
    note_color: tuple[int, int, int] = (0, 255, 255)    # Yellow
    hit_line_color: tuple[int, int, int] = (0, 0, 255)  # Red
    text_color: tuple[int, int, int] = (255, 255, 255)  # White
    
    line_thickness: int = 2
    font_scale: float = 0.5


@dataclass
class PerformanceStats:
    """Performance statistics for overlay display."""
    capture_latency_ms: float = 0.0
    detection_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    fps: float = 0.0
    notes_detected: int = 0
    actions_executed: int = 0
    
    _fps_samples: list[float] = field(default_factory=list, init=False)
    _last_frame_time: float = field(default=0.0, init=False)

    def update_fps(self, current_time: float | None = None) -> None:
        """Update FPS calculation."""
        if current_time is None:
            current_time = time.perf_counter()
        
        if self._last_frame_time > 0:
            dt = current_time - self._last_frame_time
            if dt > 0:
                self._fps_samples.append(1.0 / dt)
                if len(self._fps_samples) > 30:
                    self._fps_samples.pop(0)
                self.fps = sum(self._fps_samples) / len(self._fps_samples)
        
        self._last_frame_time = current_time


class DebugOverlay:
    """
    Debug overlay renderer for visualization.
    
    Draws lane boundaries, detected notes, timing information,
    and performance statistics on frames.
    """

    def __init__(self, config: OverlayConfig | None = None):
        self.config = config or OverlayConfig()
        self.stats = PerformanceStats()

    def render(
        self,
        frame: NDArray[np.uint8],
        layout: LaneLayout | None = None,
        notes: list[DetectedNote] | None = None,
        tracked: list[TrackedNote] | None = None,
    ) -> NDArray[np.uint8]:
        """
        Render debug overlay on a frame.
        
        Args:
            frame: BGR image to draw on.
            layout: Lane layout to visualize.
            notes: Detected notes to draw.
            tracked: Tracked notes with timing info.
        
        Returns:
            Frame with overlay rendered.
        """
        output = frame.copy()
        
        # Draw lanes
        if self.config.show_lanes and layout:
            output = self._draw_lanes(output, layout)
        
        # Draw hit line
        if self.config.show_hit_line and layout:
            output = self._draw_hit_line(output, layout)
        
        # Draw notes
        if self.config.show_notes and notes:
            output = self._draw_notes(output, notes)
        
        # Draw timing predictions
        if self.config.show_timing and tracked:
            output = self._draw_timing(output, tracked)
        
        # Draw stats
        if self.config.show_stats:
            output = self._draw_stats(output)
        
        return output

    def _draw_lanes(
        self,
        frame: NDArray[np.uint8],
        layout: LaneLayout,
    ) -> NDArray[np.uint8]:
        """Draw lane boundaries."""
        height = frame.shape[0]
        color = self.config.lane_color
        thickness = self.config.line_thickness
        
        for lane in layout.lanes:
            # Lane boundaries
            cv2.line(frame, (lane.left_bound, 0), (lane.left_bound, height), color, 1)
            cv2.line(frame, (lane.right_bound, 0), (lane.right_bound, height), color, 1)
            
            # Lane number
            cv2.putText(
                frame,
                str(lane.index),
                (lane.center_x - 5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                self.config.text_color,
                1,
            )
        
        return frame

    def _draw_hit_line(
        self,
        frame: NDArray[np.uint8],
        layout: LaneLayout,
    ) -> NDArray[np.uint8]:
        """Draw the hit line."""
        width = frame.shape[1]
        cv2.line(
            frame,
            (0, layout.hit_line_y),
            (width, layout.hit_line_y),
            self.config.hit_line_color,
            self.config.line_thickness + 1,
        )
        
        # Label
        cv2.putText(
            frame,
            "HIT",
            (5, layout.hit_line_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            self.config.hit_line_color,
            1,
        )
        
        return frame

    def _draw_notes(
        self,
        frame: NDArray[np.uint8],
        notes: list[DetectedNote],
    ) -> NDArray[np.uint8]:
        """Draw detected notes."""
        for note in notes:
            x1 = note.center_x - note.width // 2
            y1 = note.center_y - note.height // 2
            x2 = note.center_x + note.width // 2
            y2 = note.center_y + note.height // 2
            
            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                self.config.note_color,
                self.config.line_thickness,
            )
            
            # Lane label
            label = f"L{note.lane_index}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale * 0.8,
                self.config.note_color,
                1,
            )
        
        return frame

    def _draw_timing(
        self,
        frame: NDArray[np.uint8],
        tracked: list[TrackedNote],
    ) -> NDArray[np.uint8]:
        """Draw timing predictions for tracked notes."""
        current_time = time.perf_counter()
        
        for track in tracked:
            note = track.note
            
            # Calculate time until hit
            time_until_ms = (track.predicted_hit_time - current_time) * 1000
            
            # Color based on timing
            if time_until_ms < 0:
                color = (0, 0, 255)  # Red - late
            elif time_until_ms < 50:
                color = (0, 255, 255)  # Yellow - soon
            else:
                color = (0, 255, 0)  # Green - approaching
            
            # Draw timing text
            timing_text = f"{time_until_ms:.0f}ms"
            cv2.putText(
                frame,
                timing_text,
                (note.center_x - 20, note.center_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                color,
                1,
            )
            
            # Draw velocity indicator if available
            if track.velocity_y > 0:
                end_y = int(note.center_y + track.velocity_y * 0.1)
                cv2.arrowedLine(
                    frame,
                    (note.center_x, note.center_y),
                    (note.center_x, end_y),
                    color,
                    1,
                )
        
        return frame

    def _draw_stats(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Draw performance statistics."""
        y_offset = frame.shape[0] - 100
        
        stats_lines = [
            f"FPS: {self.stats.fps:.1f}",
            f"Capture: {self.stats.capture_latency_ms:.1f}ms",
            f"Detect: {self.stats.detection_latency_ms:.1f}ms",
            f"Total: {self.stats.total_latency_ms:.1f}ms",
            f"Notes: {self.stats.notes_detected}",
            f"Actions: {self.stats.actions_executed}",
        ]
        
        for i, line in enumerate(stats_lines):
            cv2.putText(
                frame,
                line,
                (10, y_offset + i * 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale * 0.8,
                self.config.text_color,
                1,
            )
        
        return frame

    def show(
        self,
        frame: NDArray[np.uint8],
        window_name: str = "idate Debug",
    ) -> int:
        """
        Show the frame in an OpenCV window.
        
        Args:
            frame: Frame to display.
            window_name: Window title.
        
        Returns:
            Key code pressed, or -1 if none.
        """
        cv2.imshow(window_name, frame)
        return cv2.waitKey(1)

    def close(self) -> None:
        """Close all OpenCV windows."""
        cv2.destroyAllWindows()


class FrameRecorder:
    """
    Records frames for debugging and analysis.
    
    Saves frames with detections to disk for later review.
    """

    def __init__(self, output_dir: str = "debug_frames"):
        import os
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._frame_count = 0

    def save(
        self,
        frame: NDArray[np.uint8],
        prefix: str = "frame",
        suffix: str = "",
    ) -> str:
        """
        Save a frame to disk.
        
        Args:
            frame: Frame to save.
            prefix: Filename prefix.
            suffix: Additional suffix before extension.
        
        Returns:
            Path to saved file.
        """
        import os
        
        timestamp = int(time.time() * 1000)
        filename = f"{prefix}_{self._frame_count:06d}_{timestamp}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".png"
        
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, frame)
        
        self._frame_count += 1
        return path

    def save_with_overlay(
        self,
        frame: NDArray[np.uint8],
        overlay: DebugOverlay,
        layout: LaneLayout | None = None,
        notes: list[DetectedNote] | None = None,
    ) -> str:
        """Save frame with debug overlay rendered."""
        rendered = overlay.render(frame, layout, notes)
        return self.save(rendered, prefix="debug")
