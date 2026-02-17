"""
Lane detection for rhythm games.

Detects lane boundaries and centers from the game screen,
either through calibration or automatic detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class Lane:
    """Represents a single lane in the rhythm game."""
    index: int
    center_x: int
    left_bound: int
    right_bound: int
    hit_line_y: int

    @property
    def width(self) -> int:
        """Get lane width in pixels."""
        return self.right_bound - self.left_bound

    def contains_x(self, x: int, tolerance: int = 0) -> bool:
        """Check if x coordinate is within this lane."""
        return (self.left_bound - tolerance) <= x <= (self.right_bound + tolerance)


@dataclass
class LaneLayout:
    """Complete lane layout configuration."""
    lanes: list[Lane]
    hit_line_y: int
    total_width: int
    total_height: int

    def get_lane_for_x(self, x: int, tolerance: int = 10) -> Lane | None:
        """Find which lane contains the given x coordinate."""
        for lane in self.lanes:
            if lane.contains_x(x, tolerance):
                return lane
        return None

    def get_lane_by_index(self, index: int) -> Lane | None:
        """Get lane by its index."""
        for lane in self.lanes:
            if lane.index == index:
                return lane
        return None

    @property
    def lane_count(self) -> int:
        """Get number of lanes."""
        return len(self.lanes)


def detect_lanes_from_config(
    lane_count: int,
    centers: list[int],
    width_tolerance: int,
    hit_line_y: int,
    frame_width: int,
    frame_height: int,
) -> LaneLayout:
    """
    Create lane layout from configuration values.
    
    Args:
        lane_count: Number of lanes.
        centers: List of lane center X coordinates.
        width_tolerance: Half-width of each lane.
        hit_line_y: Y coordinate of hit line.
        frame_width: Width of capture frame.
        frame_height: Height of capture frame.
    
    Returns:
        LaneLayout with configured lanes.
    """
    lanes = []
    for i, center_x in enumerate(centers):
        lane = Lane(
            index=i,
            center_x=center_x,
            left_bound=center_x - width_tolerance,
            right_bound=center_x + width_tolerance,
            hit_line_y=hit_line_y,
        )
        lanes.append(lane)
    
    return LaneLayout(
        lanes=lanes,
        hit_line_y=hit_line_y,
        total_width=frame_width,
        total_height=frame_height,
    )


def auto_detect_lanes(
    frame: NDArray[np.uint8],
    expected_count: int = 8,
    hit_line_ratio: float = 0.85,
) -> LaneLayout:
    """
    Automatically detect lane positions from a game frame.
    
    Uses edge detection and vertical line analysis to find lane separators.
    
    Args:
        frame: BGR image of the game screen.
        expected_count: Expected number of lanes (helps validate detection).
        hit_line_ratio: Ratio from top where hit line is expected (0.0-1.0).
    
    Returns:
        Detected LaneLayout, or estimated layout if detection fails.
    """
    height, width = frame.shape[:2]
    hit_line_y = int(height * hit_line_ratio)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Focus on a horizontal band around the hit line
    band_height = int(height * 0.1)
    band_top = max(0, hit_line_y - band_height)
    band_bottom = min(height, hit_line_y + band_height)
    band = edges[band_top:band_bottom, :]
    
    # Sum vertically to find vertical lines/separators
    vertical_sum = np.sum(band, axis=0)
    
    # Find peaks (potential lane separators)
    threshold = np.max(vertical_sum) * 0.3
    peaks = []
    in_peak = False
    peak_start = 0
    
    for i, val in enumerate(vertical_sum):
        if val > threshold and not in_peak:
            in_peak = True
            peak_start = i
        elif val <= threshold and in_peak:
            in_peak = False
            peaks.append((peak_start + i) // 2)
    
    # If we found reasonable separators, use them
    if len(peaks) >= expected_count - 1:
        # Calculate lane centers between separators
        peaks = sorted(peaks)[:expected_count + 1]
        centers = []
        for i in range(len(peaks) - 1):
            center = (peaks[i] + peaks[i + 1]) // 2
            centers.append(center)
        
        if len(centers) >= expected_count:
            centers = centers[:expected_count]
            avg_width = (peaks[-1] - peaks[0]) // expected_count
            width_tolerance = avg_width // 2
            
            return detect_lanes_from_config(
                lane_count=expected_count,
                centers=centers,
                width_tolerance=width_tolerance,
                hit_line_y=hit_line_y,
                frame_width=width,
                frame_height=height,
            )
    
    # Fallback: estimate lanes evenly across frame
    return estimate_lane_layout(
        frame_width=width,
        frame_height=height,
        lane_count=expected_count,
        hit_line_ratio=hit_line_ratio,
        margin_ratio=0.1,
    )


def estimate_lane_layout(
    frame_width: int,
    frame_height: int,
    lane_count: int = 8,
    hit_line_ratio: float = 0.85,
    margin_ratio: float = 0.1,
) -> LaneLayout:
    """
    Estimate lane positions assuming evenly distributed lanes.
    
    Args:
        frame_width: Width of the capture frame.
        frame_height: Height of the capture frame.
        lane_count: Number of lanes to create.
        hit_line_ratio: Ratio from top where hit line is (0.0-1.0).
        margin_ratio: Ratio of screen width for side margins.
    
    Returns:
        Estimated LaneLayout.
    """
    margin = int(frame_width * margin_ratio)
    playable_width = frame_width - (2 * margin)
    lane_width = playable_width // lane_count
    width_tolerance = lane_width // 2
    hit_line_y = int(frame_height * hit_line_ratio)
    
    centers = []
    for i in range(lane_count):
        center_x = margin + (i * lane_width) + (lane_width // 2)
        centers.append(center_x)
    
    return detect_lanes_from_config(
        lane_count=lane_count,
        centers=centers,
        width_tolerance=width_tolerance,
        hit_line_y=hit_line_y,
        frame_width=frame_width,
        frame_height=frame_height,
    )


def draw_lane_overlay(
    frame: NDArray[np.uint8],
    layout: LaneLayout,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> NDArray[np.uint8]:
    """
    Draw lane boundaries and hit line on a frame for visualization.
    
    Args:
        frame: BGR image to draw on.
        layout: Lane layout to visualize.
        color: BGR color for drawing.
        thickness: Line thickness.
    
    Returns:
        Frame with overlay drawn.
    """
    output = frame.copy()
    height = frame.shape[0]
    
    # Draw hit line
    cv2.line(
        output,
        (0, layout.hit_line_y),
        (layout.total_width, layout.hit_line_y),
        (0, 0, 255),
        thickness + 1,
    )
    
    # Draw lane boundaries and centers
    for lane in layout.lanes:
        # Left boundary
        cv2.line(
            output,
            (lane.left_bound, 0),
            (lane.left_bound, height),
            color,
            thickness,
        )
        # Right boundary
        cv2.line(
            output,
            (lane.right_bound, 0),
            (lane.right_bound, height),
            color,
            thickness,
        )
        # Center (dashed effect via shorter line)
        cv2.line(
            output,
            (lane.center_x, layout.hit_line_y - 20),
            (lane.center_x, layout.hit_line_y + 20),
            (255, 255, 0),
            thickness,
        )
        # Lane number
        cv2.putText(
            output,
            str(lane.index),
            (lane.center_x - 10, layout.hit_line_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    
    return output
