"""
Screen capture using MSS (cross-platform) with optional DXCam acceleration.

References:
- MSS docs: https://python-mss.readthedocs.io/
- DXCam (Windows): https://github.com/ra1nty/DXcam
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CaptureBackend(Protocol):
    """Protocol for screen capture backends."""

    def grab(self, region: tuple[int, int, int, int] | None = None) -> NDArray[np.uint8]:
        """Capture a frame from the screen."""
        ...

    def release(self) -> None:
        """Release resources."""
        ...


@dataclass
class CaptureRegion:
    """Defines a screen capture region."""
    x: int
    y: int
    width: int
    height: int
    monitor: int = 0

    def to_mss_monitor(self) -> dict[str, int]:
        """Convert to MSS monitor dict format."""
        return {
            "left": self.x,
            "top": self.y,
            "width": self.width,
            "height": self.height,
        }

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Get as (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)

    @classmethod
    def from_roi(cls, roi: list[int], monitor: int = 0) -> CaptureRegion:
        """Create from [x, y, width, height] list."""
        return cls(x=roi[0], y=roi[1], width=roi[2], height=roi[3], monitor=monitor)


@dataclass
class FrameResult:
    """Result of a frame capture operation."""
    frame: NDArray[np.uint8]
    timestamp: float
    frame_id: int
    latency_ms: float


class MSSCapture:
    """
    Screen capture backend using MSS library.
    
    MSS is cross-platform and reliable, suitable for most use cases.
    For higher FPS on Windows, consider DXCamCapture.
    """

    def __init__(self, monitor: int = 0, dpi_scale: float = 1.0):
        import mss
        self._sct = mss.mss()
        self._monitor_index = monitor + 1  # MSS uses 1-based indexing (0 = all monitors)
        self._dpi_scale = dpi_scale
        self._monitors = self._sct.monitors

    @property
    def monitor_info(self) -> dict[str, int]:
        """Get current monitor dimensions."""
        return self._monitors[self._monitor_index]

    def grab(self, region: tuple[int, int, int, int] | None = None) -> NDArray[np.uint8]:
        """
        Capture a frame from the screen.
        
        Args:
            region: Optional (x, y, width, height) to capture specific area.
                   If None, captures the entire monitor.
        
        Returns:
            BGR numpy array of the captured frame.
        """
        if region is not None:
            x, y, w, h = region
            # Apply DPI scaling
            if self._dpi_scale != 1.0:
                x = int(x * self._dpi_scale)
                y = int(y * self._dpi_scale)
                w = int(w * self._dpi_scale)
                h = int(h * self._dpi_scale)
            monitor = {"left": x, "top": y, "width": w, "height": h}
        else:
            monitor = self._monitors[self._monitor_index]

        # Grab the screen
        screenshot = self._sct.grab(monitor)
        
        # Convert to numpy array (BGRA -> BGR)
        frame = np.array(screenshot, dtype=np.uint8)
        return frame[:, :, :3]  # Drop alpha channel

    def release(self) -> None:
        """Release MSS resources."""
        self._sct.close()


class DXCamCapture:
    """
    High-performance screen capture using DXCam (Windows only).
    
    Provides significantly higher FPS than MSS on Windows by using
    DXGI Desktop Duplication API.
    """

    def __init__(self, monitor: int = 0, target_fps: int = 120):
        try:
            import dxcam
        except ImportError as e:
            raise ImportError(
                "DXCam not installed. Install with: pip install dxcam"
            ) from e

        self._camera = dxcam.create(device_idx=0, output_idx=monitor)
        self._camera.start(target_fps=target_fps, video_mode=True)
        self._target_fps = target_fps

    def grab(self, region: tuple[int, int, int, int] | None = None) -> NDArray[np.uint8]:
        """
        Capture a frame using DXCam.
        
        Args:
            region: Optional (x, y, width, height) to capture. DXCam captures
                   full screen and crops if region specified.
        
        Returns:
            BGR numpy array of the captured frame.
        """
        import cv2
        
        frame = self._camera.get_latest_frame()
        if frame is None:
            # Return black frame if capture failed
            if region:
                return np.zeros((region[3], region[2], 3), dtype=np.uint8)
            return np.zeros((1080, 1920, 3), dtype=np.uint8)

        # DXCam returns RGB, convert to BGR for OpenCV compatibility
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if region is not None:
            x, y, w, h = region
            frame = frame[y:y+h, x:x+w]

        return frame

    def release(self) -> None:
        """Stop and release DXCam resources."""
        self._camera.stop()
        del self._camera


@dataclass
class CaptureSession:
    """
    Manages a screen capture session with timing and frame tracking.
    
    Example:
        config = CaptureConfig(monitor=0, roi=[100, 200, 800, 600])
        session = CaptureSession.create(config)
        
        while running:
            result = session.capture()
            process_frame(result.frame)
        
        session.close()
    """
    backend: CaptureBackend
    region: CaptureRegion | None
    target_fps: int
    _frame_count: int = field(default=0, init=False)
    _last_capture_time: float = field(default=0.0, init=False)
    _fps_history: list[float] = field(default_factory=list, init=False)

    @classmethod
    def create(
        cls,
        monitor: int = 0,
        roi: list[int] | None = None,
        target_fps: int = 120,
        dpi_scale: float = 1.0,
        use_dxcam: bool = False,
    ) -> CaptureSession:
        """
        Create a capture session with specified configuration.
        
        Args:
            monitor: Monitor index (0 = primary).
            roi: Region of interest [x, y, width, height], or None for full screen.
            target_fps: Target frames per second.
            dpi_scale: DPI scaling factor.
            use_dxcam: Use DXCam backend (Windows only, higher performance).
        
        Returns:
            Configured CaptureSession instance.
        """
        # Choose backend
        if use_dxcam:
            backend = DXCamCapture(monitor=monitor, target_fps=target_fps)
        else:
            backend = MSSCapture(monitor=monitor, dpi_scale=dpi_scale)

        # Parse region
        region = None
        if roi and roi != [0, 0, 0, 0]:
            region = CaptureRegion.from_roi(roi, monitor)

        return cls(backend=backend, region=region, target_fps=target_fps)

    def capture(self) -> FrameResult:
        """
        Capture a single frame with timing metadata.
        
        Returns:
            FrameResult with frame data and timing information.
        """
        start_time = time.perf_counter()
        
        region_tuple = self.region.to_tuple() if self.region else None
        frame = self.backend.grab(region_tuple)
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        self._frame_count += 1
        
        # Track FPS
        if self._last_capture_time > 0:
            frame_time = end_time - self._last_capture_time
            if frame_time > 0:
                self._fps_history.append(1.0 / frame_time)
                # Keep last 60 samples for rolling average
                if len(self._fps_history) > 60:
                    self._fps_history.pop(0)
        
        self._last_capture_time = end_time

        return FrameResult(
            frame=frame,
            timestamp=end_time,
            frame_id=self._frame_count,
            latency_ms=latency_ms,
        )

    @property
    def current_fps(self) -> float:
        """Get current rolling average FPS."""
        if not self._fps_history:
            return 0.0
        return sum(self._fps_history) / len(self._fps_history)

    @property
    def frame_count(self) -> int:
        """Get total frames captured."""
        return self._frame_count

    def close(self) -> None:
        """Release capture resources."""
        self.backend.release()

    def __enter__(self) -> CaptureSession:
        return self

    def __exit__(self, *args) -> None:
        self.close()
