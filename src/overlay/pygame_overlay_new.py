"""
Real-time Transparent Overlay for iDate Revival (Pygame version)
================================================================

Uses Pygame with layered window for transparent overlay.
Cross-platform and easy to use.
"""

from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass
from typing import Callable

import pygame

# Windows layered window setup
try:
    import win32gui
    import win32con
    import win32api
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False


# Colors (RGB)
COLORS = {
    'up': (0, 255, 0),        # Green
    'down': (255, 165, 0),    # Orange
    'left': (255, 255, 0),    # Yellow
    'right': (255, 0, 255),   # Magenta
    'hand': (255, 140, 0),    # Dark orange
    'indicator': (0, 255, 255), # Cyan
}

TRANSPARENT_COLOR = (0, 0, 0)  # Black = transparent


@dataclass
class Detection:
    """Single detection for overlay."""
    class_name: str
    x: int
    y: int
    width: int
    height: int
    confidence: float


class PygameOverlay:
    """
    High-performance transparent overlay using Pygame.
    
    Features:
    - 60+ FPS update rate
    - Click-through on Windows
    - Always on top
    """
    
    def __init__(self, width: int = 0, height: int = 0):
        """
        Initialize overlay.
        
        Args:
            width: Overlay width (0 = screen width)
            height: Overlay height (0 = screen height)
        """
        # Initialize pygame
        pygame.init()
        
        # Get screen dimensions
        display_info = pygame.display.Info()
        self.width = width or display_info.current_w
        self.height = height or display_info.current_h
        
        self._running = False
        self._thread: threading.Thread | None = None
        self._detections: list[Detection] = []
        self._lock = threading.Lock()
        self._fps = 0.0
        self._show_stats = True
        
        # Font for labels
        self._font = None
        self._screen = None
    
    def start(self):
        """Start the overlay in a background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
        # Wait for window creation
        time.sleep(0.2)
    
    def stop(self):
        """Stop the overlay."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def update(self, detections: list[Detection]):
        """Update detections to render (thread-safe)."""
        with self._lock:
            self._detections = detections.copy()
    
    def get_fps(self) -> float:
        """Get current render FPS."""
        return self._fps
    
    def _run(self):
        """Main overlay thread."""
        # Create borderless, transparent window
        self._screen = pygame.display.set_mode(
            (self.width, self.height),
            pygame.NOFRAME | pygame.SRCALPHA
        )
        pygame.display.set_caption("iDate Overlay")
        
        # Make window click-through and always on top (Windows only)
        if HAS_WIN32:
            self._setup_windows_transparency()
        
        # Load font
        try:
            self._font = pygame.font.SysFont('Arial', 16)
        except:
            self._font = pygame.font.Font(None, 16)
        
        # Render loop
        clock = pygame.time.Clock()
        frame_count = 0
        fps_start = time.perf_counter()
        
        while self._running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    return
            
            # Clear screen with transparent color
            self._screen.fill(TRANSPARENT_COLOR)
            
            # Get current detections
            with self._lock:
                detections = self._detections.copy()
            
            # Draw detections
            self._draw_detections(detections)
            
            # Draw FPS counter
            if self._show_stats:
                fps_text = self._font.render(f"Overlay FPS: {self._fps:.0f}", True, (255, 255, 255))
                self._screen.blit(fps_text, (10, 10))
            
            # Update display
            pygame.display.flip()
            
            # FPS tracking
            frame_count += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                self._fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.perf_counter()
            
            # Cap at 120 FPS
            clock.tick(120)
        
        pygame.quit()
    
    def _setup_windows_transparency(self):
        """Set up Windows layered transparent click-through window."""
        hwnd = pygame.display.get_wm_info()["window"]
        
        # Extended window styles
        WS_EX_LAYERED = 0x80000
        WS_EX_TRANSPARENT = 0x20
        WS_EX_TOPMOST = 0x8
        
        # Get current style
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        
        # Add layered + transparent + topmost flags
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, 
                               style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST)
        
        # Set color key for transparency (black = transparent)
        win32gui.SetLayeredWindowAttributes(hwnd, 0x000000, 255, 0x1)  # LWA_COLORKEY
        
        # Keep window on top
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                             win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    
    def _draw_detections(self, detections: list[Detection]):
        """Draw detection boxes and labels."""
        for det in detections:
            color = COLORS.get(det.class_name, (128, 128, 128))
            
            # Calculate rectangle
            x1 = det.x - det.width // 2
            y1 = det.y - det.height // 2
            
            # Draw rectangle outline
            rect = pygame.Rect(x1, y1, det.width, det.height)
            pygame.draw.rect(self._screen, color, rect, 3)
            
            # Draw center dot
            pygame.draw.circle(self._screen, color, (det.x, det.y), 5)
            
            # Draw label
            label = f"{det.class_name} {det.confidence:.0%}"
            text = self._font.render(label, True, color)
            self._screen.blit(text, (x1, y1 - 20))


def test_overlay():
    """Test the overlay with moving detections."""
    print("Starting overlay test...")
    print("Press Ctrl+C to stop")
    
    overlay = PygameOverlay()
    overlay.start()
    
    # Simulate moving detections
    t = 0
    try:
        while True:
            import math
            
            # Create moving test detections
            detections = [
                Detection('up', 500 + int(100 * math.sin(t)), 300, 60, 60, 0.9),
                Detection('down', 500, 300 + int(100 * math.cos(t)), 60, 60, 0.85),
                Detection('left', 300, 400, 60, 60, 0.88),
                Detection('right', 700, 400, 60, 60, 0.92),
                Detection('indicator', 500 + int(200 * math.sin(t * 2)), 500, 50, 50, 0.95),
            ]
            
            overlay.update(detections)
            
            t += 0.1
            time.sleep(0.016)  # ~60 FPS update
            
            # Print stats occasionally
            if int(t * 10) % 100 == 0:
                print(f"Overlay FPS: {overlay.get_fps():.1f}")
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        overlay.stop()


if __name__ == "__main__":
    test_overlay()
