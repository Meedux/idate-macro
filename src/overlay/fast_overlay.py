"""
Real-time Transparent Overlay for iDate Revival
===============================================

Uses Win32 API to create a transparent click-through overlay
that updates at high FPS (60+) with minimal latency.
"""

from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass
from typing import Callable

import win32gui
import win32con
import win32api

# For GDI+ drawing
from ctypes import windll, byref, c_int, c_void_p, c_ulong
import struct

# Load required DLLs
gdi32 = windll.gdi32
user32 = windll.user32


# Window style constants
WS_EX_LAYERED = 0x80000
WS_EX_TRANSPARENT = 0x20
WS_EX_TOPMOST = 0x8
WS_EX_TOOLWINDOW = 0x80
WS_POPUP = 0x80000000

# Layered window constants
LWA_COLORKEY = 0x1
LWA_ALPHA = 0x2
ULW_COLORKEY = 0x1
ULW_ALPHA = 0x2

# GDI constants
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0


@dataclass
class Detection:
    """Single detection for overlay."""
    class_name: str
    x: int
    y: int
    width: int
    height: int
    confidence: float


# Colors as RGB tuples
COLORS = {
    'up': (0, 255, 0),        # Green
    'down': (255, 165, 0),    # Orange
    'left': (255, 255, 0),    # Yellow
    'right': (255, 0, 255),   # Magenta
    'hand': (255, 140, 0),    # Dark orange
    'indicator': (0, 255, 255), # Cyan
}


class FastOverlay:
    """
    High-performance transparent overlay using Win32.
    
    Features:
    - 60+ FPS update rate
    - Click-through (doesn't intercept mouse)
    - Always on top
    - Hardware-accelerated rendering via GDI
    """
    
    def __init__(self, width: int = 0, height: int = 0):
        """
        Initialize overlay.
        
        Args:
            width: Overlay width (0 = screen width)
            height: Overlay height (0 = screen height)
        """
        # Get screen dimensions
        self.screen_width = win32api.GetSystemMetrics(0)
        self.screen_height = win32api.GetSystemMetrics(1)
        
        self.width = width or self.screen_width
        self.height = height or self.screen_height
        
        self.hwnd = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._detections: list[Detection] = []
        self._lock = threading.Lock()
        self._fps = 0.0
        
        # Transparent color (will be made transparent)
        self._trans_color = 0x000000  # Black = transparent
        
        # Drawing resources
        self._hdc = None
        self._mem_dc = None
        self._bitmap = None
        self._old_bitmap = None
    
    def start(self):
        """Start the overlay in a background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
        # Wait for window creation
        time.sleep(0.1)
    
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
        # Create window class
        wndclass = win32gui.WNDCLASS()
        wndclass.style = win32con.CS_HREDRAW | win32con.CS_VREDRAW
        wndclass.lpfnWndProc = self._wnd_proc
        wndclass.hInstance = win32api.GetModuleHandle(None)
        wndclass.hCursor = win32gui.LoadCursor(None, win32con.IDC_ARROW)
        wndclass.hbrBackground = win32gui.GetStockObject(win32con.BLACK_BRUSH)
        wndclass.lpszClassName = "iDateOverlay"
        
        try:
            atom = win32gui.RegisterClass(wndclass)
        except:
            pass  # Class may already exist
        
        # Create layered window
        ex_style = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW
        style = WS_POPUP
        
        self.hwnd = win32gui.CreateWindowEx(
            ex_style,
            "iDateOverlay",
            "iDate Overlay",
            style,
            0, 0, self.width, self.height,
            None, None, win32api.GetModuleHandle(None), None
        )
        
        # Set transparency
        win32gui.SetLayeredWindowAttributes(
            self.hwnd, 
            self._trans_color,
            255,  # Alpha
            LWA_COLORKEY
        )
        
        # Show window
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
        win32gui.UpdateWindow(self.hwnd)
        
        # Create drawing resources
        self._hdc = win32gui.GetDC(self.hwnd)
        self._mem_dc = gdi32.CreateCompatibleDC(self._hdc)
        self._bitmap = gdi32.CreateCompatibleBitmap(self._hdc, self.width, self.height)
        self._old_bitmap = gdi32.SelectObject(self._mem_dc, self._bitmap)
        
        # Render loop
        frame_count = 0
        fps_start = time.perf_counter()
        
        while self._running:
            loop_start = time.perf_counter()
            
            # Process windows messages
            self._pump_messages()
            
            # Render
            self._render()
            
            # FPS tracking
            frame_count += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                self._fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.perf_counter()
            
            # Target ~120 FPS (8ms sleep)
            render_time = time.perf_counter() - loop_start
            sleep_time = 0.008 - render_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Cleanup
        gdi32.SelectObject(self._mem_dc, self._old_bitmap)
        gdi32.DeleteObject(self._bitmap)
        gdi32.DeleteDC(self._mem_dc)
        win32gui.ReleaseDC(self.hwnd, self._hdc)
        win32gui.DestroyWindow(self.hwnd)
    
    def _pump_messages(self):
        """Process Windows messages."""
        msg = win32gui.MSG()
        while win32gui.PeekMessage(byref(msg) if hasattr(msg, '__class__') else ctypes.byref(ctypes.c_int()), 
                                   self.hwnd, 0, 0, win32con.PM_REMOVE):
            win32gui.TranslateMessage(msg)
            win32gui.DispatchMessage(msg)
    
    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        """Window procedure."""
        if msg == win32con.WM_DESTROY:
            win32gui.PostQuitMessage(0)
            return 0
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)
    
    def _render(self):
        """Render detections to the overlay."""
        # Clear with transparent color
        brush = gdi32.CreateSolidBrush(self._trans_color)
        rect = (0, 0, self.width, self.height)
        # Fill rect with transparent color
        gdi32.SelectObject(self._mem_dc, brush)
        gdi32.PatBlt(self._mem_dc, 0, 0, self.width, self.height, 0x00F00021)  # PATCOPY
        gdi32.DeleteObject(brush)
        
        # Get current detections
        with self._lock:
            detections = self._detections.copy()
        
        # Draw each detection
        for det in detections:
            color = COLORS.get(det.class_name, (128, 128, 128))
            # Convert RGB to Windows BGR
            win_color = color[2] | (color[1] << 8) | (color[0] << 16)
            
            pen = gdi32.CreatePen(0, 3, win_color)  # PS_SOLID, width=3
            old_pen = gdi32.SelectObject(self._mem_dc, pen)
            
            # Draw rectangle outline
            x1 = det.x - det.width // 2
            y1 = det.y - det.height // 2
            x2 = det.x + det.width // 2
            y2 = det.y + det.height // 2
            
            # Use MoveToEx and LineTo for outline
            gdi32.MoveToEx(self._mem_dc, x1, y1, None)
            gdi32.LineTo(self._mem_dc, x2, y1)
            gdi32.LineTo(self._mem_dc, x2, y2)
            gdi32.LineTo(self._mem_dc, x1, y2)
            gdi32.LineTo(self._mem_dc, x1, y1)
            
            # Draw small filled circle at center
            brush2 = gdi32.CreateSolidBrush(win_color)
            old_brush = gdi32.SelectObject(self._mem_dc, brush2)
            gdi32.Ellipse(self._mem_dc, det.x - 5, det.y - 5, det.x + 5, det.y + 5)
            
            gdi32.SelectObject(self._mem_dc, old_brush)
            gdi32.SelectObject(self._mem_dc, old_pen)
            gdi32.DeleteObject(pen)
            gdi32.DeleteObject(brush2)
        
        # Copy to screen
        gdi32.BitBlt(self._hdc, 0, 0, self.width, self.height, 
                    self._mem_dc, 0, 0, SRCCOPY)


def test_overlay():
    """Test the overlay with dummy detections."""
    print("Starting overlay test...")
    print("Press Ctrl+C to stop")
    
    overlay = FastOverlay()
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
                Detection('indicator', 500 + int(200 * math.sin(t * 2)), 500, 50, 50, 0.95),
            ]
            
            overlay.update(detections)
            
            t += 0.1
            time.sleep(0.016)  # ~60 FPS update
            
            # Print FPS occasionally
            if int(t * 10) % 50 == 0:
                print(f"Overlay FPS: {overlay.get_fps():.1f}")
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        overlay.stop()


if __name__ == "__main__":
    test_overlay()
