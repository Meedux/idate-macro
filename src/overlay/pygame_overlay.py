"""
High-Performance Pygame Overlay - 60 FPS Real-time Detection Visualization

Uses Pygame with Windows layered window for:
- True 60fps rendering
- Hardware acceleration
- Click-through transparency
- Low CPU usage
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Any


# Windows constants
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x80000
WS_EX_TRANSPARENT = 0x20
WS_EX_TOPMOST = 0x8
WS_EX_TOOLWINDOW = 0x80
LWA_COLORKEY = 0x1
HWND_TOPMOST = -1
SWP_NOMOVE = 0x2
SWP_NOSIZE = 0x1


@dataclass
class OverlayData:
    """Data to send to overlay process."""
    indicator_pos: tuple[int, int] | None = None
    indicator_confidence: float = 0.0
    icons: list[tuple[str, int, int, float]] = field(default_factory=list)
    aligned_icons: list[str] = field(default_factory=list)
    window_rect: dict | None = None


def _get_window_rect(hwnd: int) -> dict | None:
    """Get window rectangle."""
    rect = ctypes.wintypes.RECT()
    if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return {
            "left": rect.left,
            "top": rect.top,
            "width": rect.right - rect.left,
            "height": rect.bottom - rect.top,
        }
    return None


def _overlay_process(data_queue: mp.Queue, stop_event: mp.Event, target_hwnd: int):
    """
    Separate process running Pygame overlay at 60fps.
    
    This runs in a SEPARATE PROCESS to avoid GIL and ensure 60fps.
    """
    import os
    os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'
    
    import pygame
    
    # Initialize pygame
    pygame.init()
    pygame.font.init()
    
    # Get initial window position
    rect = _get_window_rect(target_hwnd)
    if not rect:
        return
    
    # Create display - borderless window
    screen = pygame.display.set_mode(
        (rect['width'], rect['height']),
        pygame.NOFRAME | pygame.SRCALPHA
    )
    pygame.display.set_caption("iDate Overlay")
    
    # Get the pygame window handle
    info = pygame.display.get_wm_info()
    hwnd = info['window']
    
    # Make window layered, transparent, topmost, click-through
    style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    new_style = style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
    
    # Set transparent color (black)
    TRANSPARENT_COLOR = 0x000000
    ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, TRANSPARENT_COLOR, 0, LWA_COLORKEY)
    
    # Keep window always on top
    ctypes.windll.user32.SetWindowPos(
        hwnd, HWND_TOPMOST, 
        rect['left'], rect['top'], rect['width'], rect['height'],
        0
    )
    
    # Font for text
    font = pygame.font.SysFont('Consolas', 14, bold=True)
    
    # Colors
    COLOR_INDICATOR = (0, 255, 255)  # Cyan
    COLOR_ICON_NORMAL = (0, 255, 0)   # Green  
    COLOR_ICON_ALIGNED = (255, 0, 0)  # Red
    COLOR_TEXT_BG = (0, 0, 0, 180)    # Semi-transparent black
    
    clock = pygame.time.Clock()
    current_data = OverlayData()
    fps_counter = 0
    fps_time = time.time()
    display_fps = 0
    
    while not stop_event.is_set():
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()
                break
        
        # Get latest data (non-blocking, get most recent)
        while not data_queue.empty():
            try:
                current_data = data_queue.get_nowait()
            except:
                break
        
        # Update window position periodically
        new_rect = _get_window_rect(target_hwnd)
        if new_rect and new_rect != rect:
            rect = new_rect
            try:
                screen = pygame.display.set_mode(
                    (rect['width'], rect['height']),
                    pygame.NOFRAME | pygame.SRCALPHA
                )
                ctypes.windll.user32.SetWindowPos(
                    hwnd, HWND_TOPMOST,
                    rect['left'], rect['top'], rect['width'], rect['height'],
                    0
                )
            except:
                pass
        
        # Clear screen (transparent black)
        screen.fill((0, 0, 0))
        
        # Draw indicator
        if current_data.indicator_pos:
            ix, iy = current_data.indicator_pos
            
            # Vertical line
            pygame.draw.line(screen, COLOR_INDICATOR, (ix, 0), (ix, rect['height']), 3)
            
            # Circle at indicator position
            pygame.draw.circle(screen, COLOR_INDICATOR, (ix, iy), 18, 3)
            
            # Text label
            text = font.render(f"IND {current_data.indicator_confidence:.2f}", True, COLOR_INDICATOR)
            text_bg = pygame.Surface((text.get_width() + 6, text.get_height() + 4), pygame.SRCALPHA)
            text_bg.fill((0, 0, 0, 160))
            screen.blit(text_bg, (ix + 22, iy - 10))
            screen.blit(text, (ix + 25, iy - 8))
        
        # Draw icons
        for name, x, y, conf in current_data.icons:
            is_aligned = name in current_data.aligned_icons
            color = COLOR_ICON_ALIGNED if is_aligned else COLOR_ICON_NORMAL
            
            # Circle around icon
            pygame.draw.circle(screen, color, (x, y), 20, 3)
            
            # Label
            label = f">>> {name} <<<" if is_aligned else f"{name} {conf:.2f}"
            text = font.render(label, True, color)
            text_bg = pygame.Surface((text.get_width() + 6, text.get_height() + 4), pygame.SRCALPHA)
            text_bg.fill((0, 0, 0, 160))
            screen.blit(text_bg, (x + 22, y - 10))
            screen.blit(text, (x + 25, y - 8))
        
        # FPS counter (top left)
        fps_counter += 1
        now = time.time()
        if now - fps_time >= 1.0:
            display_fps = fps_counter / (now - fps_time)
            fps_counter = 0
            fps_time = now
        
        fps_text = font.render(f"Overlay: {display_fps:.0f} FPS", True, (255, 255, 0))
        fps_bg = pygame.Surface((fps_text.get_width() + 10, fps_text.get_height() + 6), pygame.SRCALPHA)
        fps_bg.fill((0, 0, 0, 200))
        screen.blit(fps_bg, (5, 5))
        screen.blit(fps_text, (10, 8))
        
        pygame.display.flip()
        clock.tick(60)  # Target 60 FPS
    
    pygame.quit()


class PygameOverlay:
    """
    60 FPS Pygame overlay running in separate process.
    
    Usage:
        overlay = PygameOverlay(target_hwnd)
        overlay.start()
        
        # In your loop:
        overlay.update(detection_result)
        
        # When done:
        overlay.stop()
    """
    
    def __init__(self, target_hwnd: int):
        self.target_hwnd = target_hwnd
        self._process: mp.Process | None = None
        self._data_queue: mp.Queue | None = None
        self._stop_event: mp.Event | None = None
    
    def start(self):
        """Start the overlay process."""
        if self._process is not None and self._process.is_alive():
            return
        
        self._data_queue = mp.Queue(maxsize=5)  # Small buffer
        self._stop_event = mp.Event()
        
        self._process = mp.Process(
            target=_overlay_process,
            args=(self._data_queue, self._stop_event, self.target_hwnd),
            daemon=True
        )
        self._process.start()
    
    def update(self, result) -> None:
        """
        Send detection result to overlay.
        
        result: DetectionResult from detector
        """
        if self._data_queue is None:
            return
        
        data = OverlayData(
            indicator_pos=result.indicator_pos,
            indicator_confidence=result.indicator_confidence,
            icons=list(result.icons),
            aligned_icons=list(result.aligned_icons)
        )
        
        # Non-blocking put - drop old data if queue full
        try:
            if self._data_queue.full():
                try:
                    self._data_queue.get_nowait()
                except:
                    pass
            self._data_queue.put_nowait(data)
        except:
            pass
    
    def stop(self):
        """Stop the overlay process."""
        if self._stop_event:
            self._stop_event.set()
        
        if self._process and self._process.is_alive():
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
        
        self._process = None
        self._data_queue = None
        self._stop_event = None
    
    def is_running(self) -> bool:
        """Check if overlay is running."""
        return self._process is not None and self._process.is_alive()


# Thread-based alternative for when multiprocessing causes issues
class PygameOverlayThreaded:
    """
    Thread-based Pygame overlay (alternative if multiprocessing fails).
    """
    
    def __init__(self, target_hwnd: int):
        self.target_hwnd = target_hwnd
        self._thread = None
        self._stop_flag = False
        self._data = OverlayData()
        self._lock = None
    
    def start(self):
        """Start overlay thread."""
        import threading
        
        if self._thread is not None and self._thread.is_alive():
            return
        
        self._stop_flag = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def _run(self):
        """Run overlay loop."""
        import os
        os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'
        
        import pygame
        
        pygame.init()
        pygame.font.init()
        
        rect = _get_window_rect(self.target_hwnd)
        if not rect:
            return
        
        screen = pygame.display.set_mode(
            (rect['width'], rect['height']),
            pygame.NOFRAME
        )
        pygame.display.set_caption("iDate Overlay")
        
        info = pygame.display.get_wm_info()
        hwnd = info['window']
        
        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        new_style = style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
        ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0x000000, 0, LWA_COLORKEY)
        ctypes.windll.user32.SetWindowPos(hwnd, HWND_TOPMOST, rect['left'], rect['top'], 0, 0, SWP_NOSIZE)
        
        font = pygame.font.SysFont('Consolas', 14, bold=True)
        clock = pygame.time.Clock()
        
        COLOR_IND = (0, 255, 255)
        COLOR_NORMAL = (0, 255, 0)
        COLOR_ALIGNED = (255, 0, 0)
        
        fps_count = 0
        fps_time = time.time()
        display_fps = 0
        
        while not self._stop_flag:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._stop_flag = True
                    break
            
            # Update position
            new_rect = _get_window_rect(self.target_hwnd)
            if new_rect:
                if new_rect['width'] != rect['width'] or new_rect['height'] != rect['height']:
                    rect = new_rect
                    screen = pygame.display.set_mode((rect['width'], rect['height']), pygame.NOFRAME)
                ctypes.windll.user32.SetWindowPos(
                    hwnd, HWND_TOPMOST, rect['left'], rect['top'], 0, 0, SWP_NOSIZE
                )
            
            screen.fill((0, 0, 0))
            
            with self._lock:
                data = self._data
            
            # Draw indicator
            if data.indicator_pos:
                ix, iy = data.indicator_pos
                pygame.draw.line(screen, COLOR_IND, (ix, 0), (ix, rect['height']), 3)
                pygame.draw.circle(screen, COLOR_IND, (ix, iy), 18, 3)
                text = font.render(f"IND {data.indicator_confidence:.2f}", True, COLOR_IND)
                screen.blit(text, (ix + 25, iy - 8))
            
            # Draw icons
            for name, x, y, conf in data.icons:
                is_aligned = name in data.aligned_icons
                color = COLOR_ALIGNED if is_aligned else COLOR_NORMAL
                pygame.draw.circle(screen, color, (x, y), 20, 3)
                label = f">>> {name} <<<" if is_aligned else f"{name} {conf:.2f}"
                text = font.render(label, True, color)
                screen.blit(text, (x + 25, y - 8))
            
            # FPS
            fps_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                display_fps = fps_count / (now - fps_time)
                fps_count = 0
                fps_time = now
            
            fps_text = font.render(f"Overlay: {display_fps:.0f} FPS", True, (255, 255, 0))
            screen.blit(fps_text, (10, 10))
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
    
    def update(self, result):
        """Update overlay data."""
        if self._lock is None:
            return
        
        data = OverlayData(
            indicator_pos=result.indicator_pos,
            indicator_confidence=result.indicator_confidence,
            icons=list(result.icons),
            aligned_icons=list(result.aligned_icons)
        )
        
        with self._lock:
            self._data = data
    
    def stop(self):
        """Stop overlay."""
        self._stop_flag = True
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
    
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
