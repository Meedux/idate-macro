"""
Win32 Real-Time Overlay for iDate Revival

FIXED overlay that properly tracks the target window using Win32 API.
Key features:
- Properly positioned ON TOP of target window (not background)
- Click-through transparency 
- Real-time position tracking at 60fps
- Uses GDI+ for smooth rendering
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import threading
import time
from dataclasses import dataclass, field


# Windows constants
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x80000
WS_EX_TRANSPARENT = 0x20
WS_EX_TOPMOST = 0x8
WS_EX_TOOLWINDOW = 0x80
WS_EX_NOACTIVATE = 0x08000000
LWA_COLORKEY = 0x1
LWA_ALPHA = 0x2
HWND_TOPMOST = ctypes.wintypes.HWND(-1)  # Must be ctypes-compatible!
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
SWP_FRAMECHANGED = 0x0020
GW_OWNER = 4

# GDI constants
PS_SOLID = 0
TRANSPARENT = 1


@dataclass
class OverlayData:
    """Data for overlay rendering."""
    indicator_pos: tuple[int, int] | None = None
    indicator_confidence: float = 0.0
    icons: list[tuple[str, int, int, float]] = field(default_factory=list)
    aligned_icons: list[str] = field(default_factory=list)
    # Track region for rhythm games (optional)
    track_y: int | None = None
    track_height: int | None = None
    hit_zone_x: int | None = None
    hit_zone_width: int = 45
    # Memory-mode game state (optional)
    score: int | None = None
    combo: int | None = None
    health: float | None = None
    is_memory_mode: bool = False


def get_window_rect(hwnd: int) -> dict | None:
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


class Win32Overlay:
    """
    Real-time overlay using Pygame with proper Win32 window management.
    
    This overlay ACTUALLY stays on top of the target window.
    """
    
    def __init__(self, target_hwnd: int, offset_x: int = 0, offset_y: int = 0):
        """
        Initialize overlay.
        
        Args:
            target_hwnd: Window handle to overlay on
            offset_x: X offset for detection positions (positive = right)
            offset_y: Y offset for detection positions (positive = down)
        """
        self.target_hwnd = target_hwnd
        self.offset_x = offset_x  # Calibration offset
        self.offset_y = offset_y  # Calibration offset
        self._thread = None
        self._stop_flag = False
        self._data = OverlayData()
        self._lock = threading.Lock()
        self._hwnd = None
    
    def start(self):
        """Start overlay thread."""
        if self._thread and self._thread.is_alive():
            return
        
        self._stop_flag = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
        # Wait for window to be created
        time.sleep(0.3)
    
    def _run(self):
        """Main overlay loop."""
        import os
        
        # Get target window position first
        rect = get_window_rect(self.target_hwnd)
        if not rect:
            print("[OVERLAY] ERROR: Cannot get target window rect")
            return
        
        # Position pygame window at target location
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{rect['left']},{rect['top']}"
        
        import pygame
        
        pygame.init()
        pygame.font.init()
        
        # Create window
        screen = pygame.display.set_mode(
            (rect['width'], rect['height']),
            pygame.NOFRAME | pygame.SRCALPHA
        )
        pygame.display.set_caption("iDate Overlay")
        
        # Get pygame window handle
        info = pygame.display.get_wm_info()
        self._hwnd = info['window']
        
        # Make it layered, always-on-top, click-through, tool window
        self._setup_window_style()
        
        # Initial position
        self._update_window_position(rect)
        
        # Setup rendering
        font = pygame.font.SysFont('Consolas', 14, bold=True)
        clock = pygame.time.Clock()
        
        COLOR_IND = (0, 255, 255)      # Cyan
        COLOR_NORMAL = (0, 255, 0)     # Green
        COLOR_ALIGNED = (255, 0, 0)    # Red
        COLOR_FPS = (255, 255, 0)      # Yellow
        
        fps_count = 0
        fps_time = time.time()
        display_fps = 0.0
        
        while not self._stop_flag:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._stop_flag = True
                    break
            
            # Update window position EVERY FRAME - no delays!
            new_rect = get_window_rect(self.target_hwnd)
            if new_rect:
                if new_rect['width'] != rect['width'] or new_rect['height'] != rect['height']:
                    # Resize if needed
                    rect = new_rect
                    try:
                        screen = pygame.display.set_mode(
                            (rect['width'], rect['height']),
                            pygame.NOFRAME | pygame.SRCALPHA
                        )
                    except:
                        pass
                
                self._update_window_position(new_rect)
                rect = new_rect
            
            # Clear screen (use color key for transparency)
            screen.fill((1, 1, 1))  # Near-black, will be transparent
            
            # Get current data
            with self._lock:
                data = self._data
            
            # === DRAW HIT ZONE (rhythm game track) ===
            track_y = getattr(data, 'track_y', None)
            hit_zone_x = getattr(data, 'hit_zone_x', None)
            if track_y is not None and hit_zone_x is not None:
                track_h = getattr(data, 'track_height', 80)
                hz_width = getattr(data, 'hit_zone_width', 45)
                
                # Draw hit zone rectangle (semi-transparent green area)
                hit_zone_rect = pygame.Surface((hz_width * 2, track_h), pygame.SRCALPHA)
                hit_zone_rect.fill((0, 255, 0, 40))  # Semi-transparent green
                screen.blit(hit_zone_rect, (hit_zone_x - hz_width, track_y))
                
                # Draw hit zone borders
                pygame.draw.rect(screen, (0, 255, 100), 
                    (hit_zone_x - hz_width, track_y, hz_width * 2, track_h), 2)
            
            # === DRAW INDICATOR ===
            if data.indicator_pos:
                ix, iy = data.indicator_pos
                # Apply calibration offset
                ix += self.offset_x
                iy += self.offset_y
                
                # Vertical line through entire height
                pygame.draw.line(screen, COLOR_IND, (ix, 0), (ix, rect['height']), 3)
                
                # Circle at indicator position
                pygame.draw.circle(screen, COLOR_IND, (ix, iy), 18, 3)
                
                # Confidence text
                text = font.render(f"IND {data.indicator_confidence:.2f}", True, COLOR_IND)
                screen.blit(text, (ix + 25, iy - 8))
            
            # === DRAW ICONS ===
            for name, x, y, conf in data.icons:
                # Apply calibration offset
                x += self.offset_x
                y += self.offset_y
                
                is_aligned = name in data.aligned_icons
                color = COLOR_ALIGNED if is_aligned else COLOR_NORMAL
                
                # Circle
                pygame.draw.circle(screen, color, (x, y), 20, 3)
                
                # Label
                if is_aligned:
                    label = f">>> {name} <<<"
                else:
                    label = f"{name} {conf:.2f}"
                text = font.render(label, True, color)
                screen.blit(text, (x + 25, y - 8))
            
            # === DRAW FPS ===
            fps_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                display_fps = fps_count / (now - fps_time)
                fps_count = 0
                fps_time = now
            
            fps_text = font.render(f"Overlay: {display_fps:.0f} FPS", True, COLOR_FPS)
            # Black background for readability
            fps_bg = pygame.Surface((fps_text.get_width() + 10, fps_text.get_height() + 6))
            fps_bg.fill((0, 0, 0))
            screen.blit(fps_bg, (5, 5))
            screen.blit(fps_text, (10, 8))
            
            # === DRAW MEMORY STATE HUD (score, combo, health) ===
            if data.is_memory_mode:
                hud_y = 30
                hud_items = []
                if data.score is not None:
                    hud_items.append((f"Score: {data.score}", (0, 200, 255)))
                if data.combo is not None:
                    combo_color = (255, 215, 0) if data.combo >= 10 else (200, 200, 200)
                    hud_items.append((f"Combo: {data.combo}", combo_color))
                if data.health is not None:
                    hp_pct = int(data.health * 100)
                    hp_color = (0, 255, 0) if hp_pct > 50 else (255, 165, 0) if hp_pct > 25 else (255, 0, 0)
                    hud_items.append((f"HP: {hp_pct}%", hp_color))
                hud_items.append(("MEM MODE", (180, 100, 255)))
                for txt, col in hud_items:
                    rendered = font.render(txt, True, col)
                    bg = pygame.Surface((rendered.get_width() + 10, rendered.get_height() + 4))
                    bg.fill((0, 0, 0))
                    screen.blit(bg, (5, hud_y))
                    screen.blit(rendered, (10, hud_y + 2))
                    hud_y += rendered.get_height() + 6
            
            pygame.display.flip()
            clock.tick(0)  # No FPS cap - run as fast as possible!
        
        pygame.quit()
    
    def _setup_window_style(self):
        """Configure window to be overlay-style."""
        if not self._hwnd:
            return
        
        user32 = ctypes.windll.user32
        
        # Get current style
        style = user32.GetWindowLongW(self._hwnd, GWL_EXSTYLE)
        
        # Add overlay styles
        new_style = style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        user32.SetWindowLongW(self._hwnd, GWL_EXSTYLE, new_style)
        
        # Set color key for transparency (RGB 0x010101 = near-black)
        user32.SetLayeredWindowAttributes(self._hwnd, 0x010101, 0, LWA_COLORKEY)
        
        # Force topmost immediately
        user32.SetWindowPos(
            self._hwnd,
            HWND_TOPMOST,
            0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED
        )
    
    def _update_window_position(self, rect: dict):
        """Update overlay position to match target window."""
        if not self._hwnd:
            return
        
        user32 = ctypes.windll.user32
        
        # Set position with HWND_TOPMOST to ensure it stays on top EVERY frame
        user32.SetWindowPos(
            self._hwnd,
            HWND_TOPMOST,
            rect['left'],
            rect['top'],
            rect['width'],
            rect['height'],
            SWP_NOACTIVATE | SWP_SHOWWINDOW
        )
        
        # Also bring to front explicitly
        user32.BringWindowToTop(self._hwnd)
    
    def update(self, result) -> None:
        """Update overlay with detection result."""
        data = OverlayData(
            indicator_pos=result.indicator_pos,
            indicator_confidence=result.indicator_confidence,
            icons=list(result.icons) if hasattr(result, 'icons') else [],
            aligned_icons=list(result.aligned_icons) if hasattr(result, 'aligned_icons') else [],
        )
        
        # Memory-mode game state
        game_state = getattr(result, 'game_state', None)
        if game_state is not None:
            data.is_memory_mode = True
            data.score = getattr(game_state, 'score', None)
            data.combo = getattr(game_state, 'combo', None)
            data.health = getattr(game_state, 'health', None)
        
        with self._lock:
            self._data = data
    
    def stop(self):
        """Stop the overlay."""
        self._stop_flag = True
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._hwnd = None
    
    def is_running(self) -> bool:
        """Check if overlay is running."""
        return self._thread is not None and self._thread.is_alive()


class DirectOverlay:
    """
    Alternative overlay using direct GDI drawing on top of target window.
    
    This approach draws directly over the target window using Windows GDI,
    which guarantees proper z-ordering.
    """
    
    def __init__(self, target_hwnd: int):
        self.target_hwnd = target_hwnd
        self._thread = None
        self._stop_flag = False
        self._data = OverlayData()
        self._lock = threading.Lock()
    
    def start(self):
        """Start overlay thread."""
        if self._thread and self._thread.is_alive():
            return
        
        self._stop_flag = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def _run(self):
        """Direct GDI overlay loop."""
        import win32gui
        import win32api
        import win32con
        
        fps_count = 0
        fps_time = time.time()
        display_fps = 0.0
        
        while not self._stop_flag:
            try:
                # Get device context for target window
                rect = get_window_rect(self.target_hwnd)
                if not rect:
                    time.sleep(0.1)
                    continue
                
                # Get DC for entire screen
                hdc = win32gui.GetDC(0)
                
                with self._lock:
                    data = self._data
                
                # Draw indicator
                if data.indicator_pos:
                    ix, iy = data.indicator_pos
                    # Offset by window position
                    screen_x = rect['left'] + ix
                    screen_y = rect['top'] + iy
                    
                    # Create cyan pen
                    pen = win32gui.CreatePen(win32con.PS_SOLID, 3, win32api.RGB(0, 255, 255))
                    old_pen = win32gui.SelectObject(hdc, pen)
                    
                    # Draw vertical line
                    win32gui.MoveToEx(hdc, screen_x, rect['top'])
                    win32gui.LineTo(hdc, screen_x, rect['top'] + rect['height'])
                    
                    # Draw circle (approximated with rectangle for speed)
                    win32gui.Ellipse(hdc, screen_x - 15, screen_y - 15, screen_x + 15, screen_y + 15)
                    
                    win32gui.SelectObject(hdc, old_pen)
                    win32gui.DeleteObject(pen)
                
                # Draw icons
                for name, x, y, conf in data.icons:
                    is_aligned = name in data.aligned_icons
                    color = win32api.RGB(255, 0, 0) if is_aligned else win32api.RGB(0, 255, 0)
                    
                    screen_x = rect['left'] + x
                    screen_y = rect['top'] + y
                    
                    pen = win32gui.CreatePen(win32con.PS_SOLID, 3, color)
                    old_pen = win32gui.SelectObject(hdc, pen)
                    
                    win32gui.Ellipse(hdc, screen_x - 18, screen_y - 18, screen_x + 18, screen_y + 18)
                    
                    win32gui.SelectObject(hdc, old_pen)
                    win32gui.DeleteObject(pen)
                
                win32gui.ReleaseDC(0, hdc)
                
                # FPS tracking
                fps_count += 1
                now = time.time()
                if now - fps_time >= 1.0:
                    display_fps = fps_count / (now - fps_time)
                    fps_count = 0
                    fps_time = now
                
                # Target 60 FPS
                time.sleep(1/60)
                
            except Exception as e:
                time.sleep(0.1)
    
    def update(self, result):
        """Update overlay data."""
        data = OverlayData(
            indicator_pos=result.indicator_pos,
            indicator_confidence=result.indicator_confidence,
            icons=list(result.icons) if hasattr(result, 'icons') else [],
            aligned_icons=list(result.aligned_icons) if hasattr(result, 'aligned_icons') else []
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
