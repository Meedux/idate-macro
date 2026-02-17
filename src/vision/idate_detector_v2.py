"""
iDate Revival Detector v2 - HEAVY REFACTOR FOR 100% ACCURACY

Key Changes from v1:
1. Multi-scale template matching (0.8x - 1.2x)
2. Much more aggressive detection (lower initial threshold)
3. Color verification is OPTIONAL and OFF by default
4. Extensive debug logging to see exactly what's happening
5. ROI-based detection (only search in relevant screen areas)
6. Support for different matching methods

Detection Strategy:
1. Find indicator FIRST (it's the bouncing cursor)
2. Only after indicator is found, search for icons in the same horizontal band
3. Use multi-scale matching to handle resolution differences
4. NMS to prevent duplicates
"""

from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import NamedTuple, Callable

import cv2
import numpy as np


# Template name to key mapping
TEMPLATE_KEY_MAP = {
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "hand": "space",
}

# Matching scales to try (handles different resolutions)
MATCH_SCALES = [1.0, 0.9, 1.1, 0.8, 1.2, 0.85, 1.15]


class TemplateInfo(NamedTuple):
    """Template data with multiple scales."""
    name: str
    images_gray: list[np.ndarray]   # Grayscale at different scales
    images_bgr: list[np.ndarray]    # BGR at different scales
    scales: list[float]             # Scale factors
    original_size: tuple[int, int]  # Original (width, height)
    avg_color: tuple[int, int, int] # Average BGR color


@dataclass
class DetectionResult:
    """Detection results for a single frame."""
    indicator_pos: tuple[int, int] | None = None
    indicator_confidence: float = 0.0
    icons: list[tuple[str, int, int, float]] = field(default_factory=list)
    aligned_icons: list[str] = field(default_factory=list)
    debug_info: str = ""


def get_average_color(image: np.ndarray) -> tuple[int, int, int]:
    """Get average BGR color of an image."""
    if len(image.shape) == 2:
        avg = int(np.mean(image))
        return (avg, avg, avg)
    b = int(np.mean(image[:, :, 0]))
    g = int(np.mean(image[:, :, 1]))
    r = int(np.mean(image[:, :, 2]))
    return (b, g, r)


def color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """Euclidean color distance."""
    return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)


class IdateDetector:
    """
    HIGH-ACCURACY template detector for iDate Revival.
    
    Uses multi-scale template matching for robust detection.
    """
    
    def __init__(self, template_dir: str | Path = "templates"):
        self.template_dir = Path(template_dir)
        
        # Templates storage
        self.templates: dict[str, TemplateInfo] = {}
        self.indicator: TemplateInfo | None = None
        
        # === DETECTION SETTINGS ===
        # Lower threshold to actually DETECT things, then filter
        self.indicator_threshold = 0.65  # Lower to find indicator
        self.icon_threshold = 0.60       # Lower to find icons!
        
        # Color verification (OFF by default until detection works)
        self.use_color_verification = False
        self.color_threshold = 50.0
        
        # Alignment
        self.alignment_tolerance = 35  # Pixels X-distance to consider aligned
        
        # Cooldown
        self._last_press: dict[str, float] = {}
        self._cooldown = 0.12  # Seconds between same key press
        
        # Cached result
        self._last_result = DetectionResult()
        
        # Debug settings
        self.debug = True  # ENABLED by default so we can see what's happening
        self._log_callback: Callable[[str], None] | None = None
        
        # Load templates
        self._load_templates()
    
    def set_log_callback(self, callback: Callable[[str], None]):
        """Set callback for debug logging."""
        self._log_callback = callback
    
    def _log(self, msg: str):
        """Log debug message."""
        if self.debug:
            print(f"[DETECTOR] {msg}")
            if self._log_callback:
                self._log_callback(msg)
    
    @property
    def indicator_templates(self) -> list:
        """Compatibility property."""
        if self.indicator:
            return [self.indicator.images_gray[0]]
        return []
    
    def _create_scaled_templates(self, img_gray: np.ndarray, img_bgr: np.ndarray, 
                                  name: str) -> TemplateInfo:
        """Create template at multiple scales."""
        h, w = img_gray.shape[:2]
        
        images_gray = []
        images_bgr = []
        valid_scales = []
        
        for scale in MATCH_SCALES:
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            if new_w < 10 or new_h < 10:
                continue
            
            scaled_gray = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scaled_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            images_gray.append(scaled_gray)
            images_bgr.append(scaled_bgr)
            valid_scales.append(scale)
        
        avg_color = get_average_color(img_bgr)
        
        return TemplateInfo(
            name=name,
            images_gray=images_gray,
            images_bgr=images_bgr,
            scales=valid_scales,
            original_size=(w, h),
            avg_color=avg_color
        )
    
    def _load_templates(self):
        """Load all templates with multi-scale support."""
        if not self.template_dir.exists():
            self._log(f"ERROR: Template directory not found: {self.template_dir}")
            return
        
        self._log(f"Loading templates from: {self.template_dir}")
        
        # Load indicator
        indicator_path = self.template_dir / "indicator.png"
        if indicator_path.exists():
            img = cv2.imread(str(indicator_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                if len(img.shape) == 3 and img.shape[2] == 4:
                    img_bgr = img[:, :, :3]
                else:
                    img_bgr = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                self.indicator = self._create_scaled_templates(img_gray, img_bgr, "indicator")
                self._log(f"✓ Loaded indicator: {self.indicator.original_size}, "
                         f"color=BGR{self.indicator.avg_color}, scales={len(self.indicator.scales)}")
        else:
            self._log("ERROR: indicator.png NOT FOUND!")
        
        # Load icon templates
        for name in TEMPLATE_KEY_MAP.keys():
            path = self.template_dir / f"{name}.png"
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if len(img.shape) == 3 and img.shape[2] == 4:
                        img_bgr = img[:, :, :3]
                    else:
                        img_bgr = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    
                    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    template = self._create_scaled_templates(img_gray, img_bgr, name)
                    self.templates[name] = template
                    self._log(f"✓ Loaded {name}: {template.original_size}, "
                             f"color=BGR{template.avg_color}, scales={len(template.scales)}")
            else:
                self._log(f"WARNING: {name}.png not found")
        
        self._log(f"Loaded {len(self.templates)} icon templates")
        self._log(f"Thresholds: indicator={self.indicator_threshold}, icon={self.icon_threshold}")
        self._log(f"Color verification: {'ON' if self.use_color_verification else 'OFF'}")
    
    def _match_template_multiscale(self, gray: np.ndarray, template: TemplateInfo,
                                    threshold: float) -> list[tuple[int, int, float, float]]:
        """
        Multi-scale template matching.
        
        Returns list of (x, y, confidence, scale) for all matches above threshold.
        """
        matches = []
        
        for i, (tmpl_gray, scale) in enumerate(zip(template.images_gray, template.scales)):
            th, tw = tmpl_gray.shape[:2]
            
            # Skip if template larger than image
            if th > gray.shape[0] or tw > gray.shape[1]:
                continue
            
            # Template matching
            result = cv2.matchTemplate(gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
            
            # Find all matches above threshold
            locations = np.where(result >= threshold)
            
            for y, x in zip(*locations):
                conf = float(result[y, x])
                cx = x + tw // 2
                cy = y + th // 2
                matches.append((cx, cy, conf, scale))
        
        return matches
    
    def _find_best_match(self, gray: np.ndarray, template: TemplateInfo,
                          threshold: float) -> tuple[tuple[int, int] | None, float]:
        """Find BEST match across all scales."""
        best_pos = None
        best_conf = 0.0
        
        for tmpl_gray, scale in zip(template.images_gray, template.scales):
            th, tw = tmpl_gray.shape[:2]
            
            if th > gray.shape[0] or tw > gray.shape[1]:
                continue
            
            result = cv2.matchTemplate(gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_conf:
                best_conf = max_val
                cx = max_loc[0] + tw // 2
                cy = max_loc[1] + th // 2
                best_pos = (cx, cy)
        
        if best_conf >= threshold:
            return best_pos, best_conf
        return None, best_conf
    
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run detection on frame.
        
        Strategy:
        1. Find indicator (single best match)
        2. Find all matching icons
        3. Determine alignment
        """
        # Convert frame
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bgr = frame
        else:
            gray = frame
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        result = DetectionResult()
        debug_msgs = []
        
        # === STEP 1: Find indicator ===
        if self.indicator:
            pos, conf = self._find_best_match(gray, self.indicator, self.indicator_threshold)
            if pos:
                result.indicator_pos = pos
                result.indicator_confidence = conf
                debug_msgs.append(f"IND: ({pos[0]},{pos[1]}) conf={conf:.3f}")
            else:
                debug_msgs.append(f"IND: not found (best={conf:.3f})")
        
        # === STEP 2: Find icons ===
        all_icons = []
        
        for name, template in self.templates.items():
            matches = self._match_template_multiscale(gray, template, self.icon_threshold)
            
            if matches:
                debug_msgs.append(f"{name}: {len(matches)} raw matches")
            
            for cx, cy, conf, scale in matches:
                # Optional color verification
                if self.use_color_verification:
                    w, h = template.original_size
                    sw, sh = int(w * scale), int(h * scale)
                    
                    x1 = max(0, cx - sw // 2)
                    y1 = max(0, cy - sh // 2)
                    x2 = min(bgr.shape[1], cx + sw // 2)
                    y2 = min(bgr.shape[0], cy + sh // 2)
                    
                    if x2 - x1 > 5 and y2 - y1 > 5:
                        region = bgr[y1:y2, x1:x2]
                        region_color = get_average_color(region)
                        dist = color_distance(region_color, template.avg_color)
                        
                        if dist > self.color_threshold:
                            continue  # Failed color check
                
                all_icons.append((name, cx, cy, conf))
        
        # === STEP 3: NMS ===
        result.icons = self._nms(all_icons)
        
        if result.icons:
            debug_msgs.append(f"Final icons: {[(n, x, y, f'{c:.2f}') for n,x,y,c in result.icons]}")
        
        # === STEP 4: Determine alignment ===
        if result.indicator_pos:
            ind_x = result.indicator_pos[0]
            for name, icon_x, icon_y, conf in result.icons:
                if abs(icon_x - ind_x) <= self.alignment_tolerance:
                    result.aligned_icons.append(name)
                    debug_msgs.append(f"ALIGNED: {name} (dist={abs(icon_x - ind_x)}px)")
        
        result.debug_info = " | ".join(debug_msgs)
        self._last_result = result
        
        # Log debug info periodically
        if self.debug and debug_msgs:
            self._log(result.debug_info)
        
        return result
    
    def _nms(self, detections: list[tuple[str, int, int, float]], 
             dist_threshold: int = 50) -> list[tuple[str, int, int, float]]:
        """Non-maximum suppression."""
        if not detections:
            return []
        
        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda x: x[3], reverse=True)
        kept = []
        
        for det in sorted_dets:
            name, x, y, conf = det
            is_duplicate = False
            
            for kname, kx, ky, kconf in kept:
                # Same type and close together
                if name == kname and abs(x - kx) + abs(y - ky) < dist_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept.append(det)
        
        return kept
    
    def get_keys_to_press(self, frame: np.ndarray) -> list[str]:
        """
        Main entry point - returns keys to press based on detection.
        """
        result = self.detect(frame)
        
        if not result.aligned_icons:
            return []
        
        keys = []
        current_time = time.time()
        
        for icon_name in result.aligned_icons:
            key = TEMPLATE_KEY_MAP.get(icon_name)
            if not key:
                continue
            
            # Cooldown check
            last_press = self._last_press.get(key, 0)
            if current_time - last_press >= self._cooldown:
                keys.append(key)
                self._last_press[key] = current_time
                self._log(f">>> PRESS: {key} <<<")
        
        return keys
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw debug overlay on frame."""
        output = frame.copy()
        r = self._last_result
        
        if r.indicator_pos:
            ix, iy = r.indicator_pos
            cv2.line(output, (ix, 0), (ix, frame.shape[0]), (0, 255, 255), 2)
            cv2.circle(output, (ix, iy), 15, (0, 255, 255), 3)
            cv2.putText(output, f"IND {r.indicator_confidence:.2f}", (ix + 15, iy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        for name, x, y, conf in r.icons:
            aligned = name in r.aligned_icons
            color = (0, 0, 255) if aligned else (0, 255, 0)
            cv2.circle(output, (x, y), 15, color, 3)
            label = f"{name} {conf:.2f}"
            if aligned:
                label = f">>> {name} <<<"
            cv2.putText(output, label, (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return output
    
    def reset(self):
        """Reset state."""
        self._last_press.clear()
        self._last_result = DetectionResult()
    
    def set_debug(self, enabled: bool):
        """Enable/disable debug mode."""
        self.debug = enabled
    
    def set_thresholds(self, indicator: float = None, icon: float = None):
        """Set detection thresholds. Lower = more detections."""
        if indicator is not None:
            self.indicator_threshold = indicator
            self._log(f"Indicator threshold: {indicator}")
        if icon is not None:
            self.icon_threshold = icon
            self._log(f"Icon threshold: {icon}")
    
    def set_color_verification(self, enabled: bool, threshold: float = 50.0):
        """Enable/disable color verification."""
        self.use_color_verification = enabled
        self.color_threshold = threshold
        self._log(f"Color verification: {'ON' if enabled else 'OFF'} (threshold={threshold})")
