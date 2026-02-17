"""
iDate Revival Detector - HIGH ACCURACY TEMPLATE MATCHING

FALSE POSITIVE PREVENTION:
1. High similarity threshold (0.90+)
2. Color verification - verify matched region color matches template
3. Template signature - store average color of each template
4. Strict NMS - prevent duplicate detections

Detection flow:
1. Find indicator template position (the bouncing cursor)
2. Find all note icons with template matching
3. Verify each match with color comparison (eliminates false positives)
4. Check if indicator X-position aligns with any verified note X-position
5. If aligned, press the corresponding key
"""

from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import NamedTuple

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


class TemplateInfo(NamedTuple):
    """Template with verification data."""
    image: np.ndarray          # Grayscale template
    image_color: np.ndarray    # BGR template for color verification
    avg_color: tuple[int, int, int]  # Average BGR color
    width: int
    height: int


@dataclass
class DetectionResult:
    """Detection results for a single frame."""
    indicator_pos: tuple[int, int] | None = None
    indicator_confidence: float = 0.0
    icons: list[tuple[str, int, int, float]] = field(default_factory=list)
    aligned_icons: list[str] = field(default_factory=list)


def get_average_color(image: np.ndarray) -> tuple[int, int, int]:
    """Get average BGR color of an image region."""
    if len(image.shape) == 2:
        avg = int(np.mean(image))
        return (avg, avg, avg)
    else:
        b = int(np.mean(image[:, :, 0]))
        g = int(np.mean(image[:, :, 1]))
        r = int(np.mean(image[:, :, 2]))
        return (b, g, r)


def color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """Calculate color distance (0-255 range)."""
    return (abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])) / 3.0


class IdateDetector:
    """
    High-accuracy template detector for iDate Revival.
    
    Uses template matching + color verification to eliminate false positives.
    """
    
    def __init__(self, template_dir: str | Path = "templates"):
        self.template_dir = Path(template_dir)
        
        # Templates: name -> TemplateInfo
        self.templates: dict[str, TemplateInfo] = {}
        self.indicator_info: TemplateInfo | None = None
        
        # === DETECTION THRESHOLDS (tuned for accuracy) ===
        self.indicator_threshold = 0.85  # Indicator needs high match
        self.icon_threshold = 0.88       # Icons need VERY high match
        
        # Color verification threshold (0-255)
        self.color_threshold = 35.0  # Max color difference allowed
        
        # Enable/disable color verification
        self.use_color_verification = True
        
        # Alignment tolerance (pixels)
        self.alignment_tolerance = 30
        
        # Cooldown between same key presses (seconds)
        self._last_press: dict[str, float] = {}
        self._cooldown = 0.15
        
        # Cached result for overlay
        self._last_result: DetectionResult = DetectionResult()
        
        # Debug mode
        self.debug = False
        
        # Load templates
        self._load_templates()
    
    @property
    def indicator_templates(self) -> list:
        """Compatibility property."""
        return [self.indicator_info.image] if self.indicator_info is not None else []
    
    def _load_template(self, filepath: Path) -> TemplateInfo | None:
        """Load a template with color information for verification."""
        if not filepath.exists():
            return None
        
        img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        
        # Handle alpha channel
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_bgr = img[:, :, :3]
        elif len(img.shape) == 3:
            img_bgr = img
        else:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Convert to grayscale for matching
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Get average color for verification
        avg_color = get_average_color(img_bgr)
        
        h, w = img_gray.shape[:2]
        
        return TemplateInfo(
            image=img_gray,
            image_color=img_bgr,
            avg_color=avg_color,
            width=w,
            height=h
        )
    
    def _load_templates(self) -> None:
        """Load all templates from directory."""
        if not self.template_dir.exists():
            print(f"[DETECTOR] Template directory not found: {self.template_dir}")
            return
        
        # Load indicator template
        indicator_path = self.template_dir / "indicator.png"
        self.indicator_info = self._load_template(indicator_path)
        if self.indicator_info:
            print(f"[DETECTOR] Loaded indicator: {self.indicator_info.width}x{self.indicator_info.height}, "
                  f"color={self.indicator_info.avg_color}")
        else:
            print(f"[DETECTOR] WARNING: indicator.png not found!")
        
        # Load note icon templates
        for name in TEMPLATE_KEY_MAP.keys():
            filepath = self.template_dir / f"{name}.png"
            info = self._load_template(filepath)
            if info:
                self.templates[name] = info
                print(f"[DETECTOR] Loaded {name}: {info.width}x{info.height}, color={info.avg_color}")
    
    def _verify_color(self, frame_bgr: np.ndarray, x: int, y: int, 
                      template: TemplateInfo) -> tuple[bool, float]:
        """
        Verify that the matched region has correct color.
        This eliminates false positives from similar shapes.
        """
        if not self.use_color_verification:
            return True, 0.0
        
        h, w = template.height, template.width
        
        # Get the matched region from the frame
        y1 = max(0, y - h // 2)
        y2 = min(frame_bgr.shape[0], y + h // 2)
        x1 = max(0, x - w // 2)
        x2 = min(frame_bgr.shape[1], x + w // 2)
        
        if y2 - y1 < 5 or x2 - x1 < 5:
            return False, 255.0
        
        region = frame_bgr[y1:y2, x1:x2]
        region_color = get_average_color(region)
        
        dist = color_distance(region_color, template.avg_color)
        
        if self.debug:
            print(f"[COLOR] region={region_color}, template={template.avg_color}, dist={dist:.1f}")
        
        return dist <= self.color_threshold, dist
    
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run full detection on frame with color verification."""
        # Keep original BGR for color verification
        if len(frame.shape) == 3:
            frame_bgr = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        result = DetectionResult()
        
        # Step 1: Find indicator
        result.indicator_pos, result.indicator_confidence = self._find_indicator(gray, frame_bgr)
        
        # Step 2: Find all note icons (with color verification)
        result.icons = self._find_icons(gray, frame_bgr)
        
        # Step 3: Determine which icons are aligned with indicator
        if result.indicator_pos:
            ind_x = result.indicator_pos[0]
            for name, icon_x, _, _ in result.icons:
                if abs(icon_x - ind_x) <= self.alignment_tolerance:
                    result.aligned_icons.append(name)
        
        self._last_result = result
        return result
    
    def _find_indicator(self, gray: np.ndarray, bgr: np.ndarray) -> tuple[tuple[int, int] | None, float]:
        """Find indicator position using template matching + color verification."""
        if self.indicator_info is None:
            return None, 0.0
        
        t = self.indicator_info
        
        if t.height > gray.shape[0] or t.width > gray.shape[1]:
            return None, 0.0
        
        result = cv2.matchTemplate(gray, t.image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= self.indicator_threshold:
            cx = max_loc[0] + t.width // 2
            cy = max_loc[1] + t.height // 2
            
            # Color verification
            if self.use_color_verification:
                is_valid, dist = self._verify_color(bgr, cx, cy, t)
                if not is_valid:
                    if self.debug:
                        print(f"[DETECTOR] Indicator rejected by color: dist={dist:.1f}")
                    return None, max_val
            
            return (cx, cy), max_val
        
        return None, max_val
    
    def _find_icons(self, gray: np.ndarray, bgr: np.ndarray) -> list[tuple[str, int, int, float]]:
        """Find all note icons using template matching + color verification."""
        all_detections = []
        
        for name, template in self.templates.items():
            if template.height > gray.shape[0] or template.width > gray.shape[1]:
                continue
            
            result = cv2.matchTemplate(gray, template.image, cv2.TM_CCOEFF_NORMED)
            
            # Find all matches above threshold
            locations = np.where(result >= self.icon_threshold)
            
            for pt in zip(*locations[::-1]):
                conf = float(result[pt[1], pt[0]])
                cx = pt[0] + template.width // 2
                cy = pt[1] + template.height // 2
                
                # Color verification - CRITICAL for eliminating false positives
                if self.use_color_verification:
                    is_valid, dist = self._verify_color(bgr, cx, cy, template)
                    if not is_valid:
                        if self.debug:
                            print(f"[DETECTOR] {name} rejected at ({cx},{cy}): color_dist={dist:.1f}")
                        continue
                
                all_detections.append((name, cx, cy, conf))
        
        return self._nms(all_detections)
    
    def _nms(self, detections: list[tuple[str, int, int, float]], dist: int = 40) -> list[tuple[str, int, int, float]]:
        """Non-maximum suppression - keep highest confidence detection per area."""
        if not detections:
            return []
        
        sorted_d = sorted(detections, key=lambda x: x[3], reverse=True)
        kept = []
        
        for det in sorted_d:
            name, x, y, conf = det
            is_duplicate = False
            
            for kept_name, kx, ky, _ in kept:
                if name == kept_name and abs(x - kx) + abs(y - ky) < dist:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept.append(det)
        
        return kept
    
    def get_keys_to_press(self, frame: np.ndarray) -> list[str]:
        """
        Main detection function.
        Returns keys to press ONLY when indicator aligns with verified icons.
        """
        result = self.detect(frame)
        
        if result.indicator_pos is None or not result.icons:
            return []
        
        ind_x = result.indicator_pos[0]
        keys_to_press = []
        current_time = time.time()
        
        for name, icon_x, icon_y, confidence in result.icons:
            x_distance = abs(icon_x - ind_x)
            
            if x_distance <= self.alignment_tolerance:
                key = TEMPLATE_KEY_MAP.get(name)
                if key:
                    last_press_time = self._last_press.get(key, 0)
                    if current_time - last_press_time >= self._cooldown:
                        keys_to_press.append(key)
                        self._last_press[key] = current_time
                        
                        if self.debug:
                            print(f"[DETECTOR] PRESS: {name} -> {key} (dist={x_distance}px)")
        
        return keys_to_press
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection visualization on frame."""
        output = frame.copy()
        r = self._last_result
        
        if r.indicator_pos:
            ix, iy = r.indicator_pos
            cv2.line(output, (ix, 0), (ix, frame.shape[0]), (0, 255, 255), 2)
            cv2.circle(output, (ix, iy), 12, (0, 255, 255), 3)
            cv2.putText(output, f"IND {r.indicator_confidence:.2f}", (ix + 15, iy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        for name, x, y, conf in r.icons:
            aligned = name in r.aligned_icons
            color = (0, 0, 255) if aligned else (0, 255, 0)
            
            cv2.circle(output, (x, y), 15, color, 3)
            label = f"{name} {conf:.2f}"
            if aligned:
                label = f">>> {name} <<<"
            cv2.putText(output, label, (x + 20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return output
    
    def reset(self) -> None:
        """Reset detection state."""
        self._last_press.clear()
        self._last_result = DetectionResult()
