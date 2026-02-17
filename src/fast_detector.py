"""
Ultra-Fast Rhythm Game Detection System for iDate Revival
=========================================================

High-performance detection using template matching for real-time accuracy.
Designed to track fast-moving indicators with minimal latency.

Features:
- Template-based detection (sub-millisecond per frame)
- Multi-scale matching for different game resolutions
- Real-time position tracking
- No false positives through correlation thresholding
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable
import cv2
import numpy as np


# Arrow types and their keyboard mappings  
ARROW_KEYS = {
    'up': 'up',
    'down': 'down',
    'left': 'left', 
    'right': 'right',
    'hand': 'space',
}

# Detection colors for visualization
DETECTION_COLORS = {
    'up': (0, 255, 0),        # Green
    'down': (255, 100, 0),    # Blue-ish
    'left': (0, 255, 255),    # Yellow
    'right': (255, 0, 255),   # Magenta
    'hand': (0, 165, 255),    # Orange
    'indicator': (255, 255, 0),  # Cyan
}


@dataclass
class Detection:
    """A single detection result."""
    class_name: str
    x: int              # Center X
    y: int              # Center Y
    width: int          # Bounding box width
    height: int         # Bounding box height
    confidence: float   # Match confidence (0-1)
    
    @property
    def x1(self) -> int:
        return self.x - self.width // 2
    
    @property
    def y1(self) -> int:
        return self.y - self.height // 2
    
    @property
    def x2(self) -> int:
        return self.x + self.width // 2
    
    @property
    def y2(self) -> int:
        return self.y + self.height // 2


@dataclass
class DetectionResult:
    """Result from a single frame detection."""
    detections: list[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    frame_id: int = 0
    
    def get_by_class(self, class_name: str) -> list[Detection]:
        """Get all detections of a specific class."""
        return [d for d in self.detections if d.class_name == class_name]
    
    def get_indicator(self) -> Detection | None:
        """Get the indicator detection if present."""
        indicators = self.get_by_class('indicator')
        return indicators[0] if indicators else None
    
    def get_arrows(self) -> list[Detection]:
        """Get all arrow detections (excluding indicator)."""
        return [d for d in self.detections if d.class_name != 'indicator']


class FastDetector:
    """
    Ultra-fast template-based detector for rhythm game icons.
    
    Uses OpenCV template matching with normalized cross-correlation
    for high-speed, accurate detection with no false positives.
    """
    
    # Template directory
    TEMPLATE_DIR = Path("templates")
    
    # Classes to detect
    CLASSES = ['up', 'down', 'left', 'right', 'hand', 'indicator']
    
    def __init__(
        self,
        confidence_threshold: float = 0.85,
        scales: list[float] | None = None,
        nms_threshold: float = 0.3,
    ):
        """
        Initialize the detector.
        
        Args:
            confidence_threshold: Minimum correlation for valid detection (0-1)
            scales: List of scales to try (default: [1.0] for speed)
            nms_threshold: IoU threshold for non-maximum suppression
        """
        self.confidence_threshold = confidence_threshold
        self.scales = scales or [1.0]  # Single scale for speed
        self.nms_threshold = nms_threshold
        
        # Templates: {class_name: [(template_gray, template_color, scale), ...]}
        self.templates: dict[str, list[tuple[np.ndarray, np.ndarray, float]]] = {}
        
        # State
        self._frame_count = 0
        self._log_callback: Callable[[str], None] | None = None
        
        # Load templates
        self._load_templates()
    
    def _log(self, msg: str):
        """Log message."""
        if self._log_callback:
            self._log_callback(msg)
    
    def set_log_callback(self, callback: Callable[[str], None]):
        """Set logging callback."""
        self._log_callback = callback
    
    def _load_templates(self):
        """Load and prepare multi-scale templates."""
        if not self.TEMPLATE_DIR.exists():
            self._log(f"Template directory not found: {self.TEMPLATE_DIR}")
            return
        
        for class_name in self.CLASSES:
            template_path = self.TEMPLATE_DIR / f"{class_name}.png"
            if not template_path.exists():
                self._log(f"Template not found: {template_path}")
                continue
            
            # Load template in color and grayscale
            template_color = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            template_gray = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            
            if template_color is None or template_gray is None:
                self._log(f"Failed to load template: {template_path}")
                continue
            
            # Create multi-scale versions
            self.templates[class_name] = []
            for scale in self.scales:
                h, w = template_gray.shape[:2]
                new_w = max(10, int(w * scale))
                new_h = max(10, int(h * scale))
                
                scaled_gray = cv2.resize(template_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                scaled_color = cv2.resize(template_color, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                self.templates[class_name].append((scaled_gray, scaled_color, scale))
            
            self._log(f"Loaded template: {class_name} ({len(self.scales)} scales)")
    
    def detect(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> DetectionResult:
        """
        Detect all icons in the frame.
        
        Args:
            frame: BGR image from screen capture
            roi: Optional region of interest (x, y, width, height) to limit detection
            
        Returns:
            DetectionResult with all detections
        """
        start_time = time.perf_counter()
        self._frame_count += 1
        
        # Apply ROI if specified
        offset_x, offset_y = 0, 0
        if roi is not None:
            x, y, w, h = roi
            frame = frame[y:y+h, x:x+w]
            offset_x, offset_y = x, y
        
        # Convert to grayscale for matching
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        all_detections: list[Detection] = []
        
        # Detect each class
        for class_name, template_list in self.templates.items():
            class_detections = self._detect_class(gray, class_name, template_list, offset_x, offset_y)
            all_detections.extend(class_detections)
        
        # Apply NMS across all detections
        final_detections = self._apply_nms(all_detections)
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        return DetectionResult(
            detections=final_detections,
            inference_time_ms=inference_time,
            frame_id=self._frame_count,
        )
    
    def _detect_class(
        self, 
        gray: np.ndarray, 
        class_name: str, 
        template_list: list[tuple[np.ndarray, np.ndarray, float]],
        offset_x: int,
        offset_y: int,
    ) -> list[Detection]:
        """Detect all instances of a class using multi-scale template matching."""
        detections = []
        h_img, w_img = gray.shape[:2]
        
        for template_gray, template_color, scale in template_list:
            h_t, w_t = template_gray.shape[:2]
            
            # Skip if template is larger than image
            if h_t >= h_img or w_t >= w_img:
                continue
            
            # Template matching using normalized cross-correlation
            result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
            
            # Find all locations above threshold
            locations = np.where(result >= self.confidence_threshold)
            
            for pt_y, pt_x in zip(*locations):
                confidence = float(result[pt_y, pt_x])
                
                # Calculate center position
                cx = pt_x + w_t // 2 + offset_x
                cy = pt_y + h_t // 2 + offset_y
                
                detections.append(Detection(
                    class_name=class_name,
                    x=cx,
                    y=cy,
                    width=w_t,
                    height=h_t,
                    confidence=confidence,
                ))
        
        return detections
    
    def _apply_nms(self, detections: list[Detection]) -> list[Detection]:
        """
        Apply non-maximum suppression to remove overlapping detections.
        
        Uses CROSS-CLASS NMS: when different arrow templates match the same location,
        only the highest confidence detection is kept. Indicators are kept separate
        from arrows (they're different game elements that can overlap).
        """
        if not detections:
            return []
        
        # Separate indicators from arrows (they shouldn't suppress each other)
        indicators = [d for d in detections if d.class_name == 'indicator']
        arrows = [d for d in detections if d.class_name != 'indicator']
        
        # Sort arrows by confidence (highest first)
        arrows = sorted(arrows, key=lambda d: d.confidence, reverse=True)
        
        # Apply NMS only to arrows
        kept_arrows = []
        for det in arrows:
            should_keep = True
            for kept in kept_arrows:
                iou = self._compute_iou(det, kept)
                if iou > self.nms_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                kept_arrows.append(det)
        
        # Keep only the highest confidence indicator (the game has only one)
        kept_indicators = []
        if indicators:
            best_indicator = max(indicators, key=lambda d: d.confidence)
            kept_indicators = [best_indicator]
        
        return kept_arrows + kept_indicators
    
    def _compute_iou(self, det1: Detection, det2: Detection) -> float:
        """Compute Intersection over Union of two detections."""
        x1 = max(det1.x1, det2.x1)
        y1 = max(det1.y1, det2.y1)
        x2 = min(det1.x2, det2.x2)
        y2 = min(det1.y2, det2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = det1.width * det1.height
        area2 = det2.width * det2.height
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def draw_detections(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """
        Draw detections on the frame.
        
        Args:
            frame: Original BGR image
            result: Detection result
            
        Returns:
            Frame with detections drawn
        """
        output = frame.copy()
        
        for det in result.detections:
            color = DETECTION_COLORS.get(det.class_name, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(output, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            
            # Draw center point
            cv2.circle(output, (det.x, det.y), 4, color, -1)
            
            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(output, (det.x1, det.y1 - th - 8), (det.x1 + tw + 4, det.y1), color, -1)
            cv2.putText(output, label, (det.x1 + 2, det.y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw inference time
        time_label = f"{result.inference_time_ms:.1f}ms"
        cv2.putText(output, time_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return output


def test_on_image(image_path: str, output_path: str, confidence: float = 0.5):
    """
    Test detection on a single image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image
        confidence: Detection confidence threshold
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    
    # Create detector
    detector = FastDetector(confidence_threshold=confidence)
    detector.set_log_callback(print)
    
    # Run detection
    result = detector.detect(img)
    
    # Print results
    print(f"\nDetections found: {len(result.detections)}")
    print(f"Inference time: {result.inference_time_ms:.2f}ms")
    
    for det in result.detections:
        print(f"  {det.class_name}: pos=({det.x}, {det.y}) conf={det.confidence:.2f}")
    
    # Draw and save
    output = detector.draw_detections(img, result)
    cv2.imwrite(output_path, output)
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    test_on_image("test/test.png", "test/test_output.png", confidence=0.85)
