"""
YOLO-based ML Detector for iDate Revival

Uses Ultralytics YOLO for highly accurate object detection.
Trained on synthetic data generated from templates.

Features:
- Real-time 60fps inference
- High accuracy with minimal false positives  
- GPU acceleration when available
- Automatic model loading/training
"""

from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np


# Class names (must match training data order)
CLASS_NAMES = ['indicator', 'up', 'down', 'left', 'right', 'hand']

# Template file mapping for verification
TEMPLATE_FILES = {
    'indicator': 'indicator.png',
    'up': 'up.png',
    'down': 'down.png',
    'left': 'left.png',
    'right': 'right.png',
    'hand': 'hand.png',
}
CLASS_TO_KEY = {
    'up': 'up',
    'down': 'down',
    'left': 'left',
    'right': 'right',
    'hand': 'space',
}


@dataclass
class DetectionResult:
    """Detection results for a single frame."""
    indicator_pos: tuple[int, int] | None = None
    indicator_confidence: float = 0.0
    icons: list[tuple[str, int, int, float]] = field(default_factory=list)
    aligned_icons: list[str] = field(default_factory=list)
    inference_time_ms: float = 0.0


class YOLODetector:
    """
    YOLO-based detector for iDate Revival.
    
    Uses a custom-trained YOLO model for accurate detection.
    Falls back to a pretrained model if custom model not available.
    """
    
    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.4,  # Lower default for better detection
        alignment_tolerance: int = 35,
        device: str = 'auto'
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to trained YOLO model (.pt file)
            confidence_threshold: Minimum confidence for detections
            alignment_tolerance: X-distance tolerance for key triggering
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.model_path = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self.alignment_tolerance = alignment_tolerance
        self.device = device
        self.inference_size = 320  # Smaller for faster inference
        
        self.model = None
        self._last_result = DetectionResult()
        
        # Key cooldown - optimized for faster response
        self._last_press: dict[str, float] = {}
        self._cooldown = 0.08  # 80ms for faster reaction
        
        # Debug
        self.debug = False
        self._log_callback: Callable[[str], None] | None = None
        
        # Template verification (multi-layered approach)
        self.use_template_verification = True
        self.template_threshold = 0.65  # Match threshold for verification
        self._templates: dict[str, np.ndarray] = {}
        self._load_templates()
        
        # Load model
        self._load_model()
    
    def _log(self, msg: str):
        """Debug logging."""
        if self.debug:
            print(f"[YOLO] {msg}")
        if self._log_callback:
            self._log_callback(msg)
    
    def set_log_callback(self, callback: Callable[[str], None]):
        """Set logging callback."""
        self._log_callback = callback
    
    def _load_templates(self):
        """
        Load templates for verification (multi-layered detection).
        Templates are used to verify YOLO detections and filter false positives.
        """
        templates_dir = Path("templates")
        if not templates_dir.exists():
            # Try relative to project root
            templates_dir = Path(__file__).parent.parent.parent / "templates"
        
        if not templates_dir.exists():
            self._log("Templates directory not found - verification disabled")
            self.use_template_verification = False
            return
        
        for class_name, filename in TEMPLATE_FILES.items():
            template_path = templates_dir / filename
            if template_path.exists():
                template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
                if template is not None:
                    self._templates[class_name] = template
                    self._log(f"Loaded template: {class_name} ({template.shape})")
        
        if self._templates:
            self._log(f"Template verification enabled ({len(self._templates)} templates)")
        else:
            self._log("No templates loaded - verification disabled")
            self.use_template_verification = False
    
    def _verify_detection_with_template(
        self, 
        frame: np.ndarray, 
        class_name: str, 
        x1: int, y1: int, x2: int, y2: int
    ) -> tuple[bool, float]:
        """
        Verify a YOLO detection using template matching.
        
        Args:
            frame: Full BGR frame
            class_name: Detected class name
            x1, y1, x2, y2: Bounding box coordinates
            
        Returns:
            (is_verified, match_score)
        """
        if not self.use_template_verification:
            return True, 1.0
        
        if class_name not in self._templates:
            return True, 1.0  # No template = auto-accept
        
        template = self._templates[class_name]
        
        # Expand ROI slightly for better matching (templates may not fit exactly)
        h, w = frame.shape[:2]
        th, tw = template.shape[:2]
        
        # Add padding (50% of template size)
        pad_x = int(tw * 0.5)
        pad_y = int(th * 0.5)
        
        rx1 = max(0, int(x1) - pad_x)
        ry1 = max(0, int(y1) - pad_y)
        rx2 = min(w, int(x2) + pad_x)
        ry2 = min(h, int(y2) + pad_y)
        
        # Crop region
        roi = frame[ry1:ry2, rx1:rx2]
        
        if roi.shape[0] < th or roi.shape[1] < tw:
            # ROI too small for template matching
            return True, 0.5  # Accept with low score
        
        try:
            # Template matching
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            is_verified = max_val >= self.template_threshold
            
            if self.debug:
                status = "OK" if is_verified else "REJECT"
                self._log(f"Template verify {class_name}: {max_val:.3f} {status}")
            
            return is_verified, max_val
            
        except Exception as e:
            self._log(f"Template match error: {e}")
            return True, 0.5  # Accept on error
    
    def _load_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            
            # Try to load custom trained model
            if self.model_path and self.model_path.exists():
                self._log(f"Loading custom model: {self.model_path}")
                self.model = YOLO(str(self.model_path))
            else:
                # Look for model in standard locations
                search_paths = [
                    Path("models/idate_best.pt"),
                    Path("runs/detect/train/weights/best.pt"),
                    Path("runs/detect/idate/weights/best.pt"),
                ]
                
                for path in search_paths:
                    if path.exists():
                        self._log(f"Found model: {path}")
                        self.model = YOLO(str(path))
                        break
                
                if self.model is None:
                    # Use pretrained YOLOv8n as fallback (requires training)
                    self._log("No trained model found! Please run training first.")
                    self._log("Using YOLOv8n base model (won't detect iDate objects)")
                    self.model = YOLO("yolov8n.pt")
            
            # Move to device
            if self.device == 'auto':
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self._log(f"Model loaded on device: {self.device}")
            self._log(f"Model classes: {self.model.names if hasattr(self.model, 'names') else 'N/A'}")
            
        except ImportError:
            raise ImportError("Ultralytics not installed! Run: pip install ultralytics")
    
    @property
    def is_trained(self) -> bool:
        """Check if using a properly trained model."""
        if self.model is None:
            return False
        # Check if model has our custom classes
        if hasattr(self.model, 'names'):
            return 'indicator' in self.model.names.values()
        return False
    
    @property
    def indicator_templates(self) -> list:
        """Compatibility property."""
        return [True] if self.is_trained else []
    
    @property
    def templates(self) -> dict:
        """Compatibility property."""
        if self.is_trained:
            return {name: True for name in CLASS_NAMES if name != 'indicator'}
        return {}
    
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run YOLO inference on frame.
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            DetectionResult with all detections
        """
        if self.model is None:
            return DetectionResult()
        
        start_time = time.perf_counter()
        
        # Run inference - optimized for speed
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
            imgsz=self.inference_size,
            half=False,  # Keep FP32 for CPU
            max_det=50   # Limit max detections
        )
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        result = DetectionResult(inference_time_ms=inference_time)
        
        if not results or len(results) == 0:
            self._last_result = result
            return result
        
        # Parse detections
        detections = results[0]
        
        if detections.boxes is None or len(detections.boxes) == 0:
            self._last_result = result
            return result
        
        boxes = detections.boxes
        
        for i in range(len(boxes)):
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            
            # Get class name
            if hasattr(self.model, 'names') and cls_id in self.model.names:
                class_name = self.model.names[cls_id]
            elif cls_id < len(CLASS_NAMES):
                class_name = CLASS_NAMES[cls_id]
            else:
                continue
            
            # === MULTI-LAYERED VERIFICATION ===
            # Verify YOLO detection with template matching to eliminate false positives
            is_verified, match_score = self._verify_detection_with_template(
                frame, class_name, x1, y1, x2, y2
            )
            
            if not is_verified:
                # False positive - skip this detection
                if self.debug:
                    self._log(f"REJECTED: {class_name} (YOLO conf={conf:.2f}, template={match_score:.2f})")
                continue
            
            # Calculate center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            if class_name == 'indicator':
                # Use highest confidence indicator
                if conf > result.indicator_confidence:
                    result.indicator_pos = (cx, cy)
                    result.indicator_confidence = conf
            else:
                result.icons.append((class_name, cx, cy, conf))
        
        # Determine aligned icons
        if result.indicator_pos:
            ind_x = result.indicator_pos[0]
            for name, icon_x, icon_y, conf in result.icons:
                if abs(icon_x - ind_x) <= self.alignment_tolerance:
                    result.aligned_icons.append(name)
        
        self._last_result = result
        
        if self.debug and (result.indicator_pos or result.icons):
            self._log(f"Detected: ind={result.indicator_pos}, icons={len(result.icons)}, aligned={result.aligned_icons}")
        
        return result
    
    def get_keys_to_press(self, frame: np.ndarray) -> list[str]:
        """
        Main detection function - returns keys to press.
        
        Args:
            frame: BGR image
            
        Returns:
            List of key names to press
        """
        result = self.detect(frame)
        
        if not result.aligned_icons:
            return []
        
        keys = []
        current_time = time.time()
        
        for icon_name in result.aligned_icons:
            key = CLASS_TO_KEY.get(icon_name)
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
        """Draw detection visualization on frame."""
        output = frame.copy()
        r = self._last_result
        
        # Draw indicator
        if r.indicator_pos:
            ix, iy = r.indicator_pos
            cv2.line(output, (ix, 0), (ix, frame.shape[0]), (0, 255, 255), 2)
            cv2.circle(output, (ix, iy), 15, (0, 255, 255), 3)
            cv2.putText(output, f"IND {r.indicator_confidence:.2f}", (ix + 15, iy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw icons
        for name, x, y, conf in r.icons:
            aligned = name in r.aligned_icons
            color = (0, 0, 255) if aligned else (0, 255, 0)
            cv2.circle(output, (x, y), 15, color, 3)
            label = f">>> {name} <<<" if aligned else f"{name} {conf:.2f}"
            cv2.putText(output, label, (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw inference time
        cv2.putText(output, f"YOLO: {r.inference_time_ms:.1f}ms", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return output
    
    def reset(self):
        """Reset detection state."""
        self._last_press.clear()
        self._last_result = DetectionResult()


def train_model(
    data_yaml: str | Path,
    output_dir: str | Path = "runs/detect/idate",
    epochs: int = 30,
    img_size: int = 320,
    batch_size: int = 16,
    base_model: str = "yolov8n.pt"
):
    """
    Train a YOLO model on the generated dataset.
    
    OPTIMIZED FOR FAST CPU TRAINING:
    - Uses yolov8n (smallest, fastest)
    - 320px images (fast training, still accurate for game icons)
    - RAM caching for 5-10x faster training
    - Early stopping (patience=10)
    - Adam optimizer (faster convergence)
    
    Args:
        data_yaml: Path to dataset configuration YAML
        output_dir: Output directory for training results
        epochs: Number of training epochs
        img_size: Training image size
        batch_size: Batch size for training
        base_model: Base YOLO model to fine-tune
    """
    from ultralytics import YOLO
    import torch
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"[TRAIN] Loading base model: {base_model}")
    print(f"[TRAIN] Device: {device}")
    model = YOLO(base_model)
    
    print(f"[TRAIN] Starting FAST CPU training...")
    print(f"[TRAIN] Dataset: {data_yaml}")
    print(f"[TRAIN] Epochs: {epochs}")
    print(f"[TRAIN] Image size: {img_size}")
    print(f"[TRAIN] Batch size: {batch_size}")
    print(f"[TRAIN] RAM caching enabled for speed")
    print("[TRAIN] Estimated time: 15-30 minutes on CPU")
    print("=" * 60)
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=str(Path(output_dir).parent),
        name=Path(output_dir).name,
        exist_ok=True,
        patience=10,  # Early stopping - stop if no improvement for 10 epochs
        save=True,
        plots=True,
        verbose=True,
        workers=0,  # Windows compatibility
        cache='ram',  # Cache images in RAM for much faster training!
        amp=False,  # Disable mixed precision for CPU stability
        optimizer='Adam',  # Faster convergence than SGD
        lr0=0.001,  # Lower LR for Adam
        lrf=0.1,
        mosaic=0.0,  # Disable mosaic for faster training
        augment=False,  # Minimal augmentation - synthetic data is already varied
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        fliplr=0.0,  # Don't flip - icons have specific orientations
        scale=0.0,
        translate=0.0,
    )
    
    # Copy best model to accessible location
    best_model = Path(output_dir) / "weights" / "best.pt"
    if best_model.exists():
        target = Path("models/idate_best.pt")
        target.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(best_model, target)
        print(f"[TRAIN] Best model saved to: {target}")
    
    return results


if __name__ == "__main__":
    # Test detection with trained model
    detector = YOLODetector()
    print(f"Model trained: {detector.is_trained}")
    print(f"Templates: {detector.templates}")
