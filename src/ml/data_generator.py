"""
YOLO Training Data Generator for iDate Revival

Generates synthetic training data from template images:
1. Creates diverse backgrounds (solid, gradient, noise)
2. Places templates at random positions with augmentation
3. Outputs YOLO format annotations (class x_center y_center width height)
4. Applies augmentation: rotation, scaling, brightness, contrast

Run this script to generate training data before training the YOLO model.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np


# Class mapping (same order as data.yaml)
CLASS_NAMES = ['indicator', 'up', 'down', 'left', 'right', 'hand']
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def create_random_background(width: int, height: int) -> np.ndarray:
    """Create a random background image."""
    bg_type = random.choice(['solid', 'gradient', 'noise', 'game_like'])
    
    if bg_type == 'solid':
        # Random solid color (dark to medium tones like game UI)
        color = [random.randint(20, 100) for _ in range(3)]
        bg = np.full((height, width, 3), color, dtype=np.uint8)
    
    elif bg_type == 'gradient':
        # Vertical or horizontal gradient
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        c1 = [random.randint(20, 80) for _ in range(3)]
        c2 = [random.randint(40, 120) for _ in range(3)]
        
        for i in range(height):
            ratio = i / height
            color = [int(c1[j] * (1 - ratio) + c2[j] * ratio) for j in range(3)]
            bg[i, :] = color
    
    elif bg_type == 'noise':
        # Noisy background
        base = random.randint(30, 70)
        bg = np.random.randint(base - 20, base + 20, (height, width, 3), dtype=np.uint8)
    
    else:  # game_like
        # Dark background with some structure
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        base_color = [random.randint(20, 50) for _ in range(3)]
        bg[:] = base_color
        
        # Add some horizontal lines (like rhythm game lanes)
        num_lanes = random.randint(4, 8)
        lane_height = height // num_lanes
        for i in range(num_lanes):
            y = i * lane_height
            line_color = [min(255, c + random.randint(10, 30)) for c in base_color]
            cv2.line(bg, (0, y), (width, y), line_color, 2)
    
    return bg


def augment_template(template: np.ndarray, alpha: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply augmentation to template image."""
    h, w = template.shape[:2]
    
    # Random scaling (0.8x to 1.4x)
    scale = random.uniform(0.8, 1.4)
    new_w = int(w * scale)
    new_h = int(h * scale)
    template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if alpha is not None:
        alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Random brightness/contrast
    brightness = random.uniform(0.7, 1.3)
    contrast = random.uniform(0.8, 1.2)
    template = np.clip(template * contrast + (brightness - 1) * 50, 0, 255).astype(np.uint8)
    
    # Random small rotation (-15 to +15 degrees)
    if random.random() < 0.3:
        angle = random.uniform(-15, 15)
        center = (new_w // 2, new_h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        template = cv2.warpAffine(template, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)
        if alpha is not None:
            alpha = cv2.warpAffine(alpha, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT)
    
    return template, alpha


def place_template_on_background(
    bg: np.ndarray, 
    template: np.ndarray, 
    alpha: np.ndarray | None,
    x: int, y: int
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Place template on background at position (x, y).
    Returns modified background and bounding box (x_center, y_center, w, h).
    """
    bg_h, bg_w = bg.shape[:2]
    t_h, t_w = template.shape[:2]
    
    # Calculate placement bounds
    x1 = max(0, x - t_w // 2)
    y1 = max(0, y - t_h // 2)
    x2 = min(bg_w, x1 + t_w)
    y2 = min(bg_h, y1 + t_h)
    
    # Adjust for edge cases
    t_x1 = 0 if x1 >= 0 else -x1
    t_y1 = 0 if y1 >= 0 else -y1
    t_x2 = t_x1 + (x2 - x1)
    t_y2 = t_y1 + (y2 - y1)
    
    if x2 <= x1 or y2 <= y1:
        return bg, None
    
    # Get regions
    template_region = template[t_y1:t_y2, t_x1:t_x2]
    bg_region = bg[y1:y2, x1:x2]
    
    if alpha is not None:
        alpha_region = alpha[t_y1:t_y2, t_x1:t_x2]
        alpha_3ch = np.stack([alpha_region] * 3, axis=2) / 255.0
        blended = (template_region * alpha_3ch + bg_region * (1 - alpha_3ch)).astype(np.uint8)
        bg[y1:y2, x1:x2] = blended
    else:
        bg[y1:y2, x1:x2] = template_region
    
    # Calculate actual bounding box
    actual_w = x2 - x1
    actual_h = y2 - y1
    x_center = x1 + actual_w // 2
    y_center = y1 + actual_h // 2
    
    return bg, (x_center, y_center, actual_w, actual_h)


def load_template(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load template image with alpha channel if available."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load template: {path}")
    
    if len(img.shape) == 3 and img.shape[2] == 4:
        # Has alpha channel
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        return bgr, alpha
    else:
        return img, None


def generate_training_data(
    template_dir: str | Path = "templates",
    output_dir: str | Path = "datasets/idate",
    num_images: int = 2000,
    img_size: tuple[int, int] = (640, 640),
    max_objects_per_image: int = 8,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    Generate YOLO training dataset from templates.
    
    Args:
        template_dir: Path to template images
        output_dir: Output directory for dataset
        num_images: Number of training images to generate
        img_size: Output image size (width, height)
        max_objects_per_image: Maximum objects per generated image
        train_ratio: Ratio of training to validation images
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    template_dir = Path(template_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    train_images = output_dir / "images" / "train"
    train_labels = output_dir / "labels" / "train"
    val_images = output_dir / "images" / "val"
    val_labels = output_dir / "labels" / "val"
    
    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load templates
    templates = {}
    for class_name in CLASS_NAMES:
        path = template_dir / f"{class_name}.png"
        if path.exists():
            templates[class_name] = load_template(path)
            print(f"[DATA] Loaded template: {class_name} ({templates[class_name][0].shape})")
        else:
            print(f"[DATA] WARNING: Template not found: {path}")
    
    if not templates:
        raise ValueError("No templates found!")
    
    print(f"\n[DATA] Generating {num_images} synthetic images...")
    
    num_train = int(num_images * train_ratio)
    
    for i in range(num_images):
        # Determine if train or val
        is_train = i < num_train
        images_dir = train_images if is_train else val_images
        labels_dir = train_labels if is_train else val_labels
        
        # Create background
        bg = create_random_background(img_size[0], img_size[1])
        
        # Randomly select how many objects to place
        num_objects = random.randint(2, max_objects_per_image)
        
        # Track placed positions to avoid overlap
        placed = []
        annotations = []
        
        # Always try to include indicator
        if 'indicator' in templates:
            template_bgr, template_alpha = templates['indicator']
            aug_template, aug_alpha = augment_template(template_bgr.copy(), 
                                                        template_alpha.copy() if template_alpha is not None else None)
            
            # Place indicator somewhere in the frame
            x = random.randint(50, img_size[0] - 50)
            y = random.randint(100, img_size[1] - 100)
            
            bg, bbox = place_template_on_background(bg, aug_template, aug_alpha, x, y)
            if bbox:
                placed.append(bbox)
                # YOLO format: class x_center y_center width height (normalized)
                x_c, y_c, w, h = bbox
                annotations.append(f"{CLASS_TO_ID['indicator']} {x_c/img_size[0]:.6f} {y_c/img_size[1]:.6f} {w/img_size[0]:.6f} {h/img_size[1]:.6f}")
        
        # Place other objects
        icon_names = [n for n in CLASS_NAMES if n != 'indicator' and n in templates]
        for _ in range(num_objects - 1):
            if not icon_names:
                break
            
            class_name = random.choice(icon_names)
            template_bgr, template_alpha = templates[class_name]
            aug_template, aug_alpha = augment_template(template_bgr.copy(),
                                                        template_alpha.copy() if template_alpha is not None else None)
            
            # Find non-overlapping position
            for attempt in range(20):
                x = random.randint(50, img_size[0] - 50)
                y = random.randint(50, img_size[1] - 50)
                
                # Check overlap with existing
                overlap = False
                for px, py, pw, ph in placed:
                    if abs(x - px) < (pw + aug_template.shape[1]) // 2 + 20:
                        if abs(y - py) < (ph + aug_template.shape[0]) // 2 + 20:
                            overlap = True
                            break
                
                if not overlap:
                    break
            
            if not overlap:
                bg, bbox = place_template_on_background(bg, aug_template, aug_alpha, x, y)
                if bbox:
                    placed.append(bbox)
                    x_c, y_c, w, h = bbox
                    annotations.append(f"{CLASS_TO_ID[class_name]} {x_c/img_size[0]:.6f} {y_c/img_size[1]:.6f} {w/img_size[0]:.6f} {h/img_size[1]:.6f}")
        
        # Save image and annotations
        img_name = f"idate_{i:05d}.png"
        label_name = f"idate_{i:05d}.txt"
        
        cv2.imwrite(str(images_dir / img_name), bg)
        with open(labels_dir / label_name, 'w') as f:
            f.write('\n'.join(annotations))
        
        if (i + 1) % 200 == 0:
            print(f"[DATA] Generated {i + 1}/{num_images} images...")
    
    # Create data.yaml
    data_yaml = output_dir / "data.yaml"
    yaml_content = f"""# iDate Revival YOLO Dataset
# Auto-generated training data from templates

path: {output_dir.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: indicator
  1: up
  2: down
  3: left
  4: right
  5: hand

# Number of classes
nc: 6
"""
    with open(data_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n[DATA] Dataset generation complete!")
    print(f"[DATA] Train images: {num_train}")
    print(f"[DATA] Val images: {num_images - num_train}")
    print(f"[DATA] Dataset config: {data_yaml}")
    
    return str(data_yaml)


def generate_negative_samples(
    output_dir: str | Path = "datasets/idate",
    num_images: int = 200,
    img_size: tuple[int, int] = (640, 640)
):
    """Generate negative samples (backgrounds without objects) to reduce false positives."""
    output_dir = Path(output_dir)
    train_images = output_dir / "images" / "train"
    train_labels = output_dir / "labels" / "train"
    
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)
    
    print(f"[DATA] Generating {num_images} negative samples...")
    
    for i in range(num_images):
        bg = create_random_background(img_size[0], img_size[1])
        
        img_name = f"negative_{i:04d}.png"
        label_name = f"negative_{i:04d}.txt"
        
        cv2.imwrite(str(train_images / img_name), bg)
        # Empty label file for negative sample
        with open(train_labels / label_name, 'w') as f:
            f.write('')
    
    print(f"[DATA] Negative samples generated!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate YOLO training data")
    parser.add_argument("--templates", default="templates", help="Template directory")
    parser.add_argument("--output", default="datasets/idate", help="Output directory")
    parser.add_argument("--num-images", type=int, default=2000, help="Number of images")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--negative", type=int, default=200, help="Number of negative samples")
    
    args = parser.parse_args()
    
    # Generate main dataset
    generate_training_data(
        template_dir=args.templates,
        output_dir=args.output,
        num_images=args.num_images,
        img_size=(args.img_size, args.img_size)
    )
    
    # Generate negative samples
    generate_negative_samples(
        output_dir=args.output,
        num_images=args.negative,
        img_size=(args.img_size, args.img_size)
    )
