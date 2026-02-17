"""
iDate YOLO Training Pipeline

Complete pipeline to train a custom YOLO model:
1. Generate synthetic training data from templates
2. Train YOLO model on generated data  
3. Export best model

Usage:
    python -m src.ml.train_yolo
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for iDate detection")
    parser.add_argument("--templates", default="templates", help="Template directory")
    parser.add_argument("--output", default="datasets/idate", help="Dataset output directory")
    parser.add_argument("--num-images", type=int, default=800, help="Number of synthetic images (800 is optimal)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (30 with early stopping)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=320, help="Image size (320 for fast CPU training)")
    parser.add_argument("--skip-generate", action="store_true", help="Skip data generation")
    parser.add_argument("--fast", action="store_true", help="Fast mode: 400 images, 15 epochs")
    
    args = parser.parse_args()
    
    # Fast mode for quick testing
    if args.fast:
        args.num_images = 400
        args.epochs = 15
        args.batch_size = 4
        print("[FAST MODE] Using reduced settings for quick testing")
    
    template_dir = Path(args.templates)
    output_dir = Path(args.output)
    
    # Step 1: Generate training data
    if not args.skip_generate:
        print("=" * 60)
        print("STEP 1: Generating Training Data")
        print("=" * 60)
        
        from .data_generator import generate_training_data, generate_negative_samples
        
        data_yaml = generate_training_data(
            template_dir=template_dir,
            output_dir=output_dir,
            num_images=args.num_images,
            img_size=(args.img_size, args.img_size)
        )
        
        # Add negative samples to reduce false positives
        generate_negative_samples(
            output_dir=output_dir,
            num_images=args.num_images // 5,
            img_size=(args.img_size, args.img_size)
        )
    else:
        data_yaml = output_dir / "data.yaml"
        if not data_yaml.exists():
            print(f"ERROR: Dataset not found at {data_yaml}")
            print("Run without --skip-generate first")
            return
    
    # Step 2: Train YOLO model
    print("\n" + "=" * 60)
    print("STEP 2: Training YOLO Model")
    print("=" * 60)
    
    from .yolo_detector import train_model
    
    train_model(
        data_yaml=data_yaml,
        output_dir="runs/detect/idate",
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model saved to: models/idate_best.pt")
    print(f"You can now run the GUI with ML detection enabled")


if __name__ == "__main__":
    main()
