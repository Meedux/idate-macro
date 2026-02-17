from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path


def _resolve_api_key(cli_key: str | None) -> str:
    key = cli_key or os.getenv("ROBOFLOW_API_KEY")
    if not key:
        raise ValueError("Missing API key. Provide --api-key or set ROBOFLOW_API_KEY.")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload local YOLO dataset to Roboflow cloud")
    parser.add_argument("--api-key", default=None, help="Roboflow API key (or use ROBOFLOW_API_KEY)")
    parser.add_argument("--workspace", default=None, help="Workspace slug (optional; auto-detect if omitted)")
    parser.add_argument("--project", default="idate-revival-ml", help="Project slug/id")
    parser.add_argument("--dataset", default="datasets/idate", help="Path to YOLO dataset root")
    parser.add_argument("--workers", type=int, default=8, help="Parallel upload workers")
    parser.add_argument("--batch-name", default=None, help="Optional upload batch name")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    data_yaml = dataset_root / "data.yaml"
    if not dataset_root.exists() or not data_yaml.exists():
        raise FileNotFoundError(f"Dataset not found or missing data.yaml: {dataset_root}")

    api_key = _resolve_api_key(args.api_key)

    print("=" * 60)
    print("Roboflow Upload: iDate YOLO Dataset")
    print("=" * 60)
    print(f"Dataset: {dataset_root.resolve()}")
    print(f"Project: {args.project}")

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)

    if args.workspace:
        workspace = rf.workspace(args.workspace)
        workspace_slug = args.workspace
    else:
        workspace = rf.workspace()
        workspace_slug = workspace.url

    print(f"Workspace: {workspace_slug}")

    batch_name = args.batch_name or f"idate-upload-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    workspace.upload_dataset(
        str(dataset_root),
        args.project,
        num_workers=max(1, args.workers),
        project_license="MIT",
        project_type="object-detection",
        batch_name=batch_name,
        num_retries=3,
        is_prediction=False,
    )

    print("=" * 60)
    print("Upload complete.")
    print(f"Open: https://app.roboflow.com/{workspace_slug}/{args.project}")
    print("Next in Roboflow: Generate dataset version -> Train.")
    print("=" * 60)


if __name__ == "__main__":
    main()
