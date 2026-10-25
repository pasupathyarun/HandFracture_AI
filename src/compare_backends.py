
"""
HandFracture AI - CPU vs GPU Consistency Test

Runs the existing hybrid pipeline using:
1. OpenVINO CPU
2. PyTorch MPS

Compares YOLO detections, bounding boxes, confidence,
IoU matching, and pipeline timings.

Outputs:
results/backend_comparison/comparison.json
results/backend_comparison/detections.csv
results/backend_comparison/matches.csv
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "src" / "improved_pipeline.py"
PIPELINE_RESULTS = ROOT / "results" / "full_pipeline"
OUTPUT = ROOT / "results" / "backend_comparison"

OUTPUT.mkdir(parents=True, exist_ok=True)


def calculate_iou(a, b):
    """Intersection over Union for XYXY boxes."""

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def run_backend(image_path, backend):
    """Run the existing pipeline without changing it."""

    print(f"\nRunning {backend.upper()}...")

    process = subprocess.run(
        [
            sys.executable,
            str(PIPELINE),
            "--image",
            str(image_path),
            "--backend",
            backend,
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )

    if process.returncode != 0:
        print(process.stdout)
        print(process.stderr)
        raise RuntimeError(f"{backend} pipeline failed")

    report_path = (
        PIPELINE_RESULTS /
        f"{image_path.stem}_option_b.json"
    )

    if not report_path.exists():
        raise FileNotFoundError(report_path)

    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)

    expected = (
        "OpenVINO CPU"
        if backend == "openvino"
        else "PyTorch MPS"
    )

    performance = report.get("live_performance", {})

    if performance.get("backend") != expected:
        raise RuntimeError(
            f"Expected {expected}, got "
            f"{performance.get('backend')}"
        )

    detections = (
        report.get("stage_3_yolov8", {})
        .get("detections", [])
    )

    normalized = []

    for index, detection in enumerate(detections, 1):
        box = detection.get("box_xyxy")

        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(
                f"Invalid bounding box in detection {index}"
            )

        normalized.append({
            "id": detection.get("id", index),
            "confidence": float(
                detection.get("confidence", 0)
            ),
            "box_xyxy": [float(v) for v in box],
        })

    return {
        "backend": expected,
        "detections": normalized,
        "performance": performance,
    }


def match_detections(cpu, gpu, threshold=0.5):
    """Greedy one-to-one IoU matching."""

    candidates = []

    for i, cpu_det in enumerate(cpu):
        for j, gpu_det in enumerate(gpu):
            iou = calculate_iou(
                cpu_det["box_xyxy"],
                gpu_det["box_xyxy"],
            )

            if iou >= threshold:
                candidates.append((iou, i, j))

    candidates.sort(reverse=True)

    used_cpu = set()
    used_gpu = set()
    matches = []

    for iou, i, j in candidates:
        if i in used_cpu or j in used_gpu:
            continue

        used_cpu.add(i)
        used_gpu.add(j)

        matches.append({
            "cpu_id": cpu[i]["id"],
            "gpu_id": gpu[j]["id"],
            "iou": round(iou, 6),
            "cpu_confidence": cpu[i]["confidence"],
            "gpu_confidence": gpu[j]["confidence"],
            "confidence_difference": round(
                cpu[i]["confidence"] -
                gpu[j]["confidence"],
                6,
            ),
        })

    cpu_only = [
        d for i, d in enumerate(cpu)
        if i not in used_cpu
    ]

    gpu_only = [
        d for j, d in enumerate(gpu)
        if j not in used_gpu
    ]

    return matches, cpu_only, gpu_only


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the hand X-ray image",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU matching threshold (default 0.5)",
    )

    args = parser.parse_args()

    if not 0 <= args.iou <= 1:
        parser.error("--iou must be between 0 and 1")

    source = Path(args.image).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    if source.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Image must be JPG, JPEG or PNG")

    # Unique filenames prevent the CPU and GPU runs
    # from overwriting one another's reports.
    run_id = uuid.uuid4().hex[:12]
    temp_dir = OUTPUT / f"inputs_{run_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    cpu_image = temp_dir / f"compare_cpu_{run_id}{source.suffix}"
    gpu_image = temp_dir / f"compare_gpu_{run_id}{source.suffix}"

    shutil.copy2(source, cpu_image)
    shutil.copy2(source, gpu_image)

    try:
        cpu = run_backend(cpu_image, "openvino")
        gpu = run_backend(gpu_image, "mps")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    cpu_detections = cpu["detections"]
    gpu_detections = gpu["detections"]

    matches, cpu_only, gpu_only = match_detections(
        cpu_detections,
        gpu_detections,
        args.iou,
    )

    cpu_time = cpu["performance"].get("full_pipeline_ms")
    gpu_time = gpu["performance"].get("full_pipeline_ms")

    summary = {
        "image": str(source),
        "run_id": run_id,
        "iou_threshold": args.iou,
        "cpu": cpu,
        "gpu": gpu,
        "matched_detections": matches,
        "cpu_only": cpu_only,
        "gpu_only": gpu_only,
        "summary": {
            "cpu_count": len(cpu_detections),
            "gpu_count": len(gpu_detections),
            "matched_count": len(matches),
            "cpu_only_count": len(cpu_only),
            "gpu_only_count": len(gpu_only),
            "cpu_pipeline_ms": cpu_time,
            "gpu_pipeline_ms": gpu_time,
        },
    }

    with (OUTPUT / "comparison.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    detection_rows = []

    for backend, detections in [
        ("OpenVINO CPU", cpu_detections),
        ("PyTorch MPS", gpu_detections),
    ]:
        for detection in detections:
            detection_rows.append({
                "backend": backend,
                "id": detection["id"],
                "confidence": detection["confidence"],
                "x1": detection["box_xyxy"][0],
                "y1": detection["box_xyxy"][1],
                "x2": detection["box_xyxy"][2],
                "y2": detection["box_xyxy"][3],
            })

    save_csv(
        OUTPUT / "detections.csv",
        detection_rows,
        ["backend", "id", "confidence", "x1", "y1", "x2", "y2"],
    )

    save_csv(
        OUTPUT / "matches.csv",
        matches,
        [
            "cpu_id",
            "gpu_id",
            "iou",
            "cpu_confidence",
            "gpu_confidence",
            "confidence_difference",
        ],
    )

    print("\n" + "=" * 55)
    print("HANDFRACTURE AI - BACKEND CONSISTENCY")
    print("=" * 55)

    print(f"CPU detections: {len(cpu_detections)}")
    print(f"GPU detections: {len(gpu_detections)}")
    print(f"Matched detections: {len(matches)}")
    print(f"CPU-only detections: {len(cpu_only)}")
    print(f"GPU-only detections: {len(gpu_only)}")

    print("\nMatched detections:")

    for match in matches:
        print(
            f"CPU #{match['cpu_id']} ↔ GPU #{match['gpu_id']} "
            f"| IoU: {match['iou']:.4f} "
            f"| CPU: {match['cpu_confidence']:.4f} "
            f"| GPU: {match['gpu_confidence']:.4f}"
        )

    print("\nCPU-only:", cpu_only)
    print("GPU-only:", gpu_only)

    print(f"\nCPU pipeline: {cpu_time} ms")
    print(f"GPU pipeline: {gpu_time} ms")

    if (
        isinstance(cpu_time, (int, float))
        and isinstance(gpu_time, (int, float))
    ):
        faster = (
            "OpenVINO CPU"
            if cpu_time < gpu_time
            else "PyTorch MPS"
        )
        print(f"Faster run: {faster}")

    print(f"\nResults saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
