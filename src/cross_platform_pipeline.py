"""
HandFracture AI - Smart Auto Backend Selection

Runs the existing improved_pipeline.py without changing
the trained models, Grad-CAM, or hybrid decision rules.

Auto:
    - CPU and supported GPU
    - one warm-up per backend
    - three measured runs per backend
    - compare median complete-pipeline duration
    - verify fracture candidate count consistency
    - choose CPU if backend results disagree

Explicit CPU/GPU:
    - one pipeline execution

Usage:
    python src/cross_platform_pipeline.py \
        --image data/test_images/hand_test.jpg \
        --backend auto
"""

import argparse
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PIPELINE = ROOT / "src" / "improved_pipeline.py"
OUTPUT_DIR = ROOT / "results" / "full_pipeline"
TEMP_DIR = ROOT / "data" / "smart_auto"

WARMUP_RUNS = 1
MEASURED_RUNS = 3
RUN_TIMEOUT_SECONDS = 180

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def supported_gpu():
    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    try:
        import openvino as ov

        if any(
            device == "GPU" or device.startswith("GPU.")
            for device in ov.Core().available_devices
        ):
            return "openvino_gpu"
    except ImportError:
        pass

    return None


def gpu_label(backend):
    if backend == "cuda":
        return "PyTorch CUDA"

    if backend == "mps":
        return "PyTorch MPS"

    if backend == "openvino_gpu":
        return "OpenVINO Intel GPU"

    return "OpenVINO CPU"


def available_backends():
    devices = ["openvino"]
    gpu = supported_gpu()

    if gpu:
        devices.append(gpu)

    return devices


def get_number(*values):
    for value in values:
        if value is None or value == "":
            continue

        try:
            result = float(value)

            if result >= 0:
                return result
        except (TypeError, ValueError):
            pass

    return None


def report_candidate_count(report):
    performance = report.get("live_performance", {})
    yolo = report.get("stage_3_yolov8", {})

    count = get_number(
        performance.get("candidates_at_threshold"),
        performance.get("fracture_candidates_at_010"),
        performance.get("candidate_count"),
        yolo.get("candidate_count")
    )

    if count is not None:
        return int(count)

    detections = yolo.get("detections", [])

    if isinstance(detections, list):
        count = 0

        for detection in detections:
            if not isinstance(detection, dict):
                continue

            confidence = get_number(
                detection.get("confidence"),
                detection.get("conf"),
                detection.get("score")
            )

            if confidence is not None and confidence >= 0.10:
                count += 1

        return count

    return None


def pipeline_duration(report, wall_ms):
    """
    Prefer an existing complete-pipeline timing.

    Fall back to subprocess wall time, which includes
    Python startup and model loading.
    """

    performance = report.get("live_performance", {})

    measured = get_number(
        performance.get("complete_pipeline_ms"),
        performance.get("full_pipeline_ms"),
        performance.get("total_pipeline_ms"),
        performance.get("pipeline_ms"),
        report.get("total_pipeline_ms")
    )

    if measured is not None:
        return measured, "reported_pipeline_ms"

    return wall_ms, "subprocess_wall_ms"


def report_path_for(image):
    return (
        OUTPUT_DIR /
        f"{image.stem}_option_b.json"
    )


def run_pipeline(image, backend):
    """
    Execute the existing pipeline as a fresh process.

    This avoids retaining GPU state between runs.
    """

    command = [
        sys.executable,
        str(ORIGINAL_PIPELINE),
        "--image",
        str(image),
        "--backend",
        backend
    ]

    started = time.perf_counter()

    process = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_SECONDS
    )

    wall_ms = (
        time.perf_counter() - started
    ) * 1000

    if process.returncode != 0:
        raise RuntimeError(
            f"{backend} execution failed.\n"
            f"STDOUT:\n{process.stdout[-3000:]}\n"
            f"STDERR:\n{process.stderr[-3000:]}"
        )

    report_file = report_path_for(image)

    if not report_file.is_file():
        raise RuntimeError(
            "Expected pipeline JSON was not generated: "
            + str(report_file)
        )

    report = json.loads(
        report_file.read_text(encoding="utf-8")
    )

    duration, timing_source = pipeline_duration(
        report,
        wall_ms
    )

    return {
        "backend": backend,
        "duration_ms": duration,
        "timing_source": timing_source,
        "wall_ms": wall_ms,
        "candidates": report_candidate_count(report),
        "report": report
    }


def temporary_image(original, backend, index):
    name = (
        f"smart_{uuid.uuid4().hex}_"
        f"{backend}_{index}{original.suffix}"
    )

    destination = TEMP_DIR / name
    shutil.copy2(original, destination)

    return destination


def remove_temporary_outputs(image):
    image.unlink(missing_ok=True)

    for path in OUTPUT_DIR.glob(
        image.stem + "*"
    ):
        if path.is_file():
            path.unlink(missing_ok=True)


def benchmark_backend(image, backend):
    durations = []
    candidates = []
    timing_sources = []

    total_runs = WARMUP_RUNS + MEASURED_RUNS

    for index in range(total_runs):
        temporary = temporary_image(
            image,
            backend,
            index
        )

        try:
            result = run_pipeline(
                temporary,
                backend
            )

            if index >= WARMUP_RUNS:
                durations.append(
                    result["duration_ms"]
                )

                candidates.append(
                    result["candidates"]
                )

                timing_sources.append(
                    result["timing_source"]
                )

        finally:
            remove_temporary_outputs(temporary)

    return {
        "backend": backend,
        "label": gpu_label(backend),
        "runs_ms": durations,
        "median_ms": statistics.median(durations),
        "candidate_counts": candidates,
        "timing_source": timing_sources[0],
        "stable_candidates": (
            len(set(candidates)) == 1
            and candidates[0] is not None
        )
    }


def choose_fastest(results):
    cpu = results["openvino"]

    gpu_results = [
        result
        for name, result in results.items()
        if name != "openvino"
    ]

    if not gpu_results:
        return "openvino", "No supported GPU available"

    gpu = gpu_results[0]

    if not cpu["stable_candidates"]:
        return (
            "openvino",
            "CPU candidate count varied between runs"
        )

    if not gpu["stable_candidates"]:
        return (
            "openvino",
            "GPU candidate count varied between runs"
        )

    if (
        cpu["candidate_counts"][0]
        != gpu["candidate_counts"][0]
    ):
        return (
            "openvino",
            "CPU/GPU candidate counts differ; "
            "CPU selected conservatively"
        )

    if cpu["timing_source"] != gpu["timing_source"]:
        return (
            "openvino",
            "Timing sources differ; "
            "CPU selected until measurements are comparable"
        )

    winner = min(
        results.values(),
        key=lambda result: result["median_ms"]
    )

    return (
        winner["backend"],
        "Lowest median complete-pipeline latency"
    )


def run_auto(image):
    results = {}
    errors = {}

    print("\nSmart Auto benchmarking")
    print("Warm-up runs:", WARMUP_RUNS)
    print("Measured runs:", MEASURED_RUNS)

    for backend in available_backends():
        print("\nBenchmarking:", gpu_label(backend))

        try:
            results[backend] = benchmark_backend(
                image,
                backend
            )

            print(
                "Median:",
                round(results[backend]["median_ms"], 3),
                "ms"
            )

            print(
                "Candidates:",
                results[backend]["candidate_counts"]
            )

        except Exception as error:
            errors[backend] = str(error)
            print("Backend failed:", error)

    if "openvino" not in results:
        raise RuntimeError(
            "OpenVINO CPU benchmark failed: "
            + errors.get("openvino", "Unknown error")
        )

    selected, reason = choose_fastest(results)

    print("\nSelected:", gpu_label(selected))
    print("Reason:", reason)

    final = run_pipeline(image, selected)

    report_file = report_path_for(image)
    report = final["report"]

    report["smart_auto"] = {
        "selected_backend": selected,
        "selected_label": gpu_label(selected),
        "selection_reason": reason,
        "benchmarks": results,
        "backend_errors": errors,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS
    }

    report_file.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    return report


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True
    )

    parser.add_argument(
        "--backend",
        choices=[
            "auto",
            "cpu",
            "gpu",
            "openvino",
            "openvino_gpu",
            "mps",
            "cuda",
            "fpga"
        ],
        default="auto"
    )

    args = parser.parse_args()

    image = Path(args.image).resolve()

    if not image.is_file():
        raise FileNotFoundError(image)

    if args.backend == "fpga":
        raise RuntimeError(
            "FPGA hardware is not configured."
        )

    print("=" * 55)
    print("HandFracture AI")
    print("System:", platform.system())
    print("Requested:", args.backend)
    print("=" * 55)

    if args.backend == "auto":
        report = run_auto(image)

    else:
        if args.backend in ("cpu", "openvino"):
            backend = "openvino"

        elif args.backend == "gpu":
            backend = supported_gpu()

            if backend is None:
                raise RuntimeError(
                    "No supported GPU was detected."
                )

        elif args.backend == "openvino_gpu":
            backend = "openvino_gpu"

        else:
            backend = args.backend

        print("Resolved:", gpu_label(backend))

        result = run_pipeline(image, backend)
        report = result["report"]

    print("\nAnalysis completed")

    performance = report.get(
        "live_performance", {}
    )

    print(
        "Backend:",
        performance.get("backend", "Unknown")
    )

    print(
        "Device:",
        performance.get("device", "Unknown")
    )

    print(
        "Complete pipeline:",
        performance.get(
            "complete_pipeline_ms",
            "See JSON report"
        )
    )


if __name__ == "__main__":
    main()