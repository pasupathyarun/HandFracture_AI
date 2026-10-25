import argparse
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import uncertainty as uncertainty_module
import yolo_gradcam as gradcam_module

from uncertainty import (
    load_model as load_resnet,
    create_preprocessing,
    generate_tta_images,
    predict_with_tta,
    calculate_uncertainty,
)
from yolo_gradcam import (
    load_model as load_gradcam_yolo,
    prepare_image,
    DetectionGradCAM,
    create_original_cam,
    mask_heatmap_to_boxes,
    create_overlay,
    draw_boxes,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "full_pipeline"
RESNET_OV = ROOT / "models" / "openvino" / "resnet18.xml"
YOLO_PT = ROOT / "models" / "yolo" / "best.pt"
YOLO_OV = ROOT / "models" / "yolo" / "best_openvino_model"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BACKEND = "cpu"
DEVICE = "cpu"
METRICS = {}

YOLO_THRESHOLD = 0.10
RESNET_THRESHOLD = 0.60
HYBRID_ENTROPY_THRESHOLD = 0.90
UNCERTAINTY_ENTROPY_THRESHOLD = 0.75
CONSISTENCY_THRESHOLD = 0.80


def openvino_devices():
    try:
        import openvino as ov
        return list(ov.Core().available_devices)
    except Exception:
        return []


def configure_backend(requested):
    global BACKEND, DEVICE

    requested = requested.lower()

    if requested == "fpga":
        raise RuntimeError("FPGA execution is not configured.")

    if requested in ("auto", "gpu"):
        if torch.cuda.is_available():
            requested = "cuda"
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            requested = "mps"
        elif any(
            d == "GPU" or d.startswith("GPU.")
            for d in openvino_devices()
        ):
            requested = "openvino_gpu"
        elif requested == "auto":
            requested = "openvino"
        else:
            raise RuntimeError("No supported GPU available.")

    if requested == "cpu":
        requested = "openvino"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable. Install CUDA-enabled PyTorch."
            )
        DEVICE = "cuda:0"

    elif requested == "mps":
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("Apple MPS is unavailable.")
        DEVICE = "mps"

    elif requested in ("openvino", "openvino_gpu"):
        target = (
            "GPU" if requested == "openvino_gpu" else "CPU"
        )
        if not any(
            d == target or d.startswith(target + ".")
            for d in openvino_devices()
        ):
            raise RuntimeError(
                f"OpenVINO {target} unavailable. "
                f"Devices: {openvino_devices()}"
            )
        # Grad-CAM still requires PyTorch gradients.
        DEVICE = "cpu"

    else:
        raise RuntimeError(f"Unknown backend: {requested}")

    BACKEND = requested

    uncertainty_module.DEVICE = DEVICE
    gradcam_module.DEVICE = DEVICE

    print(f"Backend: {BACKEND}")
    print(f"PyTorch device: {DEVICE}")

    if BACKEND == "cuda":
        print("NVIDIA GPU:", torch.cuda.get_device_name(0))


def synchronize():
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def screening_stage(image_path):
    image = Image.open(image_path).convert("RGB")

    if BACKEND in ("openvino", "openvino_gpu"):
        import openvino as ov

        if not RESNET_OV.is_file():
            raise FileNotFoundError(
                f"ResNet OpenVINO model missing: {RESNET_OV}"
            )

        target = (
            "GPU" if BACKEND == "openvino_gpu" else "CPU"
        )
        compiled = ov.Core().compile_model(
            str(RESNET_OV), target
        )

        preprocessing = create_preprocessing()
        variations = generate_tta_images(image)
        batch = torch.stack([
            preprocessing(augmented)
            for _, augmented in variations
        ]).numpy()

        start = time.perf_counter()
        logits = np.concatenate([
            np.asarray(
                compiled([sample[None, ...]])[
                    compiled.output(0)
                ]
            )
            for sample in batch
        ], axis=0)

        METRICS["resnet_inference_ms"] = round(
            (time.perf_counter() - start) * 1000, 3
        )

        exp = np.exp(
            logits - logits.max(axis=1, keepdims=True)
        )
        probabilities = exp / exp.sum(
            axis=1, keepdims=True
        )

    else:
        model = load_resnet()
        synchronize()
        start = time.perf_counter()

        _, probabilities = predict_with_tta(
            model, image
        )

        synchronize()
        METRICS["resnet_inference_ms"] = round(
            (time.perf_counter() - start) * 1000, 3
        )

    return calculate_uncertainty(probabilities)


def uncertainty_stage(screening):
    entropy = float(
        screening["normalized_predictive_entropy"]
    )
    consistency = float(
        screening["tta_consistency"]
    )

    reasons = []

    if entropy >= UNCERTAINTY_ENTROPY_THRESHOLD:
        reasons.append("High predictive entropy")

    if consistency < CONSISTENCY_THRESHOLD:
        reasons.append("Low TTA consistency")

    return {
        "normalized_entropy": entropy,
        "tta_consistency": consistency,
        "tta_disagreement": float(
            screening["tta_disagreement"]
        ),
        "review_required": bool(reasons),
        "reasons": reasons,
        "thresholds": {
            "entropy": UNCERTAINTY_ENTROPY_THRESHOLD,
            "consistency": CONSISTENCY_THRESHOLD,
        },
    }


def detection_stage(image_path):
    original_rgb, original_bgr, image_tensor, metadata = (
        prepare_image(image_path)
    )

    if BACKEND in ("openvino", "openvino_gpu"):
        model_path = YOLO_OV
        inference_device = (
            "intel:gpu"
            if BACKEND == "openvino_gpu"
            else "cpu"
        )
    else:
        model_path = YOLO_PT
        inference_device = DEVICE

    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model missing: {model_path}"
        )

    detector = YOLO(str(model_path))

    synchronize()
    start = time.perf_counter()

    results = detector.predict(
        source=original_bgr,
        imgsz=640,
        conf=0.05,
        iou=0.70,
        device=inference_device,
        verbose=False,
    )

    synchronize()

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    result = results[0]
    detections = []

    for box in result.boxes:
        detections.append({
            "box": box.xyxy[0].detach().cpu().numpy(),
            "confidence": float(box.conf[0].item()),
        })

    serialized = [
        {
            "id": index + 1,
            "confidence": float(d["confidence"]),
            "box_xyxy": [
                float(v) for v in d["box"]
            ],
        }
        for index, d in enumerate(detections)
    ]

    backend_labels = {
        "cuda": "PyTorch CUDA",
        "mps": "PyTorch MPS",
        "openvino": "OpenVINO CPU",
        "openvino_gpu": "OpenVINO Intel GPU",
    }

    METRICS.update({
        "backend": backend_labels[BACKEND],
        "device": inference_device,
        "yolo_request_ms": round(elapsed_ms, 3),
        "yolo_inference_ms": round(
            float(result.speed.get("inference", 0)), 3
        ),
        "yolo_preprocess_ms": round(
            float(result.speed.get("preprocess", 0)), 3
        ),
        "yolo_postprocess_ms": round(
            float(result.speed.get("postprocess", 0)), 3
        ),
        "fracture_candidates_at_010": sum(
            d["confidence"] >= YOLO_THRESHOLD
            for d in detections
        ),
        "highest_confidence": max(
            (d["confidence"] for d in detections),
            default=0.0,
        ),
        "input_image_shape": list(
            original_bgr.shape[:2]
        ),
        "model_input_size": 640,
    })

    return {
        "original_rgb": original_rgb,
        "original_bgr": original_bgr,
        "image_tensor": image_tensor,
        "metadata": metadata,
        "detections": detections,
        "serialized_detections": serialized,
    }


def save_visual_images(stem, original_bgr, detections, cam=None):
    boxes = draw_boxes(
        original_bgr.copy(), detections
    )
    cv2.imwrite(
        str(OUTPUT_DIR / f"{stem}_boxes.png"),
        boxes,
    )

    if cam is not None:
        heatmap = create_overlay(
            original_bgr, cam
        )
        masked = mask_heatmap_to_boxes(
            cam, detections
        )
        masked_overlay = create_overlay(
            original_bgr, masked
        )

        cv2.imwrite(
            str(OUTPUT_DIR / f"{stem}_heatmap.png"),
            heatmap,
        )
        cv2.imwrite(
            str(OUTPUT_DIR / f"{stem}_masked_heatmap.png"),
            masked_overlay,
        )


def gradcam_stage(yolo_data, stem):
    detections = yolo_data["detections"]
    original_bgr = yolo_data["original_bgr"]

    status = {
        "generated": False,
        "type": "Detection-specific box-masked Grad-CAM",
        "matched_candidate": None,
        "matching_iou": None,
        "error": None,
    }

    if not detections:
        save_visual_images(
            stem, original_bgr, detections
        )
        return original_bgr.copy(), status

    cam_generator = None

    try:
        # Use original differentiable PyTorch weights.
        # OpenVINO models cannot produce this Grad-CAM.
        gradcam_model = load_gradcam_yolo()
        cam_generator = DetectionGradCAM(
            gradcam_model
        )

        best = max(
            detections,
            key=lambda d: d["confidence"],
        )

        raw_cam, matched_index, matching_iou = (
            cam_generator.generate(
                yolo_data["image_tensor"],
                best["box"],
                yolo_data["metadata"],
            )
        )

        original_cam = create_original_cam(
            raw_cam, yolo_data["metadata"]
        )

        masked_cam = mask_heatmap_to_boxes(
            original_cam, detections
        )

        save_visual_images(
            stem, original_bgr,
            detections, original_cam
        )

        np.save(
            OUTPUT_DIR / f"{stem}_original_gradcam.npy",
            original_cam.astype(np.float32),
        )
        np.save(
            OUTPUT_DIR / f"{stem}_masked_gradcam.npy",
            masked_cam.astype(np.float32),
        )

        status.update({
            "generated": True,
            "target_layer": 6,
            "matched_candidate": int(matched_index),
            "matching_iou": float(matching_iou),
        })

        return create_overlay(
            original_bgr, masked_cam
        ), status

    except Exception as error:
        status["error"] = str(error)
        print("Grad-CAM warning:", error)
        traceback.print_exc()

        save_visual_images(
            stem, original_bgr, detections
        )

        return original_bgr.copy(), status

    finally:
        if cam_generator is not None:
            cam_generator.close()


def hybrid_stage(screening, uncertainty, detections):
    abnormal_probability = float(
        screening["mean_abnormal_probability"]
    )
    entropy = float(
        uncertainty["normalized_entropy"]
    )

    maximum = max(
        (
            float(d["confidence"])
            for d in detections
        ),
        default=0.0,
    )

    qualifying = [
        d for d in detections
        if float(d["confidence"]) >= YOLO_THRESHOLD
    ]

    yolo_positive = maximum >= YOLO_THRESHOLD

    resnet_positive = (
        abnormal_probability >= RESNET_THRESHOLD
        and entropy <= HYBRID_ENTROPY_THRESHOLD
    )

    review = yolo_positive or resnet_positive

    reasons = []

    if yolo_positive:
        reasons.append(
            f"YOLO identified {len(qualifying)} "
            "fracture candidate(s)."
        )

    if resnet_positive:
        reasons.append(
            "ResNet18 abnormality probability and "
            "entropy passed the review thresholds."
        )

    if not review:
        reasons.append(
            "Neither automatic review condition was met."
        )

    return {
        "architecture": "Option B - Validation-Selected Hybrid",
        "resnet_prediction": screening["prediction"],
        "abnormal_probability": abnormal_probability,
        "normalized_entropy": entropy,
        "yolo_max_confidence": maximum,
        "yolo_candidates_present": yolo_positive,
        "fracture_candidate_count": len(qualifying),
        "total_yolo_detections": len(detections),
        "yolo_positive": yolo_positive,
        "resnet_positive": resnet_positive,
        "high_uncertainty": bool(
            uncertainty["review_required"]
        ),
        "review_required": review,
        "status": (
            "REVIEW REQUIRED"
            if review
            else "NO AUTOMATIC REVIEW FLAG"
        ),
        "reasons": reasons,
        "interpretation": (
            "Research screening flag only. "
            "Clinical review is required to determine "
            "whether a fracture is present."
        ),
        "thresholds": {
            "yolo_confidence": YOLO_THRESHOLD,
            "resnet_abnormal_probability": RESNET_THRESHOLD,
            "maximum_normalized_entropy":
                HYBRID_ENTROPY_THRESHOLD,
        },
    }


def save_composite(image_path, original_rgb, annotated_bgr,
                   screening, uncertainty, detections, cam, decision):
    fig, axes = plt.subplots(
        1, 3, figsize=(17, 6)
    )

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(
        cv2.cvtColor(
            annotated_bgr, cv2.COLOR_BGR2RGB
        )
    )
    axes[1].set_title(
        "YOLO Detection + Grad-CAM"
        if cam["generated"]
        else "YOLO Detection"
    )
    axes[1].axis("off")

    axes[2].axis("off")

    summary = (
        "HANDFRACTURE AI\n\n"
        f"ResNet18: {screening['prediction']}\n"
        f"Abnormal probability: "
        f"{screening['mean_abnormal_probability']:.2%}\n\n"
        f"Uncertainty entropy: "
        f"{uncertainty['normalized_entropy']:.4f}\n"
        f"TTA consistency: "
        f"{uncertainty['tta_consistency']:.2%}\n\n"
        f"YOLO detections: {len(detections)}\n"
        f"Grad-CAM generated: {cam['generated']}\n\n"
        f"Final status:\n{decision['status']}\n\n"
        "Research prototype only.\n"
        "Not a clinical diagnosis."
    )

    axes[2].text(
        0.02, 0.98, summary,
        transform=axes[2].transAxes,
        va="top",
        fontsize=11,
        family="monospace",
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{image_path.stem}_option_b.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_report(image_path, screening, uncertainty,
                yolo_data, cam, decision):
    report = {
        "project": "HandFracture AI",
        "architecture": "Option B - Uncertainty-Aware Hybrid",
        "input_image": str(image_path),
        "stage_1_resnet18": {
            "dataset": "MURA XR_HAND",
            "task": "Normal/Abnormal screening",
            "class_mapping_verified": False,
            "preprocessing_verified": False,
            "results": screening,
        },
        "stage_2_uncertainty": uncertainty,
        "stage_3_yolov8": {
            "dataset": "FracAtlas hand subset",
            "task": "Fracture candidate detection",
            "detections": yolo_data[
                "serialized_detections"
            ],
        },
        "stage_4_gradcam": cam,
        "stage_5_hybrid_decision": decision,
        "live_performance": METRICS,
        "limitations": [
            "Research prototype, not a medical diagnosis.",
            "Model class mapping and preprocessing require verification.",
            "Thresholds are exploratory.",
            "No detection does not exclude fracture.",
            "YOLO confidence is not a clinical probability.",
            "Grad-CAM explanations require validation.",
        ],
    }

    path = (
        OUTPUT_DIR /
        f"{image_path.stem}_option_b.json"
    )

    path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("JSON saved:", path)


def find_default_image():
    search_dirs = [
        ROOT / "data",
        ROOT / "images",
        ROOT / "samples",
        ROOT / "assets",
        ROOT / "example_images",
    ]

    for directory in search_dirs:
        if not directory.is_dir():
            continue

        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0].resolve()

    return None


def main():
    global METRICS

    parser = argparse.ArgumentParser(
        description="HandFracture AI cross-platform inference"
    )

    parser.add_argument(
        "--image",
        help=(
            "Path to a hand X-ray image. If omitted, the script will look "
            "for a JPG/PNG in common project folders."
        ),
    )

    parser.add_argument(
        "--backend",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "gpu",
            "cuda",
            "mps",
            "openvino",
            "openvino_gpu",
            "fpga",
        ],
    )

    args = parser.parse_args()

    image_path = (
        Path(args.image).expanduser().resolve()
        if args.image
        else find_default_image()
    )

    if image_path is None or not image_path.is_file():
        raise SystemExit(
            "No X-ray image found. Pass --image <path> with a JPG/PNG file, "
            "or place one in a common project folder such as data/, images/, "
            "samples/, or assets/."
        )

    configure_backend(args.backend)
    METRICS = {}

    start = time.perf_counter()

    print("\n" + "=" * 55)
    print("HANDFRACTURE AI - HYBRID ANALYSIS")
    print("=" * 55)
    print("Image:", image_path)

    screening = screening_stage(image_path)
    uncertainty = uncertainty_stage(screening)

    yolo_data = detection_stage(image_path)

    overlay, cam = gradcam_stage(
        yolo_data, image_path.stem
    )

    annotated = draw_boxes(
        overlay, yolo_data["detections"]
    )

    decision = hybrid_stage(
        screening,
        uncertainty,
        yolo_data["serialized_detections"],
    )

    save_composite(
        image_path,
        yolo_data["original_rgb"],
        annotated,
        screening,
        uncertainty,
        yolo_data["serialized_detections"],
        cam,
        decision,
    )

    METRICS["full_pipeline_ms"] = round(
        (time.perf_counter() - start) * 1000,
        3,
    )

    save_report(
        image_path,
        screening,
        uncertainty,
        yolo_data,
        cam,
        decision,
    )

    print("\nAnalysis completed.")
    print("Backend:", BACKEND)
    print("Final decision:", decision["status"])
    print("Complete pipeline:", METRICS["full_pipeline_ms"], "ms")


if __name__ == "__main__":
    main()