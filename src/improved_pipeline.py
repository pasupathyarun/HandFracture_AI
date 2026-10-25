import argparse
import time
import torch
import uncertainty as uncertainty_module
import yolo_gradcam as gradcam_module
from ultralytics import YOLO
import json
import textwrap
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

from uncertainty import (
    load_model as load_resnet,
    create_preprocessing,
    generate_tta_images,
    predict_with_tta,
    calculate_uncertainty,
)

from yolo_gradcam import (
    load_model as load_yolo,
    prepare_image,
    run_detection,
    DetectionGradCAM,
    create_original_cam,
    mask_heatmap_to_boxes,
    create_overlay,
    draw_boxes,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_IMAGE = Path("data/test_images/hand_test.jpg")
OUTPUT_DIR = Path("results/full_pipeline")
BACKEND = "openvino"
SELECTED_DEVICE = "cpu"
LIVE_METRICS = {}

def sync_device():
    if SELECTED_DEVICE == "mps":
        torch.mps.synchronize()

def configure_backend(backend):
    global BACKEND, SELECTED_DEVICE
    if backend == "fpga":
        raise RuntimeError("FPGA inference requires configured compatible hardware; none is configured.")
    if backend == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS GPU is unavailable on this machine.")
        SELECTED_DEVICE = "mps"
    elif backend == "openvino":
        SELECTED_DEVICE = "cpu"
        import openvino as ov
        if "CPU" not in ov.Core().available_devices:
            raise RuntimeError("OpenVINO CPU device unavailable")
    else:
        raise RuntimeError("Unknown backend: " + backend)
    BACKEND = backend
    uncertainty_module.DEVICE = SELECTED_DEVICE
    gradcam_module.DEVICE = SELECTED_DEVICE


def save_heatmap_views(image_name, original_bgr, detections, cam):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = {"boxes": draw_boxes(original_bgr.copy(), detections)}
    if cam is not None:
        images["heatmap"] = create_overlay(original_bgr, cam)
        images["masked_heatmap"] = create_overlay(
            original_bgr, mask_heatmap_to_boxes(cam, detections))
    for label, img in images.items():
        cv2.imwrite(str(OUTPUT_DIR / f"{image_name}_{label}.png"), img)


# These are exploratory thresholds, not clinical cutoffs.
ENTROPY_THRESHOLD = 0.75
CONSISTENCY_THRESHOLD = 0.80


# ============================================================
# STAGE 1: RESNET18 SCREENING + TTA
# ============================================================

def run_screening(image_path):
    global LIVE_METRICS
    image = Image.open(image_path).convert("RGB")
    if BACKEND == "openvino":
        import openvino as ov
        model_path = Path("models/openvino/resnet18.xml")
        if not model_path.is_file():
            raise FileNotFoundError(f"OpenVINO ResNet model missing: {model_path}")
        compiled = ov.Core().compile_model(str(model_path), "CPU")
        preprocess = create_preprocessing()
        variations = generate_tta_images(image)
        batch = torch.stack([preprocess(im) for _, im in variations]).numpy()
        start = time.perf_counter()
        # Exported OpenVINO ResNet may have a fixed batch dimension of 1.
        logits = np.concatenate([
            np.asarray(compiled([sample[None, ...]])[compiled.output(0)])
            for sample in batch
        ], axis=0)
        LIVE_METRICS["resnet_inference_ms"] = round((time.perf_counter()-start)*1000, 3)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exp / exp.sum(axis=1, keepdims=True)
    else:
        model = load_resnet()
        sync_device()
        start = time.perf_counter()
        _, probabilities = predict_with_tta(model, image)
        sync_device()
        LIVE_METRICS["resnet_inference_ms"] = round((time.perf_counter()-start)*1000, 3)
    metrics = calculate_uncertainty(probabilities)
    return metrics


# ============================================================
# STAGE 2: UNCERTAINTY ASSESSMENT
# ============================================================

def assess_uncertainty(metrics):

    print("\n[2] UNCERTAINTY ASSESSMENT")

    entropy = metrics["normalized_predictive_entropy"]
    consistency = metrics["tta_consistency"]

    reasons = []

    if entropy >= ENTROPY_THRESHOLD:
        reasons.append(
            "High predictive entropy"
        )

    if consistency < CONSISTENCY_THRESHOLD:
        reasons.append(
            "Low TTA prediction consistency"
        )

    review_required = len(reasons) > 0

    result = {
        "normalized_entropy": float(entropy),
        "tta_consistency": float(consistency),
        "tta_disagreement": float(
            metrics["tta_disagreement"]
        ),
        "review_required": review_required,
        "reasons": reasons,
        "thresholds": {
            "entropy": ENTROPY_THRESHOLD,
            "consistency": CONSISTENCY_THRESHOLD
        }
    }

    print("Entropy:", round(entropy, 4))
    print("TTA consistency:", f"{consistency:.2%}")

    print(
        "Uncertainty review:",
        "Required" if review_required else "Not triggered"
    )

    return result


# ============================================================
# STAGE 3: YOLOV8 FRACTURE DETECTION
# ============================================================

def run_fracture_detection(image_path):
    global LIVE_METRICS
    original_rgb, original_bgr, image_tensor, metadata = prepare_image(image_path)
    model_path = (Path("models/yolo/best_openvino_model") if BACKEND == "openvino"
                  else Path("models/yolo/best.pt"))
    if not model_path.exists():
        raise FileNotFoundError(f"Detector missing: {model_path}")
    inference_detector = YOLO(str(model_path))
    sync_device()
    start = time.perf_counter()
    result = inference_detector.predict(source=original_bgr, imgsz=640, conf=0.05,
                                        iou=0.70, device=SELECTED_DEVICE, verbose=False)[0]
    sync_device()
    elapsed = (time.perf_counter() - start) * 1000
    detections = [{"box": b.xyxy[0].detach().cpu().numpy(),
                   "confidence": float(b.conf[0].item())} for b in result.boxes]
    serialized = [{"id": i+1, "confidence": float(d["confidence"]),
                   "box_xyxy": [float(x) for x in d["box"]]}
                  for i, d in enumerate(detections)]
    LIVE_METRICS.update({
        "backend": "OpenVINO CPU" if BACKEND == "openvino" else "PyTorch MPS",
        "device": SELECTED_DEVICE.upper(),
        "yolo_request_ms": round(elapsed, 3),
        "yolo_inference_ms": round(float(result.speed.get("inference", 0)), 3),
        "yolo_preprocess_ms": round(float(result.speed.get("preprocess", 0)), 3),
        "yolo_postprocess_ms": round(float(result.speed.get("postprocess", 0)), 3),
        "fracture_candidates_at_010": sum(d["confidence"] >= 0.10 for d in detections),
        "highest_confidence": max((d["confidence"] for d in detections), default=0.0),
        "input_image_shape": list(original_bgr.shape[:2]),
        "model_input_size": 640,
    })
    # Grad-CAM needs differentiable PyTorch weights even when detection used OpenVINO.
    gradcam_detector = load_yolo() if detections else None
    return {"detector": gradcam_detector, "original_rgb": original_rgb,
            "original_bgr": original_bgr, "image_tensor": image_tensor,
            "metadata": metadata, "detections": detections,
            "serialized_detections": serialized}


# ============================================================
# STAGE 4: DETECTION-SPECIFIC GRAD-CAM
# ============================================================

def run_gradcam(yolo_data, image_name):

    print("\n[4] YOLO GRAD-CAM")

    detections = yolo_data["detections"]

    original_bgr = yolo_data["original_bgr"]

    status = {
        "generated": False,
        "type": "Detection-specific box-masked Grad-CAM",
        "matched_candidate": None,
        "matching_iou": None,
        "error": None
    }

    if not detections:
        save_heatmap_views(image_name, original_bgr, detections, None)
        print("No YOLO candidates. Grad-CAM skipped.")

        return original_bgr.copy(), status

    best_detection = max(
        detections,
        key=lambda item: item["confidence"]
    )

    gradcam = None

    try:

        gradcam = DetectionGradCAM(
            yolo_data["detector"]
        )

        (
            raw_cam,
            matched_index,
            matching_iou
        ) = gradcam.generate(
            yolo_data["image_tensor"],
            best_detection["box"],
            yolo_data["metadata"]
        )

        original_cam = create_original_cam(
            raw_cam,
            yolo_data["metadata"]
        )

        masked_cam = mask_heatmap_to_boxes(
            original_cam,
            detections
        )
        save_heatmap_views(image_name, original_bgr, detections, original_cam)

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            OUTPUT_DIR / f"{image_name}_original_gradcam.npy",
            original_cam.astype(np.float32)
        )

        np.save(
            OUTPUT_DIR / f"{image_name}_masked_gradcam.npy",
            masked_cam.astype(np.float32)
        )

        overlay = create_overlay(
            original_bgr,
            masked_cam
        )

        # Diagnose attribution on the UNMASKED heatmap; never paint fake hotspots.
        box_mask = np.zeros(original_cam.shape, dtype=np.uint8)
        for item in detections:
            x1, y1, x2, y2 = [int(v) for v in item["box"]]
            x1, x2 = max(0, x1), min(box_mask.shape[1], x2)
            y1, y2 = max(0, y1), min(box_mask.shape[0], y2)
            if x2 > x1 and y2 > y1:
                box_mask[y1:y2, x1:x2] = 1
        cam_total = float(np.maximum(original_cam, 0).sum())
        box_fraction = float(box_mask.mean())
        cam_fraction = (float((original_cam * box_mask).sum() / cam_total)
                        if cam_total > 1e-9 else 0.0)
        concentration = cam_fraction / box_fraction if box_fraction > 0 else 0.0
        status.update({
            "generated": True,
            "target_layer": 6,
            "matched_candidate": int(matched_index),
            "matching_iou": float(matching_iou),
            "cam_energy_in_boxes_fraction": round(cam_fraction, 5),
            "box_area_fraction": round(box_fraction, 5),
            "cam_concentration_ratio": round(concentration, 4),
            "localization_warning": (
                "Weak attribution inside detection boxes. Heatmap may be unreliable."
                if concentration < 1.0 else
                "Attribution concentrates in boxes, but explanation validity is unverified."
            )
        })
        print("Grad-CAM box concentration ratio:", round(concentration, 4))

        print("Grad-CAM generated successfully.")

        return overlay, status

    except Exception as error:

        status["error"] = str(error)
        save_heatmap_views(image_name, original_bgr, detections, None)
        print("Grad-CAM failed:", error)

        return original_bgr.copy(), status

    finally:

        if gradcam is not None:
            gradcam.close()


# ============================================================
# STAGE 5: HYBRID DECISION ENGINE
# ============================================================

def hybrid_decision(
    screening,
    uncertainty,
    yolo_detections
):
    """
    Validation-selected hybrid review policy.

    YOLO positive:
        maximum detection confidence >= 0.10

    ResNet positive:
        abnormal probability >= 0.60
        AND normalized entropy <= 0.90

    Final review:
        YOLO positive OR ResNet positive

    These are exploratory thresholds, not clinical cutoffs.
    """

    print("\n[5] IMPROVED HYBRID DECISION ENGINE")

    YOLO_THRESHOLD = 0.10
    RESNET_THRESHOLD = 0.60
    ENTROPY_THRESHOLD = 0.90

    prediction = screening["prediction"]

    abnormal_probability = float(
        screening["mean_abnormal_probability"]
    )

    normalized_entropy = float(
        uncertainty["normalized_entropy"]
    )

    yolo_max_confidence = max(
        (
            float(detection["confidence"])
            for detection in yolo_detections
        ),
        default=0.0
    )

    yolo_positive = (
        yolo_max_confidence >= YOLO_THRESHOLD
    )

    resnet_positive = (
        abnormal_probability >= RESNET_THRESHOLD
        and normalized_entropy <= ENTROPY_THRESHOLD
    )

    review_required = (
        yolo_positive or resnet_positive
    )

    # Keep the old uncertainty flag for reporting,
    # but do not use it independently to trigger review.
    high_uncertainty = bool(
        uncertainty["review_required"]
    )

    qualifying_detections = [
        detection
        for detection in yolo_detections
        if float(detection["confidence"]) >= YOLO_THRESHOLD
    ]

    reasons = []

    if yolo_positive:
        reasons.append(
            f"YOLO detected {len(qualifying_detections)} "
            f"fracture candidate(s) at confidence >= "
            f"{YOLO_THRESHOLD:.2f}"
        )

    if resnet_positive:
        reasons.append(
            "ResNet18 abnormality signal passed the "
            "confidence and entropy gates"
        )

    if yolo_positive and prediction == "Normal":
        reasons.append(
            "Model disagreement: ResNet18 predicts Normal "
            "while YOLO detects fracture candidates"
        )

    if resnet_positive and not yolo_positive:
        reasons.append(
            "ResNet18 triggered review without a qualifying "
            "YOLO fracture detection"
        )

    if not review_required:
        reasons.append(
            "Neither hybrid review condition was met"
        )

    status = (
        "REVIEW REQUIRED"
        if review_required
        else "NO AUTOMATIC REVIEW FLAG"
    )

    if yolo_positive:
        interpretation = (
            "YOLO identified possible fracture regions "
            "requiring human assessment."
        )
    elif resnet_positive:
        interpretation = (
            "YOLO did not identify a qualifying fracture "
            "candidate, but the ResNet18 abnormality signal "
            "met the hybrid review criteria."
        )
    else:
        interpretation = (
            "Neither model met the selected hybrid review "
            "criteria. This does not rule out fracture "
            "or other abnormalities."
        )

    if high_uncertainty:
        interpretation += (
            " The original uncertainty assessment flagged "
            "the screening prediction; this flag is reported "
            "but does not independently trigger review."
        )

    result = {
        "architecture": (
            "Option B - Validation-Selected Hybrid"
        ),
        "resnet_prediction": prediction,
        "abnormal_probability": abnormal_probability,
        "normalized_entropy": normalized_entropy,
        "yolo_max_confidence": yolo_max_confidence,
        "yolo_candidates_present": yolo_positive,
        "fracture_candidate_count": len(
            qualifying_detections
        ),
        "total_yolo_detections": len(yolo_detections),
        "yolo_positive": bool(yolo_positive),
        "resnet_positive": bool(resnet_positive),
        "high_uncertainty": high_uncertainty,
        "review_required": bool(review_required),
        "status": status,
        "reasons": reasons,
        "interpretation": interpretation,
        "thresholds": {
            "yolo_confidence": YOLO_THRESHOLD,
            "resnet_abnormal_probability": RESNET_THRESHOLD,
            "maximum_normalized_entropy": ENTROPY_THRESHOLD
        }
    }

    print(
        f"ResNet18: {prediction} "
        f"(abnormal probability: {abnormal_probability:.4f})"
    )

    print(
        f"Normalized entropy: {normalized_entropy:.4f}"
    )

    print(
        f"YOLO maximum confidence: "
        f"{yolo_max_confidence:.4f}"
    )

    print("YOLO review trigger:", yolo_positive)
    print("ResNet review trigger:", resnet_positive)
    print("Original uncertainty flag:", high_uncertainty)
    print("Final status:", status)

    return result


# ============================================================
# STAGE 6: FINAL VISUALIZATION
# ============================================================

def create_visualization(
    image_path,
    original_rgb,
    annotated_bgr,
    screening,
    uncertainty,
    detections,
    gradcam_status,
    decision
):

    print("\n[6] GENERATING VISUALIZATION")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20, 7)
    )

    # Original image.
    axes[0].imshow(original_rgb)

    axes[0].set_title(
        "Original X-ray",
        fontsize=13
    )

    axes[0].axis("off")

    # Detection and Grad-CAM.
    annotated_rgb = cv2.cvtColor(
        annotated_bgr,
        cv2.COLOR_BGR2RGB
    )

    axes[1].imshow(annotated_rgb)

    if gradcam_status["generated"]:

        title = "YOLO Detection + Box-Masked Grad-CAM"

    else:

        title = "YOLO Fracture Detection"

    axes[1].set_title(title, fontsize=13)
    axes[1].axis("off")

    # Final report.
    axes[2].axis("off")

    if detections:

        highest_confidence = max(
            item["confidence"]
            for item in detections
        )

        confidence_text = (
            f"{highest_confidence:.2%}"
        )

    else:

        confidence_text = "N/A"

    reasons = decision["reasons"]

    if reasons:

        reason_text = "\n".join(
            f"- {textwrap.fill(reason, width=45)}"
            for reason in reasons
        )

    else:

        reason_text = (
            "No exploratory review threshold triggered."
        )

    summary = (
        "HANDFRACTURE AI - OPTION B\n\n"

        "STAGE 1: RESNET18 SCREENING\n"
        f"Prediction: {screening['prediction']}\n"
        f"Normal: "
        f"{screening['mean_normal_probability']:.2%}\n"
        f"Abnormal: "
        f"{screening['mean_abnormal_probability']:.2%}\n\n"

        "STAGE 2: TTA UNCERTAINTY\n"
        f"Entropy: "
        f"{uncertainty['normalized_entropy']:.4f}\n"
        f"Consistency: "
        f"{uncertainty['tta_consistency']:.2%}\n"
        f"Review flag: "
        f"{uncertainty['review_required']}\n\n"

        "STAGE 3: YOLOV8\n"
        f"Candidates: {len(detections)}\n"
        f"Highest confidence: {confidence_text}\n\n"

        "STAGE 4: GRAD-CAM\n"
        f"Generated: {gradcam_status['generated']}\n\n"

        "FINAL DECISION\n"
        f"{decision['status']}\n\n"

        "REASONS\n"
        f"{reason_text}"
    )

    axes[2].text(
        0.02,
        0.98,
        summary,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        transform=axes[2].transAxes
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_option_b.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Visualization saved:", output_path)


# ============================================================
# STAGE 7: JSON REPORT
# ============================================================

def save_report(
    image_path,
    screening,
    uncertainty,
    yolo_data,
    gradcam_status,
    decision
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "project": "HandFracture AI",

        "architecture": (
            "Option B - Uncertainty-Aware Hybrid"
        ),

        "input_image": str(image_path),

        "stage_1_resnet18": {
            "dataset": "MURA XR_HAND",
            "task": "Normal/Abnormal screening",
            "class_mapping_verified": False,
            "preprocessing_verified": False,
            "results": screening
        },

        "stage_2_uncertainty": uncertainty,

        "stage_3_yolov8": {
            "dataset": "FracAtlas hand subset",
            "task": "Fracture candidate detection",
            "detections": yolo_data[
                "serialized_detections"
            ]
        },

        "stage_4_gradcam": gradcam_status,

        "stage_5_hybrid_decision": decision,
        "live_performance": LIVE_METRICS,

        "limitations": [
            "This system is a research prototype.",
            "ResNet18 class mapping requires verification.",
            "ResNet18 preprocessing requires verification.",
            "Uncertainty thresholds are exploratory.",
            "YOLO confidence is not a calibrated clinical probability.",
            "Box-masked Grad-CAM is a visualization restriction.",
            "No YOLO detection does not exclude fracture.",
            "The models require external validation."
        ]
    }

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_option_b.json"
    )

    with open(output_path, "w") as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("JSON report saved:", output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--backend", choices=["openvino", "mps", "fpga"], default="openvino")

    args = parser.parse_args()

    configure_backend(args.backend)
    global LIVE_METRICS
    LIVE_METRICS = {}
    image_path = Path(args.image)
    total_start = time.perf_counter()

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    print("\n" + "=" * 60)
    print("HANDFRACTURE AI - OPTION B HYBRID PIPELINE")
    print("=" * 60)

    print("Input:", image_path)

    # Stage 1.
    screening = run_screening(image_path)

    # Stage 2.
    uncertainty = assess_uncertainty(screening)

    # Stage 3.
    # YOLO always executes.
    yolo_data = run_fracture_detection(image_path)

    # Stage 4.
    overlay, gradcam_status = run_gradcam(
        yolo_data,
        image_path.stem
    )

    annotated = draw_boxes(
        overlay,
        yolo_data["detections"]
    )

    # Stage 5.
    decision = hybrid_decision(
        screening,
        uncertainty,
        yolo_data["serialized_detections"]
    )

    # Stage 6.
    create_visualization(
        image_path,
        yolo_data["original_rgb"],
        annotated,
        screening,
        uncertainty,
        yolo_data["serialized_detections"],
        gradcam_status,
        decision
    )

    LIVE_METRICS["full_pipeline_ms"] = round((time.perf_counter()-total_start)*1000, 3)
    # Stage 7.
    save_report(
        image_path,
        screening,
        uncertainty,
        yolo_data,
        gradcam_status,
        decision
    )

    print("\n" + "=" * 60)
    print("FINAL HANDFRACTURE AI RESULT")
    print("=" * 60)

    print("Screening:", screening["prediction"])

    print(
        "Uncertainty review:",
        uncertainty["review_required"]
    )

    print(
        "Fracture candidates:",
        len(yolo_data["detections"])
    )

    print("Final status:", decision["status"])

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()