import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import functional as TF


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    "models/resnet18/mura_hand_resnet18_best.pth"
)

DEFAULT_IMAGE = Path(
    "data/test_images/hand_test.jpg"
)

OUTPUT_DIR = Path("results/uncertainty")

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# PROVISIONAL SETTINGS:
# Verify these against your original Colab training notebook.
CLASS_NAMES = ["Normal", "Abnormal"]

IMAGE_SIZE = 224

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Exploratory thresholds, NOT clinically validated.
ENTROPY_WARNING_THRESHOLD = 0.75
CONSISTENCY_WARNING_THRESHOLD = 0.80


# ============================================================
# LOAD RESNET18
# ============================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}"
        )

    print("\n[1] Loading ResNet18 model...")

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 2)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint, strict=True)

    model = model.to(DEVICE)
    model.eval()

    print("Model loaded successfully")
    print("Device:", DEVICE)
    print("Output classes:", CLASS_NAMES)

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image(image_path):

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    return Image.open(image_path).convert("RGB")


def create_preprocessing():

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=NORMALIZE_MEAN,
            std=NORMALIZE_STD
        )
    ])


# ============================================================
# CONSERVATIVE TEST-TIME AUGMENTATIONS
# ============================================================

def generate_tta_images(image):

    """
    Small deterministic transformations.

    Avoid horizontal/vertical flipping because anatomical
    orientation may be important.

    Avoid aggressive rotations, crops and distortions.
    """

    variations = [
        ("Original", image.copy()),

        (
            "Brightness -5%",
            TF.adjust_brightness(image, 0.95)
        ),

        (
            "Brightness +5%",
            TF.adjust_brightness(image, 1.05)
        ),

        (
            "Contrast -5%",
            TF.adjust_contrast(image, 0.95)
        ),

        (
            "Contrast +5%",
            TF.adjust_contrast(image, 1.05)
        ),

        (
            "Rotation -2°",
            TF.rotate(
                image,
                -2,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0
            )
        ),

        (
            "Rotation +2°",
            TF.rotate(
                image,
                2,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0
            )
        )
    ]

    return variations


# ============================================================
# RUN TTA INFERENCE
# ============================================================

def predict_with_tta(model, image):

    print("\n[2] Running Test-Time Augmentation...")

    preprocess = create_preprocessing()
    variations = generate_tta_images(image)

    batch = torch.stack([
        preprocess(augmented)
        for _, augmented in variations
    ]).to(DEVICE)

    with torch.inference_mode():

        logits = model(batch)

        probabilities = F.softmax(
            logits,
            dim=1
        )

    probabilities = (
        probabilities.detach()
        .cpu()
        .numpy()
    )

    for index, (name, _) in enumerate(variations):

        print(
            f"{name:20s} | "
            f"Normal: {probabilities[index, 0]:.4f} | "
            f"Abnormal: {probabilities[index, 1]:.4f}"
        )

    return variations, probabilities


# ============================================================
# UNCERTAINTY METRICS
# ============================================================

def calculate_uncertainty(probabilities):

    mean_probabilities = probabilities.mean(axis=0)

    standard_deviation = probabilities.std(
        axis=0,
        ddof=0
    )

    predicted_class = int(
        np.argmax(mean_probabilities)
    )

    prediction_label = CLASS_NAMES[predicted_class]

    confidence = float(
        mean_probabilities[predicted_class]
    )

    # Entropy of the averaged predictive distribution.
    epsilon = 1e-12

    predictive_entropy = -np.sum(
        mean_probabilities
        * np.log(mean_probabilities + epsilon)
    )

    # Normalize binary entropy to the range 0-1.
    normalized_entropy = float(
        predictive_entropy / np.log(2)
    )

    individual_predictions = np.argmax(
        probabilities,
        axis=1
    )

    consistency = float(
        np.mean(
            individual_predictions == predicted_class
        )
    )

    # Mean entropy of individual predictions.
    individual_entropy = -np.sum(
        probabilities * np.log(probabilities + epsilon),
        axis=1
    )

    expected_entropy = float(
        individual_entropy.mean()
    )

    # Mutual information across TTA views.
    # This is a disagreement statistic, not a calibrated
    # Bayesian epistemic-uncertainty estimate.
    disagreement = float(
        max(0.0, predictive_entropy - expected_entropy)
    )

    warning = (
        normalized_entropy >= ENTROPY_WARNING_THRESHOLD
        or consistency < CONSISTENCY_WARNING_THRESHOLD
    )

    return {
        "prediction": prediction_label,
        "predicted_class_index": predicted_class,

        "mean_normal_probability": float(
            mean_probabilities[0]
        ),

        "mean_abnormal_probability": float(
            mean_probabilities[1]
        ),

        "mean_predicted_class_probability": confidence,

        "std_normal_probability": float(
            standard_deviation[0]
        ),

        "std_abnormal_probability": float(
            standard_deviation[1]
        ),

        "predictive_entropy": float(
            predictive_entropy
        ),

        "normalized_predictive_entropy": normalized_entropy,

        "tta_consistency": consistency,

        "tta_disagreement": disagreement,

        "uncertainty_warning": bool(warning),

        "uncertainty_status": (
            "Review suggested"
            if warning
            else "No exploratory threshold exceeded"
        )
    }


# ============================================================
# SAVE VISUALIZATION
# ============================================================

def save_visualization(
    image,
    variations,
    probabilities,
    metrics,
    image_name
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5)
    )

    # Panel 1: Original image.
    axes[0].imshow(image)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    # Panel 2: Probability across augmentations.
    names = [name for name, _ in variations]

    x = np.arange(len(names))

    axes[1].plot(
        x,
        probabilities[:, 0],
        marker="o",
        label=CLASS_NAMES[0]
    )

    axes[1].plot(
        x,
        probabilities[:, 1],
        marker="s",
        label=CLASS_NAMES[1]
    )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        names,
        rotation=55,
        ha="right"
    )

    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Predicted probability")
    axes[1].set_title("TTA Prediction Variation")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    # Panel 3: Summary.
    axes[2].axis("off")

    summary = (
        f"Prediction: {metrics['prediction']}\n\n"
        f"Normal: "
        f"{metrics['mean_normal_probability']:.2%}\n"
        f"Abnormal: "
        f"{metrics['mean_abnormal_probability']:.2%}\n\n"
        f"Abnormal probability SD: "
        f"{metrics['std_abnormal_probability']:.4f}\n"
        f"Normalized entropy: "
        f"{metrics['normalized_predictive_entropy']:.4f}\n"
        f"TTA consistency: "
        f"{metrics['tta_consistency']:.2%}\n"
        f"TTA disagreement: "
        f"{metrics['tta_disagreement']:.6f}\n\n"
        f"Status:\n{metrics['uncertainty_status']}"
    )

    axes[2].text(
        0.05,
        0.95,
        summary,
        transform=axes[2].transAxes,
        fontsize=11,
        va="top",
        family="monospace"
    )

    axes[2].set_title("Uncertainty Summary")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{image_name}_uncertainty.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSaved visualization:", output_path)


# ============================================================
# SAVE JSON REPORT
# ============================================================

def save_json(
    variations,
    probabilities,
    metrics,
    image_name
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    individual_results = []

    for index, (name, _) in enumerate(variations):

        individual_results.append({
            "augmentation": name,
            "normal_probability": float(
                probabilities[index, 0]
            ),
            "abnormal_probability": float(
                probabilities[index, 1]
            )
        })

    report = {
        "model": "ResNet18",
        "task": "MURA hand abnormality classification",
        "method": "Deterministic Test-Time Augmentation",
        "device": DEVICE,

        "configuration": {
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "normalization_mean": NORMALIZE_MEAN,
            "normalization_std": NORMALIZE_STD,
            "training_configuration_verified": False,
            "thresholds_clinically_validated": False
        },

        "metrics": metrics,
        "tta_predictions": individual_results
    }

    output_path = (
        OUTPUT_DIR
        / f"{image_name}_uncertainty.json"
    )

    with open(output_path, "w") as file:
        json.dump(report, file, indent=4)

    print("Saved JSON report:", output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE),
        help="Path to X-ray image"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    print("\n" + "=" * 60)
    print("HANDFRACTURE AI - RESNET18 UNCERTAINTY ESTIMATION")
    print("=" * 60)

    print("Image:", image_path)
    print("Device:", DEVICE)

    print(
        "\nWARNING: Class mapping and preprocessing "
        "are provisional until checked against training."
    )

    model = load_model()

    image = load_image(image_path)

    variations, probabilities = predict_with_tta(
        model,
        image
    )

    print("\n[3] Calculating uncertainty metrics...")

    metrics = calculate_uncertainty(
        probabilities
    )

    print("\n" + "=" * 60)
    print("UNCERTAINTY RESULTS")
    print("=" * 60)

    print(
        "Prediction:",
        metrics["prediction"]
    )

    print(
        "Mean normal probability:",
        f"{metrics['mean_normal_probability']:.2%}"
    )

    print(
        "Mean abnormal probability:",
        f"{metrics['mean_abnormal_probability']:.2%}"
    )

    print(
        "Abnormal probability standard deviation:",
        f"{metrics['std_abnormal_probability']:.4f}"
    )

    print(
        "Normalized predictive entropy:",
        f"{metrics['normalized_predictive_entropy']:.4f}"
    )

    print(
        "TTA consistency:",
        f"{metrics['tta_consistency']:.2%}"
    )

    print(
        "TTA disagreement:",
        f"{metrics['tta_disagreement']:.6f}"
    )

    print(
        "Uncertainty status:",
        metrics["uncertainty_status"]
    )

    print("\n[4] Saving results...")

    save_visualization(
        image,
        variations,
        probabilities,
        metrics,
        image_path.stem
    )

    save_json(
        variations,
        probabilities,
        metrics,
        image_path.stem
    )

    print("\nUncertainty estimation completed.")


if __name__ == "__main__":
    main()
