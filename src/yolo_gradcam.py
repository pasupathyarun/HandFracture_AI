import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.tal import make_anchors


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/yolo/best.pt")
DEFAULT_IMAGE = Path("data/test_images/hand_test.jpg")
OUTPUT_DIR = Path("results/yolo_gradcam")

IMAGE_SIZE = 640
CONF_THRESHOLD = 0.05

TARGET_LAYER_INDEX = 6  # Earlier spatial feature map; validate with labeled images
MIN_MATCH_IOU = 0.50

HEATMAP_OPACITY = 0.55
GAUSSIAN_SIGMA = 9

CROP_SIZE = 130
ZOOM_SIZE = 600

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ============================================================
# MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    detector = YOLO(str(MODEL_PATH))

    detector.model.to(DEVICE)
    detector.model.eval()

    # Enable autograd for Grad-CAM.
    detector.model.requires_grad_(True)

    print("Model loaded successfully")
    print("Device:", DEVICE)
    print("Classes:", detector.names)

    print(
        "Model gradients enabled:",
        next(detector.model.parameters()).requires_grad
    )

    return detector


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(image_path):

    original_rgb = np.array(
        Image.open(image_path).convert("RGB")
    )

    original_bgr = cv2.cvtColor(
        original_rgb,
        cv2.COLOR_RGB2BGR
    )

    height, width = original_bgr.shape[:2]

    scale = min(
        IMAGE_SIZE / width,
        IMAGE_SIZE / height
    )

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(
        original_bgr,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    pad_left = (IMAGE_SIZE - new_width) // 2
    pad_top = (IMAGE_SIZE - new_height) // 2

    canvas = np.full(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        114,
        dtype=np.uint8
    )

    canvas[
        pad_top:pad_top + new_height,
        pad_left:pad_left + new_width
    ] = resized

    canvas_rgb = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2RGB
    )

    tensor = torch.from_numpy(
        np.ascontiguousarray(
            canvas_rgb.transpose(2, 0, 1)
        )
    )

    tensor = (
        tensor.unsqueeze(0)
        .float()
        .to(DEVICE)
        / 255.0
    )

    metadata = {
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "new_width": new_width,
        "new_height": new_height,
        "original_width": width,
        "original_height": height
    }

    return (
        original_rgb,
        original_bgr,
        tensor,
        metadata
    )


# ============================================================
# YOLO DETECTION
# ============================================================

def run_detection(detector, image):

    detector.model.eval()

    results = detector.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        device=DEVICE,
        verbose=False
    )

    detections = []

    for box in results[0].boxes:

        coordinates = (
            box.xyxy[0]
            .detach()
            .cpu()
            .numpy()
        )

        confidence = float(
            box.conf[0].item()
        )

        detections.append({
            "box": coordinates,
            "confidence": confidence
        })

    return detections


# ============================================================
# BOX UTILITIES
# ============================================================

def xywh_to_xyxy(boxes):

    x = boxes[..., 0]
    y = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]

    return torch.stack(
        (
            x - w / 2,
            y - h / 2,
            x + w / 2,
            y + h / 2
        ),
        dim=-1
    )


def calculate_iou(target_box, candidate_boxes):

    top_left = torch.maximum(
        target_box[:2],
        candidate_boxes[:, :2]
    )

    bottom_right = torch.minimum(
        target_box[2:],
        candidate_boxes[:, 2:]
    )

    intersection_size = (
        bottom_right - top_left
    ).clamp(min=0)

    intersection = (
        intersection_size[:, 0]
        * intersection_size[:, 1]
    )

    target_area = (
        (target_box[2] - target_box[0]).clamp(min=0)
        * (target_box[3] - target_box[1]).clamp(min=0)
    )

    candidate_areas = (
        (candidate_boxes[:, 2] - candidate_boxes[:, 0])
        .clamp(min=0)
        *
        (candidate_boxes[:, 3] - candidate_boxes[:, 1])
        .clamp(min=0)
    )

    union = (
        target_area
        + candidate_areas
        - intersection
    )

    return intersection / union.clamp(min=1e-8)


# ============================================================
# DECODE YOLO RAW PREDICTIONS
# ============================================================

def decode_raw_predictions(output, detect_head):

    if not isinstance(output, dict):
        raise RuntimeError(
            "Expected YOLO output dictionary."
        )

    scores = output.get("scores")
    raw_boxes = output.get("boxes")
    features = output.get("feats")

    if not torch.is_tensor(scores):
        raise RuntimeError("Missing score tensor.")

    if not torch.is_tensor(raw_boxes):
        raise RuntimeError("Missing box tensor.")

    if not isinstance(features, (list, tuple)):
        raise RuntimeError("Missing feature maps.")

    print("Score shape:", tuple(scores.shape))
    print("Box shape:", tuple(raw_boxes.shape))

    if (
        scores.ndim != 3
        or scores.shape[0] != 1
        or scores.shape[1] != 1
    ):
        raise RuntimeError(
            "Expected one image and one fracture class."
        )

    reg_max = int(detect_head.reg_max)

    if raw_boxes.shape[1] == 4 * reg_max:

        distances = detect_head.dfl(raw_boxes)

    elif raw_boxes.shape[1] == 4:

        distances = raw_boxes

    else:

        raise RuntimeError(
            "Unexpected box channel count."
        )

    anchors, strides = make_anchors(
        features,
        detect_head.stride,
        0.5
    )

    # Correct anchor dimensions: [2, N].
    decoded_boxes = detect_head.decode_bboxes(
        distances,
        anchors.T
    )

    # Convert grid coordinates to input-image pixels.
    decoded_boxes = (
        decoded_boxes
        * strides.T.unsqueeze(0)
    )

    # [1, 4, N] -> [1, N, 4].
    decoded_boxes = decoded_boxes.permute(
        0, 2, 1
    )

    if getattr(detect_head, "xyxy", False):

        decoded_xyxy = decoded_boxes

    else:

        decoded_xyxy = xywh_to_xyxy(
            decoded_boxes
        )

    print(
        "Decoded boxes:",
        tuple(decoded_xyxy.shape)
    )

    if decoded_xyxy.shape[1] != scores.shape[-1]:
        raise RuntimeError(
            "Box and score counts do not match."
        )

    return scores, decoded_xyxy


# ============================================================
# DETECTION-SPECIFIC GRAD-CAM
# ============================================================

class DetectionGradCAM:

    def __init__(self, detector, target_layer_index=TARGET_LAYER_INDEX):

        self.layer_index = target_layer_index
        self.model = detector.model

        self.activations = None
        self.gradients = None

        self.model.eval()
        self.model.requires_grad_(True)

        self.target_layer = self.model.model[
            self.layer_index
        ]

        self.handle = (
            self.target_layer.register_forward_hook(
                self.forward_hook
            )
        )

    def forward_hook(self, module, inputs, output):

        if not torch.is_tensor(output):
            raise RuntimeError(
                "Target layer must return a tensor."
            )

        self.activations = output

        if output.requires_grad:
            output.register_hook(
                self.gradient_hook
            )

    def gradient_hook(self, gradient):

        self.gradients = gradient

    def generate(self, image_tensor, detection_box, metadata):

        self.activations = None
        self.gradients = None

        self.model.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        self.model.eval()

        detect_head = self.model.model[-1]

        # Convert original-image box to model coordinates.
        target_box = np.asarray(
            detection_box,
            dtype=np.float32
        ).copy()

        scale = metadata["scale"]

        target_box[[0, 2]] = (
            target_box[[0, 2]] * scale
            + metadata["pad_left"]
        )

        target_box[[1, 3]] = (
            target_box[[1, 3]] * scale
            + metadata["pad_top"]
        )

        target_box = torch.tensor(
            target_box,
            device=DEVICE,
            dtype=torch.float32
        )

        # Return differentiable raw detection predictions.
        detect_head.train()

        try:

            with torch.enable_grad():

                image_tensor = (
                    image_tensor.detach()
                    .clone()
                    .requires_grad_(True)
                )

                output = self.model(image_tensor)

                scores, decoded_boxes = (
                    decode_raw_predictions(
                        output,
                        detect_head
                    )
                )

                ious = calculate_iou(
                    target_box,
                    decoded_boxes[0].detach()
                )

                matched_index = int(
                    torch.argmax(ious).item()
                )

                matching_iou = float(
                    ious[matched_index].item()
                )

                print(
                    "Matched candidate:",
                    matched_index
                )

                print(
                    "Matching IoU:",
                    round(matching_iou, 4)
                )

                if matching_iou < MIN_MATCH_IOU:

                    raise RuntimeError(
                        "Detection matching failed. "
                        f"IoU={matching_iou:.4f}"
                    )

                score = scores[
                    0,
                    0,
                    matched_index
                ]

                print(
                    "Raw fracture score:",
                    float(score.detach().item())
                )

                print(
                    "Score requires grad:",
                    score.requires_grad
                )

                if not score.requires_grad:
                    raise RuntimeError(
                        "Score is detached from autograd."
                    )

                torch.sigmoid(score).backward()  # bounded, monotonic logit target

                if self.activations is None:
                    raise RuntimeError(
                        "Activations not captured."
                    )

                if self.gradients is None:
                    raise RuntimeError(
                        "Gradients not captured."
                    )

                weights = self.gradients.mean(
                    dim=(2, 3),
                    keepdim=True
                )

                cam = (
                    self.activations * weights
                ).sum(dim=1)

                cam = torch.relu(cam)[0]

                cam = (
                    cam.detach()
                    .float()
                    .cpu()
                    .numpy()
                )

                cam = np.nan_to_num(
                    cam,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )

                cam = np.maximum(cam, 0)

                if cam.max() > 0:
                    cam = cam / cam.max()

                return (
                    cam,
                    matched_index,
                    matching_iou
                )

        finally:

            detect_head.eval()

    def close(self):

        self.handle.remove()


# ============================================================
# CREATE ORIGINAL GRAD-CAM
# ============================================================

def create_original_cam(raw_cam, metadata):

    cam = cv2.resize(
        raw_cam.astype(np.float32),
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_CUBIC
    )

    left = metadata["pad_left"]
    top = metadata["pad_top"]

    cam = cam[
        top:top + metadata["new_height"],
        left:left + metadata["new_width"]
    ]

    cam = cv2.resize(
        cam,
        (
            metadata["original_width"],
            metadata["original_height"]
        ),
        interpolation=cv2.INTER_CUBIC
    )

    cam = np.maximum(
        np.nan_to_num(cam),
        0
    )

    cam = cv2.GaussianBlur(
        cam,
        (0, 0),
        sigmaX=GAUSSIAN_SIGMA,
        sigmaY=GAUSSIAN_SIGMA
    )

    if cam.max() > 0:
        cam = cam / cam.max()

    return np.clip(
        cam,
        0,
        1
    )


# ============================================================
# MASK HEATMAP INSIDE FRACTURE BOXES
# ============================================================

def mask_heatmap_to_boxes(cam, detections):

    """
    Preserve original Grad-CAM values only inside
    YOLO-detected fracture bounding boxes.

    This is a visualization mask, not a new
    model-generated localization.
    """

    height, width = cam.shape

    mask = np.zeros(
        (height, width),
        dtype=np.float32
    )

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        x1 = max(0, min(int(np.floor(x1)), width))
        y1 = max(0, min(int(np.floor(y1)), height))

        x2 = max(0, min(int(np.ceil(x2)), width))
        y2 = max(0, min(int(np.ceil(y2)), height))

        if x2 > x1 and y2 > y1:

            mask[y1:y2, x1:x2] = 1.0

    masked_cam = cam * mask

    return masked_cam


# ============================================================
# CREATE JET HEATMAP OVERLAY
# ============================================================

def create_overlay(original_bgr, cam):

    heatmap = cv2.applyColorMap(
        np.uint8(
            np.clip(cam, 0, 1) * 255
        ),
        cv2.COLORMAP_JET
    )

    alpha = (
        cam * HEATMAP_OPACITY
    )[:, :, None]

    overlay = (
        original_bgr.astype(np.float32) * (1 - alpha)
        + heatmap.astype(np.float32) * alpha
    )

    return np.uint8(
        np.clip(overlay, 0, 255)
    )


# ============================================================
# DRAW DETECTION BOXES
# ============================================================

def draw_boxes(image, detections):

    annotated = image.copy()

    for detection in detections:

        x1, y1, x2, y2 = map(
            int,
            detection["box"]
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            1
        )

    return annotated


# ============================================================
# ZOOM FRACTURE REGION
# ============================================================

def create_zoom(image, detection_box):

    height, width = image.shape[:2]

    x1, y1, x2, y2 = map(
        int,
        detection_box
    )

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    crop_width = min(
        CROP_SIZE,
        width
    )

    crop_height = min(
        CROP_SIZE,
        height
    )

    left = max(
        0,
        min(
            center_x - crop_width // 2,
            width - crop_width
        )
    )

    top = max(
        0,
        min(
            center_y - crop_height // 2,
            height - crop_height
        )
    )

    crop = image[
        top:top + crop_height,
        left:left + crop_width
    ]

    return cv2.resize(
        crop,
        (ZOOM_SIZE, ZOOM_SIZE),
        interpolation=cv2.INTER_LINEAR
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    image_path,
    original_rgb,
    annotated,
    zoom,
    confidence,
    original_cam,
    masked_cam
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Preserve the original, unmasked explanation.
    original_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_original_gradcam.npy"
    )

    # Preserve the box-masked explanation.
    masked_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_masked_gradcam.npy"
    )

    np.save(
        original_path,
        original_cam.astype(np.float32)
    )

    np.save(
        masked_path,
        masked_cam.astype(np.float32)
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 7)
    )

    axes[0].imshow(original_rgb)
    axes[0].set_title(
        "Original X-ray",
        fontsize=13
    )

    axes[1].imshow(
        cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )
    )

    axes[1].set_title(
        "Box-Masked YOLO Grad-CAM "
        f"({confidence * 100:.2f}%)",
        fontsize=13
    )

    axes[2].imshow(
        cv2.cvtColor(
            zoom,
            cv2.COLOR_BGR2RGB
        )
    )

    axes[2].set_title(
        "Zoomed Fracture Candidate",
        fontsize=13
    )

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    image_output = (
        OUTPUT_DIR
        / f"{image_path.stem}_box_masked_gradcam.png"
    )

    plt.savefig(
        image_output,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSaved visualization:", image_output)
    print("Saved original CAM:", original_path)
    print("Saved masked CAM:", masked_path)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE),
        help="Path to input X-ray image"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    print("\n" + "=" * 55)
    print("HANDFRACTURE AI - BOX-MASKED YOLO GRAD-CAM")
    print("=" * 55)

    print("Image:", image_path)
    print("Device:", DEVICE)

    print("\n[1] Loading model...")

    detector = load_model()

    (
        original_rgb,
        original_bgr,
        image_tensor,
        metadata
    ) = prepare_image(image_path)

    print("\n[2] Detecting fracture candidates...")

    detections = run_detection(
        detector,
        original_bgr
    )

    print(
        "Detected fracture candidates:",
        len(detections)
    )

    if not detections:

        print(
            "No fracture candidate detected "
            "above the confidence threshold."
        )

        return

    for index, detection in enumerate(
        detections,
        start=1
    ):

        print(
            f"Detection {index}: "
            f"{detection['confidence'] * 100:.2f}%"
        )

    best_detection = max(
        detections,
        key=lambda item: item["confidence"]
    )

    print(
        "Selected confidence:",
        f"{best_detection['confidence'] * 100:.2f}%"
    )

    print("\n[3] Calculating detection-specific Grad-CAM...")

    gradcam = DetectionGradCAM(detector)

    try:

        (
            raw_cam,
            matched_index,
            matching_iou
        ) = gradcam.generate(
            image_tensor,
            best_detection["box"],
            metadata
        )

    finally:

        gradcam.close()

    print("\n[4] Creating original Grad-CAM...")

    original_cam = create_original_cam(
        raw_cam,
        metadata
    )

    print("\n[5] Masking heatmap to fracture boxes...")

    masked_cam = mask_heatmap_to_boxes(
        original_cam,
        detections
    )

    print(
        "Original CAM maximum:",
        float(original_cam.max())
    )

    print(
        "Masked CAM maximum:",
        float(masked_cam.max())
    )

    # Do not normalize the masked CAM again.
    # Doing so could exaggerate weak activation.
    overlay = create_overlay(
        original_bgr,
        masked_cam
    )

    annotated = draw_boxes(
        overlay,
        detections
    )

    zoom = create_zoom(
        annotated,
        best_detection["box"]
    )

    print("\n[6] Saving visualization...")

    save_results(
        image_path,
        original_rgb,
        annotated,
        zoom,
        best_detection["confidence"],
        original_cam,
        masked_cam
    )

    print("\n" + "=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)

    print("Device:", DEVICE)
    print("Grad-CAM target layer:", TARGET_LAYER_INDEX)

    print(
        "Selected detection confidence:",
        f"{best_detection['confidence'] * 100:.2f}%"
    )

    print("Matched candidate:", matched_index)
    print("Matching IoU:", round(matching_iou, 4))

    print("Gaussian sigma:", GAUSSIAN_SIGMA)
    print("Maximum opacity:", HEATMAP_OPACITY)

    print(
        "Original CAM maximum:",
        float(original_cam.max())
    )

    print(
        "Masked CAM maximum:",
        float(masked_cam.max())
    )

    print("\nBox-masked YOLO Grad-CAM completed.")


if __name__ == "__main__":
    main()