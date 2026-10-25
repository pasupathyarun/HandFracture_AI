# HandFracture AI

**Hybrid Deep Learning System for Hand X-ray Abnormality Screening, Fracture Localization, Uncertainty Estimation, and Explainable AI**

## Project Overview

HandFracture AI is a research prototype that analyzes hand X-ray images using a hybrid deep learning pipeline.

The system combines:

- **ResNet18:** Hand X-ray abnormality screening, trained using MURA.
- **YOLOv8n:** Fracture detection and localization, trained using FracAtlas.
- **Grad-CAM:** Visualization of model attention.
- **Uncertainty Estimation:** Assessment of prediction uncertainty.
- **Hybrid Decision Logic:** Integration of screening and detection results.
- **OpenVINO:** Optimized CPU inference.
- **PyTorch MPS:** Apple Silicon GPU acceleration.
- **FastAPI:** Backend API and web dashboard.

The application is intended for research and educational purposes, not independent clinical diagnosis.

## Technology Stack

- Python 3.11
- PyTorch and Torchvision
- Ultralytics YOLOv8
- OpenVINO
- FastAPI and Uvicorn
- OpenCV, NumPy, scikit-learn
- HTML, CSS, JavaScript

## Project Structure

```text
HandFracture_AI_New/
├── frontend/
│   └── index.html
├── src/
│   ├── api.py
│   ├── improved_pipeline.py
│   ├── hardware.py
│   ├── compare_backends.py
│   ├── uncertainty.py
│   └── yolo_gradcam.py
├── models/
│   ├── resnet18/
│   ├── yolo/
│   └── openvino/
├── requirements.txt
├── README.md
└── .gitignore
```

Additional source modules may be present.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/HandFracture_AI.git
cd HandFracture_AI
```

Create a Python environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows, activate using:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Web Application

Start the FastAPI backend:

```bash
uvicorn src.api:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Upload a hand X-ray image and select an available inference backend.

Backend support depends on the computer's hardware and the implementation. OpenVINO CPU and Apple MPS have been tested on the development machine.

## Run the Pipeline

Example:

```bash
python src/improved_pipeline.py --image path/to/hand_xray.jpg --backend openvino
```

For Apple Silicon GPU:

```bash
python src/improved_pipeline.py --image path/to/hand_xray.jpg --backend mps
```

## Model Evaluation

### ResNet18 — MURA XR_HAND

Previous image-level validation:

| Metric | Value |
|---|---:|
| Accuracy | 75.43% |
| Precision | 69.79% |
| Sensitivity | 70.90% |
| Specificity | 78.60% |
| F1-score | 70.34% |
| AUROC | 0.8282 |

### YOLOv8n — FracAtlas Hand

Held-out test evaluation:

| Metric | Value |
|---|---:|
| Evaluated images | 145 |
| Annotated fractures | 40 |
| Precision | 55.8% |
| Recall | 37.5% |
| mAP50 | 39.5% |
| mAP50–95 | 16.4% |
| Apple MPS inference | 2.9 ms/image |

One test image was excluded because its JPEG data was truncated.

These results represent model-specific evaluations, not a combined clinical diagnostic accuracy measurement.

## Datasets

- MURA: Musculoskeletal Radiographs dataset for abnormality screening.
- FracAtlas: Fracture dataset for detection and localization.

The full datasets are not included in this repository. Dataset access and use must follow the original providers' terms.

## Limitations

- Fracture detection recall requires further improvement.
- Predictions may include false positives and false negatives.
- Confidence and uncertainty estimates require further calibration.
- Hybrid-system performance requires additional evaluation.
- The application has not been clinically validated.

## Future Work

- Hybrid decision evaluation
- Uncertainty calibration
- Improved fracture localization
- Cross-platform inference testing
- Reproducible benchmarking

## Disclaimer

This is an academic research prototype. It is not approved as a medical device and must not be used as a substitute for professional medical interpretation.
