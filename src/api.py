import asyncio
import json
import logging
import re
import subprocess
import sys
import uuid

from pathlib import Path

import torch

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from fastapi.responses import (
    FileResponse,
    HTMLResponse,
)

from PIL import Image, UnidentifiedImageError


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PIPELINE = (
    ROOT / "src" / "cross_platform_pipeline.py"
)

FRONTEND = (
    ROOT / "frontend" / "index.html"
)

UPLOAD_DIR = (
    ROOT / "data" / "api_uploads"
)

OUTPUT_DIR = (
    ROOT / "results" / "full_pipeline"
)

BENCHMARK_FILE = (
    ROOT / "results" / "openvino"
    / "benchmark_results.json"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_BYTES = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

ALLOWED_BACKENDS = {
    "auto",
    "cpu",
    "gpu",
    "openvino",
    "openvino_gpu",
    "cuda",
    "mps",
}

PIPELINE_TIMEOUT = 1200

analysis_lock = asyncio.Lock()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("HandFractureAI")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="HandFracture AI",
    description=(
        "Hand X-ray screening, fracture candidate detection, "
        "uncertainty estimation and Grad-CAM visualization."
    ),
    version="4.0.0",
)


# ============================================================
# HARDWARE DETECTION
# ============================================================

def detect_openvino_devices():
    try:
        import openvino as ov

        core = ov.Core()

        return list(core.available_devices)

    except Exception as error:
        logger.warning(
            "OpenVINO device detection failed: %s",
            error,
        )

        return []


def detect_cuda():
    try:
        return bool(torch.cuda.is_available())

    except Exception:
        return False


def detect_mps():
    try:
        return bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )

    except Exception:
        return False


def detect_openvino_gpu():
    devices = detect_openvino_devices()

    return any(
        device == "GPU"
        or device.startswith("GPU.")
        for device in devices
    )


def supported_gpu():
    if detect_cuda():
        return "cuda"

    if detect_mps():
        return "mps"

    if detect_openvino_gpu():
        return "openvino_gpu"

    return None


def available_backends():
    backends = []

    devices = detect_openvino_devices()

    if any(
        device == "CPU"
        or device.startswith("CPU.")
        for device in devices
    ):
        backends.append("openvino")

    if detect_cuda():
        backends.append("cuda")

    if detect_mps():
        backends.append("mps")

    if detect_openvino_gpu():
        backends.append("openvino_gpu")

    return backends


def hardware_info():
    cuda = detect_cuda()
    mps = detect_mps()
    intel_gpu = detect_openvino_gpu()

    gpu = supported_gpu()

    gpu_name = None

    if cuda:
        try:
            gpu_name = torch.cuda.get_device_name(0)

        except Exception:
            gpu_name = "NVIDIA CUDA GPU"

    elif mps:
        gpu_name = "Apple Silicon GPU"

    elif intel_gpu:
        gpu_name = "Intel OpenVINO GPU"

    return {
        "cuda_available": cuda,
        "mps_available": mps,
        "openvino_gpu_available": intel_gpu,
        "gpu_available": gpu is not None,
        "gpu_name": gpu_name,
        "preferred_gpu_backend": gpu,
        "openvino_devices": detect_openvino_devices(),
        "available_backends": available_backends(),
        "auto_backend": "Smart Auto (benchmark-based)",
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
def homepage():
    if not FRONTEND.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "Frontend file not found: "
                "frontend/index.html"
            ),
        )

    return HTMLResponse(
        FRONTEND.read_text(
            encoding="utf-8",
        )
    )


# ============================================================
# HEALTH AND HARDWARE
# ============================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "hardware": hardware_info(),
    }


@app.get("/api/hardware")
def hardware():
    return hardware_info()


# ============================================================
# BACKEND VALIDATION
# ============================================================

def validate_backend(backend):
    backend = backend.strip().lower()

    if backend == "fpga":
        raise HTTPException(
            status_code=422,
            detail=(
                "FPGA inference is not configured "
                "for this project."
            ),
        )

    if backend not in ALLOWED_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported backend: {backend}",
        )

    if backend == "cuda" and not detect_cuda():
        raise HTTPException(
            status_code=422,
            detail=(
                "NVIDIA CUDA is unavailable. "
                "Check the GPU driver and "
                "CUDA-enabled PyTorch installation."
            ),
        )

    if backend == "mps" and not detect_mps():
        raise HTTPException(
            status_code=422,
            detail=(
                "Apple MPS is unavailable. "
                "MPS requires a supported Mac."
            ),
        )

    if backend == "openvino_gpu":
        if not detect_openvino_gpu():
            raise HTTPException(
                status_code=422,
                detail=(
                    "OpenVINO GPU is unavailable. "
                    "Check the Intel GPU driver "
                    "and OpenVINO installation."
                ),
            )

    if backend == "gpu":
        if supported_gpu() is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No supported GPU detected. "
                    "Select CPU or install the "
                    "required GPU runtime."
                ),
            )

    if backend in ("cpu", "openvino"):
        devices = detect_openvino_devices()

        if not any(
            device == "CPU"
            or device.startswith("CPU.")
            for device in devices
        ):
            raise HTTPException(
                status_code=422,
                detail="OpenVINO CPU device unavailable.",
            )

    return backend


# ============================================================
# RUN CROSS-PLATFORM PIPELINE
# ============================================================

def run_analysis(image_path, backend):
    if not PIPELINE.is_file():
        raise RuntimeError(
            f"Pipeline script not found: {PIPELINE}"
        )

    command = [
        sys.executable,
        str(PIPELINE),
        "--image",
        str(image_path),
        "--backend",
        backend,
    ]

    logger.info(
        "Starting pipeline | backend=%s | image=%s",
        backend,
        image_path.name,
    )

    logger.info(
        "Command: %s",
        " ".join(command),
    )

    try:
        process = subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=PIPELINE_TIMEOUT,
        )

    except subprocess.TimeoutExpired as error:
        logger.error(
            "Pipeline timeout: %s",
            error,
        )

        raise RuntimeError(
            "Pipeline exceeded the 20-minute timeout."
        ) from error

    except OSError as error:
        logger.exception(
            "Unable to launch inference process"
        )

        raise RuntimeError(
            f"Unable to start pipeline: {error}"
        ) from error

    logger.info(
        "Pipeline exit code: %s",
        process.returncode,
    )

    if process.stdout:
        logger.info(
            "\n========== PIPELINE STDOUT ==========\n%s",
            process.stdout,
        )

    if process.stderr:
        logger.warning(
            "\n========== PIPELINE STDERR ==========\n%s",
            process.stderr,
        )

    if process.returncode != 0:
        error_message = (
            process.stderr.strip()
            or process.stdout.strip()
            or "Unknown inference error"
        )

        logger.error(
            "\n"
            "========================================\n"
            "HANDFRACTURE AI PIPELINE FAILED\n"
            "========================================\n"
            "Backend: %s\n"
            "Exit code: %s\n"
            "Error:\n%s\n"
            "========================================",
            backend,
            process.returncode,
            error_message,
        )

        # The detailed traceback remains in the VS Code terminal.
        # Avoid exposing internal paths and stack traces to the UI.
        raise RuntimeError(
            f"Pipeline failed using '{backend}'. "
            "Check the VS Code server terminal "
            "for the full error traceback."
        )

    report_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_option_b.json"
    )

    if not report_path.is_file():
        raise RuntimeError(
            "Pipeline completed but did not generate "
            "the expected JSON report."
        )

    try:
        report = json.loads(
            report_path.read_text(
                encoding="utf-8",
            )
        )

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Generated pipeline report contains invalid JSON."
        ) from error

    logger.info(
        "Pipeline completed successfully | backend=%s",
        backend,
    )

    return report


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_uploaded_image(image_path):
    try:
        with Image.open(image_path) as image:
            if image.format not in ("JPEG", "PNG"):
                raise ValueError(
                    "Unsupported image format."
                )

            image.verify()

        # Verify the image can also be decoded.
        with Image.open(image_path) as image:
            image.load()

            if image.width < 32 or image.height < 32:
                raise ValueError(
                    "Image resolution is too small."
                )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid X-ray image: {error}",
        ) from error


# ============================================================
# ANALYZE UPLOADED X-RAY
# ============================================================

@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    backend: str = Form("auto"),
):
    backend = validate_backend(backend)

    extension = Path(
        file.filename or ""
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPG, JPEG and PNG "
                "images are supported."
            ),
        )

    request_id = uuid.uuid4().hex

    image_path = (
        UPLOAD_DIR
        / f"api_{request_id}{extension}"
    )

    try:
        size = 0

        # Stream the uploaded image in chunks.
        with image_path.open("wb") as output:
            while True:
                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                size += len(chunk)

                if size > MAX_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Image exceeds the "
                            "10 MB upload limit."
                        ),
                    )

                output.write(chunk)

        if size == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded image is empty.",
            )

        validate_uploaded_image(image_path)

        logger.info(
            "Upload accepted | request=%s | "
            "filename=%s | bytes=%s | backend=%s",
            request_id,
            file.filename,
            size,
            backend,
        )

        # Run one analysis at a time to avoid concurrent
        # GPU memory exhaustion and output conflicts.
        async with analysis_lock:
            try:
                report = await asyncio.to_thread(
                    run_analysis,
                    image_path,
                    backend,
                )

            except RuntimeError as error:
                logger.exception(
                    "Analysis failed | request=%s",
                    request_id,
                )

                raise HTTPException(
                    status_code=500,
                    detail=str(error),
                ) from error

        performance = report.get(
            "live_performance",
            {},
        )

        smart_auto = report.get(
            "smart_auto",
        )

        actual_backend = (
            performance.get("backend")
            or (
                smart_auto.get("selected_backend")
                if isinstance(smart_auto, dict)
                else None
            )
            or backend
        )

        # Return only files generated for this request.
        urls = {}

        for kind in (
            "boxes",
            "heatmap",
            "masked_heatmap",
        ):
            filename = (
                f"{image_path.stem}_{kind}.png"
            )

            output_file = (
                OUTPUT_DIR / filename
            )

            if output_file.is_file():
                urls[kind] = (
                    f"/api/view/{filename}"
                )

        composite_name = (
            f"{image_path.stem}_option_b.png"
        )

        composite_path = (
            OUTPUT_DIR / composite_name
        )

        logger.info(
            "Analysis successful | request=%s | "
            "actual_backend=%s",
            request_id,
            actual_backend,
        )

        return {
            "status": "completed",
            "request_id": request_id,
            "requested_hardware": backend,
            "resolved_backend": actual_backend,
            "smart_auto": smart_auto,
            "report": report,
            "live_performance": performance,
            "heatmap_urls": urls,
            "visualization_url": (
                f"/api/result/{composite_name}"
                if composite_path.is_file()
                else None
            ),
        }

    finally:
        await file.close()

        # Delete the original upload after processing.
        # Generated visualizations and reports remain.
        image_path.unlink(
            missing_ok=True,
        )


# ============================================================
# SERVE RESULT IMAGES
# ============================================================

@app.get("/api/view/{filename}")
def view_image(filename: str):
    pattern = (
        r"api_[0-9a-f]{32}_"
        r"(boxes|heatmap|masked_heatmap)\.png"
    )

    if not re.fullmatch(pattern, filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid visualization filename.",
        )

    path = OUTPUT_DIR / filename

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Visualization not found.",
        )

    return FileResponse(
        path,
        media_type="image/png",
    )


# ============================================================
# SERVE COMPOSITE RESULT
# ============================================================

@app.get("/api/result/{filename}")
def result_image(filename: str):
    pattern = (
        r"api_[0-9a-f]{32}_option_b\.png"
    )

    if not re.fullmatch(pattern, filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid result filename.",
        )

    path = OUTPUT_DIR / filename

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Result image not found.",
        )

    return FileResponse(
        path,
        media_type="image/png",
    )


# ============================================================
# SAVED OPENVINO BENCHMARKS
# ============================================================

@app.get("/api/benchmarks")
def benchmarks():
    if not BENCHMARK_FILE.is_file():
        raise HTTPException(
            status_code=404,
            detail="Saved benchmark unavailable.",
        )

    try:
        return json.loads(
            BENCHMARK_FILE.read_text(
                encoding="utf-8",
            )
        )

    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail="Benchmark JSON is invalid.",
        ) from error


# ============================================================
# OPTIONAL DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )