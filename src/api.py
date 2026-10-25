
import asyncio
import json
import re
import sys
import uuid
from pathlib import Path

import torch

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile
)

from fastapi.responses import (
    FileResponse,
    HTMLResponse
)

from PIL import Image, UnidentifiedImageError

from src.cross_platform_pipeline import (
    available_backends,
    supported_gpu
)


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
    ROOT / "results" / "openvino" /
    "benchmark_results.json"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MAX_BYTES = 10 * 1024 * 1024

analysis_lock = asyncio.Lock()

app = FastAPI(
    title="HandFracture AI",
    version="3.0.0"
)


def hardware_info():
    gpu = supported_gpu()

    return {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ),
        "gpu_available": gpu is not None,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if gpu == "cuda"
            else "Apple Silicon GPU"
            if gpu == "mps"
            else None
        ),
        "available_backends": available_backends(),
        "auto_backend": "Smart Auto (benchmark-based)"
    }


@app.get("/", response_class=HTMLResponse)
def homepage():
    return HTMLResponse(
        FRONTEND.read_text(
            encoding="utf-8"
        )
    )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "hardware": hardware_info()
    }


@app.get("/api/hardware")
def hardware():
    return hardware_info()


def run_analysis(image_path, backend):
    import subprocess

    process = subprocess.run(
        [
            sys.executable,
            str(PIPELINE),
            "--image",
            str(image_path),
            "--backend",
            backend
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=1200
    )

    if process.returncode != 0:
        print(process.stdout)
        print(process.stderr)

        raise RuntimeError(
            "Pipeline failed. Check server terminal."
        )

    report_path = (
        OUTPUT_DIR /
        f"{image_path.stem}_option_b.json"
    )

    if not report_path.is_file():
        raise RuntimeError(
            "Pipeline JSON report was not generated."
        )

    return json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    backend: str = Form("auto")
):
    backend = backend.strip().lower()

    if backend not in (
        "auto",
        "cpu",
        "gpu",
        "fpga"
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid backend selection."
        )

    if backend == "fpga":
        raise HTTPException(
            status_code=422,
            detail="FPGA hardware is not configured."
        )

    if backend == "gpu" and not supported_gpu():
        raise HTTPException(
            status_code=422,
            detail="No supported GPU available."
        )

    extension = Path(
        file.filename or ""
    ).suffix.lower()

    if extension not in (
        ".jpg",
        ".jpeg",
        ".png"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG are supported."
        )

    request_id = uuid.uuid4().hex

    image_path = (
        UPLOAD_DIR /
        f"api_{request_id}{extension}"
    )

    try:
        size = 0

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
                        detail="Maximum image size is 10 MB."
                    )

                output.write(chunk)

        try:
            with Image.open(image_path) as image:
                image.verify()

        except (
            UnidentifiedImageError,
            OSError,
            ValueError
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid image."
            )

        async with analysis_lock:
            try:
                report = await asyncio.to_thread(
                    run_analysis,
                    image_path,
                    backend
                )

            except Exception as error:
                raise HTTPException(
                    status_code=500,
                    detail=str(error)
                )

        performance = report.get(
            "live_performance", {}
        )

        smart_auto = report.get(
            "smart_auto"
        )

        actual_backend = performance.get(
            "backend"
        )

        urls = {}

        for kind in (
            "boxes",
            "heatmap",
            "masked_heatmap"
        ):
            filename = (
                f"{image_path.stem}_{kind}.png"
            )

            if (
                OUTPUT_DIR / filename
            ).is_file():
                urls[kind] = (
                    f"/api/view/{filename}"
                )

        composite = (
            OUTPUT_DIR /
            f"{image_path.stem}_option_b.png"
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
                f"/api/result/{composite.name}"
                if composite.is_file()
                else None
            )
        }

    finally:
        await file.close()
        image_path.unlink(
            missing_ok=True
        )


@app.get("/api/view/{filename}")
def view_image(filename: str):
    pattern = (
        r"api_[0-9a-f]{32}_"
        r"(boxes|heatmap|masked_heatmap)\.png"
    )

    if not re.fullmatch(
        pattern,
        filename
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid image name."
        )

    path = OUTPUT_DIR / filename

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Image not found."
        )

    return FileResponse(path)


@app.get("/api/result/{filename}")
def result_image(filename: str):
    pattern = (
        r"api_[0-9a-f]{32}_option_b\.png"
    )

    if not re.fullmatch(
        pattern,
        filename
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid result name."
        )

    path = OUTPUT_DIR / filename

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Result not found."
        )

    return FileResponse(path)


@app.get("/api/benchmarks")
def benchmarks():
    if not BENCHMARK_FILE.is_file():
        raise HTTPException(
            status_code=404,
            detail="Saved benchmark unavailable."
        )

    return json.loads(
        BENCHMARK_FILE.read_text(
            encoding="utf-8"
        )
    )
