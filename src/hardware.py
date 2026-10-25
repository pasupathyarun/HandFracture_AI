
import platform
import torch


def detect_hardware():
    """
    Detect the supported PyTorch acceleration devices
    on the computer currently running the application.
    """

    cuda_available = torch.cuda.is_available()

    mps_available = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )

    return {
        "operating_system": platform.system(),
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "cuda_device_name": (
            torch.cuda.get_device_name(0)
            if cuda_available
            else None
        ),
    }


def resolve_backend(selection="auto"):
    """
    Returns the backend configuration to use.

    auto:
        CUDA -> MPS -> OpenVINO CPU

    gpu:
        CUDA -> MPS -> error if neither exists

    cpu:
        OpenVINO CPU

    fpga:
        Not supported without compatible hardware.
    """

    selection = selection.strip().lower()
    hardware = detect_hardware()

    if selection == "fpga":
        raise RuntimeError(
            "FPGA inference is unavailable. "
            "Compatible FPGA hardware and a supported "
            "inference runtime must be configured."
        )

    if selection == "cpu":
        return {
            "selection": selection,
            "backend": "openvino",
            "backend_label": "OpenVINO CPU",
            "device": "cpu",
            "device_label": "CPU",
            "hardware": hardware,
        }

    if selection not in ("auto", "gpu"):
        raise ValueError(
            f"Unsupported hardware selection: {selection}"
        )

    if hardware["cuda_available"]:
        return {
            "selection": selection,
            "backend": "cuda",
            "backend_label": "PyTorch CUDA",
            "device": "cuda:0",
            "device_label": (
                hardware["cuda_device_name"]
                or "NVIDIA CUDA GPU"
            ),
            "hardware": hardware,
        }

    if hardware["mps_available"]:
        return {
            "selection": selection,
            "backend": "mps",
            "backend_label": "PyTorch MPS",
            "device": "mps",
            "device_label": "Apple Silicon GPU",
            "hardware": hardware,
        }

    if selection == "gpu":
        raise RuntimeError(
            "No supported GPU backend was detected. "
            "For NVIDIA GPUs, install a CUDA-enabled "
            "PyTorch build and compatible drivers. "
            "For Apple Silicon, ensure MPS is available."
        )

    return {
        "selection": selection,
        "backend": "openvino",
        "backend_label": "OpenVINO CPU",
        "device": "cpu",
        "device_label": "CPU",
        "hardware": hardware,
    }


def synchronize_device(device):
    """
    Synchronize GPU execution for accurate timing.
    """

    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)

    elif device == "mps":
        torch.mps.synchronize()


if __name__ == "__main__":
    hardware = detect_hardware()

    print("\nSystem:", hardware["operating_system"])
    print("CUDA available:", hardware["cuda_available"])
    print("MPS available:", hardware["mps_available"])
    print("NVIDIA GPU:", hardware["cuda_device_name"])

    for option in ("auto", "cpu", "gpu", "fpga"):
        print(f"\nSelection: {option}")

        try:
            result = resolve_backend(option)

            print("Backend:", result["backend_label"])
            print("Device:", result["device"])
            print("Hardware:", result["device_label"])

        except (RuntimeError, ValueError) as error:
            print("Unavailable:", error)
