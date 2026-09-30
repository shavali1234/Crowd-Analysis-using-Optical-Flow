import platform
import subprocess
import cv2


def get_cpu_name():
    """Get the actual CPU model name on Windows."""

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).Name"
            ],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        cpu_name = result.stdout.strip()

        if cpu_name:
            return cpu_name

    except Exception:
        pass

    return platform.processor() or "CPU"


def get_cuda_info():
    """Detect CUDA support through OpenCV."""

    try:
        gpu_count = cv2.cuda.getCudaEnabledDeviceCount()

        if gpu_count <= 0:
            return {
                "available": False,
                "name": "No CUDA GPU detected",
                "count": 0
            }

        try:
            device_info = cv2.cuda.DeviceInfo(0)
            gpu_name = device_info.name()
        except Exception:
            gpu_name = "NVIDIA CUDA GPU"

        return {
            "available": True,
            "name": gpu_name,
            "count": gpu_count
        }

    except Exception:
        return {
            "available": False,
            "name": "CUDA unavailable",
            "count": 0
        }


def detect_cameras(max_cameras=10):
    """
    Detect available cameras.

    Returns:
        [(index, description), ...]
    """

    cameras = []

    for index in range(max_cameras):

        cap = cv2.VideoCapture(
            index,
            cv2.CAP_DSHOW
        )

        if cap.isOpened():

            ret, frame = cap.read()

            if ret and frame is not None:
                cameras.append(
                    (index, f"Camera {index}")
                )

            cap.release()

        else:
            cap.release()

    return cameras