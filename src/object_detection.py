"""
Real-Time Object Detection using YOLOv8 (Ultralytics) + OpenCV
-----------------------------------------------------------------
Captures live video from a webcam, runs YOLO object detection on
each frame, and draws bounding boxes with class name and confidence
score. Press 'Q' to quit.
"""

import hashlib
import platform
import sys
import time
from typing import Optional

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

# ===========================================================
# CONFIGURATION
# ===========================================================
# Kept as simple module-level constants rather than a separate
# config file/class — this is a small single-purpose script, so a
# dedicated config layer would add indirection without real benefit.
# Everything a user is likely to want to change lives here.

# Model choice:
#   "yolov8n.pt" -> fastest, lowest accuracy (nano)
#   "yolov8s.pt" -> small, noticeably better accuracy, still real-time
#                   on a normal CPU laptop
#   "yolov8m.pt" -> medium, most accurate of these three, but slower
#                   on CPU-only machines (may drop below real-time)
MODEL_NAME = "yolov8s.pt"

# Minimum confidence required to display a detection (0.0 - 1.0).
CONFIDENCE_THRESHOLD = 0.4

# IoU (Intersection over Union) threshold used for Non-Max Suppression.
# Lower values remove more overlapping/duplicate boxes for the same
# object; higher values allow more overlapping boxes to survive.
IOU_THRESHOLD = 0.45

# Requested camera capture resolution. Higher resolution gives YOLO
# more detail to work with, which generally improves detection
# accuracy at the cost of some FPS.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Number of camera indexes to probe when listing available cameras.
MAX_CAMERAS_TO_CHECK = 5

# Smoothing factor for the exponential moving average used in the
# FPS display. Closer to 1.0 = smoother/slower to react; closer to
# 0.0 = noisier/faster to react. 0.9 gives a stable, readable number.
FPS_SMOOTHING = 0.9


def get_camera_backend() -> int:
    """
    Return the OpenCV capture backend appropriate for the current OS.

    DirectShow (CAP_DSHOW) is faster and more reliable for camera
    enumeration on Windows. On Linux/macOS, cv2.CAP_ANY lets OpenCV
    pick the correct native backend (e.g. V4L2 on Linux).
    """
    if platform.system() == "Windows":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def list_available_cameras(max_to_check: int = MAX_CAMERAS_TO_CHECK) -> list[int]:
    """
    Probe camera indexes 0..max_to_check-1 and return the ones that
    successfully open. Each candidate is opened just long enough to
    confirm it works, then released immediately.
    """
    backend = get_camera_backend()
    available = []
    for index in range(max_to_check):
        test_cap = cv2.VideoCapture(index, backend)
        if test_cap.isOpened():
            available.append(index)
        test_cap.release()
    return available


def select_camera(available_cameras: list[int]) -> int:
    """
    Print the available cameras and prompt the user to pick one.
    Re-prompts on invalid input instead of crashing.
    """
    print("\nAvailable cameras:")
    for idx in available_cameras:
        label = "Laptop webcam (usually index 0)" if idx == 0 else f"External/other camera (index {idx})"
        print(f"  {idx}: {label}")

    while True:
        choice = input(f"\nEnter camera index to use {available_cameras}: ").strip()
        if choice.isdigit() and int(choice) in available_cameras:
            return int(choice)
        print("Invalid choice. Please enter one of the listed numbers.")


def initialize_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    """
    Open the requested camera and configure its resolution.

    Raises RuntimeError if the camera cannot be opened, so the caller
    can decide how to handle/report the failure.
    """
    backend = get_camera_backend()
    cap = cv2.VideoCapture(camera_index, backend)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_index}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")

    return cap


def load_model(model_name: str) -> YOLO:
    """
    Load a pretrained YOLO model by name.

    If the weights file isn't present locally, Ultralytics downloads
    it automatically (requires internet on first run). Any failure
    here (bad model name, no internet on first run, corrupted file)
    is reported clearly rather than left as a raw traceback.
    """
    print(f"Loading model '{model_name}' (auto-downloads on first use)...")
    try:
        model = YOLO(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load YOLO model '{model_name}'. "
            f"Check the model name and your internet connection. Details: {exc}"
        ) from exc
    print("Model loaded.")
    return model


def get_color_for_class(class_id: int) -> tuple[int, int, int]:
    """Generate a deterministic BGR color from a class ID, so each
    object class is drawn in a consistent, distinguishable color."""
    hash_val = int(hashlib.md5(str(class_id).encode()).hexdigest(), 16)
    b = hash_val % 255
    g = (hash_val // 255) % 255
    r = (hash_val // (255 * 255)) % 255
    return (int(b), int(g), int(r))


def draw_detections(frame, result: Results, class_names: dict) -> None:
    """
    Draw a bounding box and a 'class_name confidence%' label for every
    detection in `result` directly onto `frame` (modified in place).

    Detections below CONFIDENCE_THRESHOLD are already excluded by the
    time they reach here, because the threshold is passed into the
    model() call itself (see process_frame).
    """
    frame_height, frame_width = frame.shape[:2]

    for box in result.boxes:
        # box.xyxy[0] -> [x1, y1, x2, y2] pixel coordinates of the
        # top-left and bottom-right corners of the bounding box.
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Clamp coordinates so boxes never draw outside the visible frame
        # (can happen slightly at the edges due to model rounding).
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width - 1, x2), min(frame_height - 1, y2)

        # box.conf[0] -> confidence score in [0, 1].
        confidence = float(box.conf[0])

        # box.cls[0] -> numeric class ID; class_names maps it to a
        # human-readable label (e.g. 0 -> "person").
        class_id = int(box.cls[0])
        class_name = class_names.get(class_id, f"class_{class_id}")

        label = f"{class_name} {confidence * 100:.0f}%"
        color = get_color_for_class(class_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),  # keep text on-screen near the top edge
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def process_frame(model: YOLO, frame) -> tuple[Results, float]:
    """
    Run YOLO inference on a single frame.

    Returns the Results object for the frame along with the inference
    time in milliseconds, so the caller can display both detections
    and performance information.
    """
    start = time.perf_counter()
    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
    )
    inference_ms = (time.perf_counter() - start) * 1000
    return results[0], inference_ms


def calculate_fps(prev_time: float, smoothed_fps: Optional[float]) -> tuple[float, float]:
    """
    Compute a smoothed FPS value using an exponential moving average,
    which is far more readable on screen than the raw per-frame FPS
    (which can jump around a lot frame to frame).

    Returns (current_time, smoothed_fps) — call again next frame with
    these as the new prev_time/smoothed_fps.
    """
    current_time = time.time()
    elapsed = current_time - prev_time
    instant_fps = 1 / elapsed if elapsed > 0 else 0.0

    if smoothed_fps is None:
        smoothed_fps = instant_fps
    else:
        smoothed_fps = FPS_SMOOTHING * smoothed_fps + (1 - FPS_SMOOTHING) * instant_fps

    return current_time, smoothed_fps


def draw_overlay(frame, fps: float, inference_ms: float) -> None:
    """Draw the FPS and inference-time overlay in the top-left corner."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Inference: {inference_ms:.0f} ms",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )


def run_detection_loop(model: YOLO, cap: cv2.VideoCapture) -> None:
    """
    Main capture-detect-display loop. Reads frames until the user
    presses 'Q' or the camera stops providing frames.
    """
    window_name = "YOLO Object Detection - Press Q to Quit"
    prev_time = time.time()
    smoothed_fps: Optional[float] = None
    consecutive_failures = 0
    max_consecutive_failures = 10  # tolerate brief glitches, not a dead camera

    while True:
        ret, frame = cap.read()

        if not ret:
            consecutive_failures += 1
            print(f"Warning: failed to read frame ({consecutive_failures}/{max_consecutive_failures}).")
            if consecutive_failures >= max_consecutive_failures:
                print("Error: camera appears to have disconnected. Exiting.")
                break
            continue
        consecutive_failures = 0

        result, inference_ms = process_frame(model, frame)
        draw_detections(frame, result, model.names)

        prev_time, smoothed_fps = calculate_fps(prev_time, smoothed_fps)
        draw_overlay(frame, smoothed_fps, inference_ms)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit key pressed. Closing application.")
            break


def main() -> None:
    """Entry point: set up the model and camera, run detection, clean up."""
    try:
        model = load_model(MODEL_NAME)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    available_cameras = list_available_cameras()
    if not available_cameras:
        print("Error: No cameras found. Check connections and try again.")
        sys.exit(1)

    if len(available_cameras) == 1:
        camera_index = available_cameras[0]
        print(f"\nOne camera found (index {camera_index}); using it automatically.")
    else:
        camera_index = select_camera(available_cameras)

    try:
        cap = initialize_camera(camera_index, FRAME_WIDTH, FRAME_HEIGHT)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print("Starting object detection. Press 'Q' to quit.")
    try:
        run_detection_loop(model, cap)
    finally:
        # Always release the camera and close windows, even if the
        # loop exits due to an unexpected error.
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()