"""
Real-Time Object Detection using YOLOv8 (Ultralytics) + OpenCV
-----------------------------------------------------------------
Captures live video from a webcam, runs YOLO object detection on
each frame, and draws bounding boxes with class name and confidence
score. Press 'Q' to quit.
"""

import hashlib
import platform
import statistics
import sys
import time
from typing import Optional

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

# ===========================================================
# CONFIGURATION
# ===========================================================

MODEL_NAME = "yolov8s.pt"

CONFIDENCE_THRESHOLD = 0.4

IOU_THRESHOLD = 0.45

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

MAX_CAMERAS_TO_CHECK = 5

FPS_SMOOTHING = 0.9

# --- Benchmarking ---
WARMUP_SECONDS = 3

MIN_BENCHMARK_SAMPLES = 10


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


def initialize_camera(camera_index: int, width: int, height: int) -> tuple[cv2.VideoCapture, int, int]:
    """
    Open the requested camera and configure its resolution.

    Cameras don't always support the exact resolution requested, so the
    actual negotiated resolution is read back once here (not re-queried
    every frame) and returned alongside the capture object, so callers
    — including the benchmark summary — can report what was actually
    used rather than assuming the request was honored.

    Raises RuntimeError if the camera cannot be opened, so the caller
    can decide how to handle/report the failure.

    Returns (cap, actual_width, actual_height). If the camera reports
    an invalid/zero resolution (rare, but possible with some drivers),
    width/height fall back to 0 rather than crashing; callers should
    treat 0 as "unavailable" when displaying it.
    """
    backend = get_camera_backend()
    cap = cv2.VideoCapture(camera_index, backend)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_index}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    except (cv2.error, TypeError, ValueError):
        actual_width, actual_height = 0, 0

    if actual_width <= 0 or actual_height <= 0:
        print("Warning: could not read actual camera resolution from the device.")
        actual_width, actual_height = 0, 0

    print(f"Camera resolution: {actual_width}x{actual_height}")

    return cap, actual_width, actual_height


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


def calculate_fps(prev_time: float, smoothed_fps: Optional[float]) -> tuple[float, float, float]:
    """
    Compute both a smoothed FPS (for the on-screen display, which reads
    better when it isn't jumping around every frame) and the raw
    per-frame instant FPS (used for benchmark statistics, where
    smoothing would distort the true min/max).

    Guards against divide-by-zero if two frames report the same
    timestamp (can happen on very fast frames on some systems).

    Returns (current_time, smoothed_fps, instant_fps) — call again next
    frame with current_time/smoothed_fps as the new prev_time/smoothed_fps.
    """
    current_time = time.time()
    elapsed = current_time - prev_time
    instant_fps = 1 / elapsed if elapsed > 0 else 0.0

    if smoothed_fps is None:
        smoothed_fps = instant_fps
    else:
        smoothed_fps = FPS_SMOOTHING * smoothed_fps + (1 - FPS_SMOOTHING) * instant_fps

    return current_time, smoothed_fps, instant_fps


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


def format_model_name(model_name: str) -> str:
    """Turn a weights filename like 'yolov8n.pt' into a readable label
    like 'YOLOv8n' for display in the benchmark summary."""
    base = model_name.removesuffix(".pt")
    if base.lower().startswith("yolov8"):
        return "YOLOv8" + base[len("yolov8"):]
    return base


def print_benchmark_summary(
    frames_tested: int,
    test_duration: float,
    fps_samples: list[float],
    inference_samples: list[float],
    actual_width: int,
    actual_height: int,
) -> None:
    """
    Print a clean, copy-paste-ready performance summary to the terminal.

    Only called after the user quits (Q pressed) or the app exits.
    Uses only real measurements collected after the warm-up period —
    nothing here is estimated or hardcoded. If too few samples were
    collected (e.g. the app was closed during warm-up), a clear
    message is printed instead of misleading statistics.

    Reports both the REQUESTED resolution (FRAME_WIDTH x FRAME_HEIGHT
    from config) and the ACTUAL resolution the camera negotiated
    (actual_width x actual_height from initialize_camera), since a
    webcam may not support the exact resolution requested.
    """
    print("\n" + "=" * 50)
    print("           YOLO PERFORMANCE SUMMARY")
    print("=" * 50 + "\n")

    if frames_tested < MIN_BENCHMARK_SAMPLES or not fps_samples or not inference_samples:
        print("Not enough benchmark data collected.")
        print("(Try running for longer than the warm-up period before quitting.)\n")
        print("=" * 50)
        return

    actual_resolution = (
        f"{actual_width}x{actual_height}" if actual_width > 0 and actual_height > 0 else "Unavailable"
    )

    print(f"Model                : {format_model_name(MODEL_NAME)}")
    print(f"Requested Resolution : {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"Actual Resolution    : {actual_resolution}")
    print(f"Frames Tested        : {frames_tested}")
    print(f"Test Duration        : {test_duration:.1f} seconds\n")

    print(f"Average FPS          : {statistics.mean(fps_samples):.2f} FPS")
    print(f"Min FPS              : {min(fps_samples):.2f} FPS")
    print(f"Max FPS              : {max(fps_samples):.2f} FPS\n")

    print(f"Average Inference    : {statistics.mean(inference_samples):.2f} ms")
    print(f"Min Inference        : {min(inference_samples):.2f} ms")
    print(f"Max Inference        : {max(inference_samples):.2f} ms\n")

    print(f"Confidence           : {CONFIDENCE_THRESHOLD:.2f}")
    print(f"IoU Threshold        : {IOU_THRESHOLD:.2f}")
    print("\n" + "=" * 50)


def run_detection_loop(model: YOLO, cap: cv2.VideoCapture) -> tuple[int, float, list[float], list[float]]:
    """
    Main capture-detect-display loop. Reads frames until the user
    presses 'Q' or the camera stops providing frames.

    Also runs the benchmarking logic on top of normal detection:
    - The first WARMUP_SECONDS are excluded from statistics (model
      warm-up / camera auto-exposure settling).
    - After warm-up, every frame's instant FPS and inference time
      are recorded.

    Returns the collected (benchmark_frames, test_duration, fps_samples,
    inference_samples) so the caller can print the summary AFTER the
    camera is released and windows are destroyed, per the required
    shutdown order: stop loop -> release camera -> destroy windows ->
    print benchmark summary.
    """
    window_name = "YOLO Object Detection - Press Q to Quit"
    prev_time = time.time()
    smoothed_fps: Optional[float] = None
    consecutive_failures = 0
    max_consecutive_failures = 10  # tolerate brief glitches, not a dead camera

    # --- Benchmark state ---
    loop_start_time = time.time()
    warmup_announced = False
    benchmark_start_time: Optional[float] = None
    benchmark_frames = 0
    fps_samples: list[float] = []
    inference_samples: list[float] = []

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

        prev_time, smoothed_fps, instant_fps = calculate_fps(prev_time, smoothed_fps)
        draw_overlay(frame, smoothed_fps, inference_ms)  # on-screen display unchanged

        # --- Warm-up / benchmark data collection ---
        # Only start recording once WARMUP_SECONDS have elapsed since
        # the loop began, so startup/model-warm-up effects don't skew
        # the results.
        if time.time() - loop_start_time >= WARMUP_SECONDS:
            if not warmup_announced:
                print(f"Warm-up complete ({WARMUP_SECONDS}s). Benchmark started.")
                warmup_announced = True
                benchmark_start_time = time.time()

            if instant_fps > 0:  # guard against a stray zero/invalid reading
                fps_samples.append(instant_fps)
            inference_samples.append(inference_ms)
            benchmark_frames += 1

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit key pressed. Closing application.")
            break

    test_duration = (time.time() - benchmark_start_time) if benchmark_start_time else 0.0
    return benchmark_frames, test_duration, fps_samples, inference_samples


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
        cap, actual_width, actual_height = initialize_camera(camera_index, FRAME_WIDTH, FRAME_HEIGHT)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print("Starting object detection. Press 'Q' to quit.")
    benchmark_data = (0, 0.0, [], [])
    try:
        benchmark_data = run_detection_loop(model, cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print_benchmark_summary(*benchmark_data, actual_width, actual_height)


if __name__ == "__main__":
    main()