"""
Real-Time Object Detection using YOLO (Ultralytics) + OpenCV
--------------------------------------------------------------
Captures live video from the webcam, runs YOLO object detection
on each frame, and draws bounding boxes with class name and
confidence score. Press 'Q' to quit.
"""

import time
import hashlib
import cv2
from ultralytics import YOLO

# ===========================================================
# CONFIGURATION
# ===========================================================
# Model choice:
#   "yolov8n.pt" -> fastest, lowest accuracy (nano)
#   "yolov8s.pt" -> small, noticeably better accuracy, still real-time
#                   on a normal CPU laptop (~15-30 FPS typically)
#   "yolov8m.pt" -> medium, most accurate of these three, but slower
#                   on CPU-only machines (may drop below real-time)
#
# "yolov8s.pt" is used here as the best accuracy/speed tradeoff for
# a laptop webcam demo. Change to "yolov8n.pt" if your machine is
# older/slower and you need more FPS.
MODEL_NAME = "yolov8s.pt"

# Minimum confidence required to display a detection (0.0 - 1.0).
CONFIDENCE_THRESHOLD = 0.4

# IoU (Intersection over Union) threshold used for Non-Max Suppression.
# Lower values remove more overlapping/duplicate boxes for the same
# object; higher values allow more overlapping boxes to survive.
# 0.45 is a good general-purpose default.
IOU_THRESHOLD = 0.45

# Requested camera capture resolution. Higher resolution gives YOLO
# more detail to work with, which improves detection accuracy —
# most modern webcams (laptop or external) support 1280x720.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ---------------------------------------------------------
# 1. Load the pretrained YOLO model
# ---------------------------------------------------------
# Ultralytics will auto-download the weights file the first time
# it is used (requires internet access on first run).
model = YOLO(MODEL_NAME)

# ---------------------------------------------------------
# 2. Detect available cameras and let the user choose one
# ---------------------------------------------------------
def list_available_cameras(max_to_check=5):
    """
    Tries camera indexes 0..max_to_check-1 and returns a list
    of indexes that successfully open. On Windows, cv2.CAP_DSHOW
    is used because it detects cameras faster and more reliably
    than the default backend.
    """
    available = []
    for index in range(max_to_check):
        test_cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if test_cap.isOpened():
            available.append(index)
            test_cap.release()
    return available


available_cameras = list_available_cameras()

if not available_cameras:
    print("Error: No cameras found. Check connections and try again.")
    exit()

print("\nAvailable cameras:")
for idx in available_cameras:
    label = "Laptop webcam (usually index 0)" if idx == 0 else f"External/other camera (index {idx})"
    print(f"  {idx}: {label}")

# Ask the user which camera to use
selected_index = -1
while selected_index not in available_cameras:
    choice = input(f"\nEnter camera index to use {available_cameras}: ").strip()
    if choice.isdigit() and int(choice) in available_cameras:
        selected_index = int(choice)
    else:
        print("Invalid choice. Please enter one of the listed numbers.")

# ---------------------------------------------------------
# 3. Open the selected webcam
# ---------------------------------------------------------
cap = cv2.VideoCapture(selected_index, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not access the selected webcam.")
    exit()

# Request a higher capture resolution. Not all cameras support the
# exact value requested, but most will pick the closest supported
# resolution instead of failing.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {actual_width}x{actual_height}")

print("Starting object detection. Press 'Q' to quit.")

# For the on-screen FPS counter
prev_time = time.time()

# Assigns a consistent, distinct color to each COCO class ID so
# different object types are visually easy to tell apart.
def get_color_for_class(class_id):
    """Generate a deterministic BGR color from a class ID."""
    hash_val = int(hashlib.md5(str(class_id).encode()).hexdigest(), 16)
    b = hash_val % 255
    g = (hash_val // 255) % 255
    r = (hash_val // (255 * 255)) % 255
    return (int(b), int(g), int(r))

# ---------------------------------------------------------
# 4. Main loop: read frames, run detection, display results
# ---------------------------------------------------------
while True:
    ret, frame = cap.read()  # Capture one frame from the webcam

    if not ret:
        print("Error: Failed to grab frame from webcam.")
        break

    # Run YOLO inference on the current frame.
    # conf: minimum confidence to keep a detection (filters weak results)
    # iou: overlap threshold for Non-Max Suppression (removes duplicate boxes)
    # verbose=False stops YOLO from printing logs to the console every frame.
    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
    )

    # results is a list (one entry per image passed in).
    # Since we passed a single frame, we only need results[0].
    result = results[0]

    # result.boxes contains all detected bounding boxes for this frame.
    for box in result.boxes:
        # --- Bounding box coordinates ---
        # xyxy gives [x1, y1, x2, y2] = top-left and bottom-right corners.
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # --- Confidence score ---
        # (Detections below CONFIDENCE_THRESHOLD are already filtered
        # out by the model() call above via the conf parameter.)
        confidence = float(box.conf[0])

        # --- Class name ---
        # box.cls[0] is the numeric class ID (e.g., 0, 1, 2...).
        # model.names maps that ID to a human-readable label
        # (e.g., "person", "car", "dog").
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Format label text, e.g. "person 87%"
        label = f"{class_name} {confidence * 100:.0f}%"

        # Give each object class a consistent, distinct color so
        # different types of objects are easy to tell apart at a glance.
        color = get_color_for_class(class_id)

        # --- Draw the bounding box ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # --- Draw the label above the box ---
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),  # keep text on-screen if box is near top
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    # --- Calculate and display FPS ---
    # Measures real-world frames-per-second so you can see the actual
    # performance of the model on your machine.
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    # Show the annotated frame in a window
    cv2.imshow("YOLO Object Detection - Press Q to Quit", frame)

    # Wait 1ms for a key press; exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Quit key pressed. Closing application.")
        break

# ---------------------------------------------------------
# 5. Cleanup — release webcam and close windows
# ---------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
