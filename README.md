# YOLO Real-Time Object Detection Using OpenCV

## Overview

This project performs real-time object detection using a webcam feed. It combines **OpenCV** for video capture and display with a pretrained **Ultralytics YOLOv8** model for detection. Every detected object is drawn with a bounding box, its class name, and its confidence score, while the app also reports live FPS and per-frame inference time.

**Demo files available:** [demo.mp4](demo/demo.mp4) — short demo recording | [detection_preview.png](demo/detection_preview.png) — screenshot | [demo/readme.md](demo/readme.md) — documentation

## Features

- Real-time webcam object detection
- Bounding boxes drawn around every detected object
- Object class name displayed per detection
- Confidence score displayed as a percentage
- Live FPS monitoring (smoothed) and per-frame inference time (ms)
- Automatic detection of available cameras, with a selection prompt when more than one is found
- Configurable confidence threshold
- Configurable IoU threshold
- Configurable capture resolution
- `Q` / `q` to exit
- Graceful error handling for missing cameras, failed camera opens, dropped frames, and model load failures

## Architecture

```
Webcam
  ↓
OpenCV Frame Capture
  ↓
YOLO Inference
  ↓
Detection Results (boxes, class IDs, confidence scores)
  ↓
Bounding Box + Class Name + Confidence (drawn with OpenCV)
  ↓
OpenCV Display
```

## Project Structure

```
yolo-object-detection/
│
├── demo/
│   ├── demo.mp4                 # optional short demo recording
│   └── detection_preview.png    # optional screenshot
│
├── src/
│   └── object_detection.py      # main application
│
├── .gitignore
├── requirements.txt
└── README.md
```

Key project files:
- **Main Application:** [src/object_detection.py](src/object_detection.py)
- **Dependencies:** [requirements.txt](requirements.txt)
- **Demo Folder:** [demo/](demo/) (includes optional video and screenshot)
- **Configuration:** [.gitignore](.gitignore)

Model weight files (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`) are **not** committed to this repo — Ultralytics downloads them automatically the first time the script runs.

## Requirements

- Python 3.9–3.12 (recommended; very new Python releases can lag behind PyTorch/Ultralytics compatibility)
- [OpenCV](https://pypi.org/project/opencv-python/) — webcam capture, frame display, drawing
- [Ultralytics YOLO](https://docs.ultralytics.com/) — pretrained object detection model

## Installation

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it (Windows)
venv\Scripts\activate

# 2. Activate it (macOS/Linux)
source venv/bin/activate

# 3. Install dependencies
pip install -r [requirements.txt](requirements.txt)
```

## Running

```bash
python [src/object_detection.py](src/object_detection.py)
```

On first run:
- Ultralytics downloads the YOLOv8 model weights automatically (a few MB, requires internet once).
- The terminal lists all detected cameras. If more than one is found, you'll be prompted to choose; if only one is found, it's used automatically.
- A window opens showing the live annotated video feed with bounding boxes, labels, FPS, and inference time.

## Controls

| Key | Action |
|---|---|
| `Q` / `q` | Exit the application |

## Configuration

All key parameters are defined as constants near the top of [src/object_detection.py](src/object_detection.py):

| Variable | Purpose | Default |
|---|---|---|
| `MODEL_NAME` | Which YOLOv8 weights file to load | `"[yolov8s.pt](yolov8s.pt)"` |
| `CONFIDENCE_THRESHOLD` | Minimum confidence required to display a detection | `0.4` |
| `IOU_THRESHOLD` | Overlap threshold used by Non-Max Suppression to remove duplicate boxes | `0.45` |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | Requested webcam capture resolution | `1280` / `720` |

## Performance

Actual FPS and inference time depend on several factors:

- CPU/GPU used for inference
- YOLO model size (`n` vs `s` vs `m` vs `l` vs `x`)
- Input/capture resolution
- Number of objects detected in a given frame

The app displays both a smoothed FPS value and per-frame inference time (ms) live on screen, so real performance on your machine is always visible directly in the app.

### Benchmark — measured results

Test system: Windows, Python 3.14, CPU inference (no GPU/CUDA), external USB webcam. Each row uses the app's built-in benchmark system (`WARMUP_SECONDS = 3`, `CONFIDENCE_THRESHOLD = 0.40`, `IOU_THRESHOLD = 0.45`), values copied directly from the terminal summary — none of these are estimated.

| Model | Resolution | Frames Tested | Duration | Avg FPS | Min FPS | Max FPS | Avg Inference | Min Inference | Max Inference |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | 640x480 | 236 | 25.1s | 9.41 | 4.28 | 10.01 | 103.18 ms | 96.04 ms | 230.51 ms |
| YOLOv8n | 1280x720 | 245 | 24.4s | 10.05 | 5.11 | 11.57 | 85.57 ms | 79.08 ms | 123.11 ms |
| YOLOv8s | 640x480 | 99 | 26.6s | 3.74 | 0.30 | 4.29 | 293.31 ms | 229.97 ms | 3005.44 ms |
| YOLOv8s | 1280x720 | 128 | 26.6s | 4.81 | 0.27 | 5.21 | 225.47 ms | 186.39 ms | 3093.40 ms |
| YOLOv8m | 640x480 | 50 | 28.9s | 1.67 | 0.27 | 1.83 | 641.53 ms | 542.23 ms | 3433.89 ms |
| YOLOv8m | 1280x720 | 53 | 24.8s | 2.07 | 0.26 | 2.25 | 522.13 ms | 438.25 ms | 3269.66 ms |
| YOLOv8l | 640x480 | 22 | 23.8s | 0.86 | 0.25 | 0.95 | 1248.00 ms | 1052.63 ms | 3736.51 ms |
| YOLOv8l | 1280x720 | 25 | 23.0s | 1.02 | 0.24 | 1.13 | 1055.57 ms | 878.50 ms | 3648.40 ms |
| YOLOv8x | 640x480 | 21 | 33.3s | 0.58 | 0.22 | 0.61 | 1783.53 ms | 1628.76 ms | 4207.38 ms |
| YOLOv8x | 1280x720 | 27 | 35.4s | 0.72 | 0.25 | 0.75 | 1430.70 ms | 1334.66 ms | 3437.20 ms |

### Benchmark Analysis

The benchmark confirms the expected model-size trend: average inference time increases from **YOLOv8n → YOLOv8x**, while average FPS decreases correspondingly. This reflects the increasing computational workload of larger models. **YOLOv8n at 1280×720 achieved the best result of the sweep at approximately 10 FPS average**, while none of the `m`-size or larger models reached 3 FPS.

The benchmark also showed some run-to-run performance variation. During the current benchmark session, the results were substantially more stable, with `yolov8n` showing no large inference outlier. The `s` and larger models still exhibited occasional high-latency frames. Since the code and Python environment remained unchanged, these variations are likely influenced by temporary system conditions such as background processes, Windows Defender/antivirus activity, or thermal state.

Overall, the results demonstrate that **YOLOv8n provides the best real-time performance for the tested system**, while larger models offer increased computational complexity at the cost of significantly lower throughput.

### Reproducing the Benchmark

To reproduce or extend the benchmark:

1. Change `MODEL_NAME` in [src/object_detection.py](src/object_detection.py).
2. Change `FRAME_WIDTH` and `FRAME_HEIGHT` for the desired resolution.
3. Run:

```bash
python [src/object_detection.py](src/object_detection.py)
```

4. Allow the application to run beyond the **"Warm-up complete"** message.
5. Press **Q** to stop the test.
6. Record the printed performance summary.

## Limitations

- Performance depends heavily on hardware — CPU-only inference is significantly slower than GPU inference. Measured on this test machine (CPU-only, no CUDA), `yolov8n` averaged ~9-10 FPS at both tested resolutions — usable but below the ~15-30+ FPS often seen for `yolov8n` on CPU, and background system load appears to meaningfully affect results run-to-run (see Performance above).
- The pretrained model only detects the 80 object classes it was trained on (the COCO dataset — e.g. person, car, chair, laptop, phone, bottle). It cannot recognize custom or brand-specific objects.
- Detection quality depends on webcam quality, lighting, and object distance/angle.
- Larger models (`yolov8m`, `l`, `x`) are not practical for real-time use on this CPU-only hardware — measured average FPS dropped below 2.1 FPS for `yolov8m` and below 1.1 FPS for `yolov8l`/`yolov8x`.

## Future Improvements

- GPU acceleration (CUDA) for higher FPS
- Side-by-side comparison across YOLO model sizes
- Object tracking across frames
- Training on a custom dataset for domain-specific classes
- Support for video-file input in addition to a live webcam
- Structured performance benchmarking across hardware
- Deployment on edge devices (e.g. Raspberry Pi, Jetson Nano)

## How Detection Works

1. Each webcam frame is passed to `model()`, which returns a `Results` object per input frame.
2. `result.boxes` holds every detection found in that frame.
3. For each detected box:
   - `box.xyxy[0]` → bounding box coordinates `[x1, y1, x2, y2]` (top-left and bottom-right corners, in pixels)
   - `box.conf[0]` → confidence score in `[0, 1]`, shown on screen as a percentage
   - `box.cls[0]` → numeric class ID, mapped to a readable label via `model.names`
4. Detections below `CONFIDENCE_THRESHOLD` are filtered out by passing `conf=CONFIDENCE_THRESHOLD` directly into the `model()` call, so only meaningful detections reach the drawing step.
5. `IOU_THRESHOLD` controls Non-Max Suppression — when the same object produces multiple overlapping boxes, IoU (the ratio of overlap area to union area between two boxes) is used to discard the redundant ones and keep the best one.
6. OpenCV draws each bounding box and label directly onto the frame, which is then shown with `cv2.imshow`.

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError` for `cv2` or `ultralytics` | Re-run `pip install -r [requirements.txt](requirements.txt)` inside the activated virtual environment |
| No camera found | Check physical connections; on Windows, check Settings → Privacy → Camera |
| Webcam not detected / wrong camera opens | Close other apps using the camera (Zoom, Teams, OBS); rerun and check the printed camera list |
| Low FPS / laggy video | Switch `MODEL_NAME` to `"[yolov8n.pt](yolov8n.pt)"` for faster (but slightly less accurate) inference, or lower `FRAME_WIDTH`/`FRAME_HEIGHT` |
| First run fails to start | Requires internet access once, to auto-download the model weights file |
| Detections seem inaccurate | Ensure good lighting and that the object is one of the 80 COCO classes the model was trained on |
| Camera disconnects mid-run | The app tolerates a few dropped frames and exits cleanly with a message if the camera stops responding entirely |

## Notes

- This project intentionally focuses on detection only — no object tracking, custom training, GUI framework, or database, per project scope.
- Model weights are excluded from version control via [.gitignore](.gitignore) since they are large and downloaded automatically on demand.
