# Real-Time Object Detection with YOLOv8 and OpenCV

A real-time object detection application that uses a webcam feed, detects objects using a pretrained YOLOv8 model, and displays bounding boxes, class names, and confidence scores live on screen.

<!--
Add a screenshot or short GIF of the app running here, e.g.:
![Demo](demo.gif)
-->

## Features

- Real-time object detection from a live webcam feed
- Supports selecting between multiple connected cameras (e.g. laptop webcam vs external USB webcam)
- Draws bounding boxes with class name and confidence score (%) for every detected object
- Live FPS counter to show real-world performance
- Distinct color per object class for easier visual distinction
- Clean exit — press **Q** to quit; webcam and windows are released properly
- Tunable confidence and IoU thresholds for detection quality

## Tech Stack

- **Python 3.10 / 3.11** (recommended)
- **OpenCV** — webcam capture, frame display, drawing bounding boxes/text
- **Ultralytics YOLOv8** — pretrained object detection model (trained on the COCO dataset, 80 object classes)

## Project Structure

```
.
├── object_detection.py     # Main application
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

`yolov8s.pt` (the model weights file) is **not** included in this repo — it is automatically downloaded by Ultralytics the first time the script runs.

## Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## How to Run

```bash
python object_detection.py
```

On first run:
- Ultralytics will automatically download the YOLOv8 model weights (a few MB, requires internet).
- The terminal will list all detected cameras (e.g. laptop webcam at index `0`, external webcam at index `1`).
- Enter the index of the camera you want to use.
- A window will open showing the live annotated video feed.
- Press **Q** to close the application.

## Model Choice

This project uses `yolov8s.pt` (YOLOv8 "small") rather than the nano (`yolov8n.pt`) variant.

| Model | Speed | Accuracy | Notes |
|---|---|---|---|
| yolov8n | Fastest | Lowest | Best for very limited hardware |
| **yolov8s (used here)** | Fast | Good | Best balance for a CPU laptop webcam demo |
| yolov8m | Slower | Higher | May drop below real-time on CPU-only machines |

`yolov8s.pt` was chosen as the best tradeoff between detection accuracy and real-time performance on typical laptop hardware without a dedicated GPU.

## Configuration

These values can be adjusted at the top of `object_detection.py`:

| Variable | Purpose | Default |
|---|---|---|
| `MODEL_NAME` | Which YOLOv8 weights file to load | `"yolov8s.pt"` |
| `CONFIDENCE_THRESHOLD` | Minimum confidence to display a detection | `0.4` |
| `IOU_THRESHOLD` | Overlap threshold for removing duplicate boxes | `0.45` |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | Requested webcam capture resolution | `1280x720` |

## How Detection Works (Summary)

1. Each webcam frame is passed to `model()`, which returns a `Results` object per frame.
2. `result.boxes` contains all detections for that frame.
3. For each detected box:
   - `box.xyxy[0]` → bounding box coordinates `[x1, y1, x2, y2]`
   - `box.conf[0]` → confidence score (0–1, converted to a percentage for display)
   - `box.cls[0]` → numeric class ID, mapped to a readable label via `model.names`
4. OpenCV draws the bounding box and label directly onto the frame, which is then displayed with `cv2.imshow`.

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError` for `cv2` or `ultralytics` | Re-run `pip install -r requirements.txt` inside the activated virtual environment |
| Webcam not detected / wrong camera opens | Close other apps using the camera (Zoom, Teams, OBS); rerun the script and check the printed camera list |
| Low FPS / laggy video | Switch `MODEL_NAME` to `"yolov8n.pt"` for faster (but slightly less accurate) inference |
| First run fails to start | Requires internet access once, to auto-download the model weights file |
| Detections seem inaccurate | Ensure good lighting and that the object is one of the 80 COCO classes the model was trained on (e.g. person, car, chair, laptop, phone, bottle, etc. — not brand-specific items) |

## Notes

- This project intentionally focuses only on detection — no object tracking, custom training, GUI framework, or database is included, per the project scope.
- Model weights are excluded from version control via `.gitignore` since they are large and auto-downloaded on demand.