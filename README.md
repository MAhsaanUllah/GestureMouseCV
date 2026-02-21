# 🖱️ Hand Gesture Virtual Mouse

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.6-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-0097A7?style=for-the-badge&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-Pytest-blue?style=for-the-badge&logo=pytest&logoColor=white)

**Control your computer with nothing but your hand — no mouse required.**

*A real-time Computer Vision system that transforms webcam input into full mouse control using hand landmark detection and a trained gesture classifier.*

</div>

---

## 📖 Table of Contents

- [Demo](#-demo)
- [Key Features](#-key-features)
- [Technical Architecture](#-technical-architecture)
- [ML Model Details](#-ml-model-details)
- [Gesture Reference](#-gesture-reference)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Running Tests](#-running-tests)
- [Roadmap](#-roadmap)
- [Author](#-author)

---

## 🎥 Demo

> *Point your index finger at the camera and pinch to click.*

```
Webcam → MediaPipe → Landmark Extraction → Gesture Classifier → PyAutoGUI
   ↑                                                                  ↓
Frame Overlay ←──────────────────────── Mouse / Keyboard Events ──────
```

---

## ✨ Key Features

| Feature | Description |
|--------|-------------|
| **Real-Time Cursor Control** | Moves the OS mouse cursor via index-finger tracking at 30 fps |
| **Pinch-to-Click** | Thumb–index pinch triggers left click; sustained pinch = drag |
| **Swipe Navigation** | Lateral hand swipe sends Alt+Right / Alt+Left (browser back/forward) |
| **Cursor Smoothing** | Exponential Moving Average (EMA) filter eliminates jitter |
| **On-Screen UI Panel** | Semi-transparent hover-aware buttons for quick folder/app access |
| **Configurable** | All thresholds, smoothing, and camera settings in one `config.py` |
| **CLI Interface** | `--device`, `--smoothing`, `--pinch-thresh`, `--debug` flags |
| **Structured Logging** | Console + rotating file log; debug mode for development |
| **Unit Tested** | Headless pytest suite covering gesture math & state machine |

---

## 🏗️ Technical Architecture

```
hand_gesture_virtual_mouse/
│
├── main.py                  ← Entry point (CLI, logging, error handling)
├── gesture_controller.py    ← Core engine (MediaPipe → gesture → OS events)
├── config.py                ← Centralised configuration (all magic numbers here)
├── train_model.py           ← ML training pipeline (standalone, re-runnable)
│
├── dataset/
│   └── gesture_landmarks.csv   ← 42-feature landmark dataset (21 LM × x,y)
│
├── model.h5                 ← Trained Keras model (gesture classifier)
├── scaler_mean.npy          ← StandardScaler mean (for inference normalisation)
├── scaler_scale.npy         ← StandardScaler scale
│
├── tests/
│   └── test_gesture_logic.py   ← Headless unit tests (no webcam required)
│
├── logs/                    ← Auto-created; rotating log files
├── requirements.txt
└── README.md
```

### Data Flow

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Webcam      │───▶│  MediaPipe Hands │───▶│ 21 Landmarks    │
│  (30 fps)    │    │  (real-time det) │    │ (x, y, z each)  │
└──────────────┘    └──────────────────┘    └────────┬────────┘
                                                      │
                                          ┌───────────▼──────────┐
                                          │  Gesture Interpreter  │
                                          │  · pinch distance     │
                                          │  · swipe dx ratio     │
                                          │  · EMA smoothing      │
                                          └───────────┬──────────┘
                                                      │
                                          ┌───────────▼──────────┐
                                          │    PyAutoGUI Events   │
                                          │  moveTo / click /     │
                                          │  mouseDown / hotkey   │
                                          └──────────────────────┘
```

---

## 🧠 ML Model Details

| Property | Value |
|----------|-------|
| **Framework** | TensorFlow / Keras |
| **Input** | 42 features (21 hand landmarks × x, y coordinates) |
| **Architecture** | Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Softmax |
| **Normalisation** | StandardScaler (z-score) |
| **Optimiser** | Adam (lr=0.001) |
| **Callbacks** | EarlyStopping (patience=8) + ReduceLROnPlateau |
| **Train/Val Split** | 80 / 20 (stratified) |

> The classifier is trained offline via `train_model.py` and the saved `model.h5` is used by the real-time loop at inference time.

---

## 🤌 Gesture Reference

| Gesture | Action |
|---------|--------|
| ☝️ **Index finger up** | Move cursor |
| 🤌 **Pinch** (thumb + index close) | Left click |
| ✊ **Hold pinch** | Click & drag |
| 👉 **Swipe right** | Alt + → (forward) |
| 👈 **Swipe left** | Alt + ← (back) |
| 👆 **Cursor over panel button** + pinch | Open folder / Exit |

---

## 📂 Project Structure

```
HandGesture_Virtual_Mouse/
├── main.py                 # Application entry point + CLI
├── gesture_controller.py   # GestureController class (core engine)
├── config.py               # All configuration in one place
├── train_model.py          # ML training pipeline
├── requirements.txt        # Pinned dependencies
├── dataset/
│   └── gesture_landmarks.csv
├── model.h5
├── scaler_mean.npy
├── scaler_scale.npy
├── tests/
│   ├── __init__.py
│   └── test_gesture_logic.py
└── logs/                   # Auto-created at runtime
```

---

## 🚀 Quick Start

### Prerequisites

- **OS:** Windows 10 / 11
- **Python:** 3.9.x (64-bit) — [Download](https://www.python.org/downloads/release/python-3910/)
- **Webcam:** Built-in or USB

> ⚠️ Python 3.10+ is **not** supported (TensorFlow + MediaPipe dependency conflict on that version).

### 1 — Clone & Create Virtual Environment

```powershell
git clone https://github.com/MAhsaanUllah/HandGesture_Virtual_Mouse.git
cd HandGesture_Virtual_Mouse

py -3.9 -m venv cv_env
.\cv_env\Scripts\Activate.ps1
```

### 2 — Install Dependencies

```powershell
# Install in the correct order to avoid conflicts
pip install numpy==1.23.5
pip install opencv-contrib-python==4.6.0.66
pip install pyautogui scikit-learn pandas tensorflow
pip install protobuf==3.20.3 absl-py flatbuffers attrs matplotlib
pip install mediapipe==0.10.5 --no-deps
pip install pytest  # for running tests
```

### 3 — Run

```powershell
# Standard run
python main.py

# With options
python main.py --device 1 --smoothing 0.4 --debug

# Press Q to quit
```

---

## ⚙️ Configuration

All tuneable parameters are in **`config.py`** — no magic numbers anywhere else.

```python
# config.py (examples)

gesture.smoothing_alpha = 0.35    # cursor smoothness (0=max smooth, 1=raw)
gesture.pinch_threshold = 0.04    # pinch sensitivity (smaller = tighter pinch)
gesture.click_cooldown  = 0.60    # seconds between clicks
gesture.swipe_threshold = 0.40    # swipe sensitivity

hand.min_detection_confidence = 0.75
camera.device_index = 0           # webcam index
```

Or pass overrides directly via CLI:

```powershell
python main.py --smoothing 0.5 --pinch-thresh 0.03 --device 1 --debug
```

---

## 🧪 Running Tests

The test suite runs **without a webcam or display** (ideal for CI pipelines):

```powershell
pytest tests/ -v
```

Example output:
```
tests/test_gesture_logic.py::TestEMA::test_alpha_zero_returns_prev   PASSED
tests/test_gesture_logic.py::TestEMA::test_alpha_one_returns_curr    PASSED
tests/test_gesture_logic.py::TestPinchDist::test_pythagorean         PASSED
tests/test_gesture_logic.py::TestGestureController::test_pinch_starts_drag  PASSED
...
```

---

## 🗺️ Roadmap

- [x] Real-time cursor control via index finger
- [x] Pinch-to-click and drag
- [x] Swipe navigation
- [x] Configurable thresholds
- [x] Cursor EMA smoothing
- [x] Unit test suite
- [ ] Right-click gesture (index + middle pinch)
- [ ] Scroll gesture (two-finger swipe)
- [ ] `collect_data.py` GUI for easy dataset collection
- [ ] Tkinter settings panel (adjust thresholds live)
- [ ] Package as standalone `.exe` with PyInstaller

---

## 🛠️ Tech Stack

- **[MediaPipe](https://google.github.io/mediapipe/)** — Real-time hand landmark detection
- **[OpenCV](https://opencv.org/)** — Frame capture, rendering, and UI overlay
- **[TensorFlow / Keras](https://www.tensorflow.org/)** — Gesture classification model
- **[PyAutoGUI](https://pyautogui.readthedocs.io/)** — Cross-platform OS mouse/keyboard control
- **[Scikit-learn](https://scikit-learn.org/)** — StandardScaler + LabelEncoder (training)
- **[Pytest](https://pytest.org/)** — Unit testing

---

## 👤 Author

**Muhammad Ahsaan Ullah**

- 💼 LinkedIn: [linkedin.com/in/mahsaanullah](https://www.linkedin.com/in/mahsaanullah/)
- 🐙 GitHub: [@MAhsaanUllah](https://github.com/MAhsaanUllah)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
⭐ If you found this useful, please star the repo!
</div>
