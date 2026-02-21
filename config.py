"""
config.py — Centralized configuration for Hand Gesture Virtual Mouse.

All tunable parameters live here so the rest of the codebase
stays free of magic numbers and is easy to adjust without
touching business logic.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
MODEL_PATH  = ROOT_DIR / "model.h5"
SCALER_MEAN = ROOT_DIR / "scaler_mean.npy"
SCALER_SCALE= ROOT_DIR / "scaler_scale.npy"
LOG_DIR     = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
#  Camera
# ─────────────────────────────────────────────
@dataclass
class CameraConfig:
    device_index: int  = 0          # webcam index (0 = default)
    backend: int       = None       # cv2.CAP_DSHOW on Windows, None elsewhere
    frame_width: int   = 640
    frame_height: int  = 480
    fps: int           = 30


# ─────────────────────────────────────────────
#  MediaPipe Hand Tracking
# ─────────────────────────────────────────────
@dataclass
class HandConfig:
    max_num_hands: int              = 1
    min_detection_confidence: float = 0.75
    min_tracking_confidence: float  = 0.75


# ─────────────────────────────────────────────
#  Gesture Thresholds
# ─────────────────────────────────────────────
@dataclass
class GestureConfig:
    pinch_threshold: float   = 0.04   # normalised distance (index–thumb)
    swipe_threshold: float   = 0.40   # dx ratio (index–wrist) for left/right swipe
    click_cooldown: float    = 0.60   # seconds between consecutive clicks
    nav_cooldown: float      = 1.00   # seconds between navigation swipes
    move_duration: float     = 0.00   # pyautogui moveTo duration — 0 = instant (EMA handles smoothness)
    smoothing_alpha: float   = 0.20   # EMA alpha: 0=max smooth, 1=raw. 0.20 = very smooth


# ─────────────────────────────────────────────
#  On-Screen UI Panel
# ─────────────────────────────────────────────
@dataclass
class UIConfig:
    panel_x1: int          = 10
    panel_x2: int          = 220
    panel_top: int         = 50
    button_height: int     = 42
    button_gap: int        = 58
    font_scale: float      = 0.68
    font_thickness: int    = 2
    bg_color: tuple        = (20, 20, 20)
    hover_color: tuple     = (30, 100, 200)   # bright blue highlight on hover
    text_color: tuple      = (240, 240, 240)
    cursor_color: tuple    = (0, 230, 255)    # vivid cyan dot
    cursor_radius: int     = 9


# ─────────────────────────────────────────────
#  Model Training
# ─────────────────────────────────────────────
@dataclass
class TrainConfig:
    test_size: float       = 0.20
    random_state: int      = 42
    epochs: int            = 50
    batch_size: int        = 32
    learning_rate: float   = 0.001
    dense1_units: int      = 128
    dense2_units: int      = 64
    dropout_rate: float    = 0.30
    early_stopping_patience: int = 8


# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
@dataclass
class LogConfig:
    level: str   = "INFO"           # DEBUG | INFO | WARNING | ERROR
    to_file: bool = True
    log_file: Path = LOG_DIR / "app.log"


# ─────────────────────────────────────────────
#  Assembled app config (singleton-ish)
# ─────────────────────────────────────────────
camera   = CameraConfig()
hand     = HandConfig()
gesture  = GestureConfig()
ui       = UIConfig()
training = TrainConfig()
logging  = LogConfig()
