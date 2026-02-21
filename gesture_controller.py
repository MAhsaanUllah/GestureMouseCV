"""
gesture_controller.py
─────────────────────
Core engine for the Hand Gesture Virtual Mouse.

Responsibilities
----------------
* Capture frames from the webcam.
* Detect hand landmarks via MediaPipe.
* Interpret gestures (move, pinch-click, pinch-drag, swipe L/R).
* Smooth cursor trajectory with an Exponential Moving Average (EMA).
* Dispatch PyAutoGUI mouse/keyboard events.
* Render the live overlay (landmarks, UI panel, cursor dot).

Usage
-----
    from gesture_controller import GestureController
    ctrl = GestureController()
    ctrl.run()
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

import config as cfg

# ─────────────────────────────────────────────────────────────────────────────
#  Module-level logger — handlers configured in main.py
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

pyautogui.FAILSAFE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Pure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ema(prev: float, curr: float, alpha: float) -> float:
    """Exponential Moving Average — blends raw value toward smoothed history.

    Args:
        prev:  Previous smoothed value.
        curr:  New raw value.
        alpha: Blend factor [0, 1].  0 = fully smoothed, 1 = fully raw.

    Returns:
        Smoothed value.
    """
    return alpha * curr + (1.0 - alpha) * prev


def _pinch_dist(lm_a, lm_b) -> float:
    """Normalised Euclidean distance between two MediaPipe landmarks."""
    return float(np.hypot(lm_a.x - lm_b.x, lm_a.y - lm_b.y))


# ─────────────────────────────────────────────────────────────────────────────
#  UI Buttons
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_BUTTONS = [
    ("Desktop",  str(Path.home() / "Desktop")),
    ("This PC",  "shell:MyComputerFolder"),
    ("Project",  str(Path(__file__).resolve().parent)),
    ("Exit",     "exit"),
]


# ─────────────────────────────────────────────────────────────────────────────
#  GestureController
# ─────────────────────────────────────────────────────────────────────────────

class GestureController:
    """Full hand-gesture → OS-mouse-event pipeline."""

    def __init__(
        self,
        camera_cfg=None,
        hand_cfg=None,
        gesture_cfg=None,
        ui_cfg=None,
        buttons=None,
    ):
        self._cam_cfg  = camera_cfg  or cfg.camera
        self._hand_cfg = hand_cfg    or cfg.hand
        self._gest_cfg = gesture_cfg or cfg.gesture
        self._ui_cfg   = ui_cfg      or cfg.ui
        self._buttons  = buttons     or DEFAULT_BUTTONS

        # Screen dimensions
        self._screen_w, self._screen_h = pyautogui.size()

        # ── Smoothed cursor state (screen-space pixels) ──────────────────
        self._smooth_x: float = self._screen_w / 2
        self._smooth_y: float = self._screen_h / 2

        # ── Smoothed cursor in CAMERA-SPACE (for visual dot & hit-test) ──
        self._cam_smooth_x: float = self._cam_cfg.frame_width  / 2
        self._cam_smooth_y: float = self._cam_cfg.frame_height / 2

        # ── Gesture state ────────────────────────────────────────────────
        self._dragging: bool       = False
        self._last_click: float    = 0.0
        self._last_nav: float      = 0.0
        self._last_btn_time: float = 0.0   # debounce for UI buttons
        self._btn_debounce: float  = 1.2   # seconds between button activations

        # ── Exit signal — set by button handler, checked in main loop ────
        self._exit_requested: bool = False

        # ── Resources (lazy-init in start()) ─────────────────────────────
        self._cap: Optional[cv2.VideoCapture] = None
        self._hands = None
        self._draw  = mp.solutions.drawing_utils

        logger.info(
            "GestureController initialised | screen=%dx%d",
            self._screen_w, self._screen_h,
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open webcam and initialise MediaPipe Hands."""
        import platform
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else 0
        self._cap = cv2.VideoCapture(self._cam_cfg.device_index, backend)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cam_cfg.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cam_cfg.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._cam_cfg.fps)

        mp_hands = mp.solutions.hands
        self._hands = mp_hands.Hands(
            max_num_hands=self._hand_cfg.max_num_hands,
            min_detection_confidence=self._hand_cfg.min_detection_confidence,
            min_tracking_confidence=self._hand_cfg.min_tracking_confidence,
        )
        logger.info("Camera opened (device %d).", self._cam_cfg.device_index)

    def stop(self) -> None:
        """Release webcam and destroy all OpenCV windows."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            self._cap = None
        if self._hands:
            self._hands.close()
            self._hands = None
        cv2.destroyAllWindows()
        logger.info("GestureController stopped cleanly.")

    def run(self) -> None:
        """Start + main loop. Handles clean teardown on any exit path."""
        self.start()
        logger.info("Virtual Mouse running — press Q to quit.")

        try:
            while not self._exit_requested:
                ok, frame = self._cap.read()
                if not ok:
                    logger.warning("Dropped frame — camera read failed.")
                    continue

                frame = cv2.flip(frame, 1)
                self._process_frame(frame)

                # Draw UI panel on top of everything
                self._draw_ui_panel(frame)

                # Draw smooth cursor dot (camera-space, not raw)
                cx = int(self._cam_smooth_x)
                cy = int(self._cam_smooth_y)
                if 0 <= cx < frame.shape[1] and 0 <= cy < frame.shape[0]:
                    # Outer ring
                    cv2.circle(frame, (cx, cy),
                                self._ui_cfg.cursor_radius + 3,
                                (255, 255, 255), 1)
                    # Filled dot
                    cv2.circle(frame, (cx, cy),
                                self._ui_cfg.cursor_radius,
                                self._ui_cfg.cursor_color, -1)

                cv2.imshow("Hand Gesture Virtual Mouse", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:   # Q or ESC
                    logger.info("User pressed Q/Esc — exiting.")
                    break

        finally:
            self.stop()

    # ──────────────────────────────────────────────────────────────────────
    #  Frame processing
    # ──────────────────────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> None:
        """Detect hand, update cursor, fire mouse events, update camera-space EMA."""
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            # No hand → gradually drift camera dot back to center
            return

        hand = result.multi_hand_landmarks[0]
        self._draw.draw_landmarks(
            frame, hand, mp.solutions.hands.HAND_CONNECTIONS
        )

        index = hand.landmark[8]   # index fingertip
        thumb = hand.landmark[4]   # thumb tip
        wrist = hand.landmark[0]   # wrist base

        alpha = self._gest_cfg.smoothing_alpha

        # ── Screen-space smoothing (drives the actual OS cursor) ─────────
        raw_sx = index.x * self._screen_w
        raw_sy = index.y * self._screen_h
        self._smooth_x = _ema(self._smooth_x, raw_sx, alpha)
        self._smooth_y = _ema(self._smooth_y, raw_sy, alpha)
        pyautogui.moveTo(
            int(self._smooth_x), int(self._smooth_y),
            duration=self._gest_cfg.move_duration,
        )

        # ── Camera-space smoothing (drives the visual dot in the window) ─
        h, w = frame.shape[:2]
        raw_cx = index.x * w
        raw_cy = index.y * h
        self._cam_smooth_x = _ema(self._cam_smooth_x, raw_cx, alpha)
        self._cam_smooth_y = _ema(self._cam_smooth_y, raw_cy, alpha)

        # ── Pinch state ──────────────────────────────────────────────────
        pinch = _pinch_dist(index, thumb) < self._gest_cfg.pinch_threshold
        self._handle_pinch(pinch)

        # ── Swipe navigation ─────────────────────────────────────────────
        dx = index.x - wrist.x
        self._handle_swipe(dx)

        # ── UI button hit-test (only on pinch, with debounce) ────────────
        if pinch:
            cam_cursor = (int(self._cam_smooth_x), int(self._cam_smooth_y))
            self._handle_ui_click(cam_cursor)

    # ──────────────────────────────────────────────────────────────────────
    #  Gesture handlers
    # ──────────────────────────────────────────────────────────────────────

    def _handle_pinch(self, pinch: bool) -> None:
        """Convert pinch state → mouseDown / mouseUp / click."""
        now = time.time()
        if pinch and not self._dragging:
            pyautogui.mouseDown()
            self._dragging = True
            logger.debug("mouseDown (drag start)")

        elif not pinch and self._dragging:
            pyautogui.mouseUp()
            self._dragging = False
            if now - self._last_click > self._gest_cfg.click_cooldown:
                pyautogui.click()
                self._last_click = now
                logger.debug("click fired")

    def _handle_swipe(self, dx: float) -> None:
        """Fire Alt+Right / Alt+Left on lateral hand swipe."""
        now = time.time()
        if now - self._last_nav < self._gest_cfg.nav_cooldown:
            return

        thresh = self._gest_cfg.swipe_threshold
        if dx > thresh:
            pyautogui.hotkey("alt", "right")
            self._last_nav = now
            logger.debug("Swipe RIGHT (alt+right)")
        elif dx < -thresh:
            pyautogui.hotkey("alt", "left")
            self._last_nav = now
            logger.debug("Swipe LEFT (alt+left)")

    def _handle_ui_click(self, cam_cursor: Tuple[int, int]) -> None:
        """Check smooth cursor against each button; trigger on pinch with debounce.

        FIX: Uses a time-based debounce so Exit / folder buttons do NOT
        fire repeatedly every frame while the pinch is held.
        FIX: Exit sets a flag on the controller — the main loop checks it
        and breaks cleanly, rather than raising SystemExit mid-frame.
        """
        now = time.time()
        if now - self._last_btn_time < self._btn_debounce:
            return   # still in debounce window — ignore

        cx, cy = cam_cursor
        ui = self._ui_cfg
        for i, (label, target) in enumerate(self._buttons):
            y1 = ui.panel_top + i * ui.button_gap
            y2 = y1 + ui.button_height
            if ui.panel_x1 < cx < ui.panel_x2 and y1 < cy < y2:
                logger.info("Button '%s' activated.", label)
                self._last_btn_time = now   # reset debounce

                if target == "exit":
                    logger.info("Exit button pressed — shutting down.")
                    self._exit_requested = True   # signal main loop to break
                    return

                try:
                    os.startfile(target)
                except Exception as exc:
                    logger.error("Failed to open '%s': %s", target, exc)
                break

    # ──────────────────────────────────────────────────────────────────────
    #  UI Rendering
    # ──────────────────────────────────────────────────────────────────────

    def _draw_ui_panel(self, frame: np.ndarray) -> None:
        """Render semi-transparent hover-aware button panel onto *frame*."""
        ui = self._ui_cfg
        cx = int(self._cam_smooth_x)
        cy = int(self._cam_smooth_y)

        # Determine which button (if any) the smoothed cursor is hovering
        hovering = None
        for i, _ in enumerate(self._buttons):
            y1 = ui.panel_top + i * ui.button_gap
            if ui.panel_x1 < cx < ui.panel_x2 and y1 < cy < y1 + ui.button_height:
                hovering = i
                break

        for i, (label, _) in enumerate(self._buttons):
            y1  = ui.panel_top + i * ui.button_gap
            y2  = y1 + ui.button_height
            col = ui.hover_color if hovering == i else ui.bg_color

            # Semi-transparent fill
            overlay = frame.copy()
            cv2.rectangle(overlay,
                           (ui.panel_x1, y1), (ui.panel_x2, y2),
                           col, -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

            # Thin border
            border_col = (200, 200, 200) if hovering == i else (80, 80, 80)
            cv2.rectangle(frame,
                           (ui.panel_x1, y1), (ui.panel_x2, y2),
                           border_col, 1)

            # Label text
            cv2.putText(
                frame, label,
                (ui.panel_x1 + 12, y1 + 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                ui.font_scale, ui.text_color, ui.font_thickness,
                cv2.LINE_AA,   # anti-aliased for sharper text
            )
