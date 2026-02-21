"""
tests/test_gesture_logic.py
────────────────────────────
Unit tests for the pure-logic helpers inside gesture_controller.py.

These tests do NOT require a webcam, screen, or any OpenCV window —
they validate the math and state machine in isolation, making them
safe to run in headless CI environments (e.g. GitHub Actions).

Run
---
    pytest tests/ -v
"""

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
#  Minimal stubs so gesture_controller can be imported without a real display
# ─────────────────────────────────────────────────────────────────────────────

# Stub pyautogui before importing gesture_controller
pyautogui_stub = types.ModuleType("pyautogui")
pyautogui_stub.FAILSAFE = True
pyautogui_stub.size = lambda: (1920, 1080)
pyautogui_stub.moveTo = MagicMock()
pyautogui_stub.mouseDown = MagicMock()
pyautogui_stub.mouseUp = MagicMock()
pyautogui_stub.click = MagicMock()
pyautogui_stub.hotkey = MagicMock()
sys.modules.setdefault("pyautogui", pyautogui_stub)

# Stub cv2 / mediapipe minimally
cv2_stub = types.ModuleType("cv2")
cv2_stub.CAP_DSHOW = 700
cv2_stub.VideoCapture = MagicMock()
cv2_stub.flip = MagicMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))
cv2_stub.cvtColor = MagicMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))
cv2_stub.rectangle = MagicMock()
cv2_stub.putText = MagicMock()
cv2_stub.circle = MagicMock()
cv2_stub.imshow = MagicMock()
cv2_stub.waitKey = MagicMock(return_value=ord("q"))
cv2_stub.destroyAllWindows = MagicMock()
cv2_stub.addWeighted = MagicMock(side_effect=lambda src, a, dst, b, g, dst2: None)
cv2_stub.COLOR_BGR2RGB = 4
cv2_stub.FONT_HERSHEY_SIMPLEX = 0
cv2_stub.CAP_PROP_FRAME_WIDTH = 3
cv2_stub.CAP_PROP_FRAME_HEIGHT = 4
cv2_stub.CAP_PROP_FPS = 5
sys.modules.setdefault("cv2", cv2_stub)

mp_stub = types.ModuleType("mediapipe")
mp_solutions = types.ModuleType("mediapipe.solutions")
mp_hands_stub = types.ModuleType("mediapipe.solutions.hands")
mp_hands_stub.Hands = MagicMock()
mp_hands_stub.HAND_CONNECTIONS = []
mp_drawing_stub = types.ModuleType("mediapipe.solutions.drawing_utils")
mp_drawing_stub.draw_landmarks = MagicMock()
mp_solutions.hands = mp_hands_stub
mp_solutions.drawing_utils = mp_drawing_stub
mp_stub.solutions = mp_solutions
sys.modules.setdefault("mediapipe", mp_stub)
sys.modules.setdefault("mediapipe.solutions", mp_solutions)
sys.modules.setdefault("mediapipe.solutions.hands", mp_hands_stub)
sys.modules.setdefault("mediapipe.solutions.drawing_utils", mp_drawing_stub)

# Now it is safe to import
from gesture_controller import GestureController, _ema, _pinch_dist  # noqa: E402
import config as cfg  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
#  _ema
# ─────────────────────────────────────────────────────────────────────────────

class TestEMA:
    def test_alpha_zero_returns_prev(self):
        """Alpha=0 ⟹ result equals previous value (full smoothing)."""
        assert _ema(prev=100.0, curr=200.0, alpha=0.0) == pytest.approx(100.0)

    def test_alpha_one_returns_curr(self):
        """Alpha=1 ⟹ result equals current value (no smoothing)."""
        assert _ema(prev=100.0, curr=200.0, alpha=1.0) == pytest.approx(200.0)

    def test_mid_blend(self):
        """Alpha=0.5 ⟹ exact midpoint."""
        assert _ema(prev=0.0, curr=100.0, alpha=0.5) == pytest.approx(50.0)

    def test_commutativity_does_not_hold(self):
        """EMA is NOT symmetric — prev and curr are distinct roles."""
        assert _ema(0.0, 100.0, 0.3) != _ema(100.0, 0.0, 0.3)


# ─────────────────────────────────────────────────────────────────────────────
#  _pinch_dist
# ─────────────────────────────────────────────────────────────────────────────

class TestPinchDist:
    def _landmark(self, x, y):
        lm = MagicMock()
        lm.x = x
        lm.y = y
        return lm

    def test_same_point_is_zero(self):
        a = self._landmark(0.5, 0.5)
        assert _pinch_dist(a, a) == pytest.approx(0.0)

    def test_unit_horizontal_distance(self):
        a = self._landmark(0.0, 0.0)
        b = self._landmark(1.0, 0.0)
        assert _pinch_dist(a, b) == pytest.approx(1.0)

    def test_pythagorean(self):
        """3-4-5 triangle → distance = 5 / 100 = 0.05."""
        a = self._landmark(0.00, 0.00)
        b = self._landmark(0.03, 0.04)
        assert _pinch_dist(a, b) == pytest.approx(0.05)


# ─────────────────────────────────────────────────────────────────────────────
#  GestureController — config & state machine
# ─────────────────────────────────────────────────────────────────────────────

class TestGestureController:
    @pytest.fixture
    def ctrl(self):
        return GestureController()

    def test_default_screen_size(self, ctrl):
        assert ctrl._screen_w == 1920
        assert ctrl._screen_h == 1080

    def test_initial_no_drag(self, ctrl):
        assert ctrl._dragging is False

    def test_pinch_starts_drag(self, ctrl):
        ctrl._handle_pinch(pinch=True)
        assert ctrl._dragging is True
        pyautogui_stub.mouseDown.assert_called()

    def test_release_after_drag_fires_click(self, ctrl):
        pyautogui_stub.click.reset_mock()
        ctrl._handle_pinch(pinch=True)
        ctrl._last_click = 0.0  # force cooldown to pass
        ctrl._handle_pinch(pinch=False)
        assert ctrl._dragging is False
        pyautogui_stub.click.assert_called_once()

    def test_swipe_throttled_by_cooldown(self, ctrl):
        import time
        pyautogui_stub.hotkey.reset_mock()
        ctrl._last_nav = time.time()      # reset cooldown to NOW
        ctrl._handle_swipe(dx=0.9)        # should NOT fire
        pyautogui_stub.hotkey.assert_not_called()

    def test_swipe_right_fires_alt_right(self, ctrl):
        pyautogui_stub.hotkey.reset_mock()
        ctrl._last_nav = 0.0              # cooldown expired
        ctrl._handle_swipe(dx=0.9)
        pyautogui_stub.hotkey.assert_called_once_with("alt", "right")

    def test_swipe_left_fires_alt_left(self, ctrl):
        pyautogui_stub.hotkey.reset_mock()
        ctrl._last_nav = 0.0
        ctrl._handle_swipe(dx=-0.9)
        pyautogui_stub.hotkey.assert_called_once_with("alt", "left")

    def test_custom_buttons_respected(self):
        custom = [("Custom", "/some/path"), ("Exit", "exit")]
        ctrl = GestureController(buttons=custom)
        assert ctrl._buttons == custom
