# Hand Gesture Virtual Mouse — Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions.

---

## [2.0.0] — 2026-02-21 — Professional Refactor

### Added
- `config.py` — Centralised configuration; all thresholds, paths, and
  camera settings in one file. No more magic numbers scattered across code.
- `gesture_controller.py` — `GestureController` class encapsulating the
  full pipeline (webcam → landmarks → gestures → OS events).
- CLI flags: `--device`, `--smoothing`, `--pinch-thresh`, `--debug`.
- **Cursor EMA smoothing** — configurable `smoothing_alpha` eliminates jitter.
- **Hover effect** on UI panel buttons (steel-blue highlight + alpha blend).
- **Structured logging** — console + rotating file log via Python `logging`.
- `tests/test_gesture_logic.py` — Headless pytest suite (runs in CI).
- `CHANGELOG.md` — Version history following Keep a Changelog conventions.
- `.gitignore` — Comprehensive Python + image/video/OS ignores.
- `pyproject.toml` — Modern Python packaging metadata (PEP 517/518).

### Changed
- `main.py` refactored from a script to a clean entry point with argument
  parsing, logging setup, and proper `sys.exit()` return codes.
- `train_model.py` refactored with full CLI, early stopping, LR scheduling,
  and overfitting gap reporting.
- `README.md` — Complete rewrite with architecture diagram, ML model table,
  gesture reference, and portfolio-quality documentation.

### Fixed
- `pyautogui.FAILSAFE` moved to a documented configuration decision.
- Camera backend auto-detected by OS (CAP_DSHOW on Windows only).

---

## [1.0.0] — 2026-02-01 — Initial Version

### Added
- Basic webcam capture and hand landmark detection via MediaPipe.
- Index-finger cursor control.
- Pinch-to-click and drag.
- Swipe left/right for browser navigation.
- On-screen shortcut panel (Desktop, This PC, Project, Exit).
- `train_model.py` for offline gesture classifier training.
- `model.h5`, `scaler_mean.npy`, `scaler_scale.npy` — training artefacts.
