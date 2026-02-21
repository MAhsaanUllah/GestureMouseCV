"""
main.py — Application entry point for Hand Gesture Virtual Mouse.

Run
---
    python main.py
    python main.py --debug      # verbose logging
    python main.py --device 1   # use webcam index 1
"""

import argparse
import logging
import sys
from pathlib import Path

import config as cfg
from gesture_controller import GestureController


# ─────────────────────────────────────────────────────────────────────────────
#  Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger to write to both console and a rotating file."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]

    if cfg.logging.to_file:
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(
            cfg.logging.log_file, maxBytes=2 * 1024 * 1024, backupCount=3,
            encoding="utf-8",
        )
        handlers.append(fh)

    logging.basicConfig(level=numeric, format=fmt, handlers=handlers)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hand Gesture Virtual Mouse — control your PC with gestures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--device", type=int, default=cfg.camera.device_index,
        metavar="N", help="Webcam device index (default: %(default)s)",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    p.add_argument(
        "--smoothing", type=float, default=cfg.gesture.smoothing_alpha,
        metavar="A",
        help="Cursor EMA alpha — 0.0 (max smooth) to 1.0 (raw). (default: %(default)s)",
    )
    p.add_argument(
        "--pinch-thresh", type=float, default=cfg.gesture.pinch_threshold,
        metavar="T",
        help="Pinch distance threshold in normalised coords. (default: %(default)s)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    log_level = "DEBUG" if args.debug else cfg.logging.level
    setup_logging(log_level)

    logger = logging.getLogger("main")
    logger.info("=" * 60)
    logger.info("Hand Gesture Virtual Mouse — starting up")
    logger.info("=" * 60)

    # Override config values from CLI
    cfg.camera.device_index        = args.device
    cfg.gesture.smoothing_alpha    = args.smoothing
    cfg.gesture.pinch_threshold    = args.pinch_thresh

    try:
        controller = GestureController()
        controller.run()
    except SystemExit as e:
        logger.info("Clean exit requested (code %s).", e.code)
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
        return 0
    except Exception:
        logger.exception("Unhandled exception — please report this as a bug.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
