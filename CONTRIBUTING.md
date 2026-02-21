# Contributing to Hand Gesture Virtual Mouse

Thank you for your interest in contributing! 🎉

## Getting Started

1. **Fork** the repository and clone your fork:
   ```bash
   git clone https://github.com/MAhsaanUllah/HandGesture_Virtual_Mouse.git
   cd HandGesture_Virtual_Mouse
   ```

2. **Create a virtual environment** (Python 3.9 required):
   ```powershell
   py -3.9 -m venv cv_env
   .\cv_env\Scripts\Activate.ps1
   pip install -r requirements.txt
   pip install pytest
   ```

3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Standards

- Follow **PEP 8** style conventions.
- All new functions must have **Google-style docstrings** with Args / Returns sections.
- Use **type hints** on all function signatures.
- Add or update **unit tests** for any new gesture logic.
- Run `pytest tests/ -v` and ensure all tests pass before submitting.

## Submitting a Pull Request

1. Commit your changes with a clear message:
   ```
   feat: add right-click gesture (index + middle pinch)
   fix: prevent swipe double-fire on slow hardware
   ```

2. Push to your fork and open a **Pull Request** against `main`.

3. Fill in the PR description:
   - What does this PR do?
   - How was it tested?
   - Any known limitations?

## Reporting Bugs

Open an [Issue](../../issues) with:
- Python version and OS
- Steps to reproduce
- Expected vs actual behaviour
- Any relevant log output (check `logs/app.log`)

## Feature Requests

Open an Issue tagged `enhancement`. Describe the use-case and expected UX.

---

Thank you for helping improve the project! 🚀
