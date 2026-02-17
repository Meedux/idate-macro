# Copilot instructions for `idate`

## Big picture (what actually runs)
- This repo has **three runtime paths**:
  1. **GUI production path** in `src/gui.py` (`IdateApp._main_loop`) for CV/template-based detection.
  2. **Memory read path** in `src/gui.py` (`IdateApp._main_loop_memory`) for direct process memory detection (zero-CV, sub-ms latency).
  3. **Generic CLI path** in `src/cli.py` + `src/vision/*` for lane/note experiments.
- GUI has a **mode toggle** (CV Template / Memory Read) that switches between pipelines.
- CV loop data flow: `capture (CaptureSession)` -> `detector (RhythmDetector, fallback YOLODetector)` -> `sendinput scan codes` -> `Win32 overlay`.
- Memory loop data flow: `ProcessAttacher (ReadProcessMemory)` -> `MemoryReader` -> `MemoryDetector.get_keys_to_press()` -> `sendinput scan codes` -> `Win32 overlay`.
- `RhythmDetector` (`src/rhythm_detector.py`) is the CV gameplay detector: calibrated ROI, template-first matching, optional YOLO/RF-DETR.
- `MemoryDetector` (`src/memory/memory_detector.py`) is the memory-based detector: reads process memory, no screen capture required.

## Critical architecture details
- Manual calibration is persisted in `calibration.json`; GUI calibration dialog writes this via `RhythmDetector.set_calibration()`.
- Memory patterns are persisted in `memory_patterns.json`; discovered via `src/memory/scan_interactive.py` and loaded by `MemoryReader`.
- Detection coordinates are window-relative, then rendered by overlay (`src/overlay/win32_overlay.py`); keep coordinate transforms consistent.
- Input is Windows-native: `src/input/sendinput_driver.py` (`scan_key_press`) is preferred over VK events for game compatibility.
- Global stop/start uses F12 in multiple places (`GlobalHotkey` in GUI and safety controls in runtime modules).
- Memory reading uses Windows API (`OpenProcess`, `ReadProcessMemory`, `VirtualQueryEx`) via ctypes – no external dependencies.

## Project-specific conventions
- **Do not collapse config formats** without checking call sites:
  - `configs/game_profile.toml` (GUI-oriented sections: `[track]`, `[notes]`, etc.).
  - `src/config.py` Pydantic models (CLI-oriented fields: `capture.roi`, `lanes`, `timing`).
- Template assets in `templates/*.png` are first-class dependencies (indicator + arrows).
- The codebase keeps performance-first patterns (tight loops, minimal sleeps, preallocated/cached overlay items). Preserve these when refactoring.
- GUI is Windows-first and may self-elevate to admin (`run_as_admin()` in `src/gui.py`).

## Developer workflows that matter
- Install: `pip install -e .` (or run `run_gui.bat`, which creates `.venv`, installs deps, launches GUI).
- Fast capture optional extras: `pip install -e ".[fast]"` (DXCam support).
- Main run path: `python -m src.gui`.
- Memory scanner: `python -m src.memory.scan_interactive` (interactive Cheat Engine-style scanner).
- CLI diagnostics: `python -m src.cli dry-run --show-overlay`, `python -m src.cli benchmark`.
- Train detector model: `python -m src.ml.train_yolo --fast` (or full settings).
- Roboflow upload integration: `python -m src.ml.upload_to_roboflow --dataset datasets/idate`.

## Integration points and external deps
- Detection stack uses OpenCV + NumPy + Ultralytics (`ultralytics`) and optional `rfdetr`.
- Capture backends: MSS default, DXCam optional (`src/capture/mss_capture.py`).
- Overlay/UI: `customtkinter`, tkinter, pygame-based Win32 overlay.
- Input backends include SendInput and pynput (focus/hotkey/safety utilities in `src/runtime/control.py`).

## Editing guidance for AI agents
- Prefer changes in the active GUI path unless user asks for CLI behavior.
- Preserve threaded boundaries: UI updates via `self.after(...)`; detection/input remains in worker threads.
- Keep latency-sensitive constants explicit (cooldowns, thresholds, sleep durations); avoid hidden abstractions that add overhead.
- When changing detection or overlay logic, validate with live dry-run behavior (FPS, hit-zone alignment, key spam/cooldown behavior).