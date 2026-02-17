# idate - Rhythm Game Automation

A computer vision-based automation system for rhythm games with 8-directional arrow input.

## Features

- **Full GUI Application**: Modern dark-themed interface with real-time preview and all settings configurable
- **Screen Capture**: High-speed screen capture using MSS (cross-platform) or DXCam (Windows, higher FPS)
- **Lane Detection**: Auto-detect lane positions from game visuals
- **Note Tracking**: Track notes as they approach the hit line with velocity estimation
- **Timing Fusion**: Combine CV detection with beatmap/chart timing data for better accuracy
- **Input Execution**: Windows SendInput for reliable key injection, pynput for cross-platform
- **Pattern Support**: 4-lane and 8-lane configurations with chord (multi-key) support
- **Safety Controls**: Emergency stop hotkey, window focus requirement, rate limiting

## Requirements

- Python 3.10+
- Windows (for SendInput keyboard injection; pynput works cross-platform but may not work in all games)

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd idate
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

For high-FPS capture on Windows (optional):
```bash
pip install -e ".[fast]"
```

## Quick Start - GUI (Recommended)

**Launch the GUI application:**

```bash
python -m src.gui
```

Or after installation:
```bash
idate-gui
```

The GUI provides:

### Configuration Panel (Left Side)
- **Capture Region**: Set screen coordinates (left, top, width, height)
- **Lanes**: Configure lane count (4-10), edges, and hit line position
- **Detection**: Choose method (color/contour/template), set HSV colors and tolerances
- **Timing**: Adjust hit offset, reaction time, and timing windows
- **Safety**: Enable focus requirement, set window pattern, rate limits, stop hotkey
- **Key Bindings**: Map all 8 directions to keyboard keys

### Control Panel (Right Side)
- **Mode Selection**: Dry Run (no inputs) or Live (actual key presses)
- **Start/Stop**: Control automation with buttons or F12 hotkey
- **Calibrate**: Auto-detect lane positions from current capture
- **Preview**: Live view of capture with lane/note overlay
- **Save/Load**: Export and import configurations as JSON

### Features
- Real-time FPS, action count, and latency display
- Log panel showing all events
- Visual overlay showing detected lanes and notes

---

## Quick Start - CLI

### 1. Configure the game profile

Edit `configs/game_profile.toml` to match your game's capture region and lane layout:

```toml
[capture]
left = 0
top = 0
width = 1920
height = 1080

[lanes]
count = 8
left_edge = 400
right_edge = 1520
hit_line_y = 900
```

### 2. Configure key mappings

Edit `configs/keymap.toml` to match your key bindings:

```toml
[lanes]
0 = "up_left"
1 = "up"
2 = "up_right"
3 = "left"
4 = "right"
5 = "down_left"
6 = "down"
7 = "down_right"
```

### 3. Calibrate

Run calibration to auto-detect lane positions:

```bash
python -m src.cli calibrate --lanes 8 --save calibration.json
```

### 4. Test with dry-run

Run detection without executing inputs:

```bash
python -m src.cli dry-run --duration 30 --show-overlay
```

### 5. Run automation

```bash
python -m src.cli run --show-overlay
```

Press **F12** to stop (or customize with `--stop-key`).

## Commands

### `calibrate`
Auto-detect game window and lane positions.

```bash
python -m src.cli calibrate [--window PATTERN] [--lanes N] [--save FILE]
```

### `record`
Record screen frames for reference/analysis.

```bash
python -m src.cli record [--output DIR] [--frames N] [--delay SECS]
```

### `dry-run`
Run detection without executing inputs. Useful for testing and tuning.

```bash
python -m src.cli dry-run [--duration SECS] [--show-overlay] [--save-trace FILE]
```

### `run`
Full automation mode.

```bash
python -m src.cli run [--beatmap FILE] [--show-overlay] [--stop-key KEY]
```

### `benchmark`
Test capture and detection speed.

```bash
python -m src.cli benchmark [--iterations N] [--capture-only]
```

## Architecture

```
src/
├── capture/          # Screen capture (MSS, DXCam)
├── vision/           # CV detection (lanes, notes)
├── timing/           # Beatmap parsing, timing fusion
├── input/            # Keyboard injection (SendInput, pynput)
├── patterns/         # Lane-to-key mapping, chords
├── runtime/          # Safety controls, main loop
├── debug/            # Overlay visualization, tracing
├── config.py         # Configuration models
└── cli.py            # Command-line interface
```

## Supported Input Directions

| Direction | Aliases |
|-----------|---------|
| UP | up, u |
| DOWN | down, d |
| LEFT | left, l |
| RIGHT | right, r |
| UP_LEFT | up_left, upleft, ul |
| UP_RIGHT | up_right, upright, ur |
| DOWN_LEFT | down_left, downleft, dl, low_left, lowleft |
| DOWN_RIGHT | down_right, downright, dr |

## Beatmap Formats

The system can parse timing data from:

- **JSON**: Custom format with `notes`, `bpm`, `offset` fields
- **CSV**: Simple `time_ms,lane` format
- **osu! mania**: `.osu` files (mania mode)
- **StepMania**: `.sm` and `.ssc` files

## Safety Features

1. **Emergency Stop**: Press F12 (configurable) to immediately stop automation
2. **Window Focus**: Only executes when game window is focused
3. **Rate Limiting**: Configurable maximum actions per second
4. **Dry-Run Mode**: Test detection without any input execution

## Configuration Reference

### game_profile.toml

```toml
game_name = "My Game"

[capture]
left = 0
top = 0
width = 1920
height = 1080
backend = "mss"        # "mss" or "dxcam"
target_fps = 60

[lanes]
count = 8
left_edge = 400
right_edge = 1520
hit_line_y = 900

[detection]
method = "color"       # "color", "contour", or "template"
note_color_hsv = [0, 0, 255]
note_color_tolerance = [180, 50, 50]
min_note_area = 100
max_note_area = 10000

[timing]
hit_offset_ms = 0
reaction_time_ms = 16
early_window_ms = 50
late_window_ms = 100

[safety]
require_focus = true
window_title_pattern = ".*"
max_actions_per_second = 60
stop_hotkey = "f12"

[debug]
show_overlay = false
save_frames = false
log_level = "INFO"
```

### keymap.toml

```toml
[directions]
up = "w"
down = "s"
left = "a"
right = "d"
up_left = "q"
up_right = "e"
down_left = "z"
down_right = "c"

[lanes]
0 = "up_left"
1 = "up"
2 = "up_right"
3 = "left"
4 = "right"
5 = "down_left"
6 = "down"
7 = "down_right"
```

## Troubleshooting

### Detection not working
1. Run `calibrate` to verify lane detection
2. Use `dry-run --show-overlay` to see what's being detected
3. Adjust `detection.note_color_hsv` and tolerance in config

### Keys not registering in game
1. Run as Administrator (some games require elevated privileges)
2. Try different input backend (SendInput vs pynput)
3. Check if game uses anti-cheat that blocks input injection

### Low FPS
1. Use DXCam backend on Windows: `pip install dxcam`
2. Reduce capture region size
3. Run `benchmark` to identify bottleneck

### Timing issues
1. Adjust `timing.hit_offset_ms` to compensate for input delay
2. Use beatmap file for more accurate timing
3. Increase `timing.early_window_ms` for more lenient detection

## Disclaimer

This tool is for educational and personal use only. Use of automation tools may violate the terms of service of online games. The authors are not responsible for any consequences of using this software.

## License

MIT License
