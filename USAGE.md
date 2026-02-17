# iDate Revival — Usage Guide

## Quick Start (3 Steps)

1. **Install** — Double-click `install.bat` (one time only)
2. **Launch the game** — Open iDate Revival normally
3. **Run the bot** — Double-click `run.bat`

That's it! The program will:
- Auto-detect the iDate process and window
- Attach to the game's memory
- Wait for a song to start, then auto-play

## Controls

| Action | How |
|--------|-----|
| Start automation | Click **▶ START** or press **F12** |
| Stop automation | Click **■ STOP** or press **F12** |
| Refresh detection | Click **↻ Refresh** if iDate was launched after the bot |

## How It Works

The program reads the game's memory directly — no screen capture or image recognition needed. This gives:

- **Sub-millisecond latency** — faster than any human reaction
- **Perfect accuracy** — reads note data directly from the game engine
- **Zero CPU usage** — no screenshots or image processing
- **Works minimized** — doesn't need to see the screen

### Detection Flow

1. Attaches to the `iDate.exe` process
2. Finds the game object in memory via signature scanning
3. Reads lane state flags that activate when notes reach the judgment line
4. Sends keyboard inputs (arrow keys / space) using Windows scan codes

## The Interface

```
┌──────────────────────────────────────────────┐
│  🎵 iDate Revival              Ready         │
├──────────────────────────────────────────────┤
│  Window: [auto-detected ▼]                   │
│  Process: [iDate.exe (1234) ▼]  [↻ Refresh]  │
├──────────────────────────────────────────────┤
│  [ ▶ START (F12) ]  [ ■ STOP (F12) ]        │
├──────────────────────────────────────────────┤
│  FPS: 2000 | 0.1ms      Score: 0 | Combo: 0 │
├──────────────────────────────────────────────┤
│  Log:                                        │
│  [12:00:00] ✓ Attached to iDate.exe          │
│  [12:00:01] ✓ Song playing — detecting!      │
│  [12:00:01] >>> LEFT <<<                     │
│  [12:00:01] >>> UP <<<                       │
└──────────────────────────────────────────────┘
```

## Troubleshooting

### "Failed to attach"
→ The program needs **Administrator** privileges. It auto-requests admin elevation, but if blocked:
  - Right-click `run.bat` → **Run as administrator**

### "iDate not detected"
→ Launch the game **before** starting the bot, then click **↻ Refresh**

### "Game object not found"
→ The program scans for the game object when a song starts. Make sure:
  - You're in a song (not menu/lobby)
  - The game is running normally (not minimized before loading)

### Keys aren't being pressed
→ Check the log panel:
  - If you see `>>> LEFT <<<` etc. — keys ARE being sent, but the game may not be focused
  - Click the game window to give it focus
  - If no key messages appear — the game object may not be found yet; start a song

### Overlay not showing
→ The overlay is optional visual feedback. It requires the game window to be selected.
  Not seeing it doesn't affect detection — keys are still pressed.

## Files

| File | Purpose |
|------|---------|
| `install.bat` | One-time setup: creates venv, installs dependencies |
| `run.bat` | Launches the program |
| `src/gui.py` | Main GUI application |
| `src/memory/memory_detector.py` | Core detection engine (reads game memory) |
| `src/memory/process.py` | Windows process attachment (ReadProcessMemory) |
| `src/input/sendinput_driver.py` | Keyboard input via scan codes |
| `src/overlay/win32_overlay.py` | Optional transparent overlay |

## Requirements

- **Windows 10/11**
- **Python 3.10+**
- **iDate Revival** (the game)
- **Administrator privileges**
