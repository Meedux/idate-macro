# iDate Revival — Reverse Engineering Reference

> Comprehensive analysis of `iDate_dump.exe.c` (238,336-line Ghidra decompilation)

---

## Table of Contents
1. [Game Overview](#1-game-overview)
2. [Main Game Object Structure](#2-main-game-object-structure)
3. [Note / Arrow Management](#3-note--arrow-management)
4. [Timing System](#4-timing-system)
5. [Score / Combo System](#5-score--combo-system)
6. [Game State Management](#6-game-state-management)
7. [Input Handling](#7-input-handling)
8. [BMS Chart Parser](#8-bms-chart-parser)
9. [Player / Actor System](#9-player--actor-system)
10. [Networking](#10-networking)
11. [Engine / Assets](#11-engine--assets)
12. [Global Data Constants](#12-global-data-constants)

---

## 1. Game Overview

**iDate Revival** is an online rhythm/dance game (similar to O2Jam / Audition Online).

- **Chart format**: BMS (Be-Music Source) — `%s%04d.bms` / `%s%04d_8key.bms`
- **Engine**: Custom with `gb`-prefixed classes (gbMesh, gbTransformNode, gbController, gbAnimatible, etc.)
- **Input**: DirectInput8 (keyboard), XInput (gamepad), keyboard layout mapping via registry
- **Timing**: QueryPerformanceCounter / QueryPerformanceFrequency for high-precision game clock
- **Audio**: DirectSound8, OGG files ("arrow.ogg", "Game Intro.ogg", "Game Outro.ogg"), GameSound.csv
- **Config**: GameSetting.ini `[GAMEOPTION]` section
- **Game modes**: NORMALMODE, MEETINGMODE, HUNTINGMODE, ITEMMODE, PARTNER

---

## 2. Main Game Object Structure

Constructor at **FUN_00411170** (line ~18560). The object is roughly 0x8C0+ bytes.

### Measure Array (0x34 – 0x2B3)
| Offset | Type | Description |
|--------|------|-------------|
| `+0x34` to `+0x2B3` | `undefined4*[0xA0]` | Array of 160 measure data pointers. Each entry stores a pointer to a per-measure note container. Iterated with `iVar1 = 0xa0`. |

### Sub-Structures
| Offset | Type | Description |
|--------|------|-------------|
| `+0x08` | container | BPM change events container (via `FUN_00410c00`) |
| `+0x18` | container | Beat position / timing events container |
| `+0x30` | `void*` | Score display object (0x24 bytes, created by `FUN_0047d910`) |
| `+0x2B4` | struct | Initialized via `FUN_0040bf30` |
| `+0x518` | `int` | Zero-init field |
| `+0x51c` | struct | Initialized via `FUN_0046e0f0` |

### Lane State Flags (0x560)
| Offset | Type | Description |
|--------|------|-------------|
| `+0x560` | `byte[0x40]` | Array of 4 structs × 0x10 bytes each. Lane hit-state flags, cleared on each measure advance. Initialized via sorted array setup with `FUN_004551c0`. |

### Timing Fields (0x5a0 – 0x7CC)
| Offset | Type | Description |
|--------|------|-------------|
| `+0x5a0` | `LARGE_INTEGER` | Performance frequency (from `QueryPerformanceFrequency`) |
| `+0x5a8` | `LARGE_INTEGER` | Start time reference (from `QueryPerformanceCounter`) |
| `+0x5b0` | `LARGE_INTEGER` | Current time counter |
| `+0x7c0` | `float` | Raw BPM value (from `#BPM` tag in BMS file) |
| `+0x7c4` | `float` | **Beat duration** in seconds = `DAT_0054614c / BPM`. The core timing unit. |
| `+0x7c8` | `float` | Game state float |
| `+0x7cc` | `float` | **Current song elapsed time** (seconds). Set from `FUN_0040f390()` each frame. |
| `+0x7d0` | `float` | BPM loaded from song database: `(float)(int)puVar2[0x42]` |
| `+0x7dc` | `float` | Accumulated sub-beat position (incremented by delta time each frame) |

### Game State Flags / Bytes (0x7f0 – 0x870)
| Offset | Type | Description |
|--------|------|-------------|
| `+0x7f0` | `float` | Initialized to `0x43d70000` (float 430.0) |
| `+0x7f8` | `byte` | **Is playing** flag. Checked at top of main update loop. |
| `+0x7f9` | `byte` | BMS `#MAIN` section encountered flag |
| `+0x7fb` | `byte` | **Note active** flag. Set to 1 when beat completes and notes arrive. |
| `+0x7fc` | `byte` | Game state flag |
| `+0x7fe` | `byte` | **Animation started** flag. Set to 1 on first frame of play. |
| `+0x800` | `byte` | State flag |
| `+0x801` | `byte` | Cleared in `FUN_00427fd0` setup |
| `+0x802` | `byte` | Init to 1 |
| `+0x804` | `byte` | **Finished** flag |
| `+0x808` | `int` | **Game mode type**: 10=normal, 0xb, 0xc=hunt, 0xd |
| `+0x80c` | `DWORD` | Game start timestamp (from `timeGetTime()`) |
| `+0x810` | `int` | **Scene/UI ID** — used to look up display elements |
| `+0x814` | `int` | Note counter (incremented during BMS parsing for each note entry) |
| `+0x818` | `int` | Song/music ID |
| `+0x81c` | `int` | Stage ID (used with `FUN_004ce790`) |
| `+0x820` – `+0x828` | `int[3]` | Additional constructor params |
| `+0x828` | `int` | Current BMS channel/measure being parsed |

### Core Position Tracking (0x82c – 0x838)
| Offset | Type | Description |
|--------|------|-------------|
| `+0x82c` | `int` | **Current measure index** (0 to 0x9F = 159 max). THE CORE POSITION TRACKER. |
| `+0x830` | `int` | **Game sub-state** (values: 0, 0x14=20, 0x1a=26, 0x20=32) |
| `+0x834` | `int` | **Beat position within measure** (0 to 3; when 3 → measure complete) |
| `+0x838` | `int` | Sub-beat counter (0 to 7), used for visual beat indicator. Reset alongside `+0x834`. |

### Score & Combo Fields (0x83c – 0x868)
| Offset | Type | Description |
|--------|------|-------------|
| `+0x844` | `int` | Score field |
| `+0x848` | `int` | Initialized to 7 |
| `+0x84c` | `int` | **Combo count** (passed to `FUN_00470990` for display) |
| `+0x850` | `int` | Score accumulator |
| `+0x854` | `int` | **Judgment grade** (0-4; < 5 to display combo; init to 5=no display) |
| `+0x858` | `int` | Cleared in `FUN_00427fd0` |
| `+0x85c` | `int` | Animation bounce counter (capped at 3) |

### Multiplayer Fields (0x868+)
| Offset | Type | Description |
|--------|------|-------------|
| `+0x868` | `int` | **Player count** — number of players in game room |
| `+0x870` | `byte` | Reset on measure advance |
| `+0x880` | `int` | Next measure's special event type (checked: 0xb, 0xc) |
| `+0x884` | `int` | **Previous animation/state type** (switch cases: 0xc–0x11) |
| `+0x888` | `int` | **Current animation/state type** (switch cases: 0xa–0x13) |
| `+0x890` – `+0x898` | `int[3]` | Animation frame counters |
| `+0x8b4` | `int` | Comparison ID value |

---

## 3. Note / Arrow Management

### 3.1 Note Data Structure (0x2C bytes)

Allocated with `FUN_0052ee46(0x2c)` (44 bytes), initialized via `FUN_0040b160`:

| Offset | Type | Description |
|--------|------|-------------|
| `+0x00` – `+0x0C` | `void*[4]` | 4 lane note objects (one per lane/key) |
| `+0x10` | `int` | Measure index for this note |
| `+0x11` | `byte` | Active/processed flag |
| `+0x14` | `int` | **Note beat position** (measure-relative) |
| `+0x18` | `float` | **Beat duration** at time of creation (copied from `this+0x7c4`) |
| `+0x1c` | `int` | Note state (0 = not hit) |
| `+0x20` | `float` | Position fraction within measure |
| `+0x24` | `int` | Additional note property |
| `+0x28` | `int` | Note visual type / event data |
| `+0x34` | `byte` | Default note value for "empty" beats |

### 3.2 Sub-Note / Arrow Object (0x3C bytes)

Allocated with `FUN_0052ee46(0x3c)`, initialized via `FUN_0040b850`:

| Offset | Type | Description |
|--------|------|-------------|
| `+0x00` – `+0x0C` | `int[4]` | Base fields (zeroed) |
| `+0x10` | `int` | Measure index |
| `+0x14` | `int` | **Channel/arrow type** (0xb, 0xc, 0xd, 0xe, 0xf, 0x16–0x19) |
| `+0x20` | `float` | Position fraction |
| `+0x28` | `float` | Y-position hint (set based on type: 0x43d48000=425.0, 0x43cc8000=409.0, 0x43db0000=438.0) |
| `+0x1c` | `byte` | Alive flag (init 1) |
| `+0x1d` – `+0x1f` | `byte[3]` | State flags |

### 3.3 Note Container (Circular Buffer)

The note container at `this+0x334` is a circular buffer / deque:

| Member Offset | Description |
|---------------|-------------|
| `+0x04` | Backing array pointer |
| `+0x08` | Capacity |
| `+0x0C` | Front index |
| `+0x10` | Count (number of elements) |

### 3.4 Key Functions — Note Lifecycle

| Function | Address | Line | Description |
|----------|---------|------|-------------|
| **FUN_0040b160** | `0x0040b160` | 14297 | **NoteReset** — Clears a 0x28-byte note structure. |
| **FUN_0040b190** | `0x0040b190` | 14319 | **NoteInit** — Sets `+0x14`=beat pos, `+0x18`=beat duration, `+0x1c`=0. |
| **FUN_0040b140** | `0x0040b140` | 14283 | **GetLaneNote(idx)** — Returns `this + idx*4`, bounds 0–3 (4 lanes). |
| **FUN_0040b320** | `0x0040b320` | ~14420 | **InsertArrowIntoNote** — Inserts sub-note object into note container. |
| **FUN_0040b3e0** | `0x0040b3e0` | 14481 | **SetNoteEventData** — Sets `this+0x28 = param_1`. |
| **FUN_0040b430** | `0x0040b430` | 14509 | **HasNotes** — Returns `*param_1 != 0` (true if note exists). |
| **FUN_0040b490** | `0x0040b490` | 14549 | **DestroyAllLaneNotes** — Iterates 4 lanes, frees each via `FUN_0040b070`. |
| **FUN_0040b610** | `0x0040b610` | 14589 | **PositionAllLaneNotes** — Iterates 4 lanes, positions notes visually using lane offsets (`DAT_00545e0c`, `DAT_00545e10`, `DAT_00545e14`). |
| **FUN_0040b700** | `0x0040b700` | 14657 | **SendNoteResults** — Packages 4×4 lane note results into a 0x1C-byte network packet and sends via `FUN_00457180`. Also calls `FUN_0040b4c0` to mark timestamp. |
| **FUN_0040b850** | `0x0040b850` | 14808 | **ArrowInit** — Zeroes 0x3C-byte sub-note, sets `+0x10`=−1, `+0x11`=−1, `+0x1c`=1. |
| **FUN_0040b890** | `0x0040b890` | 14830 | **ArrowSetup** — Sets measure, channel type, position. Sets Y-hint based on channel type. |

### 3.5 Key Functions — Player Note Container Operations

| Function | Address | Line | Description |
|----------|---------|------|-------------|
| **FUN_004481d0** | `0x004481d0` | 55920 | **RemoveNote** — Debug: "Remove Note - start Measure: %d". Removes expired notes from `this+0x334` container where `note_end+1 < current_pos`. Updates `this+0x348` (current note pointer). |
| **FUN_004482c0** | `0x004482c0` | 55970 | **RemoveShield** — Debug: "Remove Shield - start Measure: %d". Same structure but uses `this+0x344`. |
| **FUN_004483b0** | `0x004483b0` | 56020 | **RenderNotes** — Iterates note container (up to 5 visible), reads note type, sets display properties via `FUN_004f6980` (visual size) and `FUN_004f69d0` (visual type: 0/1/2 based on value <10 or <20). |
| **FUN_00449950** | `0x00449950` | 57162 | **AddNote** — Debug: "Add Note - start Measure: %d, e...". Adds note to `this+0x334` container via `FUN_00448800`. Sets `this+0x348` if null. |
| **FUN_00449a90** | `0x00449a90` | 57230 | **AddShield** — Debug: "Add Shield - start Measure: %d, ...". Uses `this+0x344` for shield pointer. |
| **FUN_00448780** | `0x00448780` | 56202 | **ContainerPushFront** — Inserts 0x18-byte element at front of circular buffer. Copies 6 dwords from param. |
| **FUN_00448800** | `0x00448800` | 56238 | **ContainerPushBack** — Inserts 0x18-byte element at back of circular buffer. Copies 6 dwords from param. |
| **FUN_00447060** | `0x00447060` | 54840 | **ContainerGet(idx)** — Returns element at index from circular buffer with modular arithmetic. |

---

## 4. Timing System

### 4.1 High-Precision Timer

| Function | Address | Line | Description |
|----------|---------|------|-------------|
| **FUN_0040f390** | `0x0040f390` | 17323 | **GetElapsedTime** — `QueryPerformanceCounter()` → `(current - start) / frequency`. Returns `float10` (80-bit). Called every frame in main update loop. |
| **FUN_004ef050** | `0x004ef050` | ~165235 | **InitGlobalTimer** — Initializes global timer singleton with `QueryPerformanceFrequency`. Stores freq at `DAT_005c6bd0`. |
| **FUN_004ef0d0** | `0x004ef0d0` | ~165269 | **ResetTimer** — Snapshots current `QueryPerformanceCounter` as start reference. |
| **FUN_004ef110** | `0x004ef110` | ~165295 | **UpdateTimer** — Calculates total time, delta time, and frame time from performance counters. |

### 4.2 Beat Timing Constants

| Global | Address | Role |
|--------|---------|------|
| `DAT_0054614c` | `0x0054614c` | **Seconds-per-beat numerator** — `beatDuration = DAT_0054614c / BPM`. Likely 60.0 (since BPM = beats per minute → 60/BPM = seconds per beat). |
| `DAT_00545dfc` | `0x00545dfc` | **Timing window multiplier 1** — Perfect window. Used as `beatDuration * DAT_00545dfc`. |
| `DAT_00545df8` | `0x00545df8` | **Timing window multiplier 2** — Great window. Also used in bounce animation formula. |
| `DAT_00545df4` | `0x00545df4` | **Timing window multiplier 3** — Good window. |
| `DAT_005466a0` | `0x005466a0` | Timing window multiplier for beat subindex 0 |
| `DAT_0054669c` | `0x0054669c` | Timing window multiplier for beat subindex 2 |
| `DAT_00546698` | `0x00546698` | Timing window multiplier for beat subindex 4 |
| `DAT_00546694` | `0x00546694` | Timing window multiplier for beat subindex 6 |

### 4.3 Measure Advancement Logic (lines 23435–23460)

```
elapsed = FUN_0040f390()          // Get song position
this->0x7cc = elapsed             // Store current time

if ((measureIndex + 1) * beatDuration < elapsed):
    this->0x82c++                 // Advance measure
    if (this->0x82c >= 0xA0):
        this->0x82c = 0x9F        // Cap at 159
    clear this->0x560[0..0x3F]    // Reset lane state
    FUN_0040b700(measure, '\0')   // Send note results for completed measure
```

### 4.4 BPM Change Handling (lines 23513–23527)

During playback, the game checks BPM change events stored in a separate container:
```
if (changeEvent.beat * beatDuration + currentMeasure * beatDuration < elapsed):
    this->0x7c4 = DAT_0054614c / newBPM    // Update beat duration
```

### 4.5 Game Start Delay (line ~23410)

Uses `timeGetTime()` to check when to start actual gameplay:
```
if (this->0x80c - DAT_005c1674 > threshold):
    // Begin playing
    this->0x7fe = 1    // Mark animation started
    QueryPerformanceCounter → this->0x5a8   // Set start time
```

---

## 5. Score / Combo System

### 5.1 Judgment Function

| Function | Address | Line | Description |
|----------|---------|------|-------------|
| **FUN_0040b1d0** | `0x0040b1d0` | 14339 | **JudgeNoteTiming** — Compares `param_2` (current song position) against note timing (`this+0x14` + `this+0x18`). Uses three threshold globals (`DAT_00545dfc`, `DAT_00545df8`, `DAT_00545df4`) to determine Perfect/Great/Good/Miss. Returns result from `this+0x00`, `+0x04`, `+0x08`, or `+0x0C` (4 judgment grades). |

### 5.2 Beat Animation & Visual Judgment

| Function | Address | Line | Description |
|----------|---------|------|-------------|
| **FUN_004281c0** | `0x004281c0` | 32168 | **UpdateBeatAnimation** — Calculates 8 timing windows from `beatDuration * multiplier`. Advances `this+0x834` (beat pos) when timing threshold crossed. Updates `this+0x838` (sub-beat 0–7) for visual beat indicator. Updates UI elements 0x265 and 0x263. |

### 5.3 Score Display Functions

| Function | Address | Line | Description |
|----------|---------|------|-------------|
| **FUN_0040f490** | `0x0040f490` | 17390 | **DisplayCombo** — If `this+0x854 < 5`, renders combo using UI element 0x2E6. Calls `FUN_00470990(this+0x84c, this+0x854)`. |
| **FUN_0040f3e0** | `0x0040f3e0` | 17345 | **GetMaxPlayerScore** — Iterates `this+0x868` players, reads each `player+0xC8`, returns max. |
| **FUN_0040f430** | `0x0040f430` | 17368 | **SyncAllPlayerScores** — Gets max score, iterates all players, calls `FUN_004493a0(maxScore+1)` and `FUN_00440db0(player, 0)`. |
| **FUN_004493a0** | `0x004493a0` | 56812 | **SetPlayerScore** — If solo (1 player), just resets display. Otherwise sets `this+0xC8 = param_1` (clamped ≥ 0), triggers display refresh via `FUN_00448eb0`. |

### 5.4 Score Fields Summary

| Field | Offset | Init | Role |
|-------|--------|------|------|
| Combo count | `+0x84c` | 0 | Current combo counter, passed to combo display |
| Score | `+0x850` | 0 | Score accumulator |
| Judgment grade | `+0x854` | 5 | Current judgment (0–4 = visible; 5 = hidden) |
| Max grade | `+0x848` | 7 | Max timing sub-index |
| Player score | player`+0xC8` | 0 | Per-player score (200 decimal offset) |

---

## 6. Game State Management

### 6.1 Core State Machine

| Function | Address | Line | Description |
|----------|---------|------|-------------|
| **FUN_00411170** | `0x00411170` | ~18560 | **Constructor** — Allocates and zeros the game object. Initializes measure array (160 entries), sets mode from `param_1`, stores song/stage IDs. |
| **FUN_0040eb40** | `0x0040eb40` | 16920 | **InitGame / ResetScore** — Creates score display (0x24 bytes). Sets `0x84c=0`, `0x850=0`, `0x854=5`, `0x848=7`. Loads BPM from song DB at `puVar2[0x42]` → `this+0x7d0`. Clears lane states at `0x560`. |
| **FUN_0040eca0** | `0x0040eca0` | ~16982 | **LoadStage** — Formats stage path as `stage%02d_`, loads `normal_pos.gb3` or `huntdum_pos.gb3`, sets up camera ("stagecam", "waitcam"). |
| **FUN_00418b50** | `0x00418b50` | 23370 | **MainGameUpdateLoop** — The main playing-state update. Checks `0x7f8` flag, calls actor updates, checks start delay, initializes on first frame, then calls `FUN_00427f20` and timing/measure advancement. |
| **FUN_004188e0** | `0x004188e0` | 23263 | **GameUpdateWithBounce** — Called within update loop. Manages beat-complete transitions (`0x834==3`), note activation (`0x7fb=1`), and note bounce animation using sin/cos. |
| **FUN_00412670** | `0x00412670` | 19505 | **AnimationStateMachine** — Giant switch on `this+0x888` (values 0xa–0x13). Cross-checks `this+0x884` (previous) and `0x834` (beat). Controls note activation based on state transitions. |
| **FUN_004128a0** | `0x004128a0` | 19634 | **PrepareNextMeasure** — Pre-reads `this+0x82c + 2` (two measures ahead). Sets `this+0x880` to upcoming event type. Contains animation frame stepping for event types 0xb, 0xc. |
| **FUN_00410ef0** | `0x00410ef0` | 18703 | **GameEnd / Cleanup** — Clears `0x81c`, `0x804`. Frees all 160 measure data entries. Cleans up scene objects. |

### 6.2 Measure Data Access

| Function | Address | Line | Description |
|----------|---------|------|-------------|
| **FUN_0040eb00** | `0x0040eb00` | 16891 | **GetMeasureData(idx)** — Returns `this + idx*4 + 0x34`. Bounds: 0 ≤ idx < 0xA0. |
| **FUN_0040eb20** | `0x0040eb20` | 16905 | **SetMeasureData(idx, data)** — Sets `this + idx*4 + 0x34 = data`. |
| **FUN_0040ec50** | `0x0040ec50` | ~16940 | **MeasureTransitionCleanup** — Frees resources during measure transition. |

### 6.3 Game Update Sub-calls

| Function | Address | Line | Description |
|----------|---------|------|-------------|
| **FUN_004435c0** | `0x004435c0` | — | Get actor manager |
| **FUN_00443700** | `0x00443700` | — | Update actors |
| **FUN_0044af30** | `0x0044af30` | — | Pre-frame update |
| **FUN_0044b000** | `0x0044b000` | — | Post-frame update |
| **FUN_004524c0** | `0x004524c0` | — | Secondary system update |
| **FUN_00453390** | `0x00453390` | — | Tertiary system update |
| **FUN_00427f20** | `0x00427f20` | 31928 | **UpdateCursorHitDetection** — Checks `DAT_005c0c10` flag, gets cursor position, converts screen→client coords, tests element hit at 0x203. |
| **FUN_00427fd0** | `0x00427fd0` | 31976 | **SetupGameUI** — Hides/shows UI elements (0x262, 0x26c–0x27f, etc.), sets up player name displays, initializes game score tracking. |

### 6.4 Count Active Notes

| Function | Address | Line | Description |
|----------|---------|------|-------------|
| **FUN_0040f7c0** | `0x0040f7c0` | 17603 | **CountActiveNotes** — Iterates measure array (0 to 0xA0) step 5, checks 5 entries per step, counts non-zero with gap logic. |

---

## 7. Input Handling

### 7.1 DirectInput8 Setup

| Function | Address | Line | Description |
|----------|---------|------|-------------|
| **FUN_00402880** | `0x00402880` | ~7578 | **InitKeyboard** — Calls `DirectInput8Create()`, creates keyboard device with GUID `DAT_00554a2c` (GUID_SysKeyboard). Sets data format (0x16 = `c_dfDIKeyboard`), cooperation level, acquires. Stores device at `param_1+0x234`. |
| **FUN_00402a90** | `0x00402a90` | ~7729 | **InitMouse** — Same pattern with GUID `DAT_00554a3c` (GUID_SysMouse). Device format 6 (`c_dfDIMouse`). Stores at `param_1+0x230`. |
| **FUN_00402b40** | `0x00402b40` | ~7824 | **InitAllInput** — Calls `FUN_00402880` (keyboard) then `FUN_00402a90` (mouse). |

### 7.2 Input Polling

| Function | Address | Line | Description |
|----------|---------|------|-------------|
| **FUN_00402990** | `0x00402990` | ~7651 | **PollKeyboard** — Copies current keystate (`param_1+0x04`, 0x100 bytes) to previous (`param_1+0x104`), zeroes current, calls `GetDeviceState(0x100, ...)`. Re-acquires on `DIERR_INPUTLOST` (`-0x7ff8ffe2`). |
| **FUN_00402900** | `0x00402900` | ~7611 | **PollMouse** — Copies current mouse state (`param_1+0x204`) to previous (`param_1+0x218`), zeroes current, calls `GetDeviceState(0x14, ...)`. |
| **FUN_00402a60** | `0x00402a60` | ~7789 | **PollAllInput** — Calls `FUN_00402990` (keyboard) then `FUN_00402900` (mouse). |

### 7.3 Input Device Structure

| Offset | Type | Description |
|--------|------|-------------|
| `+0x04` | `byte[0x100]` | Current keyboard state (256 bytes, one per DIK scancode) |
| `+0x104` | `byte[0x100]` | Previous keyboard state |
| `+0x204` | `DIMOUSESTATE` (0x14 bytes) | Current mouse state (lX, lY, lZ, rgbButtons[4]) |
| `+0x218` | `DIMOUSESTATE` (0x14 bytes) | Previous mouse state |
| `+0x22c` | `IDirectInput8*` | DirectInput8 interface pointer |
| `+0x230` | `IDirectInputDevice8*` | Mouse device |
| `+0x234` | `IDirectInputDevice8*` | Keyboard device |

### 7.4 XInput Support

| Function | Address | Line | Description |
|----------|---------|------|-------------|
| **FUN_004eef80** | `0x004eef80` | ~165167 | **XInputEnable** — Dynamically loads `xinput1_3.dll`, resolves `XInputEnable` via `GetProcAddress`, calls it with enable/disable flag. |

### 7.5 Key Binding Configuration

Key bindings are stored in a global config object. Offsets from config object base:

| Config Offset | Lane | Notes |
|---------------|------|-------|
| `+0x13c8` | Lane 0 (key 0) | Used by `FUN_00445720` for animation |
| `+0x13cc` | Lane 0 (key 1) | |
| `+0x13d0` | Lane 0 (key 2) | |
| `+0x13d4` | Lane 0 (key 3) | |
| `+0x13d8` | Lane 0 (key 4) | |
| `+0x13dc` | Lane 0 (key 5) | |
| `+0x13e0` | Lane 0 | Used by `FUN_00445e50` for arrow visual |
| `+0x13e4` | Lane 1 | |
| `+0x13e8` | Lane 2 | |
| `+0x13ec` | Lane 3 | |
| `+0x13f4` | Lane 5 (special) | Shield/item key |
| `+0x13f8` | Lane 6 | |
| `+0x13fc` | Lane 7 | |
| `+0x1400` | Lane 8 | |
| `+0x1404` | Lane 9 | |

### 7.6 Key Binding Setup Functions

| Function | Address | Line | Description |
|----------|---------|------|-------------|
| **FUN_00445720** | `0x00445720` | 53503 | **SetLaneAnimation** — Removes old transform node, adds via `FUN_0044d850`. Params: (player, mode, lane_index, character_id, key_config). Special handling for lane 1 (float 3.0) and lane 2 (shield setup). |
| **FUN_00445e50** | `0x00445e50` | 53978 | **SetLaneArrowKey** — Removes old transform node, adds via `FUN_0044e2c0` if `param_3 > 0`. Lane 7 → calls `FUN_004457f0` (color setup). Lane 9 → calls `FUN_00445d10` (special visual). |
| **FUN_00446e60** | `0x00446e60` | 54750 | **UpdatePlayerVisual** — If channel==5, creates/updates item visual. Calls `FUN_004f6980` for size (based on `2c4-1 + 2c8/12*2`) and `FUN_004f69d0` for type (`2c8 % 12`). |
| **FUN_00486a10** | `0x00486a10` | — | **RegisterKeyBinding** — Maps lane index to key action |

### 7.7 Keyboard Mapping

- **Registry key**: `HKLM\...\Keyboard_Mapping` (line 172065) — Custom keyboard layout
- **GetKeyboardLayout(0)** called at line 167581 for locale-aware input

---

## 8. BMS Chart Parser

### Location: Lines ~18850–19160 (within Constructor/Init function)

### 8.1 File Format
- Files: `%s%04d.bms` (4-key), `%s%04d_8key.bms` (8-key)
- Standard BMS tags parsed: `#BPM`, `#BPMxx`, `#WAVxx`, `#MAIN` section

### 8.2 Parsing Logic

1. **`#BPM` tag** (line ~19130): Extracts BPM via `atof()`, stores:
   - `this+0x7c0 = BPM`
   - `this+0x7c4 = DAT_0054614c / BPM` (beat duration)
   - `local_890[0] = BPM` (first BPM slot)

2. **`#BPMxx` tag** (line ~19105): Indexed BPM changes, stores `local_890[index] = atof(value)`.

3. **`#WAVxx` tag** (line ~19070): Loads WAV file references. Checks for "ogg" / "wav" extension, converts wide→multi-byte path.

4. **`#MAIN` data section** (line ~18900): Core note data parser.
   - Format: 3-digit measure number + 2-digit channel + multi-char hex pattern
   - Each 2-char hex value = one note event
   - Channel dispatch (switch):

   | Channel | Action |
   |---------|--------|
   | 1 | BPM change event → `FUN_00410c00(this+0x08, {bpm, fraction, measure})` |
   | 3 | Beat position event → `FUN_00410c00(this+0x18, {value, fraction, measure})` |
   | 4 | Note event → `FUN_0040b3e0(note, value)` |
   | 8 | Indexed BPM reference → Uses `local_890[(int)value]` |
   | 0xb–0xf, 0x16–0x19 | **Arrow/visual events** → Creates 0x3C sub-note via `FUN_0040b850`, initializes with `FUN_0040b890`, sets visual via `FUN_0040bb90`, inserts into note via `FUN_0040b320` |
   | 0x12, 0x13 | Ignored (empty break) |

5. **Post-parse** (line ~19145): Iterates all 160 measures, calls `FUN_0040b430` (check has notes) → `FUN_0040b610` (position all notes visually). Stores completed note data via `FUN_0040eb20`.

---

## 9. Player / Actor System

### 9.1 Player Types

| String | Address | Role |
|--------|---------|------|
| `CActorMyPlayer` | line 2919 | Local player class identifier |
| `CActorPlayer` | line 2920 | Remote player class identifier |

### 9.2 Player Object Layout (Key Fields)

| Offset | Type | Description |
|--------|------|-------------|
| `+0x10` | `int*` | Scene node / transform node |
| `+0x68` | `int` | Current animation ID (set to 0x42) |
| `+0xB0` | `int` | Player unique ID |
| `+0xC4` | `byte` | Character model ID |
| `+0xC8` | `int` | Player score (200 decimal) |
| `+0xCC` | `int` | Max score limit |
| `+0xF4` | `short*` | Player name string |
| `+0x154` | `char[...]` | Character path string (used in `%sAni/` format for animation loading) |
| `+0x200` | `int` (0xC8) | Score field (decimal 200) |
| `+0x25c` | `int` | Display state |
| `+0x268` | `int*` | Visual effect object |
| `+0x270` | `int*` | Arrow visual object |
| `+0x274` | `int*[4]` | Lane visual objects (up to 4) |
| `+0x288` | `int` | Counter |
| `+0x294` | `int` | Counter |
| `+0x298` | `int*` | UI element ref |
| `+0x29c` | `int*` | Name plate object 1 |
| `+0x2a0` | `int*` | Name plate object 2 |
| `+0x2a4` | `int*` | Item/shield visual |
| `+0x2ac` | struct | Animation container 1 |
| `+0x2b4` | struct | Animation container 2 (arrow keys) |
| `+0x2c0` | `int` | Current animation channel |
| `+0x2c4` | `int` | Animation mode |
| `+0x2c8` | `int` | Animation sub-index |
| `+0x2cc` | `float` | Y-offset for visuals |
| `+0x2e0` | `float` | Rotation angle |
| `+0x2f2` | `byte` | Effect active flag |
| `+0x2f3` | `byte` | Name plate hidden flag |
| `+0x2f5` | `byte` | State flag |
| `+0x2f6` | `byte` | Note animation active flag |
| `+0x2f7` | `byte` | Score sync flag |
| `+0x2f9` | `byte` | Arrow direction |
| `+0x308` | `int` | Stored animation param |
| `+0x30c` | `int` | Lane visual count (max 4) |
| `+0x32c` | `int` | Counter |
| `+0x334` | `container` | Note container (circular buffer) |
| `+0x338` | `int*` | Note container end sentinel |
| `+0x33c` | `int` | Note count |
| `+0x344` | `void*` | Current shield pointer |
| `+0x348` | `void*` | Current note pointer |
| `+0x360` | struct | Dance/animation controller sub-object |

### 9.3 Player Functions

| Function | Address | Line | Description |
|----------|---------|------|-------------|
| **FUN_00449d70** | `0x00449d70` | 57310 | **PlayerNoteInit** — Loads character animation from `%sAni/` path using character ID at `+0xC4`. Calls `FUN_00448b50` and `FUN_00448cd0` for note/chart loading. |
| **FUN_00449400** | `0x00449400` | 56860 | **PlayerStartNoteAnimation** — Sets `+0x2f6=1`, `+0x308=param`, loads animation data, sets animation ID to 0x42. |
| **FUN_00449520** | `0x00449520` | 56955 | **PlayerStopNoteAnimation** — Clears `+0x2f6`, reloads animation data. |
| **FUN_004489c0** | `0x004489c0` | 56350 | **PlayerResetNotes** — Clears `+0x288`, `+0x294`, `+0x32c`, `+0x290`. Destroys all note container entries. Rebuilds with 20 pre-allocated empty notes (0x14-byte each). |
| **FUN_004458d0** | `0x004458d0` | 53625 | **SyncPlayerScore** — If `+0x2f7` set, reads other player's score at `+0xC8`, updates own `+0xC8` minus 1. |

---

## 10. Networking

### 10.1 Game Commands

| Command | Lines Referenced | Description |
|---------|------------------|-------------|
| `CMD_GAMEROOM_SET_MUSIC` | 38653 | Set music for room |
| `CMD_GAMEROOM_SET_STAGE` | 38410, 38714, 39057 | Set stage |
| `CMD_GAMEROOM_MOVE` | 38795, 47932, 53218, 72061 | Player movement/actions |
| `CMD_GAME_PARTNER` | 22385–22646, 28199–28494 | Partner mode synchronization |
| `CMD_GAME_HUNT_ACCEPT` | 22802 | Hunting mode acceptance |
| `CMD_GAME_FINAL_PARTNER` | 40892–41087 | End-game partner selection |

### 10.2 Network Packet (Note Results)

In `FUN_0040b700` (line 14657), note results are packaged:
```
local_20 = { 0x1C, 0x00, 0x24, 0x09, 0x00, 0x00, 0x00, 0x00 }  // 8-byte header
local_14 through local_8 = 4×4 lane results (16 bytes)
// Total: 0x1C (28) bytes sent via FUN_00457180
```

### 10.3 Server State

| Access Pattern | Description |
|----------------|-------------|
| `FUN_00483030() + 0x477` | Server/character state byte, used for character ID lookups |
| `FUN_00483030() + 0xc10` | Cursor hit detection active flag |
| `FUN_00483030() + 0x104` | Local player object pointer |
| `FUN_00483030() + 0x108` | Secondary player object pointer |
| `FUN_00483030() + 0x440` | Local player unique ID |

---

## 11. Engine / Assets

### 11.1 gb-Engine Classes

| Class | Usage |
|-------|-------|
| `gbMesh` / `gbMeshBuffer` | 3D mesh rendering |
| `gbController` | Animation / game controller |
| `gbNode` / `gbTransformNode` | Scene graph nodes with transforms |
| `gbObject3d` | 3D objects in scene |
| `gbBase` | Base class |
| `gbAnimatible` | Animatable objects (override `Update()`) |
| `gbTexImageKey` | Texture references |
| `gbMatrix4` / `gbMatrix3` | Math matrices |
| `gbBox3` / `gbVolume3d` | Collision volumes |
| `gbStdMaterial` | Standard material |
| `gbDxStateBlock` | DirectX state management |
| `gbParam` / `gbHandle` | Generic parameters |
| `gbSkeletal` | Skeletal animation (with `GetItemTM` for bone transforms) |
| `gbTransform` | Transform (position/rotation/scale) |

### 11.2 Data Files

| File | Line | Purpose |
|------|------|---------|
| `PlayCamera.csv` | 5137 | Camera positions/animations for gameplay |
| `Stage.csv` | 5151 | Stage definitions |
| `GameColor.csv` | 5169 | Color palettes for game elements |
| `GameSound.csv` | — | Sound effect definitions |
| `GameSetting.ini` | — | User settings (PARTICLE, POSTGLOW, etc.) |
| `gameresult.xml` | 5016 | Result screen layout |
| `normal_pos.gb3` | ~17035 | Normal mode position data |
| `huntdum_pos.gb3` | ~17035 | Hunt mode position data |

### 11.3 Audio Files

| File | Context |
|------|---------|
| `arrow.ogg` | Arrow/note hit sound |
| `Game Intro.ogg` | Game start jingle |
| `Game Outro.ogg` | Game end jingle |
| `bgm_CoupleResult.ogg` | Couple mode result BGM |

---

## 12. Global Data Constants

### 12.1 Timing Constants (Critical for Memory Reading)

| Address | Name Guess | Role |
|---------|------------|------|
| `0x0054614c` | `SECONDS_PER_BEAT_FACTOR` | `60.0f` — converts BPM to seconds-per-beat |
| `0x00545dfc` | `PERFECT_WINDOW` | Timing window multiplier for Perfect judgment |
| `0x00545df8` | `GREAT_WINDOW` | Timing window multiplier for Great judgment |
| `0x00545df4` | `GOOD_WINDOW` | Timing window multiplier for Good judgment |
| `0x005466a0` | `BEAT_SUB_WINDOW_0` | Sub-beat timing for visual indicator |
| `0x0054669c` | `BEAT_SUB_WINDOW_2` | Sub-beat timing |
| `0x00546698` | `BEAT_SUB_WINDOW_4` | Sub-beat timing |
| `0x00546694` | `BEAT_SUB_WINDOW_6` | Sub-beat timing |
| `0x00546378` | `BOUNCE_ANGLE_MAX` | Used in sin/cos bounce: `(time/window)*MAX + OFFSET` |
| `0x00546374` | `BOUNCE_ANGLE_OFFSET` | Offset for bounce animation |
| `0x00545de0` | `FLOAT_ZERO` | Likely 0.0f — used as comparison baseline |
| `0x005c1674` | `GAME_START_DELAY` | Global timestamp offset for game start |

### 12.2 Visual Constants

| Address | Role |
|---------|------|
| `0x00545e0c` | Base X position for note rendering |
| `0x00545e10` | Note X spacing (per column) |
| `0x00545e14` | Note Y spacing (per lane) |
| `0x00545654` | Additional Y offset for note type 0xf |
| `0x00545e08` | Alternate Y offset |
| `0x00547638` | String: "Dummy_ID" — skeletal bone name for positioning |
| `0x00546f78` | Y-axis offset for name plate positioning |

### 12.3 Arrow Y-Position Hints (from FUN_0040b890)

| Channel Type | Float Value (hex) | Float Value (dec) | Meaning |
|-------------|-------------------|-------------------|---------|
| 0xb, 0xd, 0x17, 0x18 | `0x43d48000` | 425.0 | Standard arrow Y |
| 0xc, 0x19 | `0x43cc8000` | 409.0 | Alternate arrow Y |
| 0xe, 0xf, 0x16 | `0x43db0000` | 438.0 | High arrow Y |

---

## Summary: Most Important Offsets for Memory Reading

For the `idate` memory reader project, these are the most critical game object fields:

| Priority | Offset | Type | Name | Notes |
|----------|--------|------|------|-------|
| ★★★ | `+0x82c` | `int` | Current measure index | 0–159, the core song position |
| ★★★ | `+0x834` | `int` | Beat within measure | 0–3, triggers note activation at 3 |
| ★★★ | `+0x7cc` | `float` | Elapsed song time | Seconds since song start |
| ★★★ | `+0x7c4` | `float` | Beat duration | Seconds per beat = 60/BPM |
| ★★★ | `+0x7f8` | `byte` | Is playing | 1 when gameplay active |
| ★★★ | `+0x7fb` | `byte` | Notes active | 1 when notes are arriving |
| ★★ | `+0x888` | `int` | Animation state | 0xa–0x13, determines visual state |
| ★★ | `+0x84c` | `int` | Combo count | Current combo |
| ★★ | `+0x854` | `int` | Judgment grade | 0–4 visible, 5 hidden |
| ★★ | `+0x808` | `int` | Game mode | 10=normal, 0xc=hunt |
| ★★ | `+0x868` | `int` | Player count | Number of players |
| ★★ | `+0x560` | `byte[0x40]` | Lane hit states | 4 lanes × 0x10 bytes |
| ★ | `+0x830` | `int` | Game sub-state | Phase within game |
| ★ | `+0x838` | `int` | Sub-beat index | 0–7 visual beat indicator |
| ★ | `+0x814` | `int` | Note counter | Total notes parsed |
| ★ | `+0x804` | `byte` | Game finished | End-of-song flag |

### Input Device Key State (for reading pressed keys)

| Offset | Size | Description |
|--------|------|-------------|
| `InputDevice+0x04` | 256 bytes | Current keyboard state (DIK scancodes) |
| `InputDevice+0x104` | 256 bytes | Previous frame keyboard state |
