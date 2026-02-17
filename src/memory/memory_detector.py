"""
Memory-based note detector for iDate Revival.

Locates the game's live gameplay object in memory via vtable-signature
scanning, then polls timing / measure / lane-state fields every frame
to decide which keys to press.

All offsets derived from Ghidra decompilation of iDate_dump.exe.c.
"""
from __future__ import annotations

import struct
import time
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

log = logging.getLogger(__name__)

# ── vtable signature (first dword of the gameplay object) ────────────
# Constructor FUN_00411170:  *(void**)this = &PTR_LAB_005460b8;
VTABLE_PTR = 0x005460B8

# ── static singleton that may reference the object ──────────────────
GAME_ROOT_STATIC = 0x005C5D68  # DAT_005c5d68 — returned by FUN_00483030
IMAGE_BASE       = 0x00400000  # iDate.exe preferred base (32-bit, no ASLR)

# ── gameplay-object field offsets ────────────────────────────────────
OFF = {
    "vtable":        0x000,
    "measures":      0x034,   # ptr[160] measure data pointers
    "lanes":         0x560,   # 4 × 0x10 lane hit-state blocks
    "bpm":           0x7C0,   # float
    "beat_dur":      0x7C4,   # float  60 / bpm
    "elapsed":       0x7CC,   # float  song position (s)
    "beat_accum":    0x7DC,   # float  sub-beat accumulator
    "is_playing":    0x7F8,   # u8
    "notes_active":  0x7FB,   # u8
    "initialized":   0x7FE,   # u8
    "finished":      0x804,   # u8
    "mode":          0x808,   # i32
    "measure":       0x82C,   # i32  0-159
    "substate":      0x830,   # i32
    "beat":          0x834,   # i32  0-3
    "sub_beat":      0x838,   # i32  0-7
    "combo":         0x84C,   # i32
    "score":         0x850,   # i32
    "judgment":      0x854,   # i32
    "hp":            0x858,   # i32
    "bounce":        0x85C,   # i32
    "players":       0x868,   # i32
    "anim":          0x888,   # i32  animation state 0xa-0x13
}

LANE_STRIDE   = 0x10
NUM_LANES     = 4
MEASURE_COUNT = 160

# sub-note offsets inside 0x3C-byte structure
SN_CHANNEL  = 0x14   # i32  arrow channel
SN_HIT      = 0x1C   # u8   alive/hit flag
SN_FRACTION = 0x20   # f32  position within measure (0..1)

CHANNEL_KEY: Dict[int, str] = {
    0x0B: "left",  0x0C: "down",  0x0D: "up",   0x0E: "right", 0x0F: "space",
    0x16: "left",  0x17: "down",  0x18: "up",    0x19: "right",
}
LANE_KEY = {0: "left", 1: "down", 2: "up", 3: "right"}

# overlay helpers
LANE_X = {"left": 0.20, "down": 0.40, "up": 0.60, "right": 0.80, "space": 0.50}
OVL_W, OVL_H, HIT_Y = 800, 600, 510


# ═══════════════════ data classes ═══════════════════════════════════
@dataclass
class MemoryIcon:
    icon_type: str; x: int; y: int; confidence: float = 1.0

@dataclass
class GameState:
    is_playing: bool = False
    bpm: float = 0.0
    beat_duration: float = 0.0
    elapsed_time: float = 0.0
    measure: int = 0
    beat: int = 0
    sub_beat: int = 0
    notes_active: bool = False
    song_finished: bool = False
    combo: int = 0
    score: int = 0
    judgment: int = 0
    hp: int = 0
    anim_state: int = 0
    active_notes: int = 0          # count of notes currently visible
    lane_active: list = field(default_factory=lambda: [False]*4)

@dataclass
class DetectionResult:
    keys_to_press: List[str]       = field(default_factory=list)
    icons: List[MemoryIcon]        = field(default_factory=list)
    icons_in_hit_zone: List[str]   = field(default_factory=list)
    game_state: Optional[GameState]= None
    inference_time_ms: float       = 0.0
    timestamp: float               = 0.0


# ═══════════════════ detector ═══════════════════════════════════════
class MemoryDetector:
    """Self-contained detector with built-in process attach + cooldown."""

    # ---------------------------------------------------------------- init
    def __init__(self, cooldown_ms: int = 30, perfect_window_ms: int = 50):
        from .process import ProcessAttacher
        self._pa = ProcessAttacher()
        self._cooldown = cooldown_ms / 1000.0
        self._perfect_ms = perfect_window_ms
        self._game_base: Optional[int] = None
        self._module_base: int = IMAGE_BASE
        self._scan_done = False
        self._log_cb: Optional[Callable[[str], None]] = None

        # per-frame state
        self._last_measure = -1
        self._last_beat = -1
        self._pressed: Set[str] = set()             # keys pressed this beat
        self._last_press: Dict[str, float] = {}     # key → last press time
        self._note_cache: Dict[int, List[dict]] = {}
        self._frame = 0

    # ---------------------------------------------------------------- public API
    @property
    def process_name(self) -> str:
        return getattr(self._pa, "process_name", "(none)")

    def set_log_callback(self, cb: Callable[[str], None]):
        self._log_cb = cb

    def _log(self, msg: str):
        if self._log_cb:
            try: self._log_cb(msg)
            except: pass

    def attach_to_process(self, pid: int) -> bool:
        ok = self._pa.attach(pid)
        if ok:
            self._module_base = self._pa.get_base_address("iDate.exe") or IMAGE_BASE
            self._log(f"Attached iDate.exe PID={pid} base=0x{self._module_base:08X}")
        return ok

    def detach(self):
        self._pa.detach()
        self._game_base = None
        self._scan_done = False

    # ---------------------------------------------------------------- detect
    def detect(self, frame=None) -> DetectionResult:
        t0 = time.perf_counter()
        result = DetectionResult(timestamp=t0)

        if not self._pa.is_attached:
            return result

        # find game object once
        if self._game_base is None:
            if not self._scan_done:
                self._game_base = self._find_game_object()
                self._scan_done = True
                if self._game_base is None:
                    self._log("Game object NOT found — is a song loaded?")
            result.inference_time_ms = (time.perf_counter() - t0) * 1000
            return result

        # periodic revalidation
        self._frame += 1
        if self._frame % 300 == 0:
            if not self._validate(self._game_base):
                self._log("Game object invalidated — rescanning next frame")
                self._game_base = None
                self._scan_done = False
                self._note_cache.clear()
                return result

        # read state
        gs = self._read_state()
        if gs is None:
            result.inference_time_ms = (time.perf_counter() - t0) * 1000
            return result
        result.game_state = gs

        if not gs.is_playing:
            self._pressed.clear()
            self._last_measure = -1
            self._last_beat = -1
            self._scan_done = False        # re-scan when song starts
            result.inference_time_ms = (time.perf_counter() - t0) * 1000
            return result

        # ---- measure / beat change tracking ----
        meas_changed = gs.measure != self._last_measure
        beat_changed = gs.beat != self._last_beat
        if meas_changed or beat_changed:
            self._pressed.clear()
            if meas_changed:
                self._note_cache.pop(self._last_measure, None)
        self._last_measure = gs.measure
        self._last_beat = gs.beat

        # ---- gather keys to press ----
        keys: Set[str] = set()
        icons: List[MemoryIcon] = []

        # 1) measure data sub-notes (current + next)
        for offset in (0, 1):
            mi = gs.measure + offset
            if mi < 0 or mi >= MEASURE_COUNT:
                continue
            notes = self._get_measure_notes(mi)
            for n in notes:
                k = n["key"]
                frac = n["fraction"]
                # icon position
                ix = int(LANE_X.get(k, 0.5) * OVL_W)
                # y: approximate note scroll position
                if gs.beat_duration > 0:
                    note_time = mi * gs.beat_duration + frac * gs.beat_duration
                    delta = note_time - gs.elapsed_time
                    iy = HIT_Y - int(delta * 500)
                else:
                    iy = HIT_Y
                icons.append(MemoryIcon(k, ix, max(0, min(iy, OVL_H)), 1.0))

                # should press?
                if offset == 0 and k not in self._pressed:
                    if gs.notes_active or gs.beat >= 2:
                        keys.add(k)

        # 2) lane-state flags (catches notes even if measure parse missed)
        for li, active in enumerate(gs.lane_active):
            if active:
                k = LANE_KEY.get(li)
                if k and k not in self._pressed:
                    keys.add(k)
                    icons.append(MemoryIcon(k, int(LANE_X.get(k, .5)*OVL_W), HIT_Y, 1.0))

        # 3) if notes_active but still nothing, read next measure
        if gs.notes_active and not keys:
            mi2 = gs.measure + 1
            if 0 <= mi2 < MEASURE_COUNT:
                for n in self._get_measure_notes(mi2):
                    k = n["key"]
                    if k not in self._pressed:
                        keys.add(k)

        gs.active_notes = len(icons)
        self._pressed.update(keys)

        result.keys_to_press = sorted(keys)
        result.icons = icons
        result.icons_in_hit_zone = sorted(keys)
        result.inference_time_ms = (time.perf_counter() - t0) * 1000
        return result

    # ================================================================
    #  GAME OBJECT DISCOVERY
    # ================================================================
    def _find_game_object(self) -> Optional[int]:
        """Locate the gameplay object by vtable signature."""
        if not self._pa.is_attached:
            return None

        # ---- strategy 1: static chain via game root singleton ----
        obj = self._try_static_root()
        if obj:
            bpm = self._read_f32(obj + OFF["bpm"])
            self._log(f"Game obj via static root 0x{obj:08X}  BPM={bpm:.0f}")
            return obj

        # ---- strategy 2: scan readable memory for vtable ----
        self._log("Static pointers missed — vtable-scanning heap …")
        target_bytes = struct.pack("<I", VTABLE_PTR)
        try:
            regions = self._pa.get_readable_regions()
        except Exception:
            return None

        for base, size in regions:
            if base < 0x00600000 or base > 0x7FFF0000:
                continue
            if size > 0x08000000:   # skip >128 MB regions
                continue
            chunk_sz = min(size, 0x400000)  # read 4 MB at a time
            offset = 0
            while offset < size:
                data = self._pa.read(base + offset, min(chunk_sz, size - offset))
                if data is None:
                    break
                pos = 0
                while True:
                    idx = data.find(target_bytes, pos)
                    if idx == -1:
                        break
                    addr = base + offset + idx
                    if self._validate(addr):
                        bpm = self._read_f32(addr + OFF["bpm"])
                        self._log(f"Game obj via scan 0x{addr:08X}  BPM={bpm:.0f}")
                        return addr
                    pos = idx + 4
                offset += chunk_sz
        return None

    def _try_static_root(self) -> Optional[int]:
        """Try DAT_005c5d68 → root → probe multiple offsets for the stage object."""
        root_ptr = self._pa.read_uint32(GAME_ROOT_STATIC)
        if not root_ptr or root_ptr < 0x10000:
            return None
        # root is a big struct (0x1B5C). The gameplay object may be pointed
        # to from various offsets (it is NOT the root itself).
        for off1 in (0x104, 0x108, 0x100, 0xFC, 0x110, 0x10C):
            ptr = self._pa.read_uint32(root_ptr + off1)
            if not ptr or ptr < 0x10000:
                continue
            # direct hit?
            if self._pa.read_uint32(ptr) == VTABLE_PTR and self._validate(ptr):
                return ptr
            # one more level
            for off2 in range(0, 0x30, 4):
                inner = self._pa.read_uint32(ptr + off2)
                if inner and inner > 0x10000:
                    if self._pa.read_uint32(inner) == VTABLE_PTR and self._validate(inner):
                        return inner
        return None

    # ================================================================
    #  VALIDATION
    # ================================================================
    def _validate(self, addr: int) -> bool:
        blob = self._pa.read(addr, 0x8C0)
        if blob is None or len(blob) < 0x8C0:
            return False
        def _u32(o): return struct.unpack_from("<I", blob, o)[0]
        def _i32(o): return struct.unpack_from("<i", blob, o)[0]
        def _f32(o): return struct.unpack_from("<f", blob, o)[0]
        def _u8(o):  return blob[o]

        if _u32(OFF["vtable"]) != VTABLE_PTR:
            return False
        mode = _i32(OFF["mode"])
        if mode < 0 or mode > 20:
            return False
        m = _i32(OFF["measure"])
        if m < 0 or m >= MEASURE_COUNT:
            return False
        b = _i32(OFF["beat"])
        if b < 0 or b > 3:
            return False
        sb = _i32(OFF["sub_beat"])
        if sb < 0 or sb > 7:
            return False
        if _u8(OFF["is_playing"]):
            bpm = _f32(OFF["bpm"])
            dur = _f32(OFF["beat_dur"])
            if bpm <= 0 or bpm > 500 or dur <= 0 or dur > 10:
                return False
            if abs(dur - 60.0/bpm) > 0.5:
                return False
        return True

    # ================================================================
    #  STATE READ
    # ================================================================
    def _read_state(self) -> Optional[GameState]:
        base = self._game_base
        if base is None:
            return None

        # bulk read 0x7C0 .. 0x890 (=0xD0 bytes)
        data = self._pa.read(base + OFF["bpm"], 0xD0)
        if data is None or len(data) < 0xD0:
            return None

        def _f(off): return struct.unpack_from("<f", data, off - OFF["bpm"])[0]
        def _i(off): return struct.unpack_from("<i", data, off - OFF["bpm"])[0]
        def _u(off): return data[off - OFF["bpm"]]

        gs = GameState(
            bpm           = _f(OFF["bpm"]),
            beat_duration = _f(OFF["beat_dur"]),
            elapsed_time  = _f(OFF["elapsed"]),
            is_playing    = _u(OFF["is_playing"]) != 0,
            notes_active  = _u(OFF["notes_active"]) != 0,
            song_finished = _u(OFF["finished"]) != 0,
            measure       = _i(OFF["measure"]),
            beat          = _i(OFF["beat"]),
            sub_beat      = _i(OFF["sub_beat"]),
            combo         = _i(OFF["combo"]),
            score         = _i(OFF["score"]),
            judgment      = _i(OFF["judgment"]),
            hp            = _i(OFF["hp"]),
            anim_state    = _i(OFF["anim"]),
        )

        # lane states (separate bulk read)
        ls = self._pa.read(base + OFF["lanes"], NUM_LANES * LANE_STRIDE)
        if ls and len(ls) >= NUM_LANES * LANE_STRIDE:
            gs.lane_active = [ls[i * LANE_STRIDE] != 0 for i in range(NUM_LANES)]

        return gs

    # ================================================================
    #  MEASURE-DATA NOTE READING
    # ================================================================
    def _get_measure_notes(self, mi: int) -> List[dict]:
        if mi in self._note_cache:
            return self._note_cache[mi]
        notes = self._read_measure_notes(mi)
        self._note_cache[mi] = notes
        return notes

    def _read_measure_notes(self, mi: int) -> List[dict]:
        """Walk the measure's lane arrays and extract sub-note channels."""
        if self._game_base is None or mi < 0 or mi >= MEASURE_COUNT:
            return []

        mptr = self._pa.read_uint32(self._game_base + OFF["measures"] + mi * 4)
        if not mptr or mptr < 0x10000:
            return []

        mdata = self._pa.read(mptr, 0x30)
        if mdata is None or len(mdata) < 0x18:
            return []

        lane_ptrs = struct.unpack_from("<4I", mdata, 0)
        result: List[dict] = []

        for li, lp in enumerate(lane_ptrs):
            if not lp or lp < 0x10000:
                continue

            # Each lane object: +0x10 begin, +0x14 end of sub-note pointer array
            lhdr = self._pa.read(lp, 0x20)
            if lhdr is None or len(lhdr) < 0x18:
                continue

            arr_beg, arr_end = struct.unpack_from("<II", lhdr, 0x10)
            if arr_beg and arr_end and arr_end > arr_beg:
                cnt = (arr_end - arr_beg) // 4
                if cnt <= 0 or cnt > 64:
                    # try treating the lane ptr as a direct sub-note
                    self._try_direct_note(lp, lhdr, li, result)
                    continue
                ptrs = self._pa.read(arr_beg, cnt * 4)
                if ptrs is None:
                    continue
                for i in range(cnt):
                    snp = struct.unpack_from("<I", ptrs, i*4)[0]
                    if not snp or snp < 0x10000:
                        continue
                    sn = self._pa.read(snp, 0x3C)
                    if sn is None or len(sn) < 0x24:
                        continue
                    ch   = struct.unpack_from("<i", sn, SN_CHANNEL)[0]
                    frac = struct.unpack_from("<f", sn, SN_FRACTION)[0]
                    key  = CHANNEL_KEY.get(ch)
                    if key is None:
                        continue
                    result.append({"lane": li, "channel": ch, "key": key,
                                   "fraction": frac, "hit": sn[SN_HIT]})
            else:
                self._try_direct_note(lp, lhdr, li, result)

        # Fallback: if no sub-notes, but measure data has anim type, infer lanes
        if not result and len(mdata) >= 0x2C:
            anim = struct.unpack_from("<i", mdata, 0x28)[0]
            if anim != 0:
                for li, lp in enumerate(lane_ptrs):
                    if lp and lp > 0x10000 and li in LANE_KEY:
                        result.append({"lane": li, "channel": 0, "key": LANE_KEY[li],
                                       "fraction": 0.0, "hit": 0})
        return result

    @staticmethod
    def _try_direct_note(lp, lhdr, li, result):
        """Lane pointer might BE the sub-note (flat struct)."""
        if len(lhdr) >= 0x24:
            ch   = struct.unpack_from("<i", lhdr, SN_CHANNEL)[0]
            frac = struct.unpack_from("<f", lhdr, SN_FRACTION)[0]
            key  = CHANNEL_KEY.get(ch)
            if key:
                result.append({"lane": li, "channel": ch, "key": key,
                               "fraction": frac, "hit": lhdr[SN_HIT]})

    # ================================================================
    #  HELPERS
    # ================================================================
    def _read_f32(self, addr: int) -> float:
        d = self._pa.read(addr, 4)
        if d and len(d) == 4:
            return struct.unpack_from("<f", d, 0)[0]
        return 0.0

    def reset(self):
        self._game_base = None
        self._scan_done = False
        self._last_measure = -1
        self._last_beat = -1
        self._pressed.clear()
        self._last_press.clear()
        self._note_cache.clear()
        self._frame = 0
