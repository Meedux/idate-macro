"""
BMS (Be-Music Source) Chart Parser for iDate Revival.

Parses .bms chart files used by iDate to extract:
  - Note events with precise timing (measure, fractional beat position)
  - BPM values and mid-song BPM changes
  - Arrow/visual channel mappings

Based on reverse engineering of the iDate game executable:
  - BMS files: %s%04d.bms (4-key) / %s%04d_8key.bms (8-key)
  - Tags: #BPM, #BPMxx, #WAVxx, #MAIN
  - #MAIN data: 3-digit measure + 2-digit channel + hex note pairs
  - Channels: 1=BPM change, 3=beat event, 4=note event,
              8=indexed BPM, 0xb-0xf/0x16-0x19=arrow events

Channel → Arrow Type mapping (from FUN_0040b890):
  0x0b (11) → left        Y=425.0 (standard)
  0x0c (12) → down        Y=409.0 (alternate)
  0x0d (13) → up          Y=425.0 (standard)
  0x0e (14) → right       Y=438.0 (high)
  0x0f (15) → hand/space  Y=438.0 (high)
  0x16 (22) → left2       Y=438.0 (high)     [8-key mode]
  0x17 (23) → down2       Y=425.0 (standard) [8-key mode]
  0x18 (24) → up2         Y=425.0 (standard) [8-key mode]
  0x19 (25) → right2      Y=409.0 (alternate) [8-key mode]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Channel → key name mapping (derived from reverse-engineered exe)
# ---------------------------------------------------------------------------

CHANNEL_TO_KEY: dict[int, str] = {
    0x0B: "left",       # channel 11
    0x0C: "down",       # channel 12
    0x0D: "up",         # channel 13
    0x0E: "right",      # channel 14
    0x0F: "space",      # channel 15 (hand/special)
    # 8-key mode extensions
    0x16: "left",       # channel 22 (mirrors left)
    0x17: "down",       # channel 23 (mirrors down)
    0x18: "up",         # channel 24 (mirrors up)
    0x19: "right",      # channel 25 (mirrors right)
}

# Channels that carry note/arrow events
NOTE_CHANNELS = set(CHANNEL_TO_KEY.keys())

# Special channels
CHANNEL_BPM_CHANGE = 1
CHANNEL_BEAT_EVENT = 3
CHANNEL_NOTE_EVENT = 4
CHANNEL_BPM_INDEXED = 8


@dataclass
class BmsNote:
    """A single note event from the BMS chart."""
    measure: int            # 0-based measure index (0–159)
    channel: int            # BMS channel number
    fraction: float         # Fractional position within measure (0.0–1.0)
    value: int              # Raw hex value from BMS data
    key_name: str           # Mapped key: "left", "down", "up", "right", "space"
    time_seconds: float = 0.0   # Absolute time in seconds (computed after BPM resolution)
    beat_position: float = 0.0  # Absolute beat position


@dataclass
class BpmChange:
    """A BPM change event."""
    measure: int
    fraction: float         # Position within measure
    bpm: float
    time_seconds: float = 0.0


@dataclass
class BmsChart:
    """Parsed BMS chart with all note and timing data."""
    # Metadata
    title: str = ""
    artist: str = ""
    genre: str = ""
    file_path: str = ""

    # Base BPM (from #BPM tag)
    base_bpm: float = 130.0

    # Indexed BPM table (from #BPMxx tags, index → BPM)
    bpm_table: dict[int, float] = field(default_factory=dict)

    # BPM change events (sorted by time)
    bpm_changes: list[BpmChange] = field(default_factory=list)

    # All note events (sorted by time)
    notes: list[BmsNote] = field(default_factory=list)

    # Beat events (channel 3)
    beat_events: list[dict] = field(default_factory=list)

    # WAV references
    wav_table: dict[int, str] = field(default_factory=dict)

    # Total measures used
    total_measures: int = 0

    # Computed timing data
    _measure_times: list[float] = field(default_factory=list)

    @property
    def total_notes(self) -> int:
        return len(self.notes)

    @property
    def duration_seconds(self) -> float:
        if self.notes:
            return max(n.time_seconds for n in self.notes)
        return 0.0

    def get_notes_in_range(
        self, start_time: float, end_time: float
    ) -> list[BmsNote]:
        """Get all notes within a time window (seconds)."""
        return [
            n for n in self.notes
            if start_time <= n.time_seconds <= end_time
        ]

    def get_notes_at_measure(self, measure: int) -> list[BmsNote]:
        """Get all notes in a specific measure."""
        return [n for n in self.notes if n.measure == measure]

    def get_keys_at_time(
        self, time_seconds: float, window_ms: float = 50.0
    ) -> list[str]:
        """Get key names that should be pressed at a given time."""
        half_window = window_ms / 1000.0 / 2.0
        notes = self.get_notes_in_range(
            time_seconds - half_window,
            time_seconds + half_window,
        )
        return list({n.key_name for n in notes})

    def get_measure_start_time(self, measure: int) -> float:
        """Get the start time of a measure in seconds."""
        if measure < len(self._measure_times):
            return self._measure_times[measure]
        return 0.0


class BmsParser:
    """
    Parser for BMS chart files.

    Usage:
        parser = BmsParser()
        chart = parser.parse_file("path/to/chart.bms")
        # Get notes at a specific time
        keys = chart.get_keys_at_time(current_elapsed, window_ms=50)
    """

    def __init__(self):
        self._log_callback: Callable[[str], None] | None = None

    def set_log_callback(self, callback: Callable[[str], None]):
        self._log_callback = callback

    def _log(self, msg: str):
        if self._log_callback:
            self._log_callback(msg)

    def parse_file(self, path: str | Path) -> BmsChart:
        """Parse a BMS file and return a fully-resolved chart."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BMS file not found: {path}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        chart = self._parse_content(content)
        chart.file_path = str(path)

        # Resolve timing (compute absolute times for all events)
        self._resolve_timing(chart)

        # Sort notes by time
        chart.notes.sort(key=lambda n: (n.time_seconds, n.channel))
        chart.bpm_changes.sort(key=lambda b: (b.measure, b.fraction))

        self._log(
            f"Parsed {path.name}: {chart.total_notes} notes, "
            f"{len(chart.bpm_changes)} BPM changes, "
            f"base BPM={chart.base_bpm:.1f}, "
            f"{chart.total_measures} measures"
        )
        return chart

    def parse_string(self, content: str) -> BmsChart:
        """Parse BMS content from a string."""
        chart = self._parse_content(content)
        self._resolve_timing(chart)
        chart.notes.sort(key=lambda n: (n.time_seconds, n.channel))
        return chart

    def _parse_content(self, content: str) -> BmsChart:
        """Parse BMS content into a chart (without timing resolution)."""
        chart = BmsChart()
        in_main = False
        lines = content.splitlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            # Header tags
            if line.startswith("#"):
                tag_line = line[1:]  # Remove leading #

                # Check for #MAIN section marker
                if tag_line.upper().startswith("MAIN"):
                    in_main = True
                    continue

                # #BPM (base BPM, no index)
                bpm_match = re.match(r"^BPM\s+(\d+\.?\d*)", tag_line, re.IGNORECASE)
                if bpm_match and not re.match(r"^BPM[0-9A-Fa-f]{2}", tag_line, re.IGNORECASE):
                    chart.base_bpm = float(bpm_match.group(1))
                    chart.bpm_table[0] = chart.base_bpm
                    continue

                # #BPMxx (indexed BPM change values)
                bpmx_match = re.match(
                    r"^BPM([0-9A-Fa-f]{2})\s+(\d+\.?\d*)",
                    tag_line, re.IGNORECASE,
                )
                if bpmx_match:
                    idx = int(bpmx_match.group(1), 16)
                    bpm_val = float(bpmx_match.group(2))
                    chart.bpm_table[idx] = bpm_val
                    continue

                # #WAVxx
                wav_match = re.match(
                    r"^WAV([0-9A-Fa-f]{2})\s+(.+)",
                    tag_line, re.IGNORECASE,
                )
                if wav_match:
                    idx = int(wav_match.group(1), 16)
                    chart.wav_table[idx] = wav_match.group(2).strip()
                    continue

                # #TITLE, #ARTIST, #GENRE
                for meta_tag in ("TITLE", "ARTIST", "GENRE"):
                    meta_match = re.match(
                        rf"^{meta_tag}\s+(.+)", tag_line, re.IGNORECASE
                    )
                    if meta_match:
                        setattr(chart, meta_tag.lower(), meta_match.group(1).strip())
                        break

            # #MAIN data lines: MMMCC:HHHHH...
            # Format: 3-digit measure + 2-digit channel + colon + hex data
            if in_main:
                main_match = re.match(r"^#?(\d{3})(\d{2}):?(.+)", line)
                if not main_match:
                    # Also try without # prefix
                    main_match = re.match(r"^(\d{3})(\d{2}):?(.+)", line)
                if main_match:
                    measure = int(main_match.group(1))
                    channel = int(main_match.group(2))
                    hex_data = main_match.group(3).strip()

                    chart.total_measures = max(chart.total_measures, measure + 1)
                    self._parse_main_line(chart, measure, channel, hex_data)

        # Ensure total_measures is at least as large as max note measure
        if chart.notes:
            chart.total_measures = max(
                chart.total_measures,
                max(n.measure for n in chart.notes) + 1,
            )

        return chart

    def _parse_main_line(
        self, chart: BmsChart, measure: int, channel: int, hex_data: str
    ):
        """Parse a single #MAIN data line."""
        # Remove any whitespace or colons
        hex_data = hex_data.replace(" ", "").replace(":", "")

        # Each pair of hex characters is one event
        num_events = len(hex_data) // 2
        if num_events == 0:
            return

        for i in range(num_events):
            hex_pair = hex_data[i * 2 : i * 2 + 2]
            value = int(hex_pair, 16)

            if value == 0:
                continue  # 00 = no event at this position

            # Fractional position within the measure
            fraction = i / num_events if num_events > 0 else 0.0

            if channel == CHANNEL_BPM_CHANGE:
                # Direct BPM value (channel 1)
                chart.bpm_changes.append(BpmChange(
                    measure=measure,
                    fraction=fraction,
                    bpm=float(value),
                ))

            elif channel == CHANNEL_BPM_INDEXED:
                # Indexed BPM reference (channel 8)
                bpm = chart.bpm_table.get(value, chart.base_bpm)
                chart.bpm_changes.append(BpmChange(
                    measure=measure,
                    fraction=fraction,
                    bpm=bpm,
                ))

            elif channel == CHANNEL_BEAT_EVENT:
                # Beat position event (channel 3)
                chart.beat_events.append({
                    "measure": measure,
                    "fraction": fraction,
                    "value": value,
                })

            elif channel == CHANNEL_NOTE_EVENT:
                # Generic note event (channel 4)
                chart.notes.append(BmsNote(
                    measure=measure,
                    channel=channel,
                    fraction=fraction,
                    value=value,
                    key_name="space",  # Channel 4 = generic note
                ))

            elif channel in NOTE_CHANNELS:
                # Arrow/direction note events (channels 0xb–0xf, 0x16–0x19)
                key_name = CHANNEL_TO_KEY.get(channel, "space")
                chart.notes.append(BmsNote(
                    measure=measure,
                    channel=channel,
                    fraction=fraction,
                    value=value,
                    key_name=key_name,
                ))

            # Channels 0x12, 0x13 are ignored (as in original exe)

    def _resolve_timing(self, chart: BmsChart):
        """
        Compute absolute times for all notes and BPM changes.

        The game uses:
            beat_duration = 60.0 / BPM  (seconds per beat)
            4 beats per measure
            measure_duration = beat_duration * 4

        Measure advances when: (measureIndex + 1) * beatDuration < elapsed
        But with dynamic BPM changes, we need to accumulate time progressively.
        """
        if chart.base_bpm <= 0:
            chart.base_bpm = 130.0

        # Sort BPM changes by position
        bpm_events = sorted(
            chart.bpm_changes,
            key=lambda b: (b.measure, b.fraction),
        )

        # Build a timeline of BPM changes with absolute positions
        # Position is in "measures" (float): measure + fraction
        bpm_timeline: list[tuple[float, float]] = [(0.0, chart.base_bpm)]
        for evt in bpm_events:
            pos = evt.measure + evt.fraction
            bpm_timeline.append((pos, evt.bpm))
        bpm_timeline.sort(key=lambda x: x[0])

        # Remove duplicate positions (keep last)
        deduped: list[tuple[float, float]] = []
        for pos, bpm in bpm_timeline:
            if deduped and abs(deduped[-1][0] - pos) < 1e-9:
                deduped[-1] = (pos, bpm)
            else:
                deduped.append((pos, bpm))
        bpm_timeline = deduped

        def position_to_time(target_pos: float) -> float:
            """Convert a measure position (float) to absolute time in seconds."""
            current_time = 0.0
            current_pos = 0.0
            current_bpm = chart.base_bpm

            for i, (change_pos, change_bpm) in enumerate(bpm_timeline):
                if change_pos >= target_pos:
                    break

                if change_pos > current_pos:
                    # Accumulate time for the segment before this BPM change
                    measures_in_segment = change_pos - current_pos
                    beat_duration = 60.0 / current_bpm
                    measure_duration = beat_duration * 4.0  # 4 beats per measure
                    current_time += measures_in_segment * measure_duration
                    current_pos = change_pos

                current_bpm = change_bpm

            # Accumulate remaining time from last BPM change to target
            if target_pos > current_pos:
                measures_remaining = target_pos - current_pos
                beat_duration = 60.0 / current_bpm
                measure_duration = beat_duration * 4.0
                current_time += measures_remaining * measure_duration

            return current_time

        # Resolve BPM change times
        for evt in chart.bpm_changes:
            pos = evt.measure + evt.fraction
            evt.time_seconds = position_to_time(pos)

        # Resolve note times
        for note in chart.notes:
            pos = note.measure + note.fraction
            note.time_seconds = position_to_time(pos)
            # Beat position = measure * 4 + fraction * 4
            note.beat_position = note.measure * 4.0 + note.fraction * 4.0

        # Compute measure start times
        max_measure = max(chart.total_measures, 160)
        chart._measure_times = [
            position_to_time(float(m)) for m in range(max_measure)
        ]


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def load_chart(path: str | Path) -> BmsChart:
    """Load and parse a BMS chart file."""
    parser = BmsParser()
    return parser.parse_file(path)


def find_bms_files(directory: str | Path) -> list[Path]:
    """Find all .bms files in a directory (recursive)."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.rglob("*.bms"))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.memory.bms_parser <path_to.bms>")
        print("\nSearching for .bms files in current directory...")
        bms_files = find_bms_files(".")
        if bms_files:
            for f in bms_files[:10]:
                print(f"  {f}")
        else:
            print("  No .bms files found")
        sys.exit(0)

    path = sys.argv[1]
    parser = BmsParser()
    parser.set_log_callback(print)
    chart = parser.parse_file(path)

    print(f"\n=== Chart Summary ===")
    print(f"Title:    {chart.title}")
    print(f"Artist:   {chart.artist}")
    print(f"BPM:      {chart.base_bpm}")
    print(f"Notes:    {chart.total_notes}")
    print(f"Measures: {chart.total_measures}")
    print(f"Duration: {chart.duration_seconds:.1f}s")
    print(f"BPM changes: {len(chart.bpm_changes)}")

    if chart.bpm_table:
        print(f"\nBPM Table:")
        for idx, bpm in sorted(chart.bpm_table.items()):
            print(f"  [{idx:02X}] = {bpm}")

    print(f"\nFirst 20 notes:")
    for note in chart.notes[:20]:
        print(
            f"  [{note.measure:03d}+{note.fraction:.3f}] "
            f"ch={note.channel:02X} key={note.key_name:6s} "
            f"time={note.time_seconds:.3f}s val={note.value:02X}"
        )

    if chart.bpm_changes:
        print(f"\nBPM Changes:")
        for bc in chart.bpm_changes[:10]:
            print(
                f"  [{bc.measure:03d}+{bc.fraction:.3f}] "
                f"BPM={bc.bpm:.1f} time={bc.time_seconds:.3f}s"
            )

    # Show key distribution
    from collections import Counter
    key_counts = Counter(n.key_name for n in chart.notes)
    print(f"\nKey distribution:")
    for key, count in key_counts.most_common():
        print(f"  {key:6s}: {count}")
