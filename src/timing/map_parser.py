"""
Beatmap/chart file parser for timing metadata.

Supports common rhythm game chart formats for hybrid CV + timing detection.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class NoteEvent:
    """A single note event from a beatmap."""
    time_ms: float           # Time in milliseconds from song start
    lane: int                # Lane index (0-based)
    hold_duration_ms: float = 0  # Duration if hold note, 0 for tap
    note_type: str = "tap"   # tap, hold, slide, etc.

    @property
    def end_time_ms(self) -> float:
        """Get end time for hold notes."""
        return self.time_ms + self.hold_duration_ms


@dataclass
class BPMChange:
    """BPM change event in a beatmap."""
    time_ms: float
    bpm: float


@dataclass
class ParsedBeatmap:
    """Parsed beatmap with timing information."""
    title: str = ""
    artist: str = ""
    source_file: str = ""
    audio_file: str = ""
    
    # Timing
    offset_ms: float = 0  # Audio offset
    initial_bpm: float = 120
    bpm_changes: list[BPMChange] = field(default_factory=list)
    
    # Notes
    notes: list[NoteEvent] = field(default_factory=list)
    lane_count: int = 4
    
    # Duration
    duration_ms: float = 0

    @property
    def note_count(self) -> int:
        """Get total note count."""
        return len(self.notes)

    def get_notes_in_range(
        self,
        start_ms: float,
        end_ms: float,
    ) -> list[NoteEvent]:
        """Get notes within a time range."""
        return [
            n for n in self.notes
            if start_ms <= n.time_ms <= end_ms
        ]

    def get_notes_for_lane(self, lane: int) -> list[NoteEvent]:
        """Get all notes for a specific lane."""
        return [n for n in self.notes if n.lane == lane]

    def get_bpm_at_time(self, time_ms: float) -> float:
        """Get BPM at a specific time."""
        current_bpm = self.initial_bpm
        for change in self.bpm_changes:
            if change.time_ms <= time_ms:
                current_bpm = change.bpm
            else:
                break
        return current_bpm


class BeatmapParser:
    """
    Parser for various beatmap/chart formats.
    
    Supported formats:
    - JSON-based charts (generic)
    - Simple CSV timing files
    - osu! mania charts (.osu)
    - StepMania/SM files (.sm, .ssc) - basic support
    """

    @staticmethod
    def parse(file_path: str | Path) -> ParsedBeatmap:
        """
        Parse a beatmap file, auto-detecting format.
        
        Args:
            file_path: Path to beatmap file.
        
        Returns:
            Parsed beatmap data.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix == ".json":
            return BeatmapParser._parse_json(path)
        elif suffix == ".csv":
            return BeatmapParser._parse_csv(path)
        elif suffix == ".osu":
            return BeatmapParser._parse_osu(path)
        elif suffix in (".sm", ".ssc"):
            return BeatmapParser._parse_stepmania(path)
        else:
            raise ValueError(f"Unsupported beatmap format: {suffix}")

    @staticmethod
    def _parse_json(path: Path) -> ParsedBeatmap:
        """Parse JSON-format beatmap."""
        with path.open() as f:
            data = json.load(f)
        
        beatmap = ParsedBeatmap(
            title=data.get("title", ""),
            artist=data.get("artist", ""),
            source_file=str(path),
            audio_file=data.get("audio", ""),
            offset_ms=data.get("offset", 0),
            initial_bpm=data.get("bpm", 120),
            lane_count=data.get("lanes", 4),
        )
        
        # Parse notes
        for note_data in data.get("notes", []):
            note = NoteEvent(
                time_ms=note_data.get("time", 0),
                lane=note_data.get("lane", 0),
                hold_duration_ms=note_data.get("duration", 0),
                note_type=note_data.get("type", "tap"),
            )
            beatmap.notes.append(note)
        
        # Parse BPM changes
        for bpm_data in data.get("bpm_changes", []):
            change = BPMChange(
                time_ms=bpm_data.get("time", 0),
                bpm=bpm_data.get("bpm", 120),
            )
            beatmap.bpm_changes.append(change)
        
        # Sort notes by time
        beatmap.notes.sort(key=lambda n: n.time_ms)
        
        if beatmap.notes:
            beatmap.duration_ms = beatmap.notes[-1].end_time_ms
        
        return beatmap

    @staticmethod
    def _parse_csv(path: Path) -> ParsedBeatmap:
        """
        Parse simple CSV timing file.
        
        Expected format: time_ms,lane[,duration_ms]
        """
        beatmap = ParsedBeatmap(
            title=path.stem,
            source_file=str(path),
        )
        
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                
                try:
                    time_ms = float(parts[0])
                    lane = int(parts[1])
                    duration = float(parts[2]) if len(parts) > 2 else 0
                    
                    note = NoteEvent(
                        time_ms=time_ms,
                        lane=lane,
                        hold_duration_ms=duration,
                        note_type="hold" if duration > 0 else "tap",
                    )
                    beatmap.notes.append(note)
                    
                    # Track max lane
                    if lane >= beatmap.lane_count:
                        beatmap.lane_count = lane + 1
                except ValueError:
                    continue
        
        beatmap.notes.sort(key=lambda n: n.time_ms)
        if beatmap.notes:
            beatmap.duration_ms = beatmap.notes[-1].end_time_ms
        
        return beatmap

    @staticmethod
    def _parse_osu(path: Path) -> ParsedBeatmap:
        """
        Parse osu! mania beatmap format.
        
        Basic parser for mania mode charts.
        """
        beatmap = ParsedBeatmap(source_file=str(path))
        
        content = path.read_text(encoding="utf-8", errors="ignore")
        lines = content.split("\n")
        
        section = ""
        column_count = 4
        
        for line in lines:
            line = line.strip()
            
            # Section headers
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].lower()
                continue
            
            if not line or line.startswith("//"):
                continue
            
            # Metadata
            if section == "metadata":
                if line.startswith("Title:"):
                    beatmap.title = line[6:].strip()
                elif line.startswith("Artist:"):
                    beatmap.artist = line[7:].strip()
            
            # Difficulty settings
            elif section == "difficulty":
                if line.startswith("CircleSize:"):
                    try:
                        column_count = int(float(line[11:]))
                        beatmap.lane_count = column_count
                    except ValueError:
                        pass
            
            # General settings
            elif section == "general":
                if line.startswith("AudioFilename:"):
                    beatmap.audio_file = line[14:].strip()
            
            # Timing points
            elif section == "timingpoints":
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        time_ms = float(parts[0])
                        beat_length = float(parts[1])
                        
                        if beat_length > 0:  # Regular timing point
                            bpm = 60000 / beat_length
                            if not beatmap.bpm_changes:
                                beatmap.initial_bpm = bpm
                            beatmap.bpm_changes.append(
                                BPMChange(time_ms=time_ms, bpm=bpm)
                            )
                    except ValueError:
                        pass
            
            # Hit objects (mania notes)
            elif section == "hitobjects":
                parts = line.split(",")
                if len(parts) >= 5:
                    try:
                        x = int(parts[0])
                        time_ms = float(parts[2])
                        hit_type = int(parts[3])
                        
                        # Calculate lane from X position
                        lane = int(x * column_count / 512)
                        lane = max(0, min(lane, column_count - 1))
                        
                        # Check for hold note (type 128)
                        duration = 0
                        if hit_type & 128:  # Hold note
                            if ":" in parts[5]:
                                end_time = float(parts[5].split(":")[0])
                                duration = end_time - time_ms
                        
                        note = NoteEvent(
                            time_ms=time_ms,
                            lane=lane,
                            hold_duration_ms=duration,
                            note_type="hold" if duration > 0 else "tap",
                        )
                        beatmap.notes.append(note)
                    except (ValueError, IndexError):
                        pass
        
        beatmap.notes.sort(key=lambda n: n.time_ms)
        if beatmap.notes:
            beatmap.duration_ms = beatmap.notes[-1].end_time_ms
        
        return beatmap

    @staticmethod
    def _parse_stepmania(path: Path) -> ParsedBeatmap:
        """
        Parse StepMania chart format (.sm/.ssc).
        
        Basic support for dance/pump style charts.
        """
        beatmap = ParsedBeatmap(source_file=str(path))
        
        content = path.read_text(encoding="utf-8", errors="ignore")
        
        # Extract metadata
        title_match = re.search(r"#TITLE:([^;]*);", content)
        if title_match:
            beatmap.title = title_match.group(1).strip()
        
        artist_match = re.search(r"#ARTIST:([^;]*);", content)
        if artist_match:
            beatmap.artist = artist_match.group(1).strip()
        
        music_match = re.search(r"#MUSIC:([^;]*);", content)
        if music_match:
            beatmap.audio_file = music_match.group(1).strip()
        
        offset_match = re.search(r"#OFFSET:([^;]*);", content)
        if offset_match:
            try:
                beatmap.offset_ms = float(offset_match.group(1)) * 1000
            except ValueError:
                pass
        
        # Parse BPMs
        bpm_match = re.search(r"#BPMS:([^;]*);", content)
        if bpm_match:
            bpm_str = bpm_match.group(1)
            for pair in bpm_str.split(","):
                parts = pair.strip().split("=")
                if len(parts) == 2:
                    try:
                        beat = float(parts[0])
                        bpm = float(parts[1])
                        if not beatmap.bpm_changes:
                            beatmap.initial_bpm = bpm
                        # Note: would need to convert beats to ms using BPM
                        beatmap.bpm_changes.append(
                            BPMChange(time_ms=beat * (60000 / bpm), bpm=bpm)
                        )
                    except ValueError:
                        pass
        
        # Parse notes (simplified - full parsing is complex)
        notes_match = re.search(r"#NOTES:[^:]*:[^:]*:[^:]*:[^:]*:[^:]*:([^;]*);", content)
        if notes_match:
            notes_data = notes_match.group(1)
            beatmap.lane_count = 4  # Default for dance
            
            # Split into measures
            measures = notes_data.strip().split(",")
            
            current_beat = 0
            bpm = beatmap.initial_bpm if beatmap.initial_bpm else 120
            
            for measure in measures:
                lines = [l.strip() for l in measure.strip().split("\n") if l.strip()]
                if not lines:
                    continue
                
                beats_per_line = 4.0 / len(lines)
                
                for line in lines:
                    if len(line) >= beatmap.lane_count:
                        for lane, char in enumerate(line[:beatmap.lane_count]):
                            if char in "124":  # Tap, hold start, roll
                                time_ms = current_beat * (60000 / bpm)
                                note = NoteEvent(
                                    time_ms=time_ms,
                                    lane=lane,
                                    note_type="tap" if char == "1" else "hold",
                                )
                                beatmap.notes.append(note)
                    
                    current_beat += beats_per_line
        
        beatmap.notes.sort(key=lambda n: n.time_ms)
        if beatmap.notes:
            beatmap.duration_ms = beatmap.notes[-1].end_time_ms
        
        return beatmap


def create_timing_file(
    notes: list[NoteEvent],
    output_path: str | Path,
    format: str = "csv",
) -> None:
    """
    Create a timing file from note events.
    
    Args:
        notes: List of note events.
        output_path: Path to write file.
        format: Output format ('csv' or 'json').
    """
    path = Path(output_path)
    
    if format == "csv":
        with path.open("w") as f:
            f.write("# time_ms,lane,duration_ms\n")
            for note in sorted(notes, key=lambda n: n.time_ms):
                f.write(f"{note.time_ms},{note.lane},{note.hold_duration_ms}\n")
    
    elif format == "json":
        data = {
            "notes": [
                {
                    "time": n.time_ms,
                    "lane": n.lane,
                    "duration": n.hold_duration_ms,
                    "type": n.note_type,
                }
                for n in sorted(notes, key=lambda n: n.time_ms)
            ]
        }
        with path.open("w") as f:
            json.dump(data, f, indent=2)
