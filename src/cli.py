"""
Command-line interface for idate rhythm game automation.

Commands:
- calibrate: Auto-detect game window and lane positions
- record: Record screen region for reference
- dry-run: Run detection without executing inputs
- run: Full automation mode
- benchmark: Test capture and detection speed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


def get_arg_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="idate",
        description="Rhythm game automation with computer vision",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/game_profile.toml",
        help="Path to game profile config",
    )
    parser.add_argument(
        "--keymap",
        type=str,
        default="configs/keymap.toml",
        help="Path to keymap config",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Calibrate command
    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Auto-detect game window and lane positions",
    )
    calibrate_parser.add_argument(
        "--window",
        type=str,
        help="Window title pattern to detect",
    )
    calibrate_parser.add_argument(
        "--lanes",
        type=int,
        default=8,
        help="Number of lanes (default: 8)",
    )
    calibrate_parser.add_argument(
        "--save",
        type=str,
        help="Save calibration to file",
    )
    
    # Record command
    record_parser = subparsers.add_parser(
        "record",
        help="Record screen region for reference",
    )
    record_parser.add_argument(
        "--output", "-o",
        type=str,
        default="debug_frames",
        help="Output directory for frames",
    )
    record_parser.add_argument(
        "--frames", "-n",
        type=int,
        default=60,
        help="Number of frames to record",
    )
    record_parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Delay before recording starts (seconds)",
    )
    
    # Dry-run command
    dryrun_parser = subparsers.add_parser(
        "dry-run",
        help="Run detection without executing inputs",
    )
    dryrun_parser.add_argument(
        "--duration", "-d",
        type=float,
        default=30.0,
        help="Duration to run (seconds)",
    )
    dryrun_parser.add_argument(
        "--show-overlay",
        action="store_true",
        help="Show debug overlay window",
    )
    dryrun_parser.add_argument(
        "--save-trace",
        type=str,
        help="Save event trace to file",
    )
    
    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Full automation mode",
    )
    run_parser.add_argument(
        "--beatmap",
        type=str,
        help="Path to beatmap/chart file for timing",
    )
    run_parser.add_argument(
        "--show-overlay",
        action="store_true",
        help="Show debug overlay window",
    )
    run_parser.add_argument(
        "--stop-key",
        type=str,
        default="f12",
        help="Key to stop automation (default: f12)",
    )
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Test capture and detection speed",
    )
    benchmark_parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=100,
        help="Number of iterations",
    )
    benchmark_parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Benchmark capture only (no detection)",
    )
    
    return parser


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Run calibration command."""
    from .config import GameProfile
    from .capture.mss_capture import MSSCapture
    from .vision.lane_detector import auto_detect_lanes, draw_lane_overlay
    import cv2
    
    print("Starting calibration...")
    
    # Load config if exists
    config_path = Path(args.config)
    if config_path.exists():
        profile = GameProfile.from_toml(config_path)
    else:
        from .config import CaptureConfig, LanesConfig, DetectionConfig, TimingConfig, SafetyConfig, DebugConfig
        profile = GameProfile(
            game_name="Unknown",
            capture=CaptureConfig(left=0, top=0, width=800, height=600),
            lanes=LanesConfig(count=args.lanes),
            detection=DetectionConfig(),
            timing=TimingConfig(),
            safety=SafetyConfig(),
            debug=DebugConfig(),
        )
    
    # Capture a frame
    with MSSCapture() as capture:
        result = capture.grab()
        if result is None:
            print("Failed to capture screen")
            return 1
        
        frame = result.frame
    
    # Auto-detect lanes
    print(f"Detecting {args.lanes} lanes...")
    layout = auto_detect_lanes(frame, args.lanes)
    
    if layout:
        print(f"Detected {len(layout.lanes)} lanes:")
        for lane in layout.lanes:
            print(f"  Lane {lane.index}: x={lane.center_x}, bounds=[{lane.left_bound}, {lane.right_bound}]")
        print(f"Hit line Y: {layout.hit_line_y}")
        
        # Show preview
        preview = draw_lane_overlay(frame, layout)
        cv2.imshow("Calibration Preview", preview)
        print("\nPress any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save if requested
        if args.save:
            save_path = Path(args.save)
            calibration = {
                "lanes": [
                    {
                        "index": l.index,
                        "center_x": l.center_x,
                        "left_bound": l.left_bound,
                        "right_bound": l.right_bound,
                    }
                    for l in layout.lanes
                ],
                "hit_line_y": layout.hit_line_y,
            }
            with save_path.open("w") as f:
                json.dump(calibration, f, indent=2)
            print(f"Calibration saved to {save_path}")
    else:
        print("Failed to detect lanes. Try adjusting the capture region.")
        return 1
    
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    """Run record command."""
    from .capture.mss_capture import MSSCapture
    from .debug.overlay import FrameSaver
    import cv2
    
    output_dir = Path(args.output)
    saver = FrameSaver(output_dir)
    
    print(f"Recording {args.frames} frames to {output_dir}")
    print(f"Starting in {args.delay} seconds...")
    time.sleep(args.delay)
    
    print("Recording...")
    with MSSCapture() as capture:
        for i in range(args.frames):
            result = capture.grab()
            if result:
                path = saver.save(result.frame)
                print(f"  Saved frame {i+1}/{args.frames}: {path.name}")
            else:
                print(f"  Frame {i+1}/{args.frames}: capture failed")
    
    print(f"Recording complete. Frames saved to {output_dir}")
    return 0


def cmd_dryrun(args: argparse.Namespace) -> int:
    """Run dry-run command."""
    from .config import GameProfile, KeyMap
    from .capture.mss_capture import CaptureSession
    from .vision.lane_detector import detect_lanes_from_config
    from .vision.note_tracker import NoteTracker
    from .runtime.control import SafetyController, PerformanceMonitor
    from .debug.overlay import DebugOverlay, OverlayWindow
    from .debug.trace import Tracer, setup_logging
    
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1
    
    profile = GameProfile.from_toml(config_path)
    layout = detect_lanes_from_config(profile.lanes)
    
    # Setup components
    tracker = NoteTracker(layout)
    monitor = PerformanceMonitor()
    tracer = Tracer(enabled=bool(args.save_trace))
    
    overlay = DebugOverlay() if args.show_overlay else None
    window = OverlayWindow() if args.show_overlay else None
    
    print(f"Starting dry-run for {args.duration} seconds...")
    print("Press Ctrl+C to stop")
    
    monitor.start()
    tracer.start()
    
    try:
        with CaptureSession(profile.capture) as session:
            start_time = time.perf_counter()
            frame_count = 0
            
            while time.perf_counter() - start_time < args.duration:
                frame_start = time.perf_counter()
                
                # Capture
                result = session.grab()
                if result is None:
                    continue
                
                capture_time = (time.perf_counter() - frame_start) * 1000
                
                # Detect
                detect_start = time.perf_counter()
                detections = tracker.detect(result.frame)
                tracked = tracker.update(detections, result.timestamp)
                detect_time = (time.perf_counter() - detect_start) * 1000
                
                # Record metrics
                monitor.record_frame(capture_time + detect_time)
                monitor.record_detection(detect_time)
                tracer.set_frame(frame_count)
                tracer.trace_frame(capture_time)
                
                for det in detections:
                    tracer.trace_detection(det.lane_index, det.center_x, det.center_y)
                
                # Show overlay
                if overlay and window:
                    frame = overlay.render(result.frame, layout, detections, tracked, monitor)
                    window.show(frame)
                    if window.wait_key(1) == 27:  # ESC
                        break
                
                frame_count += 1
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        if window:
            window.close()
    
    # Print summary
    summary = monitor.get_summary()
    print("\n--- Dry-Run Summary ---")
    print(f"Duration: {summary['elapsed_seconds']:.1f}s")
    print(f"Frames: {summary['frame_count']}")
    print(f"Avg FPS: {summary['avg_fps']:.1f}")
    print(f"Avg Frame Time: {summary['avg_frame_time_ms']:.1f}ms")
    print(f"Avg Detection Time: {summary['avg_detection_time_ms']:.1f}ms")
    
    # Save trace
    if args.save_trace:
        trace_path = Path(args.save_trace)
        tracer.save(trace_path)
        print(f"Trace saved to {trace_path}")
    
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run full automation command."""
    from .config import GameProfile, KeyMap
    from .capture.mss_capture import CaptureSession
    from .vision.lane_detector import detect_lanes_from_config
    from .vision.note_tracker import NoteTracker, NoteState
    from .timing.map_parser import BeatmapParser
    from .timing.fusion import TimingFusion
    from .input.sendinput_driver import InputScheduler
    from .patterns.chords import PatternMapper, create_default_8lane_mapper
    from .runtime.control import SafetyController, PerformanceMonitor
    from .debug.overlay import DebugOverlay, OverlayWindow
    from .debug.trace import Tracer, setup_logging
    
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    
    # Load configs
    config_path = Path(args.config)
    keymap_path = Path(args.keymap)
    
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1
    
    profile = GameProfile.from_toml(config_path)
    keymap = KeyMap.from_toml(keymap_path) if keymap_path.exists() else None
    
    layout = detect_lanes_from_config(profile.lanes)
    
    # Setup components
    tracker = NoteTracker(layout)
    fusion = TimingFusion() if args.beatmap else None
    scheduler = InputScheduler(max_actions_per_sec=profile.safety.max_actions_per_second)
    mapper = create_default_8lane_mapper()
    safety = SafetyController(
        stop_hotkey=args.stop_key,
        require_focus=profile.safety.require_focus,
        window_pattern=profile.safety.window_title_pattern,
    )
    monitor = PerformanceMonitor()
    
    overlay = DebugOverlay() if args.show_overlay else None
    window = OverlayWindow() if args.show_overlay else None
    
    # Load beatmap if provided
    if args.beatmap:
        beatmap_path = Path(args.beatmap)
        if beatmap_path.exists():
            beatmap = BeatmapParser.parse(beatmap_path)
            print(f"Loaded beatmap: {beatmap.title} ({len(beatmap.notes)} notes)")
        else:
            print(f"Beatmap not found: {beatmap_path}")
            fusion = None
    
    print(f"Starting automation. Press {args.stop_key.upper()} to stop.")
    
    safety.start()
    monitor.start()
    scheduler.start()
    
    try:
        with CaptureSession(profile.capture) as session:
            while not safety.should_stop:
                frame_start = time.perf_counter()
                
                # Check if we can execute
                can_exec, reason = safety.can_execute()
                if not can_exec:
                    if reason != "Window not focused":
                        time.sleep(0.01)
                    continue
                
                # Capture
                result = session.grab()
                if result is None:
                    continue
                
                # Detect notes
                detect_start = time.perf_counter()
                detections = tracker.detect(result.frame)
                tracked = tracker.update(detections, result.timestamp)
                detect_time = (time.perf_counter() - detect_start) * 1000
                
                # Schedule actions for notes at hit line
                for track in tracked:
                    if track.state == NoteState.AT_HIT_LINE:
                        # Get key pattern for this lane
                        pattern = mapper.get_pattern(track.note.lane_index)
                        if pattern:
                            keys = [k.value for k in pattern.keys]
                            success = scheduler.schedule_now(keys)
                            
                            # Record action
                            latency = (time.perf_counter() - track.predicted_hit_time) * 1000
                            safety.record_action(track.note.lane_index, keys, success, latency)
                            monitor.record_action(latency, hit=success)
                
                # Update metrics
                frame_time = (time.perf_counter() - frame_start) * 1000
                monitor.record_frame(frame_time)
                monitor.record_detection(detect_time)
                
                # Show overlay
                if overlay and window:
                    frame = overlay.render(result.frame, layout, detections, tracked, monitor)
                    window.show(frame)
                    if window.wait_key(1) == 27:  # ESC
                        break
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        safety.stop()
        scheduler.stop()
        if window:
            window.close()
    
    # Print summary
    summary = monitor.get_summary()
    print("\n--- Run Summary ---")
    print(f"Duration: {summary['elapsed_seconds']:.1f}s")
    print(f"Frames: {summary['frame_count']}")
    print(f"Avg FPS: {summary['avg_fps']:.1f}")
    print(f"Actions: {summary['action_count']}")
    print(f"Hit Rate: {summary['hit_rate']:.1f}%")
    print(f"Avg Latency: {summary['avg_latency_ms']:.1f}ms")
    
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run benchmark command."""
    from .config import GameProfile
    from .capture.mss_capture import MSSCapture
    from .vision.lane_detector import detect_lanes_from_config
    from .vision.note_tracker import NoteTracker
    import statistics
    
    # Load config if exists
    config_path = Path(args.config)
    if config_path.exists():
        profile = GameProfile.from_toml(config_path)
    else:
        from .config import CaptureConfig
        profile = GameProfile(
            game_name="Benchmark",
            capture=CaptureConfig(left=0, top=0, width=800, height=600),
            lanes=None,
            detection=None,
            timing=None,
            safety=None,
            debug=None,
        )
    
    print(f"Running benchmark ({args.iterations} iterations)...")
    
    capture_times: list[float] = []
    detect_times: list[float] = []
    
    with MSSCapture(profile.capture) as capture:
        # Warmup
        for _ in range(10):
            capture.grab()
        
        # Setup detector if needed
        tracker = None
        if not args.capture_only and profile.lanes:
            layout = detect_lanes_from_config(profile.lanes)
            tracker = NoteTracker(layout)
        
        # Benchmark
        for i in range(args.iterations):
            # Capture
            start = time.perf_counter()
            result = capture.grab()
            capture_time = (time.perf_counter() - start) * 1000
            capture_times.append(capture_time)
            
            # Detect
            if tracker and result:
                start = time.perf_counter()
                detections = tracker.detect(result.frame)
                detect_time = (time.perf_counter() - start) * 1000
                detect_times.append(detect_time)
            
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{args.iterations}...")
    
    # Print results
    print("\n--- Benchmark Results ---")
    print(f"Capture:")
    print(f"  Mean: {statistics.mean(capture_times):.2f}ms")
    print(f"  Median: {statistics.median(capture_times):.2f}ms")
    print(f"  Std Dev: {statistics.stdev(capture_times):.2f}ms")
    print(f"  Min: {min(capture_times):.2f}ms")
    print(f"  Max: {max(capture_times):.2f}ms")
    print(f"  Theoretical FPS: {1000 / statistics.mean(capture_times):.1f}")
    
    if detect_times:
        print(f"\nDetection:")
        print(f"  Mean: {statistics.mean(detect_times):.2f}ms")
        print(f"  Median: {statistics.median(detect_times):.2f}ms")
        print(f"  Std Dev: {statistics.stdev(detect_times):.2f}ms")
        print(f"  Min: {min(detect_times):.2f}ms")
        print(f"  Max: {max(detect_times):.2f}ms")
        
        total_mean = statistics.mean(capture_times) + statistics.mean(detect_times)
        print(f"\nTotal Pipeline:")
        print(f"  Mean: {total_mean:.2f}ms")
        print(f"  Theoretical FPS: {1000 / total_mean:.1f}")
    
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = get_arg_parser()
    args = parser.parse_args(argv)
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    if args.command is None:
        parser.print_help()
        return 0
    
    commands = {
        "calibrate": cmd_calibrate,
        "record": cmd_record,
        "dry-run": cmd_dryrun,
        "run": cmd_run,
        "benchmark": cmd_benchmark,
    }
    
    cmd_func = commands.get(args.command)
    if cmd_func:
        return cmd_func(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
