"""
Interactive Memory Scanner – run this while playing the game.

Usage:
    python -m src.memory.scan_interactive

Workflow:
    1. Pick a target process from the list.
    2. Play the game and use scan commands to find memory values:
       - first <value>   – First scan for an exact value
       - next <value>    – Filter for a new exact value
       - changed         – Keep only changed values
       - unchanged       – Keep only unchanged values
       - increased       – Keep only increased values
       - decreased       – Keep only decreased values
       - monitor         – Live-watch top candidates
       - refresh         – Re-read values without filtering
       - type <type>     – Set value type (int32, float, etc.)
       - save            – Export candidates to memory_patterns.json
       - reset           – Start over
       - quit            – Exit
"""

from __future__ import annotations

import json
import os
import sys
import time
import msvcrt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.memory.process import ProcessAttacher, enumerate_processes
from src.memory.scanner import MemoryScanner, ScanMode, ValueType

PATTERNS_FILE = Path("memory_patterns.json")

BANNER = r"""
+==================================================================+
|                  iDate Memory Scanner v1.0                       |
|  Attach to a game process and discover memory addresses.         |
|  Run this in the background while you play the game.             |
+==================================================================+
"""

HELP_TEXT = """
Commands:
  first <value>    First scan for an exact value
  first_unknown    First scan (store ALL values, filter later)
  next <value>     Filter: keep only addresses with this exact value
  changed          Filter: keep values that changed since last scan
  unchanged        Filter: keep values that stayed the same
  increased        Filter: keep values that increased
  decreased        Filter: keep values that decreased
  between <lo> <hi> Filter: keep values in range [lo, hi]
  monitor [N]      Live-watch top N candidates (default 20)
  refresh          Re-read values without filtering
  type <type>      Set value type: int8/uint8/int16/uint16/int32/uint32/
                                   int64/uint64/float/double
  list [N]         Show top N candidates (default 30)
  save [label]     Export candidates to memory_patterns.json
  load             Load previously saved patterns
  reset            Clear all scan data, start over
  help             Show this help
  quit / exit      Exit scanner
"""


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def pick_process() -> int | None:
    """Interactive process picker. Returns PID or None."""
    processes = enumerate_processes()
    # Filter to interesting processes (skip system/service ones)
    interesting = [
        p for p in processes
        if p.pid > 4
        and not p.name.lower().startswith(("svchost", "csrss", "services", "lsass", "wininit",
                                            "winlogon", "dwm", "smss", "registry",
                                            "fontdrvhost", "conhost", "sihost", "runtimebroker"))
    ]
    interesting.sort(key=lambda p: p.name.lower())

    # Deduplicate by name, keep lowest PID
    seen: dict[str, list] = {}
    for p in interesting:
        key = p.name.lower()
        if key not in seen:
            seen[key] = []
        seen[key].append(p)

    display = []
    for name_procs in seen.values():
        for p in name_procs:
            display.append(p)

    print("\n  Available processes:\n")
    print(f"  {'#':<5} {'PID':<8} {'Process Name'}")
    print(f"  {'─'*5} {'─'*8} {'─'*40}")

    for idx, p in enumerate(display, 1):
        print(f"  {idx:<5} {p.pid:<8} {p.name}")

    print(f"\n  Total: {len(display)} processes")
    print()

    while True:
        choice = input("  Enter # or PID (or 'search <name>'): ").strip()
        if not choice:
            continue
        if choice.lower().startswith("search "):
            query = choice[7:].lower()
            matches = [p for p in display if query in p.name.lower()]
            if not matches:
                print(f"  No process matching '{query}'")
                continue
            print()
            for idx, p in enumerate(matches, 1):
                print(f"  {idx:<5} {p.pid:<8} {p.name}")
            print()
            sub = input("  Enter # from filtered list: ").strip()
            try:
                si = int(sub)
                if 1 <= si <= len(matches):
                    return matches[si - 1].pid
            except ValueError:
                pass
            continue
        if choice.lower() in ("quit", "exit", "q"):
            return None

        try:
            num = int(choice)
            if 1 <= num <= len(display):
                return display[num - 1].pid
            # Maybe they entered a raw PID
            for p in interesting:
                if p.pid == num:
                    return p.pid
            print(f"  Invalid number: {num}")
        except ValueError:
            # Try as name search
            matches = [p for p in display if choice.lower() in p.name.lower()]
            if len(matches) == 1:
                return matches[0].pid
            elif matches:
                print(f"  Multiple matches, be more specific:")
                for p in matches:
                    print(f"    PID {p.pid}: {p.name}")
            else:
                print(f"  Not found: {choice}")


def parse_value(text: str, vtype: ValueType) -> int | float | None:
    """Parse a value string according to value type."""
    text = text.strip()
    try:
        if vtype in (ValueType.FLOAT, ValueType.DOUBLE):
            return float(text)
        if text.startswith("0x") or text.startswith("0X"):
            return int(text, 16)
        return int(text)
    except ValueError:
        return None


def format_value(val: int | float | None, vtype: ValueType) -> str:
    if val is None:
        return "N/A"
    if vtype in (ValueType.FLOAT, ValueType.DOUBLE):
        return f"{val:.4f}"
    return str(val)


def do_monitor(scanner: MemoryScanner, count: int = 20):
    """Live-monitor top candidates. Press any key to stop."""
    print(f"\n  Live monitoring top {count} candidates. Press any key to stop...\n")
    try:
        while True:
            results = scanner.monitor(count=count)
            # Clear and redraw
            lines = []
            lines.append(f"  {'Address':<20} {'Current':<16} {'Previous':<16}")
            lines.append(f"  {'─'*20} {'─'*16} {'─'*16}")
            for r in results:
                addr = f"0x{r.address:016X}"
                cur = format_value(r.value, scanner._value_type)
                prev = format_value(r.previous_value, scanner._value_type)
                changed = " *" if r.previous_value is not None and r.value != r.previous_value else ""
                lines.append(f"  {addr:<20} {cur:<16} {prev:<16}{changed}")
            lines.append(f"\n  [{len(results)} addresses | Press any key to stop]")

            # Move cursor up and overwrite
            sys.stdout.write(f"\033[{len(lines) + 2}A")
            for line in lines:
                sys.stdout.write(f"\r{line:<80}\n")
            sys.stdout.flush()

            time.sleep(0.25)

            # Check for keypress (non-blocking on Windows)
            if msvcrt.kbhit():
                msvcrt.getch()
                break

    except KeyboardInterrupt:
        pass
    print("\n  Monitoring stopped.\n")


def do_save(scanner: MemoryScanner, label: str = ""):
    """Save current candidates to memory_patterns.json."""
    exported = scanner.export_candidates(max_count=200)
    if not exported:
        print("  No candidates to save.")
        return

    # Load existing
    existing: dict = {}
    if PATTERNS_FILE.exists():
        try:
            with open(PATTERNS_FILE, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    if "discovered" not in existing:
        existing["discovered"] = {}

    key = label if label else f"scan_{int(time.time())}"
    existing["discovered"][key] = {
        "value_type": scanner._value_type.value,
        "scan_count": scanner.scan_count,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidates": exported,
    }

    with open(PATTERNS_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"  Saved {len(exported)} candidates under label '{key}' to {PATTERNS_FILE}")


def do_load():
    """Load and display saved patterns."""
    if not PATTERNS_FILE.exists():
        print(f"  No saved patterns ({PATTERNS_FILE} not found)")
        return

    with open(PATTERNS_FILE, "r") as f:
        data = json.load(f)

    discovered = data.get("discovered", {})
    confirmed = data.get("confirmed", {})

    if confirmed:
        print("\n  Confirmed patterns:")
        for name, info in confirmed.items():
            print(f"    {name}: {info}")

    if discovered:
        print("\n  Discovered scan results:")
        for label, info in discovered.items():
            count = len(info.get("candidates", []))
            ts = info.get("timestamp", "?")
            vtype = info.get("value_type", "?")
            print(f"    [{label}] {count} candidates ({vtype}) at {ts}")
    print()


def main():
    clear_screen()
    print(BANNER)

    # Step 1: Pick process
    pid = pick_process()
    if pid is None:
        print("  Cancelled.")
        return

    # Step 2: Attach
    attacher = ProcessAttacher()
    if not attacher.attach(pid):
        print(f"  ERROR: Failed to attach to PID {pid}. Run as Administrator?")
        return

    print(f"\n  ✓ Attached to {attacher.process_name} (PID {attacher.pid})")
    print(f"  ✓ Found {len(attacher.modules)} modules")
    base = attacher.get_base_address()
    if base:
        print(f"  ✓ Base address: 0x{base:016X}")
    print()

    # Step 3: Interactive scan loop
    scanner = MemoryScanner(attacher)
    vtype = ValueType.INT32
    print(f"  Value type: {vtype.value} (change with 'type <type>')")
    print(f"  Type 'help' for commands.\n")

    try:
        while True:
            prompt = f"  [{scanner.candidate_count} candidates | scan #{scanner.scan_count}] > "
            try:
                user_input = input(prompt).strip()
            except EOFError:
                break

            if not user_input:
                continue

            parts = user_input.split()
            cmd = parts[0].lower()

            # ─── first scan ─────────────────────────────────
            if cmd == "first" and len(parts) >= 2:
                val = parse_value(parts[1], vtype)
                if val is None:
                    print(f"  Invalid value: {parts[1]}")
                    continue
                print(f"  Scanning for {vtype.value} == {val}...")
                scanner.reset()
                scanner._value_type = vtype

                def progress(p, msg):
                    sys.stdout.write(f"\r  [{p*100:.0f}%] {msg:<60}")
                    sys.stdout.flush()

                scanner.set_progress_callback(progress)
                count = scanner.first_scan(val, vtype, ScanMode.EXACT)
                print(f"\n  Found {count} candidates.\n")

            elif cmd == "first_unknown":
                print(f"  Scanning ALL {vtype.value} values (unknown initial scan)...")
                scanner.reset()
                scanner._value_type = vtype

                def progress(p, msg):
                    sys.stdout.write(f"\r  [{p*100:.0f}%] {msg:<60}")
                    sys.stdout.flush()

                scanner.set_progress_callback(progress)
                count = scanner.first_scan(None, vtype, ScanMode.UNKNOWN)
                print(f"\n  Stored {count} values.\n")

            # ─── next scan (exact) ──────────────────────────
            elif cmd == "next" and len(parts) >= 2:
                val = parse_value(parts[1], vtype)
                if val is None:
                    print(f"  Invalid value: {parts[1]}")
                    continue
                count = scanner.next_scan(val, ScanMode.EXACT)
                print(f"  {count} candidates remaining.\n")

            # ─── filter commands ────────────────────────────
            elif cmd == "changed":
                count = scanner.next_scan(mode=ScanMode.CHANGED)
                print(f"  {count} candidates remaining.\n")

            elif cmd == "unchanged":
                count = scanner.next_scan(mode=ScanMode.UNCHANGED)
                print(f"  {count} candidates remaining.\n")

            elif cmd == "increased":
                count = scanner.next_scan(mode=ScanMode.INCREASED)
                print(f"  {count} candidates remaining.\n")

            elif cmd == "decreased":
                count = scanner.next_scan(mode=ScanMode.DECREASED)
                print(f"  {count} candidates remaining.\n")

            elif cmd == "between" and len(parts) >= 3:
                lo = parse_value(parts[1], vtype)
                hi = parse_value(parts[2], vtype)
                if lo is None or hi is None:
                    print("  Invalid range values.")
                    continue
                count = scanner.next_scan(lo, ScanMode.BETWEEN, hi)
                print(f"  {count} candidates remaining.\n")

            # ─── monitoring ─────────────────────────────────
            elif cmd == "monitor":
                n = int(parts[1]) if len(parts) >= 2 else 20
                do_monitor(scanner, n)

            elif cmd == "refresh":
                count = scanner.refresh()
                print(f"  Refreshed {count} candidates.\n")

            # ─── listing ────────────────────────────────────
            elif cmd == "list":
                n = int(parts[1]) if len(parts) >= 2 else 30
                candidates = scanner.candidates[:n]
                if not candidates:
                    print("  No candidates.")
                else:
                    print(f"\n  {'Address':<20} {'Value':<16} {'Previous':<16}")
                    print(f"  {'─'*20} {'─'*16} {'─'*16}")
                    for c in candidates:
                        addr = f"0x{c.address:016X}"
                        val = format_value(c.value, vtype)
                        prev = format_value(c.previous_value, vtype)
                        print(f"  {addr:<20} {val:<16} {prev:<16}")
                    if scanner.candidate_count > n:
                        print(f"  ... and {scanner.candidate_count - n} more")
                print()

            # ─── value type ─────────────────────────────────
            elif cmd == "type":
                if len(parts) < 2:
                    print(f"  Current: {vtype.value}")
                    print(f"  Options: {', '.join(v.value for v in ValueType if v != ValueType.BYTE_ARRAY)}")
                    continue
                try:
                    new_type = ValueType(parts[1].lower())
                    vtype = new_type
                    print(f"  Value type set to: {vtype.value}\n")
                except ValueError:
                    print(f"  Unknown type: {parts[1]}")
                    print(f"  Options: {', '.join(v.value for v in ValueType if v != ValueType.BYTE_ARRAY)}")

            # ─── persistence ────────────────────────────────
            elif cmd == "save":
                label = parts[1] if len(parts) >= 2 else ""
                do_save(scanner, label)

            elif cmd == "load":
                do_load()

            # ─── misc ───────────────────────────────────────
            elif cmd == "reset":
                scanner.reset()
                print("  Scan data cleared.\n")

            elif cmd == "help":
                print(HELP_TEXT)

            elif cmd in ("quit", "exit", "q"):
                break

            elif cmd == "modules":
                for mod in attacher.modules[:30]:
                    print(f"  0x{mod.base_address:016X}  {mod.size:>10}  {mod.name}")
                print()

            elif cmd == "read":
                # read <hex_address> [size] – raw memory dump
                if len(parts) < 2:
                    print("  Usage: read <address> [size]")
                    continue
                try:
                    addr = int(parts[1], 16) if parts[1].startswith("0x") else int(parts[1])
                    sz = int(parts[2]) if len(parts) >= 3 else 64
                    data = attacher.read(addr, sz)
                    if data:
                        # Hex dump
                        for row_off in range(0, len(data), 16):
                            hex_part = " ".join(f"{b:02X}" for b in data[row_off:row_off + 16])
                            ascii_part = "".join(
                                chr(b) if 32 <= b < 127 else "." for b in data[row_off:row_off + 16]
                            )
                            print(f"  0x{addr + row_off:016X}  {hex_part:<48} {ascii_part}")
                    else:
                        print("  Read failed.")
                except ValueError:
                    print("  Invalid address.")
                print()

            else:
                print(f"  Unknown command: {cmd}. Type 'help' for commands.")

    except KeyboardInterrupt:
        pass
    finally:
        attacher.detach()
        print("\n  Detached. Goodbye.\n")


if __name__ == "__main__":
    main()
