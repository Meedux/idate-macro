"""
iDate Revival Automation GUI  —  Simplified Edition

One-click automation: the program auto-detects the iDate process
and window, attaches via memory reading, and starts playing.

Usage:
  1.  Launch iDate Revival (the game)
  2.  Run this program  (it will auto-elevate to admin)
  3.  Click  **▶ START**  (or press F12)
  4.  Switch to the game window — automation begins instantly
  5.  Press F12 again (or click STOP) to stop
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import os
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path

import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ──────────── Win32 constants ────────────
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x80000
WS_EX_TRANSPARENT = 0x20
WS_EX_TOOLWINDOW = 0x80
LWA_COLORKEY = 0x1
SW_SHOW = 5
VK_F12 = 0x7B
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100


# ──────────────────── helpers ────────────────────

def is_admin() -> bool:
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        return False


def run_as_admin():
    """Re-launch as admin."""
    if sys.platform == "win32":
        script = os.path.abspath(sys.argv[0])
        params = " ".join([f'"{a}"' for a in sys.argv[1:]])
        ret = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, f'"{script}" {params}', None, SW_SHOW
        )
        if ret > 32:
            sys.exit(0)
        else:
            sys.exit(1)


def enumerate_windows() -> list[tuple[int, str]]:
    """Return all visible windows as (hwnd, title) pairs."""
    windows: list[tuple[int, str]] = []

    def callback(hwnd, _):
        if ctypes.windll.user32.IsWindowVisible(hwnd):
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value
                if title and title.strip():
                    windows.append((hwnd, title))
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
    ctypes.windll.user32.EnumWindows(WNDENUMPROC(callback), 0)
    return windows


def get_window_rect(hwnd: int) -> dict | None:
    rect = ctypes.wintypes.RECT()
    if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return {
            "left": rect.left,
            "top": rect.top,
            "width": rect.right - rect.left,
            "height": rect.bottom - rect.top,
        }
    return None


def focus_window(hwnd: int) -> bool:
    try:
        ctypes.windll.user32.ShowWindow(hwnd, 9)
        time.sleep(0.05)
        ctypes.windll.user32.SetForegroundWindow(hwnd)
        return True
    except Exception:
        return False


def find_idate_window() -> tuple[int, str] | None:
    """Try to auto-detect the iDate game window."""
    for hwnd, title in enumerate_windows():
        lower = title.lower()
        # Match "iDate" as a standalone window title (not VS Code or other editors)
        if any(skip in lower for skip in ("visual studio", "vscode", "code -", "notepad", "explorer")):
            continue
        if "idate" in lower or "i-date" in lower or "i date" in lower:
            return (hwnd, title)
    return None


def find_idate_process() -> tuple[int, str] | None:
    """Try to auto-detect the iDate process (executable name contains 'idate')."""
    from .memory.process import enumerate_processes
    for p in enumerate_processes():
        name_lower = p.name.lower()
        # Match idate.exe, iDate.exe, i-date.exe etc.
        if ("idate" in name_lower or "i-date" in name_lower) and name_lower.endswith(".exe"):
            return (p.pid, p.name)
    return None


# ──────────────────── widgets ────────────────────


class LogPanel(ctk.CTkFrame):
    """Compact log panel."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        header = ctk.CTkFrame(self, height=28)
        header.pack(fill="x", padx=5, pady=(5, 3))
        header.pack_propagate(False)
        ctk.CTkLabel(header, text="Log", font=("Segoe UI", 11, "bold")).pack(side="left")
        ctk.CTkButton(header, text="Clear", width=50, height=22,
                       command=self._clear).pack(side="right")

        self.textbox = ctk.CTkTextbox(self, font=("Consolas", 10), wrap="word",
                                       state="disabled")
        self.textbox.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        tb = self.textbox._textbox
        tb.tag_configure("time", foreground="#888888")
        tb.tag_configure("info", foreground="#00BFFF")
        tb.tag_configure("key", foreground="#00FF00")
        tb.tag_configure("error", foreground="#FF4444")
        tb.tag_configure("detect", foreground="#FFD700")
        tb.tag_configure("ok", foreground="#00FF88")
        self._max_lines = 400

    def log(self, message: str, tag: str = "info"):
        self.textbox.configure(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        tb = self.textbox._textbox
        tb.insert("end", f"[{ts}] ", "time")
        tb.insert("end", f"{message}\n", tag)
        lines = int(tb.index("end-1c").split(".")[0])
        if lines > self._max_lines:
            tb.delete("1.0", f"{lines - self._max_lines}.0")
        tb.see("end")
        self.textbox.configure(state="disabled")

    def _clear(self):
        self.textbox.configure(state="normal")
        self.textbox._textbox.delete("1.0", "end")
        self.textbox.configure(state="disabled")


class GlobalHotkey:
    """Global F12 hotkey via low-level keyboard hook."""

    def __init__(self, callback):
        self.callback = callback
        self._hook = None
        self._thread: threading.Thread | None = None
        self._running = False
        self.HOOKPROC = ctypes.CFUNCTYPE(
            ctypes.c_longlong, ctypes.c_int, ctypes.c_ulonglong, ctypes.c_longlong
        )
        self._hook_proc = self.HOOKPROC(self._keyboard_proc)
        fn = ctypes.windll.user32.CallNextHookEx
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong, ctypes.c_longlong]
        fn.restype = ctypes.c_longlong
        self._CallNextHookEx = fn

    def _keyboard_proc(self, nCode, wParam, lParam):
        try:
            if nCode >= 0 and wParam == WM_KEYDOWN:
                vk = ctypes.c_ulong.from_address(lParam).value & 0xFF
                if vk == VK_F12:
                    threading.Thread(target=self.callback, daemon=True).start()
        except Exception:
            pass
        return self._CallNextHookEx(self._hook, nCode, wParam, lParam)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._hook_thread, daemon=True)
        self._thread.start()

    def _hook_thread(self):
        self._hook = ctypes.windll.user32.SetWindowsHookExW(
            WH_KEYBOARD_LL, self._hook_proc, None, 0
        )
        if not self._hook:
            return
        msg = ctypes.wintypes.MSG()
        while self._running:
            ret = ctypes.windll.user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if ret in (0, -1):
                break
            ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
            ctypes.windll.user32.DispatchMessageW(ctypes.byref(msg))
        if self._hook:
            ctypes.windll.user32.UnhookWindowsHookEx(self._hook)
            self._hook = None

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            ctypes.windll.user32.PostThreadMessageW(
                self._thread.ident, 0x0012, 0, 0
            )


# ─────────────────── main app ───────────────────


class IdateApp(ctk.CTk):
    """Simplified iDate Revival automation GUI."""

    def __init__(self):
        super().__init__()

        admin_tag = " [ADMIN]" if is_admin() else ""
        self.title(f"iDate Revival{admin_tag}")
        self.geometry("600x520")
        self.minsize(500, 420)

        # State
        self._running = False
        self._stop_event = threading.Event()
        self._selected_hwnd: int | None = None
        self._selected_pid: int | None = None
        self._windows: dict[str, int] = {}
        self._process_list: list[tuple[int, str]] = []
        self._memory_detector = None
        self._pygame_overlay = None

        self._hotkey = GlobalHotkey(self._toggle_from_hotkey)
        self._hotkey.start()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Auto-detect on startup
        self.after(100, self._auto_detect)

    # ─────────────── UI ─────────────────

    def _build_ui(self):
        # ── Title / status banner ──
        banner = ctk.CTkFrame(self, fg_color="#1a1a2e", corner_radius=8)
        banner.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            banner, text="🎵 iDate Revival", font=("Segoe UI", 18, "bold"),
        ).pack(side="left", padx=15, pady=8)

        self.status_label = ctk.CTkLabel(
            banner, text="Initializing...", text_color="#aaaaaa",
            font=("Segoe UI", 11),
        )
        self.status_label.pack(side="right", padx=15, pady=8)

        # ── Target selection (Window + Process on same row) ──
        sel_frame = ctk.CTkFrame(self)
        sel_frame.pack(fill="x", padx=10, pady=5)

        # Window
        win_col = ctk.CTkFrame(sel_frame, fg_color="transparent")
        win_col.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkLabel(win_col, text="Window", font=("Segoe UI", 10)).pack(anchor="w")
        self.window_combo = ctk.CTkComboBox(
            win_col, values=["(auto-detect)"], width=220,
            command=self._on_window_select,
        )
        self.window_combo.pack(fill="x")

        # Process
        proc_col = ctk.CTkFrame(sel_frame, fg_color="transparent")
        proc_col.pack(side="left", fill="x", expand=True, padx=(5, 0))
        ctk.CTkLabel(proc_col, text="Process", font=("Segoe UI", 10)).pack(anchor="w")
        self.process_combo = ctk.CTkComboBox(
            proc_col, values=["(auto-detect)"], width=220,
            command=self._on_process_select,
        )
        self.process_combo.pack(fill="x")

        # Refresh button
        ctk.CTkButton(
            sel_frame, text="↻ Refresh", width=80,
            command=self._refresh_all,
        ).pack(side="right", padx=(10, 0), pady=(15, 0))

        # ── Big START / STOP buttons ──
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)

        self.start_btn = ctk.CTkButton(
            btn_frame, text="▶  START  (F12)", height=50,
            font=("Segoe UI", 16, "bold"),
            fg_color="#228B22", hover_color="#006400",
            command=self._start,
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.stop_btn = ctk.CTkButton(
            btn_frame, text="■  STOP  (F12)", height=50,
            font=("Segoe UI", 16, "bold"),
            fg_color="#B22222", hover_color="#8B0000",
            command=self._stop, state="disabled",
        )
        self.stop_btn.pack(side="left", fill="x", expand=True, padx=(5, 0))

        # ── Live stats bar ──
        stats = ctk.CTkFrame(self, height=30)
        stats.pack(fill="x", padx=10, pady=(0, 5))
        stats.pack_propagate(False)

        self.fps_label = ctk.CTkLabel(stats, text="FPS: --", font=("Consolas", 10))
        self.fps_label.pack(side="left", padx=10)
        self.score_label = ctk.CTkLabel(stats, text="Score: 0  |  Combo: 0",
                                         font=("Consolas", 10))
        self.score_label.pack(side="right", padx=10)

        # ── Log panel ──
        self.log_panel = LogPanel(self)
        self.log_panel.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # ─────────── auto-detect ─────────────

    def _auto_detect(self):
        """Auto-detect iDate window and process on startup."""
        self.log_panel.log(
            "Admin ✓" if is_admin() else "⚠ Not admin — Run as Administrator for best results!",
            "ok" if is_admin() else "error",
        )

        # Refresh lists
        self._refresh_all(silent=True)

        # Try auto-detect
        found_window = False
        found_process = False

        # Window
        result = find_idate_window()
        if result:
            hwnd, title = result
            self._selected_hwnd = hwnd
            display = title[:50]
            self.window_combo.set(display)
            self.log_panel.log(f"Window auto-detected: {display}", "ok")
            found_window = True

        # Process
        result = find_idate_process()
        if result:
            pid, name = result
            self._selected_pid = pid
            self.process_combo.set(f"{name} ({pid})")
            self.log_panel.log(f"Process auto-detected: {name} (PID {pid})", "ok")
            found_process = True

        if found_window and found_process:
            self.status_label.configure(text="Ready — Press START or F12", text_color="#00FF88")
            self.log_panel.log("✓ Ready! Press START or F12 to begin.", "ok")
        elif found_process:
            self.status_label.configure(text="Ready — Window not found, select manually",
                                         text_color="#FFD700")
            self.log_panel.log("Process found but window not detected. Select window manually.", "info")
        else:
            self.status_label.configure(text="Launch iDate first, then click Refresh",
                                         text_color="#FF8888")
            self.log_panel.log("iDate not detected. Launch the game first, then click Refresh.", "info")

        self.log_panel.log("Press F12 anytime to Start/Stop", "info")

    # ─────────── refresh ─────────────────

    def _refresh_all(self, silent: bool = False):
        """Refresh both window and process lists."""
        self._refresh_windows(silent)
        self._refresh_processes(silent)

    def _refresh_windows(self, silent: bool = False):
        windows = enumerate_windows()
        self._windows.clear()
        titles: list[str] = []
        for hwnd, title in windows:
            display = title[:50] + "..." if len(title) > 50 else title
            self._windows[display] = hwnd
            titles.append(display)
        self.window_combo.configure(values=titles or ["No windows found"])
        if not silent:
            self.log_panel.log(f"Found {len(titles)} windows", "info")

    def _refresh_processes(self, silent: bool = False):
        from .memory.process import enumerate_processes as enum_procs
        procs = enum_procs()
        skip = (
            "svchost", "csrss", "services", "lsass", "wininit",
            "winlogon", "dwm", "smss", "registry", "fontdrvhost",
            "conhost", "sihost", "runtimebroker", "system",
        )
        self._process_list = [
            (p.pid, p.name) for p in procs
            if p.pid > 4 and not p.name.lower().startswith(skip)
        ]
        self._process_list.sort(key=lambda x: x[1].lower())
        vals = [f"{nm} ({pid})" for pid, nm in self._process_list]
        self.process_combo.configure(values=vals or ["No processes found"])
        if not silent:
            self.log_panel.log(f"Found {len(self._process_list)} processes", "info")

    # ─────────── selection callbacks ─────

    def _on_window_select(self, selection: str):
        self._selected_hwnd = self._windows.get(selection)

    def _on_process_select(self, selection: str):
        for pid, name in self._process_list:
            if f"{name} ({pid})" == selection:
                self._selected_pid = pid
                self.log_panel.log(f"Selected: {name} (PID {pid})", "info")
                return

    # ─────────── start / stop ────────────

    def _toggle_from_hotkey(self):
        if self._running:
            self.after(0, self._stop)
        else:
            self.after(0, self._start)

    def _start(self):
        if self._running:
            return

        # Auto-detect if nothing selected
        if not self._selected_pid:
            result = find_idate_process()
            if result:
                self._selected_pid = result[0]
                self.process_combo.set(f"{result[1]} ({result[0]})")
                self.log_panel.log(f"Auto-attached to {result[1]}", "ok")
            else:
                self.log_panel.log("No iDate process found! Launch the game first.", "error")
                self.status_label.configure(text="Launch iDate first!", text_color="#FF4444")
                return

        if not self._selected_hwnd:
            result = find_idate_window()
            if result:
                self._selected_hwnd = result[0]
                self.log_panel.log(f"Auto-detected window: {result[1][:40]}", "ok")

        self._running = True
        self._stop_event.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="Running...", text_color="#00FF88")
        self.log_panel.log("▶ Starting automation...", "ok")

        if self._selected_hwnd:
            focus_window(self._selected_hwnd)

        thread = threading.Thread(target=self._main_loop, daemon=True)
        thread.start()

    def _stop(self):
        if not self._running:
            return
        self._stop_event.set()
        self._running = False

        if self._pygame_overlay:
            self._pygame_overlay.stop()
            self._pygame_overlay = None

        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Stopped", text_color="#aaaaaa")
        self.log_panel.log("■ Stopped", "info")

    # ─────────── main loop ───────────────

    def _main_loop(self):
        """Detection loop — pure memory reading, no screen capture."""
        try:
            from .input.sendinput_driver import scan_key_press
            from .overlay.win32_overlay import Win32Overlay
            from .memory.memory_detector import MemoryDetector

            detector = MemoryDetector(
                cooldown_ms=30,
                perfect_window_ms=50,
            )

            def log_cb(msg):
                self.after(0, lambda m=msg: self.log_panel.log(m, "detect"))
            detector.set_log_callback(log_cb)

            # Attach
            if not detector.attach_to_process(self._selected_pid):
                self.after(0, lambda: self.log_panel.log(
                    "Failed to attach! Run as Administrator.", "error"
                ))
                return

            self.after(0, lambda: self.log_panel.log(
                f"✓ Attached to {detector.process_name}", "ok"
            ))

            self._memory_detector = detector

            # Overlay
            overlay = None
            if self._selected_hwnd:
                try:
                    overlay = Win32Overlay(self._selected_hwnd, offset_x=0, offset_y=0)
                    overlay.start()
                    time.sleep(0.2)
                    self._pygame_overlay = overlay
                except Exception as e:
                    self.after(0, lambda err=str(e): self.log_panel.log(
                        f"Overlay failed (non-fatal): {err}", "error"
                    ))

            fps_time = time.perf_counter()
            fps_count = 0
            action_count = 0
            log_cooldown: dict[str, float] = {}
            last_playing: bool | None = None
            last_state_msg = ""

            while not self._stop_event.is_set():
                det_result = detector.detect(frame=None)
                gs = det_result.game_state
                # Get keys after detect() so state is fresh
                keys = []
                now_t = time.time()
                for k in det_result.icons_in_hit_zone:
                    mapped = k if k in ('left','right','up','down','space') else k
                    last = detector._last_press.get(mapped, 0)
                    if now_t - last >= detector._cooldown:
                        keys.append(mapped)
                        detector._last_press[mapped] = now_t

                # State transition logging
                if gs:
                    if gs.is_playing != last_playing:
                        last_playing = gs.is_playing
                        if gs.is_playing:
                            self.after(0, lambda: self.log_panel.log(
                                "✓ Song playing — detecting notes!", "ok"
                            ))
                        else:
                            self.after(0, lambda: self.log_panel.log(
                                "Waiting for song to start...", "info"
                            ))

                    # Game object status (log once)
                    if detector._game_base:
                        state_msg = f"obj=0x{detector._game_base:08X}"
                        if state_msg != last_state_msg:
                            last_state_msg = state_msg
                            self.after(0, lambda s=state_msg: self.log_panel.log(
                                f"Game object: {s}", "detect"
                            ))

                # Press keys
                now_t = time.time()
                for key in keys:
                    scan_key_press(key, duration_ms=12)
                    action_count += 1
                    last = log_cooldown.get(key, 0)
                    if now_t - last >= 0.12:
                        self.after(0, lambda k=key: self.log_panel.log(
                            f">>> {k.upper()} <<<", "key"
                        ))
                        log_cooldown[key] = now_t

                # Overlay
                if overlay:
                    overlay.update(self._make_overlay_result(det_result))

                # FPS
                fps_count += 1
                now = time.perf_counter()
                if now - fps_time >= 1.0:
                    fps = fps_count / (now - fps_time)
                    inf = det_result.inference_time_ms
                    score = gs.score if gs else 0
                    combo = gs.combo if gs else 0
                    notes = gs.active_notes if gs else 0
                    lanes = gs.lane_active if gs else [False]*4
                    lane_str = "".join("■" if l else "□" for l in lanes)
                    fps_count = 0
                    fps_time = now
                    self.after(0, lambda f=fps, ms=inf, n=notes, ls=lane_str: (
                        self.fps_label.configure(
                            text=f"FPS: {f:.0f} | {ms:.1f}ms | {n} notes {ls}"
                        ),
                    ))
                    self.after(0, lambda s=score, c=combo, a=action_count: (
                        self.score_label.configure(
                            text=f"Score: {s}  |  Combo: {c}  |  Keys: {a}"
                        ),
                    ))

                time.sleep(0.0005)

        except Exception as e:
            import traceback
            self.after(0, lambda err=str(e): self.log_panel.log(
                f"Error: {err}", "error"
            ))
        finally:
            self._running = False
            if self._memory_detector:
                self._memory_detector.detach()
                self._memory_detector = None
            if self._pygame_overlay:
                self._pygame_overlay.stop()
                self._pygame_overlay = None
            self.after(0, lambda: (
                self.start_btn.configure(state="normal"),
                self.stop_btn.configure(state="disabled"),
                self.status_label.configure(text="Stopped", text_color="#aaaaaa"),
            ))

    @staticmethod
    def _make_overlay_result(det_result):
        """Convert MemoryDetectionResult → overlay-compatible object."""
        class _R:
            def __init__(self):
                self.indicator_pos = None
                self.indicator_confidence = 0.0
                self.icons: list[tuple] = []
                self.aligned_icons: list[str] = []
                self.track_y = None
                self.track_height = None
                self.hit_zone_x = None
                self.hit_zone_width = 45
                self.game_state = None

        r = _R()
        if hasattr(det_result, "indicator_pos"):
            r.indicator_pos = det_result.indicator_pos
        if hasattr(det_result, "indicator_confidence"):
            r.indicator_confidence = det_result.indicator_confidence

        if hasattr(det_result, "icons"):
            for icon in det_result.icons:
                if hasattr(icon, "icon_type"):
                    r.icons.append((icon.icon_type, icon.x, icon.y, icon.confidence))
                else:
                    r.icons.append(icon)

        if hasattr(det_result, "icons_in_hit_zone"):
            r.aligned_icons = list(det_result.icons_in_hit_zone)
        if hasattr(det_result, "game_state"):
            r.game_state = det_result.game_state
        return r

    # ─────────── cleanup ─────────────────

    def _on_close(self):
        self._stop_event.set()
        self._running = False
        self._hotkey.stop()
        if self._pygame_overlay:
            self._pygame_overlay.stop()
            self._pygame_overlay = None
        self.destroy()


# ──────────────── entry point ────────────────


def main():
    if not is_admin():
        print("Requesting admin privileges...")
        run_as_admin()
        return
    app = IdateApp()
    app.mainloop()


if __name__ == "__main__":
    main()
