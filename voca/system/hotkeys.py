import re
import subprocess
from typing import Callable, Optional, Union
from pynput import keyboard
import threading
from voca.utils.logging import get_logger


CAPTURING_TRIGGER = threading.Event()
MODIFIERS = {"ctrl", "alt", "shift", "super", "fn"}
# Linux exposes KEY_FN under this XKB keysym when it has a mapping.
FN_KEYSYM = 0x100811D0


class TriggerMatcher:
    """Track physical keys, then match their normalized names."""

    def __init__(self, names, mode, start, finish=None, cancel=None):
        self.names = frozenset(names)
        self.mode = mode
        self.release_toggle = mode == "toggle" and self.names <= MODIFIERS
        self.start, self.finish, self.cancel = start, finish, cancel
        self.down = {}
        self.armed = False
        self.blocked = False

    def event(self, physical, name, pressed):
        if pressed:
            if physical in self.down:
                return
            self.down[physical] = name
        else:
            if physical not in self.down:
                return
            del self.down[physical]
        held = set(self.down.values())
        trigger_held = held & self.names
        if CAPTURING_TRIGGER.is_set():
            if self.armed and self.mode == "hold":
                self.cancel()
            self.armed = False
            self.blocked = bool(trigger_held)
            return
        if trigger_held and held - self.names and not self.armed:
            self.blocked = True
        if self.armed and held - self.names:
            if self.mode == "hold":
                self.cancel()
            self.armed = False
            self.blocked = True
        if self.armed and self.mode == "hold" and held != self.names:
            self.armed = False
            self.blocked = True
            self.finish()
        if not trigger_held:
            if self.armed and self.release_toggle:
                self.start()
            self.armed = self.blocked = False
        elif pressed and not self.armed and not self.blocked and held == self.names:
            self.armed = True
            if not self.release_toggle:
                self.start()

    def reset(self):
        if self.armed and self.mode == "hold":
            self.cancel()
        self.armed = False
        self.down.clear()


# Also match right-side variants
_MODIFIER_VARIANTS = {
    keyboard.Key.ctrl_l: "ctrl",
    keyboard.Key.ctrl_r: "ctrl",
    keyboard.Key.alt_l: "alt",
    keyboard.Key.alt_r: "alt",
    keyboard.Key.alt_gr: "alt",
    keyboard.Key.shift: "shift",
    keyboard.Key.shift_l: "shift",
    keyboard.Key.shift_r: "shift",
    keyboard.Key.cmd: "super",
    keyboard.Key.cmd_l: "super",
    keyboard.Key.cmd_r: "super",
}


class PhysicalKeyboardListener(keyboard.Listener):
    """Use X11 keycodes for identity even if Shift changes the keysym."""

    def __init__(self, dispatch):
        self.dispatch = dispatch
        self.pending_release = None
        self.release_timer = None
        self.event_lock = threading.Lock()
        self.xorg = type(self).__mro__[1].__module__.endswith("._xorg")
        super().__init__(on_press=lambda key, injected=False: self._portable(key, True, injected),
                         on_release=lambda key, injected=False: self._portable(key, False, injected))

    def _portable(self, key, pressed, injected):
        if not injected and not self.xorg:
            self.dispatch(key, pressed)

    def _flush_release(self):
        if self.pending_release:
            physical, _, key = self.pending_release
            self.pending_release = None
            self.dispatch(key, False, physical)

    def _release_after_repeat_window(self):
        with self.event_lock:
            if self.running:
                self._flush_release()

    def _handle_message(self, display, event, injected):
        if injected or event.type not in (2, 3):
            return
        with self.event_lock:
            if self.release_timer:
                self.release_timer.cancel()
            if event.type == 2 and self.pending_release and self.pending_release[:2] == (event.detail, event.time):
                self.pending_release = None
                return
            self._flush_release()
            try:
                key = self._event_to_key(display, event)
            except IndexError:
                key = None
            if event.type == 2:
                self.dispatch(key, True, event.detail)
            else:
                self.pending_release = (event.detail, event.time, key)
                self.release_timer = threading.Timer(0.008, self._release_after_repeat_window)
                self.release_timer.daemon = True
                self.release_timer.start()


class HotkeyManager:
    """Cross-platform global hotkey manager supporting keyboard and mouse inputs."""

    def __init__(self, callback: Callable[[], None], finish=None, cancel=None):
        """
        Initialize hotkey manager.

        Args:
            callback: Function to call when hotkey is pressed
        """
        self.callback = callback
        self.finish = finish or (lambda: None)
        self.cancel = cancel or (lambda: None)
        self.mode = 'toggle'
        self.matchers = []
        self.current_hotkey: Optional[str] = None
        self.keyboard_listener: Optional[Union[keyboard.GlobalHotKeys, keyboard.Listener]] = None
        self._xinput_procs: list[subprocess.Popen] = []
        self._xinput_stop = threading.Event()
        self.is_listening = False
        self._lock = threading.Lock()
        self._is_mouse_hotkey = False
        self._extra_hotkeys: dict = {}  # key_combination -> callback
        self.logger = get_logger(__name__)

    @staticmethod
    def parse_binding(binding):
        aliases = {"control": "ctrl", "cmd": "super", "command": "super",
                   "win": "super", "option": "alt", "esc": "escape", "return": "enter"}
        parts = binding.lower().replace(" ", "").split("+")
        names = [aliases.get(part, part) for part in parts]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Choose a key or a chord with distinct keys.")
        for name in names:
            if name in MODIFIERS or len(name) == 1 or name in keyboard.Key.__members__:
                continue
            if len(names) == 1 and re.fullmatch(r"button[0-9]+", name) and int(name[6:]) >= 8:
                continue
            raise ValueError(f"Unsupported trigger key: {name}")
        return names

    def set_hotkey(self, key_combination: str, mode: str = "toggle") -> None:
        """
        Set the hotkey combination.

        Args:
            key_combination: Hotkey string like 'ctrl+shift+v', 'f12', or 'button9'
        """
        names = self.parse_binding(key_combination)
        if mode not in {"toggle", "hold"}:
            raise ValueError("Choose Hold to record or Press to toggle.")
        if self.is_listening:
            self.stop_listening()
        with self._lock:
            self.current_hotkey = key_combination
            self.mode = mode
            self._is_mouse_hotkey = self._is_mouse_button(names[0])
            self.matchers = [TriggerMatcher(names, mode, self._on_hotkey_pressed,
                                            self.finish, self.cancel)]
            for combo, callback in self._extra_hotkeys.items():
                extra = self.parse_binding(combo)
                if set(extra) != set(names):
                    self.matchers.append(TriggerMatcher(extra, "toggle", callback))
            self.keyboard_listener = PhysicalKeyboardListener(self._keyboard_event)

    def _keyboard_event(self, key, pressed, physical=None):
        name = _MODIFIER_VARIANTS.get(key)
        if name is None:
            if getattr(key, "vk", None) == FN_KEYSYM:
                name = "fn"
            else:
                name = getattr(key, "name", None) or getattr(key, "char", None)
                if name:
                    name = name.lower()
                    if len(name) == 1 and 1 <= ord(name) <= 26:
                        name = chr(ord(name) + 96)
        if physical is None:
            physical = name if isinstance(key, keyboard.KeyCode) and name else getattr(key, "vk", None)
        if physical is None:
            physical = name if isinstance(key, keyboard.KeyCode) and name else key
        self._dispatch(("keyboard", physical), name, pressed)

    def _dispatch(self, physical, name, pressed):
        for matcher in self.matchers:
            matcher.event(physical, name, pressed)

    def start_listening(self) -> None:
        """Start listening for hotkey presses."""
        with self._lock:
            if self.is_listening:
                return

            if not (self.keyboard_listener or self._is_mouse_hotkey):
                raise RuntimeError("No hotkey set. Call set_hotkey() first.")

            try:
                if self._is_mouse_hotkey:
                    self._start_xinput_listeners()
                if self.keyboard_listener:
                    self.keyboard_listener.start()

                self.is_listening = True

            except Exception as e:
                raise RuntimeError(f"Failed to start hotkey listener: {e}")

    def stop_listening(self) -> None:
        """Stop listening for hotkey presses."""
        with self._lock:
            if not self.is_listening:
                return

            try:
                if self.keyboard_listener:
                    self.keyboard_listener.stop()
                    self.keyboard_listener = None
                self._stop_xinput_listeners()
                for matcher in self.matchers:
                    matcher.reset()
            except Exception as e:
                self.logger.error(f"Error stopping listener: {e}")

        self.is_listening = False

    def _on_hotkey_pressed(self) -> None:
        """Internal callback when hotkey is pressed."""
        try:
            self.callback()
        except Exception as e:
            self.logger.error(f"Error in hotkey callback: {e}")

    def is_hotkey_listening(self) -> bool:
        """Check if currently listening for hotkeys."""
        return self.is_listening

    def get_current_hotkey(self) -> Optional[str]:
        """Get the current hotkey combination."""
        return self.current_hotkey

    def _is_mouse_button(self, key_combination: str) -> bool:
        """Check if the hotkey is a mouse button."""
        return key_combination.startswith("button") and key_combination[6:].isdigit()

    def register_hotkey(self, key_combination: str, callback: Callable[[], None]) -> None:
        """
        Register an additional hotkey with its own callback.
        Must be called before start_listening() or will require restart.

        Args:
            key_combination: Hotkey string like 'ctrl+alt+space'
            callback: Function to call when this hotkey is pressed
        """
        self._extra_hotkeys[key_combination] = callback

    def _start_xinput_listeners(self) -> None:
        """Spawn xinput test processes on all pointer devices to detect the target button."""
        self._xinput_stop.clear()
        self._xinput_procs = []
        try:
            out = subprocess.check_output(["xinput", "list", "--short"], text=True)
        except FileNotFoundError:
            self.logger.error("xinput not found — mouse button hotkeys require xinput")
            return
        for line in out.splitlines():
            if "slave  pointer" not in line or "Virtual" in line or "XTEST" in line:
                continue
            m = re.search(r"id=(\d+)", line)
            if not m:
                continue
            dev_id = m.group(1)
            proc = subprocess.Popen(
                ["xinput", "test", dev_id],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            self._xinput_procs.append(proc)
            t = threading.Thread(
                target=self._read_xinput, args=(proc, dev_id), daemon=True
            )
            t.start()

    def _read_xinput(self, proc: subprocess.Popen, device) -> None:
        """Read xinput output and fire callback on target button press."""
        try:
            for line in proc.stdout:
                if self._xinput_stop.is_set():
                    break
                match = re.fullmatch(r"button (press|release)\s+(\d+)", line.strip())
                if match:
                    action, button = match.groups()
                    if f"button{button}" == self.current_hotkey.lower().strip():
                        self._dispatch(("mouse", device, button), f"button{button}", action == "press")
        except Exception:
            pass

    def _stop_xinput_listeners(self) -> None:
        """Terminate all xinput test processes."""
        self._xinput_stop.set()
        for proc in self._xinput_procs:
            try:
                proc.terminate()
            except Exception:
                pass
        self._xinput_procs = []

    @staticmethod
    def get_suggested_hotkeys() -> list:
        """Get a list of suggested hotkey combinations that rarely conflict."""
        return [
            "f12",  # Function keys rarely used by apps
            "f11",  # Another safe function key
            "button9",  # Mouse side button (forward)
            "button8",  # Mouse side button (back)
            "ctrl+f12",  # Modified function keys
            "alt+f12",  # Even safer with modifier
            "shift+f12",  # Alternative modifier
            "ctrl+alt+q",  # Uncommon combination
        ]
