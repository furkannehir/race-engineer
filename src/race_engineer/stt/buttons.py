"""Digital push-to-talk bindings for keyboard, mouse, wheels, and controllers."""

import importlib
import os
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from race_engineer.config import PttBindingConfig, SttConfig
from race_engineer.core.speech_input import SpeechInputError

_MOUSE_LABELS = {
    0x01: "Mouse left",
    0x02: "Mouse right",
    0x04: "Mouse middle",
    0x05: "Mouse back",
    0x06: "Mouse forward",
}
ControlType = Literal["button", "hat"]
HatValue = tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]]


class ButtonInput(Protocol):
    def is_down(self) -> bool: ...

    def close(self) -> None: ...


def legacy_virtual_key(name: str) -> int:
    special = {"SPACE": 0x20, "RCTRL": 0xA3, "RALT": 0xA5}
    if name in special:
        return special[name]
    if name.startswith("F") and name[1:].isdigit() and 1 <= int(name[1:]) <= 24:
        return 0x70 + int(name[1:]) - 1
    raise ValueError("unsupported push-to-talk key")


def legacy_binding(name: str) -> PttBindingConfig:
    return PttBindingConfig(kind="keyboard", code=legacy_virtual_key(name), label=name)


def effective_binding(config: SttConfig) -> PttBindingConfig:
    return config.ptt_binding or legacy_binding(config.ptt_key)


def binding_label(config: SttConfig) -> str:
    return effective_binding(config).label


def compact_binding_label(binding: PttBindingConfig) -> str:
    if binding.kind == "joystick":
        if binding.control == "hat":
            directions = {
                (-1, 0): "←",
                (1, 0): "→",
                (0, 1): "↑",
                (0, -1): "↓",
                (-1, 1): "↖",
                (1, 1): "↗",
                (-1, -1): "↙",
                (1, -1): "↘",
            }
            assert binding.hat_value is not None
            return f"H{binding.code + 1}{directions[binding.hat_value]}"
        return f"B{binding.code + 1}"
    if binding.kind == "mouse":
        return {0x01: "M1", 0x02: "M2", 0x04: "M3", 0x05: "M4", 0x06: "M5"}[
            binding.code
        ]
    aliases = {
        "RIGHT CTRL": "RCTRL",
        "RIGHT ALT": "RALT",
        "BACKSPACE": "BKSP",
        "PAGE UP": "PGUP",
        "PAGE DOWN": "PGDN",
        "RETURN": "ENTER",
    }
    normalized = aliases.get(binding.label.upper(), binding.label.upper().replace(" ", ""))
    return normalized if len(normalized) <= 7 else f"K{binding.code:02X}"


def mouse_binding(code: int) -> PttBindingConfig:
    return PttBindingConfig(kind="mouse", code=code, label=_MOUSE_LABELS[code])


def _load_pygame() -> Any:
    # SDL normally requires a video event queue. Its dummy driver gives us that queue
    # without opening a window, while the explicit hint keeps wheel events flowing in-game.
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        return importlib.import_module("pygame")
    except (ImportError, OSError) as error:
        raise SpeechInputError("push_to_talk_controller_dependency_missing") from error


@dataclass(frozen=True)
class _OpenJoystick:
    index: int
    guid: str
    name: str
    handle: Any


class _SdlButtonSystem:
    def __init__(self, pygame_module: Any | None = None) -> None:
        self._pygame = pygame_module or _load_pygame()
        self._devices: list[_OpenJoystick] = []
        self._owns_display = not self._pygame.display.get_init()
        self._owns_joystick = not self._pygame.joystick.get_init()
        try:
            if self._owns_display:
                self._pygame.display.init()
            if self._pygame.display.get_surface() is None:
                hidden = getattr(self._pygame, "HIDDEN", 0)
                self._pygame.display.set_mode((1, 1), flags=hidden)
            if self._owns_joystick:
                self._pygame.joystick.init()
            self._pump()
            for index in range(self._pygame.joystick.get_count()):
                handle = self._pygame.joystick.Joystick(index)
                handle.init()
                self._devices.append(
                    _OpenJoystick(index, str(handle.get_guid()), str(handle.get_name()), handle)
                )
        except Exception as error:
            self.close()
            raise SpeechInputError("push_to_talk_controller_open_failed") from error

    @property
    def devices(self) -> tuple[_OpenJoystick, ...]:
        return tuple(self._devices)

    def _pump(self) -> None:
        self._pygame.event.pump()

    def pressed(self) -> set[tuple[int, str, int, int, int]]:
        try:
            self._pump()
            buttons = {
                (device.index, "button", button, 0, 0)
                for device in self._devices
                for button in range(int(device.handle.get_numbuttons()))
                if device.handle.get_button(button)
            }
            hats = {
                (device.index, "hat", hat, int(value[0]), int(value[1]))
                for device in self._devices
                for hat in range(int(device.handle.get_numhats()))
                if (value := device.handle.get_hat(hat)) != (0, 0)
            }
            return buttons | hats
        except Exception as error:
            raise SpeechInputError("push_to_talk_controller_read_failed") from error

    def resolve(self, binding: PttBindingConfig) -> _OpenJoystick:
        exact = [
            device
            for device in self._devices
            if device.index == binding.device_index
            and device.guid.lower() == (binding.device_guid or "").lower()
            and device.name == binding.device_name
        ]
        if exact:
            return exact[0]
        matches = [
            device
            for device in self._devices
            if device.guid.lower() == (binding.device_guid or "").lower()
            and device.name == binding.device_name
        ]
        if len(matches) == 1:
            return matches[0]
        reason = (
            "push_to_talk_device_ambiguous"
            if len(matches) > 1
            else "push_to_talk_device_unavailable"
        )
        raise SpeechInputError(reason)

    def close(self) -> None:
        for device in self._devices:
            with suppress(Exception):
                device.handle.quit()
        self._devices.clear()
        try:
            if self._owns_joystick:
                self._pygame.joystick.quit()
            if self._owns_display:
                self._pygame.display.quit()
        except Exception:
            pass


class JoystickButtonInput:
    """Held-state reader for one device-qualified SDL joystick button."""

    def __init__(
        self, binding: PttBindingConfig, *, pygame_module: Any | None = None
    ) -> None:
        self._system = _SdlButtonSystem(pygame_module)
        try:
            self._device = self._system.resolve(binding)
            self._button = binding.code
            self._control = binding.control
            self._hat_value = binding.hat_value
            count = (
                self._device.handle.get_numbuttons()
                if self._control == "button"
                else self._device.handle.get_numhats()
            )
            if self._button >= int(count):
                raise SpeechInputError("push_to_talk_button_unavailable")
        except Exception:
            self._system.close()
            raise

    def is_down(self) -> bool:
        try:
            self._system._pump()
            if self._control == "hat":
                return tuple(self._device.handle.get_hat(self._button)) == self._hat_value
            return bool(self._device.handle.get_button(self._button))
        except Exception as error:
            raise SpeechInputError("push_to_talk_controller_read_failed") from error

    def close(self) -> None:
        self._system.close()


class JoystickBindingDetector:
    """Detect rising button edges while deliberately ignoring every analog axis."""

    def __init__(self, *, pygame_module: Any | None = None) -> None:
        self._system = _SdlButtonSystem(pygame_module)
        self._previous = self._system.pressed()

    def poll(self) -> PttBindingConfig | None:
        current = self._system.pressed()
        newly_pressed = sorted(current - self._previous)
        self._previous = current
        if not newly_pressed:
            return None
        index, raw_control, code, x, y = newly_pressed[0]
        control: ControlType = "hat" if raw_control == "hat" else "button"
        device = next(item for item in self._system.devices if item.index == index)
        hat_value = cast(HatValue, (x, y)) if control == "hat" else None
        directions = {
            (-1, 0): "left",
            (1, 0): "right",
            (0, 1): "up",
            (0, -1): "down",
            (-1, 1): "up-left",
            (1, 1): "up-right",
            (-1, -1): "down-left",
            (1, -1): "down-right",
        }
        control_label = (
            f"Hat {code + 1} {directions[hat_value]}"
            if hat_value is not None
            else f"Button {code + 1}"
        )
        return PttBindingConfig(
            kind="joystick",
            code=code,
            label=f"{device.name} · {control_label}",
            control=control,
            hat_value=hat_value,
            device_guid=device.guid,
            device_name=device.name,
            device_index=device.index,
        )

    def close(self) -> None:
        self._system.close()
