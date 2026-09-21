import pytest

from race_engineer.config import PttBindingConfig
from race_engineer.core.speech_input import SpeechInputError
from race_engineer.stt.buttons import JoystickBindingDetector, JoystickButtonInput

GUID = "0123456789abcdef0123456789abcdef"


class FakeHandle:
    def __init__(self, name="Wheel", guid=GUID, buttons=3, hats=1):
        self.name, self.guid = name, guid
        self.buttons = [False] * buttons
        self.hats = [(0, 0)] * hats
        self.closed = False

    def init(self):
        pass

    def quit(self):
        self.closed = True

    def get_guid(self):
        return self.guid

    def get_name(self):
        return self.name

    def get_numbuttons(self):
        return len(self.buttons)

    def get_button(self, index):
        return self.buttons[index]

    def get_numhats(self):
        return len(self.hats)

    def get_hat(self, index):
        return self.hats[index]


class FakeDisplay:
    def __init__(self):
        self.initialized = False
        self.surface = None

    def get_init(self):
        return self.initialized

    def init(self):
        self.initialized = True

    def get_surface(self):
        return self.surface

    def set_mode(self, size, flags=0):
        self.surface = (size, flags)

    def quit(self):
        self.initialized = False


class FakeJoystickApi:
    def __init__(self, handles):
        self.handles = handles
        self.initialized = False

    def get_init(self):
        return self.initialized

    def init(self):
        self.initialized = True

    def quit(self):
        self.initialized = False

    def get_count(self):
        return len(self.handles)

    def Joystick(self, index):
        return self.handles[index]


class FakeEventQueue:
    def __init__(self):
        self.pumps = 0

    def pump(self):
        self.pumps += 1


class FakePygame:
    HIDDEN = 1

    def __init__(self, handles):
        self.display = FakeDisplay()
        self.joystick = FakeJoystickApi(handles)
        self.event = FakeEventQueue()


def wheel_binding(index=0, button=1):
    return PttBindingConfig(
        kind="joystick",
        code=button,
        label=f"Wheel · Button {button + 1}",
        device_guid=GUID,
        device_name="Wheel",
        device_index=index,
    )


def test_detector_uses_a_rising_digital_button_edge_and_ignores_held_at_open():
    handle = FakeHandle()
    handle.buttons[1] = True
    pygame = FakePygame([handle])
    detector = JoystickBindingDetector(pygame_module=pygame)
    assert detector.poll() is None
    handle.buttons[1] = False
    assert detector.poll() is None
    handle.buttons[1] = True
    assert detector.poll() == wheel_binding()
    detector.close()
    assert handle.closed


def test_reader_tracks_held_state_and_recovers_a_uniquely_reindexed_device():
    handle = FakeHandle()
    pygame = FakePygame([handle])
    reader = JoystickButtonInput(wheel_binding(index=7, button=2), pygame_module=pygame)
    assert not reader.is_down()
    handle.buttons[2] = True
    assert reader.is_down()
    assert pygame.event.pumps >= 3
    reader.close()


def test_digital_hat_direction_can_be_bound_but_axes_are_never_read():
    handle = FakeHandle()
    detector = JoystickBindingDetector(pygame_module=FakePygame([handle]))
    handle.hats[0] = (0, 1)
    binding = detector.poll()
    assert binding is not None
    assert binding.control == "hat" and binding.hat_value == (0, 1)
    detector.close()

    handle = FakeHandle()
    reader = JoystickButtonInput(binding, pygame_module=FakePygame([handle]))
    assert not reader.is_down()
    handle.hats[0] = (0, 1)
    assert reader.is_down()
    handle.hats[0] = (1, 0)
    assert not reader.is_down()
    reader.close()


def test_reader_fails_closed_for_ambiguous_or_missing_device_and_button():
    with pytest.raises(SpeechInputError, match="device_ambiguous"):
        JoystickButtonInput(
            wheel_binding(index=7), pygame_module=FakePygame([FakeHandle(), FakeHandle()])
        )
    with pytest.raises(SpeechInputError, match="device_unavailable"):
        JoystickButtonInput(wheel_binding(), pygame_module=FakePygame([]))
    with pytest.raises(SpeechInputError, match="button_unavailable"):
        JoystickButtonInput(
            wheel_binding(button=9), pygame_module=FakePygame([FakeHandle(buttons=2)])
        )
