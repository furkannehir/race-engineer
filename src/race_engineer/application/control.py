"""Small, UI-independent control channel for a live session.

Only Events cross threads. Status notifications contain no audio or conversation text.
"""

import threading
from collections.abc import Callable


class LiveControl:
    def __init__(self, notify: Callable[[str, str], None] = lambda *_: None) -> None:
        self.stop = threading.Event()
        self.muted = threading.Event()
        self.notify = notify

    def emit(self, topic: str, value: str) -> None:
        self.notify(topic, value)
