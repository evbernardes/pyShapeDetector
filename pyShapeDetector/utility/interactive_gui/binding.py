from open3d.visualization import gui
from typing import Union, Callable

KEY_LEFT_CONTROL = gui.KeyName.LEFT_CONTROL
KEY_LEFT_SHIFT = gui.KeyName.LEFT_SHIFT

KEY_TEXT_LEFT_CONTROL = "LCtrl"
KEY_TEXT_LEFT_SHIFT = "LShift"


def _get_key_name(key):
    if isinstance(key, gui.KeyName):
        return str(key).split(".")[1]
    elif isinstance(key, int):
        return chr(key)


class Binding:
    @property
    def key(self):
        return self._key

    @property
    def lctrl(self):
        return self._lctrl

    @property
    def lshift(self):
        return self._lshift

    @property
    def description(self):
        return self._description

    @property
    def callback(self):
        return self._callback

    @property
    def menu(self):
        return self._menu

    @property
    def key_instruction(self):
        if self.key is None:
            return None

        sequence = []
        if self.lctrl:
            sequence.append(KEY_TEXT_LEFT_CONTROL)

        if self._lshift:
            sequence.append(KEY_TEXT_LEFT_SHIFT)

        sequence.append(_get_key_name(self.key))

        return " + ".join(sequence)

    @property
    def description_and_instruction(self):
        if self.key_instruction is None:
            return self.description
        else:
            return self.description + " (" + self.key_instruction + ")"

    def __init__(
        self,
        description: str,
        callback: Callable,
        key: None,
        lctrl: bool = False,
        lshift: bool = False,
        menu: Union[str, None] = None,
    ):
        self._description = description
        self._callback = callback
        self._key = key
        self._lctrl = lctrl
        self._lshift = lshift
        self._menu = menu
