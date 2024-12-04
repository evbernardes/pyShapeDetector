from open3d.visualization import gui
from typing import Union, Callable

KEY_EXTRA_FUNCTIONS = gui.KeyName.LEFT_CONTROL
KEY_MODIFIER = gui.KeyName.LEFT_SHIFT


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
    def extra_functions(self):
        return self._extra_functions

    @property
    def modifier(self):
        return self._modifier

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
            return ""
        line = "("
        if self.extra_functions:
            line += f"{_get_key_name(KEY_EXTRA_FUNCTIONS)} + "
        if self.modifier:
            line += f"[{_get_key_name(KEY_MODIFIER)}] + "
        line += f"{_get_key_name(self.key)})"
        return line

    @property
    def description_and_instruction(self):
        instruction = self.key_instruction
        if instruction == "":
            return self.description
        else:
            return self.description + " " + instruction

    def __init__(
        self,
        description: str,
        callback: Callable,
        key: None,
        extra_functions: bool = False,
        modifier: bool = False,
        menu: Union[str, None] = None,
    ):
        self._description = description
        self._callback = callback
        self._key = key
        self._extra_functions = extra_functions
        self._modifier = modifier
        self._menu = menu
