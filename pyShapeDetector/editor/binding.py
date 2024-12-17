from open3d.visualization import gui
from typing import Union, Callable
from .editor_app import Editor
from pathlib import Path

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
    def menu_id(self):
        return self._menu_id

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
    def creates_window(self):
        return self._creates_window

    @property
    def key_instruction(self):
        if self.key is None:
            return None

        sequence = []
        if self.lctrl:
            sequence.append(KEY_TEXT_LEFT_CONTROL)

        if self.lshift:
            sequence.append(KEY_TEXT_LEFT_SHIFT)

        sequence.append(_get_key_name(self.key))

        return " + ".join(sequence)

    @property
    def menu_item_description(self):
        """Get line for menu item. If binding creates a window, a ... is added."""
        if self.creates_window:
            description = self.description + "..."
        else:
            description = self.description

        if self.key_instruction is None:
            return description
        else:
            return description + " (" + self.key_instruction + ")"

    def add_to_menu(self, editor_instance: Editor, set_checked=False):
        if self.menu_id is None:
            self._menu_id = next(editor_instance._submenu_id_generator)

        if self.menu is None:
            return

        menu = editor_instance._get_submenu_from_path(Path(self.menu))
        editor_instance._settings.print_debug(
            f"Assigned id {self.menu_id} to item '{self.menu_item_description}' on menu '{self.menu}'.",
            require_verbose=True,
        )
        menu.add_item(self.menu_item_description, self.menu_id)
        menu.set_checked(self.menu_id, set_checked)
        editor_instance._window.set_on_menu_item_activated(self.menu_id, self.callback)

    def __init__(
        self,
        description: str,
        callback: Callable,
        key=None,
        lctrl: bool = False,
        lshift: bool = False,
        menu: Union[str, None] = None,
        creates_window: bool = False,
        menu_id: Union[None, int] = None,
    ):
        self._description = description
        self._callback = callback
        self._key = key
        self._lctrl = lctrl
        self._lshift = lshift
        self._menu = menu
        self._creates_window = creates_window
        self._menu_id = None
