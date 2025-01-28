import warnings
from typing import TYPE_CHECKING
from open3d.visualization import gui
from .binding import Binding, KEY_LEFT_CONTROL, KEY_LEFT_SHIFT

if TYPE_CHECKING:
    from .editor_app import Editor


class Hotkeys:
    """Class to create a hotkey system with all bindings."""

    _is_lctrl_pressed: bool = False
    _is_lshift_pressed: bool = False
    _bindings_map: dict[str, Binding] = {}

    def add_one_binding(self, binding: Binding):
        key = (binding.key, binding.lctrl, binding.lshift)
        if binding.key is None:
            return

        if key in self._bindings_map:
            warnings.warn(
                f"hotkey {binding.key_instruction} previously assigned to "
                f"{self._bindings_map[key].description}, resetting it to "
                "{binding.description}."
            )
            self._bindings_map[key]._key = None
            self._bindings_map[key]._lctrl = False
            self._bindings_map[key]._lshift = False

        self._bindings_map[key] = binding

    def add_multiple_bindings(self, bindings: list["Binding"]):
        for binding in bindings:
            self.add_one_binding(binding)

    @property
    def help_text(self) -> str:
        return "\n\n".join(
            [
                f"({binding.key_instruction}):\n- {binding.description}"
                for binding in self._bindings_map.values()
            ]
        )

    def __init__(self, editor_instance: "Editor"):
        self._editor_instance = editor_instance

    @property
    def bindings_map(self):
        return self._bindings_map

    def _on_key(self, event):
        self._editor_instance._settings.print_debug(
            f"Key: {event.key}, type: {event.type}",
            require_verbose=True,
        )

        # if event.key == gui.KeyName.ESCAPE:
        #     return gui.Widget.EventCallbackResult.HANDLED

        # First check if extra functions flag (LCtrl) is being pressed...
        if event.key == KEY_LEFT_CONTROL:
            self._is_lctrl_pressed = event.type == gui.KeyEvent.Type.DOWN
            return gui.Widget.EventCallbackResult.HANDLED

        # .. or modifier flag (LShift) is being pressed...
        if event.key == KEY_LEFT_SHIFT:
            self._is_lshift_pressed = event.type == gui.KeyEvent.Type.DOWN
            return gui.Widget.EventCallbackResult.HANDLED

        # ... if not, ignore every release
        if not event.type == gui.KeyEvent.Type.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED

        # If down key, check if it's one of the callbacks:
        binding = self.bindings_map.get(
            (event.key, self._is_lctrl_pressed, self._is_lshift_pressed)
        )

        if binding is not None:
            binding.callback()
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED
