import warnings
from open3d.visualization import gui
from .editor_app import Editor
from .binding import Binding, KEY_LEFT_CONTROL, KEY_LEFT_SHIFT, _get_key_name


class Hotkeys:
    def add_one_binding(self, binding: Binding):
        key = (binding.key, binding.lctrl)
        if binding.key is None:
            return

        if key in self._bindings_map:
            if binding.lctrl:
                extra_text = f"{_get_key_name(KEY_LEFT_CONTROL)} + "
            else:
                extra_text = ""
            warnings.warn(
                f"hotkey {extra_text}{binding} previously assigned to function "
                f"{self._bindings_map[key].description}, resetting it to {binding.description}."
            )
            self._bindings_map[key]._key = None

        self._bindings_map[key] = binding

    def __init__(self, editor_instance: Editor):
        self._editor_instance = editor_instance
        self._is_lctrl_pressed = False
        self._is_lshift_pressed = False
        self._bindings_map = {}

        # First, add internal bindings
        for binding in editor_instance._internal_functions.bindings:
            self.add_one_binding(binding)

        # Then, get extension bindings
        for extension in editor_instance.extensions:
            self.add_one_binding(extension.as_binding)

    @property
    def bindings_map(self):
        return self._bindings_map

    def _on_key(self, event):
        self._editor_instance.print_debug(
            f"Key: {event.key}, type: {event.type}", require_verbose=True
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
        binding = self.bindings_map.get((event.key, self._is_lctrl_pressed))

        if binding is not None:
            binding.callback()
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED
