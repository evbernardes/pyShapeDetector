from abc import ABC
import copy
import time
import numpy as np
from open3d.visualization import gui
from .interactive_gui import AppWindow
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
    def description(self):
        return self._description

    @property
    def callback(self):
        return self._callback

    @property
    def uses_modifier(self):
        return self._uses_modifier

    @property
    def menu(self):
        return self._menu

    @property
    def key_instruction(self):
        if self.key is None:
            return ""
        line = "("
        if self.key[1]:
            line += f"{_get_key_name(KEY_EXTRA_FUNCTIONS)} + "
        if self.uses_modifier:
            line += f"[{_get_key_name(KEY_MODIFIER)}] + "
        line += f"{_get_key_name(self.key[0])})"
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
        key: Union[None, tuple] = None,
        uses_modifier: int = False,
        menu: Union[str, None] = None,
    ):
        self._description = description
        self._callback = callback
        self._key = key
        self._uses_modifier = uses_modifier
        self._menu = menu


class Hotkeys:
    def __init__(self, app_instance: AppWindow):
        self._app_instance = app_instance
        self._is_extra_functions = False
        self._is_modifier_pressed = False
        bindings = {
            (gui.KeyName.ESCAPE, False): Binding(
                description="Quit",
                callback=self._cb_quit_app,
                uses_modifier=False,
                menu="File",
            ),
            (gui.KeyName.SPACE, False): Binding(
                description="Selected/unselect current",
                callback=self._cb_toggle,
                uses_modifier=True,
                menu=None,
            ),
            (gui.KeyName.LEFT, False): Binding(
                description="[Fast] Previous",
                callback=self._cb_previous,
                uses_modifier=True,
                menu=None,
            ),
            (gui.KeyName.RIGHT, False): Binding(
                description="[Fast] Next",
                callback=self._cb_next,
                uses_modifier=True,
                menu=None,
            ),
            (gui.KeyName.DELETE, False): Binding(
                description="Delete elements",
                callback=self._cb_delete,
                uses_modifier=False,
                menu="Edit",
            ),
            (gui.KeyName.C, True): Binding(
                description="Copy",
                callback=self._cb_copy,
                uses_modifier=False,
                menu="Edit",
            ),
            (gui.KeyName.V, True): Binding(
                description="Paste",
                callback=self._cb_paste,
                uses_modifier=False,
                menu="Edit",
            ),
            (gui.KeyName.H, False): Binding(
                description="Show Help",
                callback=self._cb_toggle_help_panel,
                uses_modifier=False,
                menu=None,  # This is added later
            ),
            (gui.KeyName.I, False): Binding(
                description="Show Info",
                callback=self._cb_toggle_info_panel,
                uses_modifier=False,
                menu=None,  # This is added later
            ),
            (gui.KeyName.P, False): Binding(
                description="Show Preferences",
                callback=self._cb_toggle_settings_panel,
                uses_modifier=False,
                menu=None,  # This is added later
            ),
            (gui.KeyName.ENTER, True): Binding(
                description="Repeat last function",
                callback=self._cb_repeat_last_function,
                uses_modifier=False,
                menu="Edit",
            ),
            (gui.KeyName.Z, True): Binding(
                description="Undo [Redo]",
                callback=self._cb_undo_redo,
                uses_modifier=True,
                menu="Edit",
            ),
            (gui.KeyName.U, True): Binding(
                description="[Un]Hide",
                callback=self._cb_hide_unhide,
                uses_modifier=True,
                menu=None,
            ),
            (gui.KeyName.A, True): Binding(
                description="[Un]select all",
                callback=self._cb_toggle_all,
                uses_modifier=True,
                menu=None,
            ),
            (gui.KeyName.L, True): Binding(
                description="[Un]select last",
                callback=self._cb_toggle_last,
                uses_modifier=True,
                menu=None,
            ),
            (gui.KeyName.T, True): Binding(
                description="[Un]select type",
                callback=self._cb_toggle_type,
                uses_modifier=True,
                menu=None,
            ),
        }

        # Create menu entries for all hotkeys that contain 'menu', except extensions:
        for key, binding in bindings.items():
            binding._key = key
            if binding.menu is not None:
                app_instance._add_menu_item(
                    binding.menu,
                    binding.description_and_instruction,
                    binding.callback,
                )

        extension_key_mappings = app_instance.extension_key_mappings

        for key, extension in extension_key_mappings.items():
            bindings[key, True] = Binding(
                key=(key, True),
                description=extension.name,
                callback=extension.run,
                uses_modifier=False,
            )

        for key, _ in bindings.keys():
            if key not in extension_key_mappings:
                continue

            extension = extension_key_mappings[key]

            hotkey = extension.hotkey
            if hotkey is None:
                continue

            extension._name += (
                f" ({_get_key_name(KEY_EXTRA_FUNCTIONS)} + {chr(hotkey)})"
            )

        self._bindings = bindings

    @property
    def bindings(self):
        return self._bindings

    def find_binding(self, desc: str):
        for binding in self.bindings.values():
            if binding.description == desc:
                return binding
        return None

    def get_instructions(self):
        return (
            "\n\n".join(
                [
                    f"{binding.key_instruction}:\n- {binding.description}"
                    for binding in self.bindings.values()
                ]
            )
            + f"\n\n({_get_key_name(KEY_EXTRA_FUNCTIONS)}):"
            + "\n- Enables mouse selection [and toggling]"
        )

    def _on_key(self, event):
        self._app_instance.print_debug(
            f"Key: {event.key}, type: {event.type}", require_verbose=True
        )

        # if event.key == gui.KeyName.ESCAPE:
        #     return gui.Widget.EventCallbackResult.HANDLED

        # First check if extra functions flag (left ctrl) is being pressed...
        if event.key == KEY_EXTRA_FUNCTIONS:
            self._is_extra_functions = event.type == gui.KeyEvent.Type.DOWN
            return gui.Widget.EventCallbackResult.HANDLED

        # .. or modifier flag (left shift) is being pressed...
        if event.key == KEY_MODIFIER:
            self._is_modifier_pressed = event.type == gui.KeyEvent.Type.DOWN
            return gui.Widget.EventCallbackResult.HANDLED

        # ... if not, ignore every release
        if not event.type == gui.KeyEvent.Type.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED

        # If down key, check if it's one of the callbacks:
        binding = self.bindings.get((event.key, self._is_extra_functions))

        if binding is not None:
            binding.callback()
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def _cb_quit_app(self):
        window = self._app_instance._window
        em = window.theme.font_size
        button_separation_width = 2 * int(round(0.5 * em))

        dlg = gui.Dialog("Quit Dialog")

        def _on_close_yes():
            self._app_instance._closing_app = True
            window.close()

        def _on_close_no():
            window.close_dialog()

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Close Shape Detector?"))

        yes = gui.Button("Yes")
        yes.set_on_clicked(_on_close_yes)

        no = gui.Button("No")
        no.set_on_clicked(_on_close_no)

        h = gui.Horiz()
        h.add_stretch()
        h.add_child(yes)
        h.add_fixed(button_separation_width)
        h.add_child(no)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        window.show_dialog(dlg)

    def _cb_toggle(self):
        """Toggle the current highlighted element between selected/unselected."""
        app_instance = self._app_instance
        if not app_instance.select_filter(app_instance.current_element):
            return

        app_instance.is_current_selected = not app_instance.is_current_selected
        app_instance._update_current_idx()

    def _cb_next(self):
        """Highlight next element in list."""
        delta = 1 + 4 * self._is_modifier_pressed
        app_instance = self._app_instance
        app_instance._update_current_idx(
            min(app_instance.i + delta, len(app_instance.elements) - 1)
        )

    def _cb_delete(self):
        """Delete current elements."""

        def delete(elements):
            return []

        self._app_instance._apply_function_to_elements(delete)

    def _cb_copy(self):
        """Save elements to be copied."""
        app_instance = self._app_instance
        copied_elements = copy.deepcopy(
            [elem["raw"] for elem in app_instance.elements if elem["selected"]]
        )
        app_instance.print_debug(f"Copying {len(copied_elements)} elements.")
        app_instance._copied_elements = copied_elements

    def _cb_paste(self):
        app_instance = self._app_instance

        def paste(elements):
            return elements + app_instance._copied_elements

        app_instance.print_debug(
            f"Pasting {len(app_instance._copied_elements)} elements."
        )
        self._app_instance._apply_function_to_elements(paste)
        app_instance._copied_elements = []

    def _cb_previous(self):
        """Highlight previous element in list."""
        delta = 1 + 4 * self._is_modifier_pressed
        app_instance = self._app_instance
        app_instance._update_current_idx(max(app_instance.i - delta, 0))

    def _cb_toggle_all(self):
        """Toggle the all elements between all selected/all unselected."""
        self._app_instance._toggle_indices(None)

    def _cb_toggle_last(self):
        """Toggle the elements from last output."""
        app_instance = self._app_instance
        if len(app_instance._past_states) == 0:
            return

        num_outputs = app_instance._past_states[-1]["num_outputs"]
        app_instance._toggle_indices(slice(-num_outputs, None))

    def _cb_toggle_type(self):
        from ..helpers_visualization import get_inputs

        app_instance = self._app_instance
        elems_raw = [elem["raw"] for elem in app_instance.elements]

        types = set([t for elem in elems_raw for t in elem.__class__.mro()])
        types.discard(ABC)
        types.discard(object)
        types = list(types)
        types_names = [type_.__name__ for type_ in types]

        try:
            (selected_type_name,) = get_inputs(
                {"type": [types_names, types_names[0]]}, window_name="Select type"
            )
        except KeyboardInterrupt:
            return

        selected_type = types[types_names.index(selected_type_name)]
        idx = np.where([isinstance(elem, selected_type) for elem in elems_raw])[0]
        app_instance._toggle_indices(idx)

    # def _cb_finish_process(self, vis, action, mods):
    #     """Signal ending."""
    #     if action == 1:
    #         return
    #     self.finish = True
    #     vis.close()

    # def _cb_center_current(self, vis, action, mods):
    #     if not self._is_modifier_extra or action == 1:
    #         return

    #     ctr = self._vis.get_view_control()
    #     ctr.set_lookat(self._current_bbox.get_center())
    #     ctr.set_front([0, 0, -1])  # Define the camera front direction
    #     ctr.set_up([0, 1, 0])  # Define the camera "up" direction
    #     ctr.set_zoom(0.1)  # Adjust zoom level if necessary

    def _cb_repeat_last_function(self):
        app_instance = self._app_instance
        func = app_instance._last_used_function

        if func is not None:
            app_instance.print_debug(f"Re-applying last function: {func}")
            app_instance._apply_function_to_elements(func, update_parameters=False)

    def _cb_toggle_help_panel(self):
        self._app_instance._menu_help._on_help_toggle()

    def _cb_toggle_info_panel(self):
        self._app_instance._menu_help._on_info_toggle()

    def _cb_toggle_settings_panel(self):
        self._app_instance._settings._on_menu_toggle()

    def _cb_hide_unhide(self):
        """Hide selected elements or unhide all hidden elements."""

        app_instance = self._app_instance
        indices = app_instance.selected_indices

        if self._is_modifier_pressed:
            # UNHIDE
            app_instance._insert_elements(
                app_instance._hidden_elements, selected=True, to_gui=True
            )
            app_instance._hidden_elements = []

        else:
            # HIDE SELECTED
            app_instance._hidden_elements += app_instance._pop_elements(
                indices, from_gui=True
            )
            app_instance.selected = False

        # TODO: find a way to make hiding work with undoing
        app_instance._past_states = []
        app_instance._future_states = []
        app_instance._update_plane_boundaries()

    def _cb_undo_redo(self):
        app_instance = self._app_instance
        if self._is_modifier_pressed:
            # REDO
            if len(app_instance._future_states) == 0:
                return

            future_state = app_instance._future_states.pop()

            modified_elements = future_state["modified_elements"]
            indices = future_state["indices"]

            app_instance.print_debug(
                f"Redoing last operation, removing {len(indices)} inputs and "
                f"resetting {len(modified_elements)} inputs."
            )

            input_elements = [app_instance.elements[i]["raw"] for i in indices]
            app_instance._save_state(indices, input_elements, len(modified_elements))

            app_instance.i = future_state["current_index"]
            app_instance._pop_elements(indices, from_gui=True)
            app_instance._insert_elements(modified_elements, to_gui=True)

            app_instance._update_current_idx(len(app_instance.elements) - 1)
        else:
            # UNDO
            if len(app_instance._past_states) == 0:
                return

            last_state = app_instance._past_states.pop()
            indices = last_state["indices"]
            elements = last_state["elements"]
            num_outputs = last_state["num_outputs"]
            num_elems = len(app_instance.elements)

            app_instance.print_debug(
                f"Undoing last operation, removing {num_outputs} outputs and "
                f"resetting {len(elements)} inputs."
            )

            indices_to_pop = range(num_elems - num_outputs, num_elems)
            modified_elements = app_instance._pop_elements(
                indices_to_pop, from_gui=True
            )
            app_instance._insert_elements(elements, indices, selected=True, to_gui=True)

            app_instance._future_states.append(
                {
                    "modified_elements": modified_elements,
                    "indices": indices,
                    "current_index": app_instance.i,
                }
            )

            while (
                len(app_instance._future_states)
                > app_instance._settings.number_redo_states
            ):
                app_instance._future_states.pop(0)

            if len(indices) > 0:
                app_instance._update_current_idx(indices[-1])

        app_instance._update_plane_boundaries()

    def _cb_quit(self):
        pass
