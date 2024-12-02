from abc import ABC
import time
import numpy as np
from open3d.visualization import gui
from .helpers import get_key_name

KEY_EXTRA_FUNCTIONS = gui.KeyName.LEFT_CONTROL
KEY_MODIFIER = gui.KeyName.LEFT_SHIFT


class Hotkeys:
    def __init__(self, app_instance, function_key_mappings={}):
        self._app_instance = app_instance
        self._is_extra_functions = False
        self._is_modifier_pressed = False
        self._bindings = {
            (gui.KeyName.SPACE, False): {
                "desc": "Toggle",
                "callback": self._cb_toggle,
                "modifier": True,
            },
            (gui.KeyName.LEFT, False): {
                "desc": "[Fast] Previous",
                "callback": self._cb_previous,
                "modifier": True,
            },
            (gui.KeyName.RIGHT, False): {
                "desc": "[Fast] Next",
                "callback": self._cb_next,
                "modifier": True,
            },
            (gui.KeyName.DELETE, False): {
                "desc": "Delete elements",
                "callback": self._cb_delete,
                "modifier": False,
            },
            (gui.KeyName.C, True): {
                "desc": "Copy",
                "callback": self._cb_copy,
                "modifier": False,
            },
            (gui.KeyName.V, True): {
                "desc": "Paste",
                "callback": self._cb_paste,
                "modifier": False,
            },
            (gui.KeyName.H, False): {
                "desc": "Show help",
                "callback": self._cb_toggle_help_panel,
                "modifier": True,
            },
            (gui.KeyName.I, False): {
                "desc": "Print info",
                "callback": self._cb_print_info,
                "modifier": True,
            },
            (gui.KeyName.P, False): {
                "desc": "Show Preferences",
                "callback": self._cb_toggle_settings_panel,
                "modifier": True,
            },
            (gui.KeyName.ENTER, True): {
                "desc": "Repeat last function",
                "callback": self._cb_repeat_last_function,
                "modifier": True,
            },
            (gui.KeyName.Z, True): {
                "desc": "Undo [Redo]",
                "callback": self._cb_undo_redo,
                "modifier": True,
            },
            (gui.KeyName.U, True): {
                "desc": "[Un]Hide",
                "callback": self._cb_hide_unhide,
                "modifier": True,
            },
            (gui.KeyName.A, True): {
                "desc": "[Un]select all",
                "callback": self._cb_toggle_all,
                "modifier": True,
            },
            (gui.KeyName.L, True): {
                "desc": "[Un]select last",
                "callback": self._cb_toggle_last,
                "modifier": True,
            },
            (gui.KeyName.T, True): {
                "desc": "[Un]select type",
                "callback": self._cb_toggle_type,
                "modifier": True,
            },
        }

        for key, function_descriptor in function_key_mappings.items():
            func = function_descriptor["function"]
            _callback = lambda f=func: self._app_instance._apply_function_to_elements(f)

            self._bindings[(key, True)] = {
                "desc": function_descriptor["name"],
                "callback": _callback,
                "modifier": False,
            }

        for key, _ in self._bindings.keys():
            if key not in function_key_mappings:
                continue

            function_descriptor = function_key_mappings[key]

            hotkey = function_descriptor.get("hotkey")
            if hotkey is None:
                continue

            function_descriptor[
                "name"
            ] += f" ({get_key_name(KEY_EXTRA_FUNCTIONS)} + {chr(hotkey)})"

    @property
    def bindings(self):
        return self._bindings

    def get_instructions(self):
        def get_action_line(key, values):
            line = "("
            if key[1]:
                line += f"{get_key_name(KEY_EXTRA_FUNCTIONS)} + "
            if values["modifier"]:
                line += f"[{get_key_name(KEY_MODIFIER)}] + "
            line += f"{get_key_name(key[0])}):\n- {values['desc']}"
            return line

        # function_key_mappings_info = [
        #     f"({get_key_name(KEY_EXTRA_FUNCTIONS)} + {chr(key)}: \n- {value['name']}"
        #     for key, value in self._app_instance.function_key_mappings.items()
        # ]

        return (
            # "******************** KEYS: ***********************\n"
            "\n\n".join(
                [get_action_line(key, values) for key, values in self._bindings.items()]
            )
            + f"\n\n({get_key_name(KEY_EXTRA_FUNCTIONS)}):"
            + "\n- Enables mouse selection [and toggling]"
            # + "\n\n".join(function_key_mappings_info)
            # + "\n**************************************************"
        )

    def _on_key(self, event):
        self._app_instance.print_debug(
            f"Key: {event.key}, type: {event.type}", require_verbose=True
        )

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
            binding["callback"]()
            return gui.Widget.EventCallbackResult.HANDLED

        # if self._is_extra_functions:
        #     if function_descriptor := self._function_key_mappings.get(event.key):
        #         self._app_instance._apply_function_to_elements(
        #             function_descriptor["function"]
        #         )

        # for _, (key, cb_name, extra_functions, _) in KEYS_ACTIONS.items():
        #     if event.key == key:
        #         if extra_functions != self._is_extra_functions:
        #             return gui.Widget.EventCallbackResult.IGNORED

        #         getattr(self, cb_name)()
        #         return gui.Widget.EventCallbackResult.HANDLED

        # for function_key, function_descriptor in self.function_key_mappings.items():
        #     if event.key == function_key:
        #         self._apply_function_to_elements(function_descriptor["function"])
        #         return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

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
        copied_elements = [
            elem["raw"] for elem in app_instance.elements if elem["selected"]
        ]
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

    def _cb_print_info(self):
        app_instance = self._app_instance
        elem = app_instance.current_element

        print()
        print(
            f"Current element: ({app_instance.i + 1}/{len(app_instance.elements)}): {elem['raw']}"
        )
        app_instance.print_debug(f"drawable: {elem['drawable']}")
        app_instance.print_debug(
            f"Current index: {app_instance.i}, old index: {app_instance.i_old}"
        )
        print(f"Current selected: {app_instance.is_current_selected}")
        print(f"Current bbox: {app_instance._current_bbox}")
        print(f"{len(app_instance.elements)} current elements")
        print(f"{len(app_instance._fixed_elements)} fixed elements")
        print(f"{len(app_instance._hidden_elements)} hidden elements")
        print(f"{len(app_instance._past_states)} past states (for undoing)")
        print(f"{len(app_instance._future_states)} future states (for redoing)")
        time.sleep(0.5)

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
            app_instance._apply_function_to_elements(func)

    def _cb_toggle_help_panel(self):
        self._app_instance._menu_help._on_menu_toggle()

    def _cb_toggle_settings_panel(self):
        self._app_instance._settings._on_menu_toggle()

    # def _cb_functions_menu(self):
    #     if self.functions is None or len(self.functions) == 0:
    #         warnings.warn("No functions, cannot call menu.")
    #         return

    #     from ..helpers_visualization import select_function_with_gui

    #     try:
    #         func = select_function_with_gui(self.functions, self._last_used_function)
    #     except KeyboardInterrupt:
    #         return

    #     self.print_debug(f"Chosen function from menu: {func}")

    #     if func is not None:
    #         self._apply_function_to_elements(func)

    def _cb_hide_unhide(self):
        """Hide selected elements or unhide all hidden elements."""

        app_instance = self._app_instance
        indices = app_instance.selected_indices

        if self._is_modifier_pressed:
            # UNHIDE
            app_instance._insert_elements(
                app_instance._hidden_elements, selected=True, to_vis=True
            )
            app_instance._hidden_elements = []

        else:
            # HIDE SELECTED
            app_instance._hidden_elements += app_instance._pop_elements(
                indices, from_vis=True
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
            app_instance._pop_elements(indices, from_vis=True)
            app_instance._insert_elements(modified_elements, to_vis=True)

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
                indices_to_pop, from_vis=True
            )
            app_instance._insert_elements(elements, indices, selected=True, to_vis=True)

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

            app_instance._update_current_idx(indices[-1])

        app_instance._update_plane_boundaries()
