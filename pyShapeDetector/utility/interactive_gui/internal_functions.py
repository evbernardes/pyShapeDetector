from abc import ABC
import copy
import numpy as np
from open3d.visualization import gui
from .editor_app import Editor
from .binding import Binding


class InternalFunctions:
    @property
    def is_lshift_pressed(self):
        return self._editor_instance._hotkeys._is_lshift_pressed

    @property
    def bindings(self):
        return self._bindings

    def __init__(self, editor_instance: Editor):
        self._editor_instance = editor_instance
        self._bindings = [
            Binding(
                key=gui.KeyName.ESCAPE,
                lctrl=False,
                lshift=False,
                description="Quit",
                callback=self._cb_quit_app,
                menu="File",
            ),
            Binding(
                key=gui.KeyName.SPACE,
                lctrl=False,
                lshift=False,
                description="Selected/unselect current",
                callback=self._cb_toggle,
                menu=None,
            ),
            Binding(
                key=gui.KeyName.LEFT,
                lctrl=False,
                lshift=False,
                description="[Fast] Previous",
                callback=self._cb_previous,
                menu=None,
            ),
            Binding(
                key=gui.KeyName.RIGHT,
                lctrl=False,
                lshift=False,
                description="[Fast] Next",
                callback=self._cb_next,
                menu=None,
            ),
            Binding(
                key=gui.KeyName.DELETE,
                lctrl=False,
                lshift=False,
                description="Delete elements",
                callback=self._cb_delete,
                menu="Edit",
            ),
            Binding(
                key=gui.KeyName.C,
                lctrl=True,
                lshift=False,
                description="Copy",
                callback=self._cb_copy,
                menu="Edit",
            ),
            Binding(
                key=gui.KeyName.V,
                lctrl=True,
                lshift=False,
                description="Paste",
                callback=self._cb_paste,
                menu="Edit",
            ),
            Binding(
                key=gui.KeyName.H,
                lctrl=False,
                lshift=False,
                description="Show Help",
                callback=self._cb_toggle_help_panel,
                menu=None,  # This is added later
            ),
            Binding(
                key=gui.KeyName.I,
                lctrl=False,
                lshift=False,
                description="Show Info",
                callback=self._cb_toggle_info_panel,
                menu=None,  # This is added later
            ),
            Binding(
                key=gui.KeyName.P,
                lctrl=False,
                lshift=False,
                description="Show Preferences",
                callback=self._cb_toggle_settings_panel,
                menu=None,  # This is added later
            ),
            Binding(
                key=gui.KeyName.R,
                lctrl=True,
                lshift=False,
                description="Repeat last function",
                callback=self._cb_repeat_last_function,
                menu="Edit",
            ),
            Binding(
                key=gui.KeyName.Z,
                lctrl=True,
                lshift=True,
                description="Undo [Redo]",
                callback=self._cb_undo_redo,
                menu="Edit",
            ),
            Binding(
                key=gui.KeyName.U,
                lctrl=True,
                lshift=True,
                description="[Un]Hide",
                callback=self._cb_hide_unhide,
                menu=None,
            ),
            Binding(
                key=gui.KeyName.A,
                lctrl=True,
                lshift=True,
                description="[Un]select all",
                callback=self._cb_toggle_all,
                menu=None,
            ),
            Binding(
                key=gui.KeyName.L,
                lctrl=True,
                lshift=True,
                description="[Un]select last",
                callback=self._cb_toggle_last,
                menu=None,
            ),
            Binding(
                key=gui.KeyName.T,
                lctrl=True,
                lshift=False,
                description="[Un]select per type",
                callback=self._cb_toggle_type,
                menu=None,
            ),
        ]

        self._dict = {binding.description: binding for binding in self._bindings}

    def create_menu_items(self):
        for binding in self.bindings:
            if binding.menu is not None:
                self._editor_instance._add_menu_item(
                    binding.menu,
                    binding.description_and_instruction,
                    binding.callback,
                )

    def _cb_quit_app(self):
        window = self._editor_instance._window
        em = window.theme.font_size
        button_separation_width = 2 * int(round(0.5 * em))

        dlg = gui.Dialog("Quit Dialog")

        def _on_close_yes():
            self._editor_instance._closing_app = True
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
        editor_instance = self._editor_instance
        if not editor_instance.select_filter(editor_instance.current_element):
            return

        editor_instance.is_current_selected = not editor_instance.is_current_selected
        editor_instance._update_current_idx()

    def _cb_next(self):
        """Highlight next element in list."""
        delta = 1 + 4 * self.is_lshift_pressed
        editor_instance = self._editor_instance
        editor_instance._update_current_idx(
            min(editor_instance.i + delta, len(editor_instance.elements) - 1)
        )

    def _cb_delete(self):
        """Delete current elements."""

        def delete(elements):
            return []

        self._editor_instance._apply_function_to_elements(delete)

    def _cb_copy(self):
        """Save elements to be copied."""
        editor_instance = self._editor_instance
        copied_elements = copy.deepcopy(
            [elem["raw"] for elem in editor_instance.elements if elem["selected"]]
        )
        editor_instance.print_debug(f"Copying {len(copied_elements)} elements.")
        editor_instance._copied_elements = copied_elements

    def _cb_paste(self):
        editor_instance = self._editor_instance

        def paste(elements):
            return elements + editor_instance._copied_elements

        editor_instance.print_debug(
            f"Pasting {len(editor_instance._copied_elements)} elements."
        )
        self._editor_instance._apply_function_to_elements(paste)
        editor_instance._copied_elements = []

    def _cb_previous(self):
        """Highlight previous element in list."""
        delta = 1 + 4 * self.is_lshift_pressed
        editor_instance = self._editor_instance
        editor_instance._update_current_idx(max(editor_instance.i - delta, 0))

    def _cb_toggle_all(self):
        """Toggle the all elements between all selected/all unselected."""
        self._editor_instance._toggle_indices(None)

    def _cb_toggle_last(self):
        """Toggle the elements from last output."""
        editor_instance = self._editor_instance
        if len(editor_instance._past_states) == 0:
            return

        num_outputs = editor_instance._past_states[-1]["num_outputs"]
        editor_instance._toggle_indices(slice(-num_outputs, None))

    def _cb_toggle_type(self):
        editor_instance = self._editor_instance
        window = editor_instance._window
        em = window.theme.font_size
        separation_height = int(round(0.5 * em))
        elems_raw = [elem["raw"] for elem in editor_instance.elements]

        temp_window = editor_instance.app.create_window("Select type", 200, 400)
        temp_window.show_menu(False)

        def _callback(_type, value):
            elems_raw = [elem["raw"] for elem in editor_instance.elements]
            idx = np.where([isinstance(elem, _type) for elem in elems_raw])[0].tolist()
            editor_instance._toggle_indices(idx, to_value=value)

        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Select types to select:"))

        types = set([t for elem in elems_raw for t in elem.__class__.mro()])
        types.discard(ABC)
        types.discard(object)
        types = list(types)
        # type_names = [type_.__name__ for type_ in types]

        for _type in types:
            element = gui.Checkbox(_type.__name__)
            element.checked = False

            element.set_on_checked(lambda value, t=_type: _callback(t, value))
            dlg_layout.add_child(element)
            dlg_layout.add_fixed(separation_height)
            # elements.append(element)

        def _on_ok():
            window.close_dialog()

        ok = gui.Button("Ok")
        ok.set_on_clicked(_on_ok)

        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        temp_window.add_child(dlg_layout)

    def _cb_repeat_last_function(self):
        editor_instance = self._editor_instance
        func = editor_instance._last_used_function

        if func is not None:
            editor_instance.print_debug(f"Re-applying last function: {func}")
            editor_instance._apply_function_to_elements(func, update_parameters=False)

    def _cb_toggle_help_panel(self):
        self._editor_instance._menu_help._on_help_toggle()

    def _cb_toggle_info_panel(self):
        self._editor_instance._menu_help._on_info_toggle()

    def _cb_toggle_settings_panel(self):
        self._editor_instance._settings._on_menu_toggle()

    def _cb_hide_unhide(self):
        """Hide selected elements or unhide all hidden elements."""

        editor_instance = self._editor_instance
        indices = editor_instance.selected_indices

        if self.is_lshift_pressed:
            # UNHIDE
            editor_instance._insert_elements(
                editor_instance._hidden_elements, selected=True, to_gui=True
            )
            editor_instance._hidden_elements = []

        else:
            # HIDE SELECTED
            editor_instance._hidden_elements += editor_instance._pop_elements(
                indices, from_gui=True
            )
            editor_instance.selected = False

        # TODO: find a way to make hiding work with undoing
        editor_instance._past_states = []
        editor_instance._future_states = []
        editor_instance._update_plane_boundaries()

    def _cb_undo_redo(self):
        editor_instance = self._editor_instance
        if self.is_lshift_pressed:
            # REDO
            if len(editor_instance._future_states) == 0:
                return

            future_state = editor_instance._future_states.pop()

            modified_elements = future_state["modified_elements"]
            indices = future_state["indices"]

            editor_instance.print_debug(
                f"Redoing last operation, removing {len(indices)} inputs and "
                f"resetting {len(modified_elements)} inputs."
            )

            input_elements = [editor_instance.elements[i]["raw"] for i in indices]
            editor_instance._save_state(indices, input_elements, len(modified_elements))

            editor_instance.i = future_state["current_index"]
            editor_instance._pop_elements(indices, from_gui=True)
            editor_instance._insert_elements(modified_elements, to_gui=True)

            editor_instance._update_current_idx(len(editor_instance.elements) - 1)
        else:
            # UNDO
            if len(editor_instance._past_states) == 0:
                return

            last_state = editor_instance._past_states.pop()
            indices = last_state["indices"]
            elements = last_state["elements"]
            num_outputs = last_state["num_outputs"]
            num_elems = len(editor_instance.elements)

            editor_instance.print_debug(
                f"Undoing last operation, removing {num_outputs} outputs and "
                f"resetting {len(elements)} inputs."
            )

            indices_to_pop = range(num_elems - num_outputs, num_elems)
            modified_elements = editor_instance._pop_elements(
                indices_to_pop, from_gui=True
            )
            editor_instance._insert_elements(
                elements, indices, selected=True, to_gui=True
            )

            editor_instance._future_states.append(
                {
                    "modified_elements": modified_elements,
                    "indices": indices,
                    "current_index": editor_instance.i,
                }
            )

            while (
                len(editor_instance._future_states)
                > editor_instance._settings.number_redo_states
            ):
                editor_instance._future_states.pop(0)

            if len(indices) > 0:
                editor_instance._update_current_idx(indices[-1])

        editor_instance._update_plane_boundaries()
