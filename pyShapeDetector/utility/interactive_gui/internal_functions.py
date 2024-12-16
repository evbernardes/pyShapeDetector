from abc import ABC
import copy
import numpy as np
import warnings
import traceback
from pathlib import Path
from open3d.visualization import gui
from .editor_app import Editor
from .binding import Binding


class InternalFunctions:
    @property
    def bindings(self):
        return self._bindings

    def __init__(self, editor_instance: Editor):
        self._editor_instance = editor_instance
        self._bindings = [
            Binding(
                key=gui.KeyName.I,
                lctrl=True,
                lshift=False,
                description="Import",
                callback=self._cb_import,
                menu="File",
            ),
            Binding(
                key=gui.KeyName.O,
                lctrl=True,
                lshift=False,
                description="Open scene",
                callback=self._cb_open_scene,
                menu="File",
            ),
            Binding(
                key=gui.KeyName.S,
                lctrl=True,
                lshift=False,
                description="Save scene as",
                callback=self._cb_save_scene,
                menu="File",
            ),
            Binding(
                key=gui.KeyName.S,
                lctrl=True,
                lshift=True,
                description="Save scene as",
                callback=self._cb_save_scene_as,
                menu="File",
            ),
            Binding(
                key=gui.KeyName.I,
                lctrl=True,
                lshift=True,
                description="Import all from directory",
                callback=self._cb_import_all_from_directory,
                menu="File",
            ),
            Binding(
                key=gui.KeyName.E,
                lctrl=True,
                lshift=False,
                description="Export current element",
                callback=self._cb_export_current_element,
                menu="File",
            ),
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
                description="Previous",
                callback=self._cb_previous,
                menu=None,
            ),
            Binding(
                key=gui.KeyName.LEFT,
                lctrl=True,
                lshift=False,
                description="Fast Previous",
                callback=self._cb_fast_previous,
                menu=None,
            ),
            Binding(
                key=gui.KeyName.RIGHT,
                lctrl=False,
                lshift=False,
                description="Next",
                callback=self._cb_next,
                menu=None,
            ),
            Binding(
                key=gui.KeyName.RIGHT,
                lctrl=True,
                lshift=False,
                description="Fast Next",
                callback=self._cb_fast_next,
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
                description="Show Hotkeys",
                callback=self._cb_toggle_hotkeys_panel,
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
                description="Repeat last extension",
                callback=self._cb_repeat_last_extension,
                menu="Edit",
            ),
            Binding(
                key=gui.KeyName.Z,
                lctrl=True,
                lshift=False,
                description="Undo",
                callback=self._cb_undo,
                menu="Edit",
            ),
            Binding(
                key=gui.KeyName.Z,
                lctrl=True,
                lshift=True,
                description="Redo",
                callback=self._cb_redo,
                menu="Edit",
            ),
            Binding(
                key=gui.KeyName.U,
                lctrl=True,
                lshift=False,
                description="Hide toggled elements",
                callback=self._cb_hide,
                menu=None,
            ),
            Binding(
                key=gui.KeyName.U,
                lctrl=True,
                lshift=True,
                description="Unhide all hidden elements",
                callback=self._cb_unhide,
                menu=None,
            ),
            Binding(
                key=gui.KeyName.A,
                lctrl=True,
                lshift=False,
                description="Select all",
                callback=self._cb_select_all,
                menu=None,
            ),
            Binding(
                key=gui.KeyName.A,
                lctrl=True,
                lshift=True,
                description="Unselect all",
                callback=self._cb_unselect_all,
                menu=None,
            ),
            Binding(
                key=gui.KeyName.L,
                lctrl=True,
                lshift=False,
                description="Select last",
                callback=self._cb_select_last,
                menu=None,
            ),
            Binding(
                key=gui.KeyName.L,
                lctrl=True,
                lshift=True,
                description="Unselect last",
                callback=self._cb_unselect_last,
                menu=None,
            ),
            Binding(
                key=gui.KeyName.T,
                lctrl=True,
                lshift=False,
                description="[Un]select per type",
                callback=self._cb_toggle_type,
                menu="Edit",
                creates_window=True,
            ),
        ]

        self._dict = {binding.description: binding for binding in self._bindings}

    def _cb_import(self):
        from .io import element, _load_one_element

        editor_instance = self._editor_instance
        window = self._editor_instance._window

        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load", window.theme)

        for type, extensions in element.items():
            if type == "all":
                continue
            dlg.add_filter(extensions["all"], extensions["all_description"])

        dlg.add_filter(element["all"], "All recognized files")
        dlg.add_filter("", "All files")

        def _on_file_dialog_cancel():
            editor_instance._close_dialog()

        def _on_load_dialog_done(filename):
            element = _load_one_element(filename)
            if element is not None:
                try:
                    self._editor_instance.elements.insert_multiple(element, to_gui=True)
                except Exception:
                    warnings.warn("Failed to insert imported file.")
                    traceback.print_exc()
            editor_instance._close_dialog()

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(_on_file_dialog_cancel)
        dlg.set_on_done(_on_load_dialog_done)
        window.show_dialog(dlg)

    def _cb_open_scene(self):
        from .io import _open_scene

        editor_instance = self._editor_instance
        window = self._editor_instance._window

        dlg = gui.FileDialog(
            gui.FileDialog.OPEN,
            "Select scene file to open",
            window.theme,
        )

        dlg.add_filter(".sdscene", "Shape Detector Scene (.sdscene)")

        def _on_cancel():
            editor_instance._close_dialog()

        def _on_done(path):
            try:
                _open_scene(path, editor_instance)
                editor_instance._scene_file_path = path
            except:
                warnings.warn(f"Could not open file on path '{path}'.")
            finally:
                editor_instance._close_dialog()

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(_on_cancel)
        dlg.set_on_done(_on_done)
        window.show_dialog(dlg)

    def _cb_save_scene(self):
        from .io import _save_scene

        editor_instance = self._editor_instance
        path = editor_instance._scene_file_path

        if path is None:
            self._cb_save_scene_as()
        else:
            _save_scene(path, editor_instance)

    def _cb_save_scene_as(self):
        from .io import _save_scene

        editor_instance = self._editor_instance
        window = self._editor_instance._window

        dlg = gui.FileDialog(
            gui.FileDialog.SAVE,
            "Save scene to file",
            window.theme,
        )

        def _on_cancel():
            editor_instance._close_dialog()

        def _on_done(path):
            _save_scene(path, editor_instance)
            editor_instance._scene_file_path = path
            editor_instance._close_dialog()

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(_on_cancel)
        dlg.set_on_done(_on_done)
        window.show_dialog(dlg)

    def _cb_import_all_from_directory(self):
        from .io import RECOGNIZED_EXTENSION, _load_one_element

        editor_instance = self._editor_instance
        window = self._editor_instance._window

        dlg = gui.FileDialog(
            gui.FileDialog.OPEN_DIR,
            "Choose directory to load everything",
            window.theme,
        )

        def _on_cancel():
            editor_instance._close_dialog()

        def _on_done(filename):
            path = Path(filename)

            subpaths = [
                subpath
                for subpath in path.glob("*.*")
                if subpath.suffix in RECOGNIZED_EXTENSION["all"].split()
            ]

            elements = [_load_one_element(subpath) for subpath in subpaths]
            elements = [element for element in elements if element is not None]
            try:
                self._editor_instance.elements.insert_multiple(elements, to_gui=True)
            except Exception:
                warnings.warn("Failed to insert imported files.")
                traceback.print_exc()
            editor_instance._close_dialog()

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(_on_cancel)
        dlg.set_on_done(_on_done)
        window.show_dialog(dlg)

    def _cb_export_current_element(self):
        from .io import _write_one_element

        editor_instance = self._editor_instance
        window = self._editor_instance._window

        dlg = gui.FileDialog(
            gui.FileDialog.SAVE,
            "Export current element to file",
            window.theme,
        )

        def _on_cancel():
            editor_instance._close_dialog()

        def _on_done(filename):
            current_element = editor_instance.elements.current_element
            if current_element is None:
                return

            if _write_one_element(current_element, filename) is not None:
                editor_instance._close_dialog()

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(_on_cancel)
        dlg.set_on_done(_on_done)
        window.show_dialog(dlg)

    def _cb_quit_app(self):
        window = self._editor_instance._window
        em = window.theme.font_size
        button_separation_width = 2 * int(round(0.5 * em))

        dlg = gui.Dialog("Quit Dialog")

        def _on_close_yes():
            self._editor_instance._closing_app = True
            window.close()

        def _on_close_no():
            self._editor_instance._close_dialog()

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Close Shape Detector?"))

        yes = gui.Button("Yes [Enter]")
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

        def _on_key_event(event):
            if event.type == gui.KeyEvent.DOWN and event.key == gui.KeyName.ENTER:
                _on_close_yes()
                return True
            return False

        window.set_on_key(_on_key_event)

    def _cb_toggle(self):
        """Toggle the current highlighted element between selected/unselected."""
        editor_instance = self._editor_instance
        elem = editor_instance.elements.current_element
        if elem is None:
            return

        editor_instance.elements.is_current_selected = (
            not editor_instance.elements.is_current_selected
        )
        editor_instance.elements.update_current_index()

    def _cb_delete(self):
        """Delete selected elements."""
        # Implementing as an extension to save state
        editor_instance = self._editor_instance
        indices = editor_instance.elements.selected_indices
        input_elements = [editor_instance.elements[i].raw for i in indices]

        try:
            editor_instance.print_debug(f"Deleting elements at indices {indices}.")
            editor_instance.elements.pop_multiple(indices, from_gui=True)
        except Exception:
            warnings.warn(f"Could not delete elements at indices {indices}.")
            traceback.print_exc()
            return

        editor_instance._save_state(indices, input_elements, 0)
        editor_instance._future_states = []
        editor_instance._update_plane_boundaries()
        editor_instance._update_BBOX_and_axes()

    def _cb_copy(self):
        """Save elements to be copied."""
        editor_instance = self._editor_instance
        copied_elements = copy.deepcopy(
            [elem.raw for elem in editor_instance.elements if elem.selected]
        )
        editor_instance.print_debug(f"Copying {len(copied_elements)} elements.")
        editor_instance._copied_elements = copied_elements

    def _cb_paste(self):
        editor_instance = self._editor_instance

        try:
            editor_instance.print_debug(
                f"Pasting {len(editor_instance._copied_elements)} elements."
            )
            editor_instance.elements.insert_multiple(
                editor_instance._copied_elements, to_gui=True
            )
        except Exception:
            warnings.warn(
                f"Could not paste {len(editor_instance._copied_elements)} elements."
            )
            traceback.print_exc()

        editor_instance._save_state([], [], len(editor_instance._copied_elements))
        editor_instance._future_states = []
        editor_instance._update_plane_boundaries()

    def _shift_current(self, delta):
        """Shifts 'current index' pointer checking if within limits"""
        editor_instance = self._editor_instance
        new_idx = min(
            max(editor_instance.elements.current_index + delta, 0),
            len(editor_instance.elements) - 1,
        )
        editor_instance.elements.update_current_index(new_idx)

    def _cb_next(self):
        self._shift_current(+1)

    def _cb_fast_next(self):
        self._shift_current(+5)

    def _cb_previous(self):
        self._shift_current(-1)

    def _cb_fast_previous(self):
        self._shift_current(-5)

    def _cb_select_all(self):
        """Toggle the all elements to selected."""
        self._editor_instance.elements.toggle_indices(None, to_value=True)

    def _cb_unselect_all(self):
        """Toggle the all elements between to unselected."""
        self._editor_instance.elements.toggle_indices(None, to_value=False)

    def _cb_select_last(self):
        """Toggle the elements from last output to selected."""
        editor_instance = self._editor_instance
        if len(editor_instance._past_states) == 0:
            return

        num_outputs = editor_instance._past_states[-1]["num_outputs"]
        editor_instance.elements.toggle_indices(
            slice(-num_outputs, None), to_value=True
        )

    def _cb_unselect_last(self):
        """Toggle the elements from last output to unselected."""
        editor_instance = self._editor_instance
        if len(editor_instance._past_states) == 0:
            return

        num_outputs = editor_instance._past_states[-1]["num_outputs"]
        editor_instance.elements.toggle_indices(
            slice(-num_outputs, None), to_value=False
        )

    def _cb_toggle_type(self):
        editor_instance = self._editor_instance
        window = editor_instance._window
        em = window.theme.font_size
        separation_height = int(round(0.5 * em))
        elems_raw = [elem.raw for elem in editor_instance.elements]

        temp_window = editor_instance.app.create_window("Select type", 200, 400)
        temp_window.show_menu(False)
        self._editor_instance._temp_windows.append(temp_window)

        def _callback(_type, value):
            is_type = [isinstance(elem.raw, _type) for elem in editor_instance.elements]
            indices = np.where(is_type)[0].tolist()
            editor_instance.elements.toggle_indices(indices, to_value=value)

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
            temp_window.close()
            self._editor_instance._temp_windows.remove(temp_window)

        ok = gui.Button("Ok")
        ok.set_on_clicked(_on_ok)

        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        temp_window.add_child(dlg_layout)

    def _cb_repeat_last_extension(self):
        editor_instance = self._editor_instance
        ext = editor_instance._last_used_extension

        if ext is not None:
            editor_instance.print_debug(f"Re-applying last function: {ext.name}")
            ext._apply_to_elements()

    def _cb_toggle_hotkeys_panel(self):
        self._editor_instance._menu_help._on_help_toggle()

    def _cb_toggle_info_panel(self):
        self._editor_instance._menu_help._on_info_toggle()

    def _cb_toggle_settings_panel(self):
        self._editor_instance._settings._on_menu_toggle()

    def _cb_hide(self):
        """Hide selected elements."""

        editor_instance = self._editor_instance
        indices = editor_instance.elements.selected_indices
        if len(indices) == 0:
            return

        editor_instance._elements_hidden += editor_instance.elements.pop_multiple(
            indices, from_gui=True
        )
        editor_instance.elements.is_selected = False

        # TODO: find a way to make hiding work with undoing
        editor_instance._past_states = []
        editor_instance._future_states = []
        editor_instance._update_plane_boundaries()

    def _cb_unhide(self):
        """Unhide all hidden elements."""

        editor_instance = self._editor_instance
        hidden_elements = editor_instance._elements_hidden

        if len(hidden_elements) == 0:
            return

        editor_instance.elements.insert_multiple(
            hidden_elements.pop_multiple(range(len(hidden_elements))),
            selected=True,
            to_gui=True,
        )

        # TODO: find a way to make hiding work with undoing
        editor_instance._past_states = []
        editor_instance._future_states = []
        editor_instance._update_plane_boundaries()

    def _cb_undo(self):
        editor_instance = self._editor_instance

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

        modified_elements = editor_instance.elements.pop_multiple(
            indices_to_pop, from_gui=True
        )

        editor_instance.elements.insert_multiple(
            elements, indices, selected=True, to_gui=True
        )

        editor_instance._future_states.append(
            {
                "modified_elements": modified_elements,
                "indices": indices,
                "current_index": editor_instance.elements.current_index,
            }
        )

        while len(editor_instance._future_states) > editor_instance._get_preference(
            "number_redo_states"
        ):
            editor_instance._future_states.pop(0)

        if len(indices) > 0:
            editor_instance.elements.update_current_index(indices[-1])

        editor_instance._update_plane_boundaries()

    def _cb_redo(self):
        editor_instance = self._editor_instance

        if len(editor_instance._future_states) == 0:
            return

        future_state = editor_instance._future_states.pop()

        modified_elements = future_state["modified_elements"]
        indices = future_state["indices"]

        editor_instance.print_debug(
            f"Redoing last operation, removing {len(indices)} inputs and "
            f"resetting {len(modified_elements)} inputs."
        )

        input_elements = [editor_instance.elements[i].raw for i in indices]
        editor_instance._save_state(indices, input_elements, len(modified_elements))

        editor_instance.elements.current_index = future_state["current_index"]
        editor_instance.elements.pop_multiple(indices, from_gui=True)
        editor_instance.elements.insert_multiple(modified_elements, to_gui=True)

        editor_instance.elements.update_current_index(len(editor_instance.elements) - 1)

        editor_instance._update_plane_boundaries()
