from abc import ABC
import copy
import numpy as np
import warnings
import traceback
from importlib.util import find_spec
from typing import List, Union, TYPE_CHECKING
from pathlib import Path
from open3d.visualization import gui
from .binding import Binding

if has_pyperclip := find_spec("pyperclip"):
    import pyperclip

if TYPE_CHECKING:
    from .editor_app import Editor


class InternalFunctions:
    @property
    def bindings(self):
        return self._bindings

    def __init__(self, editor_instance: "Editor"):
        self._editor_instance = editor_instance
        self._bindings = [
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
                description="Save scene",
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
                lshift=False,
                description="Import",
                callback=self._cb_import,
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
                # menu_id=100,
                key=gui.KeyName.E,
                lctrl=True,
                lshift=False,
                description="Export current element",
                callback=self._cb_export_current_element,
                menu="File/Export",
            ),
            Binding(
                # menu_id=101,
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
                menu="Edit",
            ),
            Binding(
                key=gui.KeyName.U,
                lctrl=True,
                lshift=True,
                description="Unhide all hidden elements",
                callback=self._cb_unhide,
                menu="Edit",
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

    def _cb_import(self, path=None):
        from .io import _load_one_element, RECOGNIZED_EXTENSION

        editor_instance = self._editor_instance

        if path is not None:
            element = _load_one_element(path)
            self._editor_instance._settings.print_debug(f"Loaded element: {element}")
            if element is not None:
                try:
                    self._editor_instance.element_container.insert_multiple(
                        element, to_gui=True
                    )
                    editor_instance._update_info()
                    editor_instance._reset_camera()
                except Exception:
                    warnings.warn(f"Failed to imported from file '{path}'.")
                    traceback.print_exc()
            return

        window = self._editor_instance._main_window

        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load", window.theme)
        if editor_instance._scene_file_path is not None:
            try:
                dlg.set_path(Path(editor_instance._scene_file_path).parent.as_posix())
            except Exception:
                pass

        # First all recognized files...
        dlg.add_filter(RECOGNIZED_EXTENSION["all"], "All recognized files")

        # ... then general types...
        for type_name, extensions in RECOGNIZED_EXTENSION.items():
            if type_name == "all":
                continue

            dlg.add_filter(extensions["all"], extensions["all_description"])

        # ... finally specific extensions
        for type_name, extensions in RECOGNIZED_EXTENSION.items():
            if type_name == "all":
                continue

            for extension, description in extensions.items():
                if extension[0] != ".":
                    continue
                dlg.add_filter(extension, description)

        dlg.add_filter("", "All files")

        def _on_file_dialog_cancel():
            editor_instance._close_dialog()

        def _on_load_dialog_done(path):
            self._cb_import(path)
            editor_instance._close_dialog()

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(_on_file_dialog_cancel)
        dlg.set_on_done(_on_load_dialog_done)
        window.show_dialog(dlg)

    def _cb_open_scene(self, path=None):
        from .io import _open_scene, SCENE_FILE_EXTENSION

        editor_instance = self._editor_instance

        if path is not None:
            try:
                _open_scene(path, editor_instance)
                editor_instance._scene_file_path = path
            except Exception:
                warnings.warn(f"Could not open scene on file '{path}'.")
                traceback.print_exc()
            # finally:
            # editor_instance._close_dialog()
            return

        window = self._editor_instance._main_window

        dlg = gui.FileDialog(
            gui.FileDialog.OPEN, "Select scene file to open", window.theme
        )
        if editor_instance._scene_file_path is not None:
            try:
                dlg.set_path(Path(editor_instance._scene_file_path).parent.as_posix())
            except Exception:
                pass

        dlg.add_filter(
            f"{SCENE_FILE_EXTENSION}", f"Shape Detector Scene ({SCENE_FILE_EXTENSION})"
        )

        def _on_cancel():
            editor_instance._close_dialog()

        def _on_done(path):
            try:
                self._cb_open_scene(path)
            except Exception:
                warnings.warn(f"Could not open file on path '{path}'.")
                traceback.print_exc()
            finally:
                editor_instance._close_dialog()

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(_on_cancel)
        dlg.set_on_done(_on_done)
        window.show_dialog(dlg)

    def _cb_save_scene(self, quitting=False):
        from .io import _save_scene

        editor_instance = self._editor_instance
        path = editor_instance._scene_file_path

        if path is None:
            self._cb_save_scene_as(quitting)
        else:

            def _save_thread():
                _save_scene(path, editor_instance)
                editor_instance._close_dialog()

            editor_instance._create_simple_dialog(
                title_text=f"Saving scene to {path}...",
                create_button=False,
            )

            editor_instance.app.run_in_thread(_save_thread)
            if quitting:
                self._editor_instance._closing_app = True
                self._editor_instance._main_window.close()

    def _cb_save_scene_as(self, quitting=False):
        from .io import _create_overwrite_warning, SCENE_FILE_EXTENSION

        editor_instance = self._editor_instance
        window = self._editor_instance._main_window

        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Save scene to file", window.theme)
        if editor_instance._scene_file_path is not None:
            try:
                dlg.set_path(Path(editor_instance._scene_file_path).parent.as_posix())
            except Exception:
                pass

        dlg.add_filter(
            f"{SCENE_FILE_EXTENSION}", f"Shape Detector Scene ({SCENE_FILE_EXTENSION})"
        )

        def _on_cancel():
            editor_instance._close_dialog()

        def _on_done(path):
            # path = Path(path)
            if Path(path).exists() and path != editor_instance._scene_file_path:
                _create_overwrite_warning(editor_instance, path, quitting)
            else:
                editor_instance._scene_file_path = path
                self._cb_save_scene()

                if quitting:
                    self._editor_instance._closing_app = True
                    self._editor_instance._main_window.close()
                else:
                    editor_instance._close_dialog()

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(_on_cancel)
        dlg.set_on_done(_on_done)
        window.show_dialog(dlg)

    def _cb_import_all_from_directory(self):
        from .io import RECOGNIZED_EXTENSION, _load_one_element

        editor_instance = self._editor_instance
        window = self._editor_instance._main_window

        dlg = gui.FileDialog(
            gui.FileDialog.OPEN_DIR, "Choose directory to load everything", window.theme
        )
        if editor_instance._scene_file_path is not None:
            try:
                dlg.set_path(Path(editor_instance._scene_file_path).parent.as_posix())
            except Exception:
                pass

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
                elements_flattened = []
                for element in elements:
                    if isinstance(element, list):
                        elements_flattened += element
                    else:
                        elements_flattened.append(element)

                self._editor_instance.element_container.insert_multiple(
                    elements_flattened, to_gui=True
                )
            except Exception:
                warnings.warn("Failed to insert imported files.")
                traceback.print_exc()
            editor_instance._close_dialog()
            editor_instance._update_info()

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(_on_cancel)
        dlg.set_on_done(_on_done)
        window.show_dialog(dlg)

    def _cb_export_current_element(self):
        from .io import _write_one_element

        editor_instance = self._editor_instance
        window = self._editor_instance._main_window

        dlg = gui.FileDialog(
            gui.FileDialog.SAVE,
            "Export current element to file",
            window.theme,
        )
        if editor_instance._scene_file_path is not None:
            try:
                dlg.set_path(Path(editor_instance._scene_file_path).parent.as_posix())
            except Exception:
                pass

        def _on_cancel():
            editor_instance._close_dialog()

        def _on_done(filename):
            current_element = editor_instance.element_container.current_element
            if current_element is None:
                return

            if _write_one_element(current_element, filename) is not None:
                editor_instance._close_dialog()

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(_on_cancel)
        dlg.set_on_done(_on_done)
        window.show_dialog(dlg)

    def _cb_quit_app(self):
        window = self._editor_instance._main_window
        em = window.theme.font_size
        button_separation_width = 2 * int(round(0.5 * em))

        dlg = gui.Dialog("Quit Dialog")

        def _on_quit():
            self._editor_instance._closing_app = True
            window.close()

        def _on_quit_and_save():
            self._cb_save_scene(quitting=True)

        def _on_close_no():
            self._editor_instance._close_dialog()

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Close Shape Detector?"))

        save_and_quit = gui.Button("Save and quit [Enter]")
        save_and_quit.set_on_clicked(_on_quit_and_save)

        quit_without_saving = gui.Button("Quit without saving")
        quit_without_saving.set_on_clicked(_on_quit)

        cancel = gui.Button("Cancel")
        cancel.set_on_clicked(_on_close_no)

        h = gui.Horiz()
        h.add_stretch()
        h.add_child(save_and_quit)
        h.add_fixed(button_separation_width)
        h.add_child(quit_without_saving)
        h.add_fixed(button_separation_width)
        h.add_child(cancel)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        window.show_dialog(dlg)

        def _on_key_event(event):
            if event.type == gui.KeyEvent.DOWN and event.key == gui.KeyName.ENTER:
                _on_quit_and_save()
                return True
            return False

        window.set_on_key(_on_key_event)

    def _cb_toggle(self):
        """Toggle the current highlighted element between selected/unselected."""
        editor_instance = self._editor_instance
        elem = editor_instance.element_container.current_element
        if elem is None:
            return

        elem.is_selected = not elem.is_selected
        editor_instance.element_container.update_current_index()
        editor_instance._update_extra_elements(planes_boundaries=False)

    def _cb_delete(self):
        """Delete selected elements."""
        # Implementing as an extension to save state
        editor_instance = self._editor_instance
        indices = editor_instance.element_container.selected_indices
        if len(indices) == 0:
            return
        input_elements = [editor_instance.element_container[i].raw for i in indices]

        try:
            self._editor_instance._settings.print_debug(
                f"Deleting elements at indices {indices}."
            )
            editor_instance.element_container.pop_multiple(indices, from_gui=True)
        except Exception:
            warnings.warn(f"Could not delete elements at indices {indices}.")
            traceback.print_exc()
            return

        current_state = {
            "indices": copy.deepcopy(indices),
            "elements": copy.deepcopy(input_elements),
            "num_outputs": 0,
            "current_index": editor_instance.element_container.current_index,
            "operation": "delete",
        }
        editor_instance._save_state(current_state)
        editor_instance._update_extra_elements(planes_boundaries=True)

    def _cb_copy(self):
        """Save elements to be copied."""
        editor_instance = self._editor_instance
        copied_elements = copy.deepcopy(
            [elem.raw for elem in editor_instance.element_container if elem.is_selected]
        )
        editor_instance._settings.print_debug(
            f"Copying {len(copied_elements)} elements."
        )
        editor_instance._copied_elements = copied_elements

        if has_pyperclip:
            try:
                full_text = ",\n".join([str(elem) for elem in copied_elements])
                pyperclip.copy(full_text)

            except Exception:
                # Not important
                pass

    def _cb_paste(self):
        editor_instance = self._editor_instance

        try:
            editor_instance._settings.print_debug(
                f"Pasting {len(editor_instance._copied_elements)} elements.",
            )
            editor_instance.element_container.insert_multiple(
                editor_instance._copied_elements, to_gui=True
            )
        except Exception:
            warnings.warn(
                f"Could not paste {len(editor_instance._copied_elements)} elements."
            )
            traceback.print_exc()

        current_state = {
            "indices": [],
            "elements": [],
            "num_outputs": len(editor_instance._copied_elements),
            "current_index": editor_instance.element_container.current_index,
            "operation": "paste",
        }

        editor_instance._save_state(current_state)
        editor_instance._update_extra_elements(planes_boundaries=True)

    def _shift_current(self, delta: int):
        """Shifts 'current index' pointer checking if within limits"""
        if delta == 0:
            # should not happen though
            return

        elements = self._editor_instance.element_container
        unhidden_indices = elements.unhidden_indices
        current_index = elements.current_index

        try:
            new_position = unhidden_indices.index(current_index) + delta
            new_position = max(0, min(new_position, len(unhidden_indices) - 1))
            elements.update_current_index(unhidden_indices[new_position])
            self._editor_instance._update_extra_elements(planes_boundaries=False)
        except Exception:
            warnings.warn("Could not shift index.")
            traceback.print_exc()

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
        self._editor_instance.element_container.toggle_indices(None, to_value=True)
        self._editor_instance._update_info()

    def _cb_unselect_all(self):
        """Toggle the all elements between to unselected."""
        self._editor_instance.element_container.toggle_indices(None, to_value=False)
        self._editor_instance._update_info()

    def _cb_select_last(self):
        """Toggle the elements from last output to selected."""
        editor_instance = self._editor_instance
        if len(editor_instance._past_states) == 0:
            return

        last_state = editor_instance._past_states[-1]
        if "num_outputs" not in last_state:
            return

        if (num_outputs := last_state["num_outputs"]) == 0:
            return

        editor_instance.element_container.toggle_indices(
            slice(-num_outputs, None), to_value=True
        )

        self._editor_instance._update_info()

    def _cb_unselect_last(self):
        """Toggle the elements from last output to unselected."""
        editor_instance = self._editor_instance
        if len(editor_instance._past_states) == 0:
            return

        if (num_outputs := editor_instance._past_states[-1]["num_outputs"]) == 0:
            return

        print(num_outputs)

        editor_instance.element_container.toggle_indices(
            slice(-num_outputs, None), to_value=False
        )

        self._editor_instance._update_info()

    def _cb_toggle_type(self):
        editor_instance = self._editor_instance
        window = editor_instance._main_window
        em = window.theme.font_size
        separation_height = int(round(0.5 * em))
        elems_raw = [elem.raw for elem in editor_instance.element_container]

        temp_window = editor_instance.app.create_window("Select type", 200, 400)
        temp_window.show_menu(False)
        self._editor_instance._temp_windows.append(temp_window)

        def _callback(_type, value):
            is_type = [
                isinstance(elem.raw, _type)
                for elem in editor_instance.element_container
            ]
            indices = np.where(is_type)[0].tolist()
            editor_instance.element_container.toggle_indices(indices, to_value=value)
            self._editor_instance._update_info()

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
        temp_window.size_to_fit()

    def _cb_repeat_last_extension(self):
        editor_instance = self._editor_instance
        ext = editor_instance._last_used_extension

        if ext is not None:
            editor_instance._settings.print_debug(
                f"Re-applying last function: {ext.name}"
            )
            ext._apply_to_elements_in_thread()

    def _cb_hide(
        self,
        indices: Union[List[int], None] = None,
        to_future: bool = False,
        delete_future: bool = True,
    ):
        """Hide selected elements."""

        editor_instance = self._editor_instance
        elements = editor_instance.element_container

        if indices is None:
            indices = elements.selected_indices
        elif len(indices) == 0:
            return

        for idx in indices:
            elem = editor_instance.element_container[idx]
            elem.is_hidden = True
            elem.is_selected = False

        if elements.current_index in indices:
            new_index = elements.get_closest_unhidden_index()
            elements.update_current_index(new_index)

        editor_instance._update_extra_elements(planes_boundaries=False)

        editor_instance._save_state(
            {"indices": indices, "operation": "hide"},
            to_future=to_future,
            delete_future=delete_future,
        )

    def _cb_unhide(
        self,
        indices: Union[List[int], None] = None,
        to_future: bool = False,
        delete_future: bool = True,
    ):
        """Unhide all hidden elements."""

        editor_instance = self._editor_instance
        print(indices)

        if indices is None:
            indices = editor_instance.element_container.hidden_indices
        elif len(indices) == 0:
            return

        for idx in indices:
            elem = editor_instance.element_container[idx]
            elem.is_hidden = False

        editor_instance._update_extra_elements(planes_boundaries=False)

        editor_instance._save_state(
            {"indices": indices, "operation": "unhide"},
            to_future=to_future,
            delete_future=delete_future,
        )

    def _cb_undo(self):
        editor_instance = self._editor_instance

        if len(editor_instance._past_states) == 0:
            return

        last_state = editor_instance._past_states.pop()
        indices = last_state["indices"]

        if last_state["operation"] == "unhide":
            self._cb_hide(indices, to_future=True)
            return

        if last_state["operation"] == "hide":
            self._cb_unhide(indices, to_future=True)
            return

        if last_state["operation"] == "transformation":
            indices = last_state["indices"]
            transformation_matrix = last_state["transformation_matrix"]
            transformation_matrix_inverse = np.linalg.inv(transformation_matrix)
            for index in indices:
                editor_instance.element_container[index].transform(
                    transformation_matrix_inverse
                )
            editor_instance._save_state(last_state, to_future=True, delete_future=False)
            editor_instance.element_container.update_current_index(
                last_state["current_index"]
            )
            editor_instance._update_extra_elements(planes_boundaries=True)
            return

        elements = last_state["elements"]
        num_outputs = last_state["num_outputs"]
        num_elems = len(editor_instance.element_container)

        editor_instance._settings.print_debug(
            f"Undoing last operation, removing {num_outputs} outputs and "
            f"resetting {len(elements)} inputs.",
        )

        indices_to_pop = range(num_elems - num_outputs, num_elems)

        modified_elements = editor_instance.element_container.pop_multiple(
            indices_to_pop, from_gui=True
        )

        editor_instance.element_container.insert_multiple(
            elements, indices, is_selected=True, to_gui=True
        )

        # editor_instance._future_states.append()
        current_state = {
            "modified_elements": modified_elements,
            "indices": indices,
            "current_index": editor_instance.element_container.current_index,
            "operation": "undo",
        }
        editor_instance._save_state(current_state, to_future=True, delete_future=False)

        if len(indices) > 0:
            editor_instance.element_container.update_current_index(indices[-1])

        editor_instance._update_extra_elements(planes_boundaries=True)

    def _cb_redo(self):
        editor_instance = self._editor_instance

        if len(editor_instance._future_states) == 0:
            return

        future_state = editor_instance._future_states.pop()
        indices = future_state["indices"]

        if future_state["operation"] == "unhide":
            self._cb_hide(indices, delete_future=False)
            return

        if future_state["operation"] == "hide":
            self._cb_unhide(indices, delete_future=False)
            return

        if future_state["operation"] == "transformation":
            indices = future_state["indices"]
            transformation_matrix = future_state["transformation_matrix"]
            for index in indices:
                editor_instance.element_container[index].transform(
                    transformation_matrix
                )
            editor_instance._save_state(
                future_state, to_future=False, delete_future=False
            )
            editor_instance.element_container.update_current_index(
                future_state["current_index"]
            )
            editor_instance._update_extra_elements(planes_boundaries=True)
            return

        modified_elements = future_state["modified_elements"]

        editor_instance._settings.print_debug(
            f"Redoing last operation, removing {len(indices)} inputs and "
            f"resetting {len(modified_elements)} inputs.",
        )

        input_elements = [editor_instance.element_container[i].raw for i in indices]

        current_state = {
            "indices": copy.deepcopy(indices),
            "elements": copy.deepcopy(input_elements),
            "num_outputs": len(modified_elements),
            "current_index": editor_instance.element_container.current_index,
            "operation": "redo",
        }
        editor_instance._save_state(current_state, delete_future=False)

        editor_instance.element_container.current_index = future_state["current_index"]
        editor_instance.element_container.pop_multiple(indices, from_gui=True)
        editor_instance.element_container.insert_multiple(
            modified_elements, to_gui=True
        )
        editor_instance.element_container.update_current_index(
            len(editor_instance.element_container) - 1
        )
        editor_instance._update_extra_elements(planes_boundaries=True)
