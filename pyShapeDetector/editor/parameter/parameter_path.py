#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-01-23 12:51:07

@author: evbernardes
"""
from typing import Callable, Union, TYPE_CHECKING
from pathlib import Path
from open3d.visualization import gui
from .parameter import ParameterBase

if TYPE_CHECKING:
    from ..editor_app import Editor


class ParameterPath(ParameterBase[Path]):
    """Parameter for Paths

    Seting "editor_instance" allows the creation of a dialog to search for path.

    Attributes
    ----------
    is_reference
    valid_arguments
    type
    value
    type_name
    label
    subpanel
    on_update
    editor_instance

    Methods
    -------
    _warn_unused_parameters
    _callback
    _update_internal_element
    _reset_values_and_limits
    _update_references
    _enable_internal_element
    get_gui_widget
    create_reference
    create_from_dict
    """

    _type = Path
    _editor_instance = None

    @property
    def editor_instance(self) -> "Editor":
        return self._editor_instance

    @editor_instance.setter
    def editor_instance(self, editor_instance: "Editor"):
        from ..editor_app import Editor

        if not isinstance(editor_instance, Editor):
            raise TypeError(f"Expected Editor, got {editor_instance}.")
        self._editor_instance = editor_instance

    @ParameterBase.value.setter
    def value(self, new_value):
        self._value = Path(new_value)
        # if self.is_reference:
        self._update_internal_element()

    def _update_internal_element(self):
        if self.internal_element is None:
            return
        self.internal_element.text_value = str(self.value)

    # def _callback(self, value):
    #     self.value = value
    #     self.on_update(self.value)
    #     self._update_references()

    def _on_open_dir(self):
        editor_instance = self.editor_instance
        window = editor_instance._main_window

        dlg = gui.FileDialog(
            gui.FileDialog.OPEN_DIR, "Chose Directory to export...", window.theme
        )
        if self.value is not Path():
            try:
                dlg.set_path(Path(self.value).parent.as_posix())
            except Exception:
                pass

        def _on_cancel():
            editor_instance._close_dialog()

        def _on_done(directory):
            self.value = directory
            # if current_element is None:
            #     return

            # if _write_one_element(current_element, directory) is not None:
            editor_instance._close_dialog()

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(_on_cancel)
        dlg.set_on_done(_on_done)
        window.show_dialog(dlg)

    def get_gui_widget(self, font_size):
        text_edit = self._internal_element = gui.TextEdit()
        self._update_internal_element()

        # text_edit.set_on_text_changed(self._callback)
        text_edit.set_on_value_changed(self._callback)
        self._enable_internal_element(not self.is_reference)

        # label = gui.Label(self.label)
        if self.editor_instance is None:
            return text_edit

        button_open_dir = gui.Button(f"Open {self.label}...")
        button_open_dir.set_on_clicked(self._on_open_dir)

        element = gui.VGrid(1, 0.25 * font_size)
        # label_and_button = gui.Horiz(0.25 * font_size)
        # label_and_button.add_child(label)
        # label_and_button.add_child(button_open_dir)
        element.add_child(button_open_dir)
        element.add_child(text_edit)

        return element

    def __init__(
        self,
        label: str,
        default: Union[str, Path] = "",
        on_update: Callable = None,
        subpanel: Union[str, None] = None,
        **other_kwargs,
    ):
        super().__init__(label=label, on_update=on_update, subpanel=subpanel)
        self.value = default
        self._warn_unused_parameters(other_kwargs)
