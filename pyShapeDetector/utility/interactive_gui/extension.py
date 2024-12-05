#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import warnings
import inspect
from typing import Union, Callable

from open3d.visualization import gui

from .editor_app import Editor
from .parameter import PARAMETER_TYPE_DICTIONARY
from .helpers import get_pretty_name
from .binding import Binding


class Extension:
    DEFAULT_MENU_NAME = "Misc functions"

    @property
    def binding(self):
        return self._binding

    @property
    def function(self):
        return self._function

    @property
    def name(self):
        return self._name

    @property
    def menu(self):
        return self._menu

    @property
    def parameters(self):
        return self._parameters

    @property
    def parameters_kwargs(self):
        return {key: param.value for key, param in self.parameters.items()}

    @property
    def hotkey(self):
        return self._hotkey

    @property
    def lctrl(self):
        return self._lctrl

    @property
    def lshift(self):
        return self._lshift

    @property
    def hotkey_number(self):
        if self.hotkey is None:
            return None
        return int(chr(self.hotkey))

    def _set_name(self, descriptor: dict):
        name = descriptor.get("name", get_pretty_name(self.function))
        if not isinstance(name, str):
            raise TypeError("Name expected to be string.")
        self._name = name

    def _set_menu(self, descriptor: dict):
        menu = descriptor.get("menu", Extension.DEFAULT_MENU_NAME)
        if not isinstance(menu, str):
            raise TypeError("Menu expected to be string.")
        self._menu = menu

    def _set_hotkey(self, descriptor: dict):
        hotkey = descriptor.get("hotkey", None)
        lctrl = descriptor.get("lctrl", None)
        lshift = descriptor.get("lshift", None)

        if hotkey is None:
            if lctrl is not None:
                warnings.warn(
                    "Hotkey for extension {self.name} set to 'None', "
                    "ignoring 'lctrl' input."
                )

            if lshift is not None:
                warnings.warn(
                    "Hotkey for extension {self.name} set to 'None', "
                    "ignoring 'lshift' input."
                )

            self._hotkey = None
            self._lctrl = False
            self._lshift = False
            return

        if not isinstance(hotkey, int) or not (0 <= hotkey <= 9):
            warnings.warn(
                f"Expected integer hotkey between 0 and 9, got {hotkey}. "
                "Ignoring hotkey"
            )
            self._hotkey = None
            self._lctrl = False
            self._lshift = False
            return

        self._hotkey = ord(str(hotkey))

        if lctrl is None:
            self._lctrl = False
        else:
            self._lctrl = bool(lctrl)

        if lshift is None:
            self._lshift = False
        else:
            self._lshift = bool(lshift)

    def _set_parameters(self, descriptor: dict):
        signature = inspect.signature(self.function)
        parsed_parameters = {}
        parameter_descriptors = descriptor.get("parameters", {})

        if not isinstance(parameter_descriptors, dict):
            raise TypeError("parameters expected to be dict.")

        for key, parameter in parameter_descriptors.items():
            if key not in signature.parameters.keys():
                raise ValueError(
                    f"Function '{self.function.__name__}' from extension '{self.name}' does not take parameter '{key}'."
                )
            parameter_type = PARAMETER_TYPE_DICTIONARY[parameter.get("type")]
            parsed_parameters[key] = parameter_type(key, parameter)

        self._parameters = parsed_parameters

    def __init__(self, function_or_descriptor: Union[Callable, dict]):
        if isinstance(function_or_descriptor, dict):
            if "function" not in function_or_descriptor:
                raise ValueError("Dict descriptor does not contain 'function'.")
            descriptor = copy.copy(function_or_descriptor)
        elif callable(function_or_descriptor):
            descriptor = {"function": function_or_descriptor}
        else:
            raise TypeError("Input should be either a dict descriptor or a function.")

        self._function = descriptor["function"]
        self._set_name(descriptor)
        self._set_menu(descriptor)
        self._set_hotkey(descriptor)
        self._set_parameters(descriptor)

        self._binding = Binding(
            key=self.hotkey,
            lctrl=self.lctrl,
            lshift=self.lshift,
            description=self.name,
            menu=self.menu,
            callback=self.run,
            creates_window=len(self.parameters) > 0,
        )

    def add_to_application(self, editor_instance: Editor):
        self._editor_instance = editor_instance

        if editor_instance._extensions is None:
            editor_instance._extensions = []

        editor_instance._extensions.append(self)

    def add_menu_item(self):
        self.binding.add_to_menu(self._editor_instance)

    def update_in_separate_window(self):
        editor_instance = self._editor_instance

        if len(self.parameters) == 0:
            return gui.Widget.EventCallbackResult.IGNORED

        app = editor_instance.app

        temp_window = app.create_window(
            f"Parameter selection for {self.name}", 400, 600
        )
        temp_window.show_menu(False)
        em = temp_window.theme.font_size

        self._accepted = False

        separation_height = int(round(0.5 * em))
        button_separation_width = 2 * separation_height

        # dlg = gui.Dialog("Parameter selection")
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))

        label = gui.Label("Enter parameters:")
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(label)
        h.add_stretch()
        dlg_layout.add_child(h)

        previous_values = {}
        for key, param in self.parameters.items():
            previous_values[key] = copy.copy(param.value)
            param._reset_values_and_limits(editor_instance)
            dlg_layout.add_child(param.get_gui_element(temp_window))
            dlg_layout.add_fixed(separation_height)

        def _on_accept():
            self._accepted = True
            temp_window.close()

            self._editor_instance._apply_function_to_elements(
                self, update_parameters=False
            )

        def _on_cancel():
            temp_window.close()

        def _on_close():
            if not self._accepted:
                for key, param in self.parameters.items():
                    param.value = previous_values[key]

            return True

        accept = gui.Button("Accept")
        accept.set_on_clicked(_on_accept)
        cancel = gui.Button("Cancel")
        cancel.set_on_clicked(_on_cancel)
        temp_window.set_on_close(_on_close)

        h = gui.Horiz()
        h.add_stretch()
        h.add_child(accept)
        h.add_fixed(button_separation_width)
        h.add_child(cancel)
        h.add_stretch()
        dlg_layout.add_child(h)
        temp_window.add_child(dlg_layout)

        return gui.Widget.EventCallbackResult.HANDLED

    def run(self):
        event_result = self.update_in_separate_window()
        if event_result is gui.Widget.EventCallbackResult.HANDLED:
            return

        self._editor_instance._apply_function_to_elements(self, update_parameters=False)
