#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import warnings
import inspect
import traceback
from typing import Union, Callable

from open3d.visualization import gui

from .editor_app import Editor
from .parameter import PARAMETER_TYPE_DICTIONARY
from .helpers import get_pretty_name
from .binding import Binding

VALID_INPUTS = ("current", "selected", "global")


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
    def inputs(self):
        return self._inputs

    @property
    def keep_inputs(self):
        return self._keep_inputs

    @property
    def select_outputs(self):
        return self._select_outputs

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

        if isinstance(hotkey, str) and str.isnumeric(hotkey):
            hotkey = int(hotkey)

        if isinstance(hotkey, int) and (0 <= hotkey <= 9):
            self._hotkey = ord(str(hotkey))
        elif isinstance(hotkey, str) and str.isalpha(hotkey):
            self._hotkey = getattr(gui.KeyName, hotkey.upper())
        else:
            warnings.warn(
                f"Hotkey expected to be either letter or integer between 0 and 9"
                f", got {hotkey}. Ignoring hotkey"
            )
            self._hotkey = None
            self._lctrl = False
            self._lshift = False
            return

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

    def _set_misc(self, descriptor: dict):
        inputs = descriptor.get("inputs", "selected")
        if inputs not in VALID_INPUTS:
            raise ValueError(
                f"Possible values for 'inputs' are {VALID_INPUTS}, got {inputs}."
            )
        self._inputs = inputs
        self._keep_inputs = bool(descriptor.get("keep_inputs", False))
        select_outputs = bool(descriptor.get("select_outputs", False))

        if select_outputs and inputs == "current":
            raise ValueError(
                "'select_outputs' should not be True for inputs of type 'current'."
            )
        self._select_outputs = select_outputs

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
        self._set_misc(descriptor)

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

    def run(self):
        event_result = self._create_extension_window()
        if event_result is gui.Widget.EventCallbackResult.HANDLED:
            return

        self._apply_to_elements()

    def _apply_to_elements(self):
        editor_instance = self._editor_instance

        if self.inputs == "current":
            indices = [editor_instance.i]
            input_elements = editor_instance.current_element["raw"]
        elif self.inputs == "selected":
            indices = editor_instance.selected_indices
            input_elements = [editor_instance.elements[i]["raw"] for i in indices]
        elif self.inputs == "current":
            indices = list(range(len(editor_instance.elements)))
            input_elements = [editor_instance.elements[i]["raw"] for i in indices]
        else:
            raise RuntimeError(
                f"Invalid input instruction {self.inputs} "
                f"found in extension {self.name}."
            )

        # Debug lines
        editor_instance.print_debug(f"Applying {self.name} to {len(indices)} elements")
        editor_instance.print_debug(f"Extension has input type {self.inputs}")
        editor_instance.print_debug(f"Indices: {indices}.", require_verbose=True)
        if len(self.parameters) > 0:
            editor_instance.print_debug(f"Parameters: {self.parameters}.")

        try:
            output_elements = self.function(input_elements, **self.parameters_kwargs)
        except KeyboardInterrupt:
            return
        except Exception as e:
            warnings.warn(
                f"Failed to apply {self.name} extension to "
                f"elements in indices {indices}, got:"
            )
            traceback.print_exc()

        # assures it's a list
        if isinstance(output_elements, tuple):
            output_elements = list(output_elements)
        elif not isinstance(output_elements, list):
            output_elements = [output_elements]

        if self.inputs == "current" and len(output_elements) != 1:
            warnings.warn(
                "Only one output expected for extensions with "
                f"'current' input type, got {len(output_elements)}."
            )
            return

        editor_instance._save_state(indices, input_elements, len(output_elements))

        if self.inputs == "current":
            editor_instance._insert_elements(
                output_elements,
                to_gui=True,
                selected=input_elements["selected"],
            )
            editor_instance._update_current_idx(len(self.elements) - 1)
        else:
            editor_instance._insert_elements(
                output_elements, to_gui=True, selected=self.select_outputs
            )

        if not self.keep_inputs:
            assert (
                editor_instance._pop_elements(indices, from_gui=True) == input_elements
            )

        editor_instance._last_used_extension = self
        editor_instance._future_states = []
        editor_instance._update_plane_boundaries()

    def _create_extension_window(self):
        editor_instance = self._editor_instance

        if len(self.parameters) == 0:
            return gui.Widget.EventCallbackResult.IGNORED

        app = editor_instance.app

        temp_window = app.create_window(
            f"Parameter selection for {self.name}", 400, 600
        )
        temp_window.show_menu(False)
        em = temp_window.theme.font_size

        self._ran_at_least_once = False

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

        def _on_apply():
            self._ran_at_least_once = True
            self._apply_to_elements()

        def _on_close():
            if not self._ran_at_least_once:
                for key, param in self.parameters.items():
                    param.value = previous_values[key]

            temp_window.close()

        apply = gui.Button("Apply")
        apply.set_on_clicked(_on_apply)
        close = gui.Button("Close")
        close.set_on_clicked(_on_close)

        h = gui.Horiz()
        h.add_stretch()
        h.add_child(apply)
        h.add_fixed(button_separation_width)
        h.add_child(close)
        h.add_stretch()
        dlg_layout.add_child(h)
        temp_window.add_child(dlg_layout)

        return gui.Widget.EventCallbackResult.HANDLED
