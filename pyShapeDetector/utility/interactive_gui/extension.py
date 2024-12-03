#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import warnings
import inspect
import traceback
import time
from typing import Union, Callable

from open3d.visualization import gui

from .interactive_gui import AppWindow
from .helpers import get_pretty_name


class Parameter:
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        if not isinstance(new_name, str):
            raise TypeError(
                f"Expected string as name for Parameter, got {type(new_name)}."
            )
        self._name = new_name

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, new_type):
        if not isinstance(new_type, type):
            raise TypeError("parameter descriptor has invalid type.")

        self._type = new_type

    @property
    def limits(self):
        return self._limits

    @limits.setter
    def limits(self, new_limits):
        if self.type not in (int, float):
            self._limits = None
            return

        if new_limits is None:
            self._limits = None
            return

        if not isinstance(new_limits, (list, tuple)) or len(new_limits) != 2:
            raise TypeError(
                f"Limits should be list or tuple of 2 elements, got {new_limits}."
            )

        new_limits = (min(new_limits), max(new_limits))
        self._limits = new_limits

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        if new_value is not None:
            new_value = self.type(new_value)

        if self.type is bool:
            if new_value is None:
                new_value = False

        elif self.type in (int, float):
            if self.limits is None:
                if new_value is None:
                    raise TypeError(
                        "No limits or default value for parameter of type {self.type}."
                    )
            else:
                if new_value is None:
                    new_value = self.limits[0]
                elif not (self.limits[0] <= new_value <= self.limits[1]):
                    warnings.warn(
                        f"Default value not in limits for parameter {self.name}, resetting it."
                    )
                    new_value = self.limits[0]

        elif self.type is str:
            if new_value is None:
                new_value = ""

        else:
            raise NotImplementedError(
                "Not implemented for parameters of type {self.type}."
            )

        self._value = new_value

    @property
    def limit_setter(self):
        return self._limit_setter

    @limit_setter.setter
    def limit_setter(self, new_setter):
        if new_setter is None:
            self._limit_setter = None
            return
        elif callable(new_setter):
            if self.type not in (int, float):
                self._limit_setter = None
                raise TypeError(
                    "Limit setter only valid for parameters of type int and float."
                )
            self._limit_setter = new_setter
        else:
            warnings.warn(
                f"Input limit setter for {self.name} is invalid, "
                f"expected function and got {type(new_setter)}."
            )
            self._limit_setter = None

    def _gui_callback(self, value):
        if self.type is int and isinstance(value, str):
            value = float(value)
        self._value = self.type(value)

    def _reset_values_and_limits(self, app_instance: AppWindow):
        if self.limit_setter is None:
            return

        try:
            self.limits = self.limit_setter(app_instance.selected_raw_elements)
        except Exception:
            warnings.warn(f"Could not reset limits of parameter {self.name}:")
            traceback.print_exc()
        finally:
            # Recheck changed limits
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            self.value = self.value

    def get_gui_element(self, window):
        em = window.theme.font_size
        label = gui.Label(self.name)

        if self.type is bool:
            element = gui.Checkbox(self.name + "?")
            element.checked = self.value
            element.set_on_checked(self._gui_callback)
        elif self.type is int and self.limits is not None:
            slider = gui.Slider(gui.Slider.INT)
            slider.set_limits(*self.limits)
            slider.int_value = self.value
            slider.set_on_value_changed(self._gui_callback)

            element = gui.VGrid(2, 0.25 * em)
            element.add_child(label)
            element.add_child(slider)

        elif self.type is float and self.limits is not None:
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(*self.limits)
            slider.double_value = self.value
            slider.set_on_value_changed(self._gui_callback)

            element = gui.VGrid(2, 0.25 * em)
            element.add_child(label)
            element.add_child(slider)

        else:
            # Text field for general inputs
            text_edit = gui.TextEdit()
            text_edit.placeholder_text = str(self.value)
            text_edit.set_on_value_changed(self._gui_callback)

            element = gui.VGrid(2, 0.25 * em)
            element.add_child(label)
            element.add_child(text_edit)

        return element

    def __init__(self, key, parameter_descriptor: dict):
        if not isinstance(parameter_descriptor, dict):
            raise TypeError("parameter descriptor should be a dictionary")

        self.name = key
        self.type = parameter_descriptor["type"]
        # self._set_setters(parameter_descriptor)
        self.limit_setter = parameter_descriptor.get("limit_setter")
        if self.limit_setter is None:
            self.limits = parameter_descriptor.get("limits")
            self.value = parameter_descriptor.get("default")
        else:
            if "default" in parameter_descriptor or "limits" in parameter_descriptor:
                warnings.warn(
                    f"Limit setter defined for parameter {self.name}, "
                    "ignoring default value and limits."
                )
            self._value = None
            self._limits = None


class Extension:
    DEFAULT_MENU_NAME = "Misc functions"

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

        if hotkey is None:
            self._hotkey = hotkey
            return

        if not isinstance(hotkey, int) or not (0 <= hotkey <= 9):
            warnings.warn(
                f"Expected integer hotkey between 0 and 9, got {hotkey}. "
                "Ignoring hotkey"
            )
            self._hotkey = None
            return

        self._hotkey = ord(str(hotkey))

    def _set_parameters(self, descriptor: dict):
        signature = inspect.signature(self.function)
        parsed_parameters = {}
        parameter_descriptors = descriptor.get("parameters", {})

        if not isinstance(parameter_descriptors, dict):
            raise TypeError("parameters expected to be dict.")

        for key, parameter in parameter_descriptors.items():
            if key not in signature.parameters.keys():
                raise ValueError(
                    f"Function '{self.name}' does not take parameter {key}."
                )
            parsed_parameters[key] = Parameter(key, parameter)

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

    def add_to_application(self, app_instance: AppWindow):
        self._app_instance = app_instance
        # Check whether hotkey has already been assigned extension
        if app_instance.extensions is not None and self.hotkey is not None:
            current_hotkeys = [ext.hotkey for ext in app_instance.extensions]

            if self.hotkey in current_hotkeys:
                idx = current_hotkeys.index(self.hotkey)
                warnings.warn(
                    f"hotkey {self.hotkey_number} previously assigned to function "
                    f"{app_instance.extensions[idx].name}, resetting it to {self.name}."
                )
                app_instance.extensions[idx]._hotkey = None

        if self.menu not in app_instance.extension_submenu_names:
            app_instance._extension_submenu_names += [self.menu]

        if app_instance._extensions is None:
            app_instance._extensions = []
        app_instance._extensions.append(self)

    def update_in_separate_window(self):
        app_instance = self._app_instance

        if len(self.parameters) == 0:
            return gui.Widget.EventCallbackResult.IGNORED

        app = app_instance.app

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
            param._reset_values_and_limits(app_instance)
            dlg_layout.add_child(param.get_gui_element(temp_window))
            dlg_layout.add_fixed(separation_height)

        def _on_accept():
            self._accepted = True
            temp_window.close()

            self._app_instance._apply_function_to_elements(
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

        self._app_instance._apply_function_to_elements(self, update_parameters=False)
