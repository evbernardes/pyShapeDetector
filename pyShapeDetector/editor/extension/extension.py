#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-03 10:42:00

@author: evbernardes
"""
import time
import copy
import warnings
import inspect
import traceback
from threading import Thread
from typing import Union, Callable, TYPE_CHECKING

from open3d.visualization import gui

if TYPE_CHECKING:
    from ..editor_app import Editor
from ..parameter import (
    ParameterBase,
    ParameterPath,
    ParameterPanel,
    ParameterCurrentElement,
)
from ..binding import Binding
from ..settings import Settings

# Defines minimun amount of seconds between two extension runs
MIN_TIME_BETWEEN_RUNS = 0.2
INPUT_TYPES_WITH_ELEMENTS = ("current", "selected", "global", "internal")
INPUT_TYPES_ONLY_PARAMETERS = ("none",)
ALL_VALID_INPUT_TYPES = INPUT_TYPES_WITH_ELEMENTS + INPUT_TYPES_ONLY_PARAMETERS


def _get_pretty_name(label: Union[Callable, str]):
    if callable(label):
        label = label.__name__
    words = label.replace("_", " ").split()
    result = []
    for word in words:
        if word.isupper():  # Keep existing UPPERCASE values as is
            result.append(word)
        else:  # Capitalize other words
            result.append(word.capitalize())

    return " ".join(result)


class Extension:
    DEFAULT_MENU_NAME = "Misc extensions"
    _editor_instance = None

    @property
    def binding(self) -> Binding:
        "Binding linking extension to menu item and hotkey"
        return self._binding

    @property
    def function(self) -> Callable:
        "Extension's main function"
        return self._function

    @property
    def name(self) -> str:
        "Name and description of extension"
        return self._name

    @property
    def menu(self) -> Union[str, None]:
        "Path of the extension menu item"
        return self._menu

    @property
    def parameters(self) -> dict[str, ParameterBase]:
        "Dictionary of parameters for panel creation"
        return self._parameters

    @property
    def inputs(self) -> str:
        "Type of inputs that should be inputed in extension"
        return self._inputs

    @property
    def keep_inputs(self) -> bool:
        "If False, remove inputs from elements"
        return self._keep_inputs

    @property
    def select_outputs(self) -> bool:
        "If True, new output elements will be pre-selected"
        return self._select_outputs

    @property
    def cancellable(self) -> bool:
        "If True, function can be cancelled"
        return self._cancellable

    @property
    def parameters_kwargs(self) -> dict[str]:
        kwargs = {}
        for key, param in self.parameters.items():
            if isinstance(param, ParameterCurrentElement):
                kwargs[
                    key
                ] = self._editor_instance.element_container.current_element.raw
            else:
                kwargs[key] = param.value
        return kwargs

    @property
    def hotkey(self) -> Union[None, int, gui.KeyName]:
        return self._hotkey

    @property
    def lctrl(self) -> bool:
        return self._lctrl

    @property
    def lshift(self) -> bool:
        return self._lshift

    @property
    def hotkey_number(self) -> int:
        if self.hotkey is None:
            return None
        return int(chr(self.hotkey))

    def _set_name(self, descriptor: dict):
        name = descriptor.get("name", _get_pretty_name(self.function))
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

    def _set_parameters(self, descriptor: dict, settings: Settings):
        signature = inspect.signature(self.function)

        parsed_parameters = {}
        parameter_descriptors = descriptor.get("parameters", {})

        if not isinstance(parameter_descriptors, dict):
            raise TypeError("parameters expected to be dict.")

        function_params = list(signature.parameters)
        descriptor_params = list(parameter_descriptors)
        expected_args_len = len(parameter_descriptors)

        expects_elements_as_input = self.inputs in INPUT_TYPES_WITH_ELEMENTS
        # print(f"{self.name=}")
        # print(f"{self.inputs =}")
        # print(f"{INPUT_TYPES_WITH_ELEMENTS=}")
        # print(f"{expects_elements_as_input=}")

        if expects_elements_as_input:
            expected_args_len += 1
            # function_params_set.remove(list(signature.parameters)[0])

        # print(f"{function_params=}")
        # print(f"{descriptor_params=}")

        missing_in_function = set(descriptor_params).difference(set(function_params))
        # print(f"{missing_in_function=}")
        if len(missing_in_function) > 0:
            raise ValueError(
                f"Extension '{self.name}'s descriptor contains following "
                f"missing parameters from function signature: {missing_in_function}."
            )

        # missing_in_descriptor = descriptor_params_set.difference(function_params_set)
        # missing_in_descriptor = function_params.difference(descriptor_params)
        missing_in_descriptor = set(function_params).difference(set(descriptor_params))
        # print(f"{missing_in_descriptor=}")
        if len(missing_in_descriptor) > int(expects_elements_as_input):
            # first_argument = list(signature.parameters)[0]
            # if first_argument not in descriptor_params_set:
            #     missing_in_descriptor.remove(first_argument)

            raise ValueError(
                f"Extension '{self.name}'s function signature expects following "
                f"missing parameters from descriptor: {missing_in_descriptor}."
            )

        if len(signature.parameters) != expected_args_len:
            raise ValueError(
                f"Invalid number of arguments for function in Extension '{self.name}', with "
                f"input type '{self.inputs}'. Expected {expected_args_len} "
                f"parameters, got '{len(signature.parameters)}'. Maybe missing elements?"
            )

        for key, parameter_descriptor in parameter_descriptors.items():
            if "label" not in parameter_descriptor:
                parameter_descriptor["label"] = _get_pretty_name(key)

            if parameter_descriptor["type"] == "preference":
                if key not in settings._dict:
                    raise KeyError(
                        f"Parameter '{key}' of type 'settings' in extension "
                        f"'{self.name}' could not found in internal setting "
                        "options."
                    )
                parsed_parameters[key] = settings._dict[key].create_reference()
                parsed_parameters[key]._subpanel = parameter_descriptor.get(
                    "subpanel", None
                )
                continue

            if key not in signature.parameters.keys():
                raise ValueError(
                    f"Function '{self.function.__name__}' from extension '{self.name}' does not take parameter '{key}'."
                )

            parsed_parameters[key] = ParameterBase.create_from_dict(
                key, parameter_descriptor
            )

            if isinstance(parsed_parameters[key], ParameterPath):
                parsed_parameters[key].editor_instance = self._editor_instance

        self._parameters = parsed_parameters

    def _set_misc(self, descriptor: dict):
        inputs = descriptor.get("inputs", "selected")

        if inputs is None:
            inputs = "none"

        if inputs not in ALL_VALID_INPUT_TYPES:
            raise ValueError(
                f"Possible values for 'inputs' are {ALL_VALID_INPUT_TYPES}, got {inputs}."
            )

        is_internal = inputs == "internal"

        select_outputs = bool(descriptor.get("select_outputs", False))

        # TODO: Discover yhy I put this here before:
        # if select_outputs and inputs == "current":
        #     raise ValueError(
        #         "'select_outputs' should not be True for inputs of type 'current'."
        #     )

        if (cancellable := descriptor.get("cancellable")) is None:
            cancellable = not is_internal
        else:
            cancellable = bool(cancellable)

        if cancellable and is_internal:
            warnings.warn(
                "Cancellable set to 'True' for internal extension, "
                "this might create undefined behaviour."
            )

        self._cancellable = cancellable
        self._inputs = inputs
        self._keep_inputs = bool(descriptor.get("keep_inputs", False))
        self._select_outputs = select_outputs

        # if self._cancellable and self.type

    def __init__(
        self,
        function_or_descriptor: Union[Callable, dict],
        settings: Settings,
        editor_instance: "Editor" = None,
    ):
        if editor_instance is not None:
            self._editor_instance = editor_instance

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
        self._set_misc(descriptor)
        self._set_parameters(descriptor, settings)

        self._binding = Binding(
            key=self.hotkey,
            lctrl=self.lctrl,
            lshift=self.lshift,
            description=self.name,
            menu=self.menu,
            callback=self.run,
            creates_window=len(self.parameters) > 0,
        )

        self._extension_window_opened: bool = False

    def add_to_application(self, editor_instance: Union[None, "Editor"] = None):
        if editor_instance is None and self._editor_instance is None:
            raise RuntimeError("No editor instance to add extension to.")

        if editor_instance is not None and self._editor_instance is not None:
            warnings.warn(
                "Assigning new editor instance to extension with pre-defined instance."
            )
            self._editor_instance = editor_instance

        editor_instance = self._editor_instance

        if editor_instance._extensions is None:
            editor_instance._extensions = []

        editor_instance._extensions.append(self)

    def add_menu_item(self):
        self.binding.add_to_menu(self._editor_instance)

    def run(self):
        _on_window = self._editor_instance._settings.get_setting("extensions_on_window")
        _empty_extensions_on_panel_window = self._editor_instance._settings.get_setting(
            "empty_extensions_on_panel_window"
        )

        if len(self.parameters) == 0 and not _empty_extensions_on_panel_window:
            self._apply_to_elements_in_thread()
        else:
            if self._extension_window_opened and _on_window:
                warnings.warn(
                    f"Tried opening extension '{self.name}' but it's already opened."
                )
                return

            try:
                self._create_extension_window()
            except Exception:
                warnings.warn("Could not create extension window.")
                traceback.print_exc()

    def _apply_to_elements_in_thread(self):
        editor_instance = self._editor_instance
        if editor_instance._running_extension:
            return

        editor_instance._running_extension = True

        def _on_cancel():
            self._cancelled = True
            editor_instance._close_dialog()
            warnings.warn(f"User cancelled extension {self.name}.")

        self._cancelled = False
        editor_instance._create_simple_dialog(
            title_text=f"Applying {self.name}...",
            create_button=self.cancellable,
            button_text="Cancel",
            button_callback=_on_cancel,
        )

        editor_instance._settings.print_debug("Running extension in thread...")
        # editor_instance.app.run_in_thread(self._worker)

        Thread(target=self._worker).start()

    def _stop(self):
        """Graciously stop"""
        editor_instance = self._editor_instance
        editor_instance._close_dialog()
        editor_instance._time_last_used_extension = time.time()
        editor_instance._running_extension = False

    def _worker(self):
        editor_instance = self._editor_instance

        # TODO: without this time sleep, a Segmentation fault might happen
        # running extensions repeatedly and fast
        # I have no idea why :)
        time_last = editor_instance._time_last_used_extension
        elapsed = time.time() - time_last
        editor_instance._settings.print_debug(
            f"{time_last=}s, {elapsed=}s", require_verbose=True
        )
        if (diff := MIN_TIME_BETWEEN_RUNS - elapsed) > 0:
            editor_instance._settings.print_debug(
                f"Waiting {diff}s", require_verbose=True
            )
            time.sleep(diff)

        if self.inputs == "none":
            indices = []
            input_elements = []
        elif self.inputs == "current":
            indices = [editor_instance.element_container.current_index]
            input_elements = [editor_instance.element_container.current_element.raw]
        elif self.inputs == "selected":
            indices = editor_instance.element_container.selected_indices
            input_elements = [editor_instance.element_container[i].raw for i in indices]
        elif self.inputs == "global":
            indices = list(range(len(editor_instance.element_container)))
            input_elements = [editor_instance.element_container[i].raw for i in indices]
        elif self.inputs == "internal":
            indices = None
            input_elements = editor_instance
        else:
            # This should never happen
            self._stop()
            raise RuntimeError(
                f"Invalid input instruction {self.inputs} "
                f"found in extension {self.name}."
            )

        # DEBUG LINES
        settings = editor_instance._settings
        if self.inputs == "internal":
            settings.print_debug(f"Applying internal extension {self.name}")
        else:
            settings.print_debug(f"Applying {self.name} to {len(indices)} elements")
            settings.print_debug(f"Extension has input type {self.inputs}")
            settings.print_debug(f"Indices: {indices}.", require_verbose=True)
        if len(self.parameters) > 0:
            settings.print_debug(f"Parameters: {list(self.parameters.values())}.")

        # MAIN FUNCTION EXECUTION
        try:
            editor_instance._settings.print_debug("Trying to run extension function...")
            kwargs = self.parameters_kwargs
            if self.inputs == "none":
                output_elements = self.function(**kwargs)
            else:
                output_elements = self.function(input_elements, **kwargs)
            editor_instance._settings.print_debug("Extension function complete!")
        except KeyboardInterrupt:
            self._stop()
            return
        except Exception as e:
            if self.inputs == "internal":
                warnings.warn(
                    f"Failed to apply {self.name} internal extension to, got:"
                )
            else:
                warnings.warn(
                    f"Failed to apply {self.name} extension to "
                    f"elements in indices {indices}, got:"
                )
            self._stop()
            editor_instance._create_simple_dialog(
                title_text=f"Extension '{self.name}' failed: \n\n{e}."
            )
            return

        if self._cancelled:
            settings.print_debug(
                f"Extensions '{self.name}' thread ended, but call was canceled. "
                "Ignoring output."
            )
            return

        # Assuring output is a list
        try:
            if self.inputs == "internal":
                pass
            elif isinstance(output_elements, tuple):
                output_elements = list(output_elements)
            elif not isinstance(output_elements, list):
                output_elements = [output_elements]
            if self.inputs == "current" and len(output_elements) != 1:
                warnings.warn(
                    "Only one output expected for extensions with "
                    f"'current' input type, got {len(output_elements)}."
                )
                self._stop()
                return
        except Exception:
            warnings.warn(f"Error with output elements {output_elements}.")
            traceback.print_exc()
            self._stop()
            return

        # Saving state for undoing purposes
        try:
            if self.inputs != "internal":
                current_state = {
                    "indices": copy.deepcopy(indices),
                    "elements": copy.deepcopy(input_elements),
                    "num_outputs": len(output_elements),
                    "current_index": editor_instance.element_container.current_index,
                    "operation": "extension",
                }
                editor_instance._save_state(current_state)
        except Exception:
            warnings.warn("Could not save state! Ignoring extension output.")
            traceback.print_exc()
            self._stop()
            return

        try:
            if self.inputs == "current":
                editor_instance.element_container.insert_multiple(
                    output_elements,
                    to_gui=True,
                    is_selected=editor_instance.element_container.current_element.is_selected,
                )
                editor_instance.element_container.update_current_index(
                    len(editor_instance.element_container) - 1
                )
            elif self.inputs != "internal":
                # editor_instance._insert_elements(
                editor_instance.element_container.insert_multiple(
                    output_elements, to_gui=True, is_selected=self.select_outputs
                )
        except Exception:
            warnings.warn("Could not insert output elements!")
            self._stop()
            return

        try:
            if self.inputs != "internal" and not self.keep_inputs and len(indices) > 0:
                assert (
                    editor_instance.element_container.pop_multiple(
                        indices, from_gui=True
                    )
                    == input_elements
                )
        except Exception:
            warnings.warn("Could not remove input elements!")
            traceback.print_exc()
            self._stop()
            return

        editor_instance._last_used_extension = self
        # editor_instance._future_states = []
        editor_instance.element_container.update_current_index()
        editor_instance._update_plane_boundaries()
        editor_instance._update_info()
        editor_instance._update_BBOX_and_axes()
        self._stop()
        # editor_instance._set_gray_overlay(False)

    def _create_extension_window(self):
        _on_panel = not self._editor_instance._settings.get_setting(
            "extensions_on_window"
        )
        editor_instance = self._editor_instance

        if _on_panel and editor_instance._set_extension_panel_open(self.name, True):
            """ If True, it was already opened and set visibility to True."""
            return

        em = editor_instance._main_window.theme.font_size

        self._ran_at_least_once = False

        previous_values = {}
        for key, param in self.parameters.items():
            previous_values[key] = copy.copy(param.value)

        separation_width = em
        separation_height = int(round(0.1 * em))

        parameter_panel = ParameterPanel(
            self.parameters, separation_width, separation_height
        )

        panel = parameter_panel.panel

        window_width = 500
        window_height = 50 * len(panel.get_children())

        if not _on_panel:
            temp_window = editor_instance.app.create_window(
                self.name, window_width, window_height
            )
            temp_window.show_menu(False)
            self._editor_instance._temp_windows.append(temp_window)
            self._extension_window_opened = True

        def _on_apply_button():
            self._ran_at_least_once = True
            self._apply_to_elements_in_thread()

        def _on_close_button():
            if not _on_panel:
                temp_window.close()
            else:
                self._editor_instance._set_extension_panel_open(self.name, False)

        def _on_refresh_limits():
            if len(editor_instance.element_container.selected_indices) > 0:
                for param in self.parameters.values():
                    param._reset_values_and_limits(editor_instance.element_container)

        def _on_close_window():
            if not self._ran_at_least_once:
                for key, param in self.parameters.items():
                    param.value = previous_values[key]

            self._editor_instance._temp_windows.remove(temp_window)
            self._extension_window_opened = False
            return True

        apply = gui.Button("Apply")
        apply.set_on_clicked(_on_apply_button)
        apply.vertical_padding_em = 0
        close = gui.Button("Close")
        close.set_on_clicked(_on_close_button)
        close.vertical_padding_em = 0
        refresh = gui.Button("Refresh limits")
        refresh.set_on_clicked(_on_refresh_limits)
        refresh.vertical_padding_em = 0

        callbacks = {
            "apply": _on_apply_button,
            "close": _on_close_button,
            "refresh": _on_refresh_limits,
        }

        h = gui.Horiz()
        h.add_stretch()
        h.add_child(apply)
        h.add_stretch()
        h.add_child(close)
        h.add_stretch()
        if parameter_panel.has_limit_setters:
            h.add_child(refresh)
            h.add_stretch()
        panel.add_child(h)

        if not _on_panel:
            temp_window.add_child(panel)
            temp_window.set_on_close(_on_close_window)

        else:
            editor_instance._add_extension_panel(self.name, panel, callbacks)

        _on_refresh_limits()
