#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-01-27 14:38:17

@author: evbernardes
"""
import warnings
from typing import Union, Callable, TYPE_CHECKING, Generator
from pathlib import Path
import numpy as np
from open3d.visualization import gui

if TYPE_CHECKING:
    from .editor_app import Editor

KEY_LEFT_CONTROL = gui.KeyName.LEFT_CONTROL
KEY_LEFT_SHIFT = gui.KeyName.LEFT_SHIFT

KEY_TEXT_LEFT_CONTROL = "LCtrl"
KEY_TEXT_LEFT_SHIFT = "LShift"


def _get_key_name(key) -> str:
    if isinstance(key, gui.KeyName):
        return str(key).split(".")[1]
    elif isinstance(key, int):
        return chr(key)


class Binding:
    @property
    def menu_id(self) -> Union[None, int]:
        return self._menu_id

    @property
    def key(self) -> Union[gui.KeyName, int]:
        return self._key

    @property
    def lctrl(self) -> bool:
        return self._lctrl

    @property
    def lshift(self) -> bool:
        return self._lshift

    @property
    def description(self) -> str:
        return self._description

    @property
    def callback(self) -> Callable:
        return self._callback

    @property
    def menu(self) -> str:
        return self._menu

    @property
    def creates_window(self) -> bool:
        return self._creates_window

    @property
    def key_instruction(self) -> str:
        """Creates detailed instruction line with item name and hotkeys"""
        if self.key is None:
            return None

        sequence = []
        if self.lctrl:
            sequence.append(KEY_TEXT_LEFT_CONTROL)

        if self.lshift:
            sequence.append(KEY_TEXT_LEFT_SHIFT)

        sequence.append(_get_key_name(self.key))

        return " + ".join(sequence)

    @property
    def menu_item_description(self) -> str:
        """Get line for menu item. If binding creates a window, a ... is added."""
        if self.creates_window:
            description = self.description + "..."
        else:
            description = self.description

        if self.key_instruction is None:
            return description
        else:
            return description + " (" + self.key_instruction + ")"

    def add_to_menu(self, editor_instance: "Editor", set_checked: bool = False):
        if self.menu_id is None:
            self._menu_id = next(editor_instance._submenu_id_generator)

        if self.menu is None:
            return

        menu = editor_instance._get_submenu_from_path(Path(self.menu))
        editor_instance._settings.print_debug(
            f"Assigned id {self.menu_id} to item '{self.menu_item_description}' on menu '{self.menu}'.",
            require_verbose=True,
        )
        menu.add_item(self.menu_item_description, self.menu_id)
        menu.set_checked(self.menu_id, set_checked)
        editor_instance._main_window.set_on_menu_item_activated(
            self.menu_id, self.callback
        )

    @staticmethod
    def _set_binding_ids_with_generator(
        bindings: list["Binding"], id_generator: Generator[int, None, None]
    ) -> dict[int, "Binding"]:
        bindings_per_id: dict[int, Binding] = {}

        for binding in bindings:
            menu_id = binding.menu_id

            if menu_id is None:
                menu_id = next(id_generator)
                if not isinstance(menu_id, int):
                    raise TypeError(
                        f"Id generator yielded {menu_id}, expected integer. "
                    )
            elif not isinstance(menu_id, int):
                raise TypeError(f"Pre-defined menu id {menu_id} is not an integer. ")

            binding_existant = bindings_per_id.get(menu_id, None)

            if binding_existant is not None:
                if binding_existant == binding:
                    continue
                else:
                    warnings.warn(
                        f"Pre-defined menu id {menu_id} already set to binding "
                        f"{binding_existant}. Resetting it to {binding}."
                    )
                    binding_existant._menu_id = None
            binding._menu_id = menu_id
            bindings_per_id[menu_id] = binding
        return bindings_per_id

    @staticmethod
    def _add_multiple_to_menu(
        bindings: list["Binding"], editor_instance: "Editor"
    ) -> list["Binding"]:
        bindings_per_id = Binding._set_binding_ids_with_generator(
            bindings, editor_instance._submenu_id_generator
        )

        successfully_added = []
        for idx in np.sort(list(bindings_per_id.keys())):
            binding = bindings_per_id[idx]
            try:
                binding.add_to_menu(editor_instance)
                successfully_added.append(binding)
            except Exception:
                warnings.warn(f"Could not add binding {binding}.")

        return successfully_added

    def __init__(
        self,
        description: str,
        callback: Callable,
        key: Union[gui.KeyName, int] = None,
        lctrl: bool = False,
        lshift: bool = False,
        menu: Union[str, None] = None,
        creates_window: bool = False,
        menu_id: Union[None, int] = None,
    ):
        self._description = description
        self._callback = callback
        self._key = key
        self._lctrl = lctrl
        self._lshift = lshift
        self._menu = menu
        self._creates_window = creates_window
        self._menu_id = menu_id
