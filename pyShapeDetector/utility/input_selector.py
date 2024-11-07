#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:40:10 2024

@author: ebernardes
"""
import warnings
import tkinter as tk
from tkinter import ttk


class InputSelector:
    """
    A GUI-based input collector that allows users to input values for
    specified variables with type checking and default values.

    Attributes:
    ----------
    _specs : dict
        A dictionary where each key is the variable name, and each value is a
        tuple (expected_type, default_value), specifying the type and default
        value for the input variable.

    Methods:
    -------
    add_argument(var_name, expected_type, default_value)
        Adds a new input variable to the specification.

    remove_argument(name)
        Removes an input variable from the specification by name.

    get_results()
        Opens the GUI to collect inputs if they haven't been collected, and
        returns the list of user-provided or default values.

    run()
        Opens the GUI window to collect inputs for the specified variables.
    """

    def __init__(self, input_specs={}, window_name="Enter values"):
        if not isinstance(window_name, str):
            window_name = str(window_name)
        self._window_name = window_name

        if not isinstance(input_specs, dict):
            raise ValueError(f"input_spec should be a dict, got {type(input_specs)}.")

        self._specs = {}
        for var_name, (expected_type, default_value) in input_specs.items():
            self.add_argument(var_name, expected_type, default_value)

    def add_argument(self, var_name, expected_type, default_value):
        if not isinstance(expected_type, type):
            raise ValueError(
                f"expected_type should be a type, got {type(expected_type)}."
            )

        if not isinstance(default_value, expected_type):
            if isinstance(default_value, float) and expected_type == int:
                warnings.warn("Expected int and got float, rounding...")
                default_value = int(default_value)
            elif isinstance(default_value, int) and expected_type == float:
                default_value = float(default_value)
            else:
                raise ValueError(
                    f"type of default value should be 'expected_type', got {type(default_value)}."
                )

        if not isinstance(var_name, str):
            raise ValueError(f"name expected to be a string, got {type(var_name)}.")

        if var_name in self._specs:
            raise RuntimeError(f"'{var_name}' already exists.")

        self._specs[var_name] = (expected_type, default_value)

    def remove_argument(self, name):
        self._specs.pop(name)

    def get_results(self):
        if not hasattr(self, "_results") or len(self._results) == 0:
            self._run()

        return [val for val in self._results.values()]

    def _on_accept(self, event=None):
        for var_name, (expected_type, default_value) in self._specs.items():
            user_input = self._input_vars[var_name].get()

            # Validate and convert input based on expected type
            if expected_type == str:
                self._results[var_name] = user_input or default_value
            else:
                try:
                    # Try to convert to the specified type (int or float)
                    self._results[var_name] = expected_type(user_input)
                except ValueError:
                    # Use default if conversion fails
                    self._results[var_name] = default_value

        self._root.destroy()  # Close the window after submission

    def _get_input_vars(self):
        input_vars = {}

        # Create labels and entry fields based on input_spec
        for row, (var_name, (expected_type, default_value)) in enumerate(
            self._specs.items()
        ):
            if expected_type == bool:
                # Boolean variables will use IntVar (0 for False, 1 for True)
                input_vars[var_name] = tk.IntVar(value=int(default_value))

                # Label for the checkbox
                text = f"Check {var_name} (bool):"
                ttk.Label(self._root, text=text).grid(
                    row=row, column=0, padx=5, pady=5, sticky="e"
                )

                # Checkbox for boolean input
                checkbox = ttk.Checkbutton(self._root, variable=input_vars[var_name])
                checkbox.grid(row=row, column=1, padx=5, pady=5)
            else:
                # Initialize a StringVar to hold user input for each field
                input_vars[var_name] = tk.StringVar()

                # Label for each entry field
                text = f"Enter {var_name} ({expected_type.__name__}):"
                ttk.Label(self._root, text=text).grid(
                    row=row, column=0, padx=5, pady=5, sticky="e"
                )

                # Entry widget for user input
                entry = ttk.Entry(self._root, textvariable=input_vars[var_name])
                entry.grid(row=row, column=1, padx=5, pady=5)

                # Set the default value in the entry field
                entry.insert(0, str(default_value))

        self._input_vars = input_vars

    def _run(self):
        if len(self._specs) == 0:
            raise RuntimeError("No input specified.")

        self._results = {}

        self._root = tk.Tk()
        self._root.title(self._window_name)
        self._get_input_vars()

        # Submit button
        self._root.bind("<Return>", self._on_accept)
        accept_button = ttk.Button(self._root, text="Accept", command=self._on_accept)
        accept_button.grid(row=len(self._specs), column=0, columnspan=2, pady=10)

        # Run the application
        self._root.mainloop()

        if len(self._results) == 0:
            raise KeyboardInterrupt("Action cancelled, no input.")
