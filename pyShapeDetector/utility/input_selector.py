#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:40:10 2024

@author: ebernardes
"""
import tkinter as tk
from tkinter import ttk


class InputSelector:
    """
    A GUI-based input collector that allows users to input values for
    specified variables with type checking and default values.

    Attributes:
    ----------
    _spec : dict
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

    def __init__(self, input_spec={}):
        if not isinstance(input_spec, dict):
            raise ValueError(f"input_spec should be a dict, got {type(input_spec)}.")

        self._spec = {}
        for var_name, (expected_type, default_value) in input_spec.items():
            self.add_argument(var_name, expected_type, default_value)

    def add_argument(self, var_name, expected_type, default_value):
        if not isinstance(expected_type, type):
            raise ValueError(
                f"expected_type should be a type, got {type(expected_type)}."
            )

        if not isinstance(default_value, expected_type):
            raise ValueError(
                f"type of default value should be 'expected_type', got {type(default_value)}."
            )

        if not isinstance(var_name, str):
            raise ValueError(f"name expected to be a string, got {type(var_name)}.")

        if var_name in self._spec:
            raise RuntimeError(f"'{var_name}' already exists.")

        self._spec[var_name] = (expected_type, default_value)

    def remove_argument(self, name):
        self._spec.pop(name)

    def get_results(self):
        if not hasattr(self, "_results") or len(self._results) == 0:
            self._run()

        return [val for val in self._results.values()]

    def _on_submit(self):
        for var_name, (expected_type, default_value) in self._spec.items():
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
            self._spec.items()
        ):
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

        # Set up the main window
        self._results = {}
        self._root = tk.Tk()
        self._root.title("Enter values.")

        # Submit button
        submit_button = ttk.Button(self._root, text="Submit", command=self._on_submit)
        submit_button.grid(row=len(self._spec), column=0, columnspan=2, pady=10)

        # Run the application
        self._root.mainloop()
