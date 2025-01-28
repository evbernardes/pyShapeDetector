# Parameters

This module defines classes to conveniently map variables to graphical widgets.
These elements are used, for example, in the Preferences and extensions.

To create an element for an `int` variable named `"number"`, for example, it can be created directly as an instance of the class `ParameterNumeric`:

```python
from pyShapeDetector.editor.parameter import ParameterNumeric

number_parameter = ParameterNumeric("number", default=1, limits=[0, 10])
```

Or it can be created from a `key` and a `dict` descriptor: 

```python
from pyShapeDetector.editor.parameter import ParameterBase

parameter_descriptor = {
    "type": int,
    "default": 1,
    "limits": [0, 10]
}
number_parameter = ParameterBase.create_from_dict("number", parameter_descriptor)
```