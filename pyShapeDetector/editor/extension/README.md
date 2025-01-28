# Extension

This module implements extensions that can be defined from Python dictionaries.

## Creating an extension

### Attributes
Many attributes can be set when defining a new extension.

#### Main attributes:

- `function`, (`Callable`): This is the actual function that will be applied when the extension runs.
- `inputs`, (`str`, optional): Defines which inputs should be entered to the extension's `function`. The options are:
  - `"selected"` **(Default)**: all elements flagged with `selected=True`.
  - `"current"`: only the current element is used as input.
  - `"none"`: no input element.
  - `"global"`: all elements are used, regardless of being selected or not, hidden or not, etc.
  - `"internal"`: Used to define internal extensions that act directly on the editor instance itself. **Use with caution**.

#### Miscellaneous attributes:
- `name` (`str`, optional): The name of the function. **Default**: `function`'s label.
- `menu` (`str`, optional): The path on the menu bar where the Extension item should be created. **Default**: `"Misc extensions"`
- `selected_outputs` (`bool`, optional): It `True`, outputs from this extension will be flagged as `selected=True`. **Default**: `False`.

#### Hotkey attributes:

- `hotkey` (`str`, optional): Defines a letter or number to be used as a hotkey.
- `lctrl` (`bool`, optional): If `True`, the hotkey requires `[LEFT CONTROL]` to be pressed. Default: `False`.
- `lshift` (`bool`, optional): If `True`, the hotkey requires `[LEFT SHIFT]` to be pressed. Default: `False`.

### Parameters:

A special dictionary mapping `str` parameter keys to subdicts defining different parameters. These parameters define gui widgets to retrieve values used for the function.

In order to define parameters, the name of the parameter must be the exact same name of some variable defined in the signature of the extension's internal `function`.

For example, for a `function` defined as:
```python
import numpy as np
from pyShapeDetector.primitives import Sphere

def create_sphere(radius, center)
    return Sphere.from_center_radius(center, radius)
```

The extension `parameters` descriptor should be set, for example, as:

```python
parameters = {
    "radius": {
        "type": float,
        "default": 1,
        "limits": (0, 5),
    },
    "center": {"type": np.ndarray, "default": [0.0, 0.0, 0.0]},
}
```

### Adding extension to editor
In order to add the extension, define extension descriptor dict and add it to the extension before running it.
For the `create_sphere` example:

```python
from pyShapeDetector.editor import Editor

extension = {
    "function": create_sphere,
    "inputs": "none",
    "parameters": parameters,
}

editor_instance = Editor()
editor_instance.add_extension(extension)
editor_instance.run()
```

### Examples
For a complete set of possible parameters, see: [Parameter](https://github.com/evbernardes/pyShapeDetector/tree/main/pyShapeDetector/editor/parameter/README.md).
For more examples of extension descriptor definitions, see: [Default Extensions](https://github.com/evbernardes/pyShapeDetector/tree/main/pyShapeDetector/editor/extension/default_extensions)

