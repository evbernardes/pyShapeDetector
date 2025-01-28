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

## Common attributes
All types of parameters have the following common attributes:
- `on_update`, type `Callable`: Defines a function to be called whenever the variable is changed.  
- `subpanel`, type `str`: Defines a sub-panel for the variable, useful when creating panels with many variables.

## Types of parameters

### `ParameterNumeric`
This class is used to define `int` or `float` type parameters.

#### Attributes
- `limits`, optional: Defines limits for value.
- `default`: Non-optional if `limits` is not given.
- `use_slider`, optional: Graphical widget uses as slider if `True`, only if limits are available. Default value is `False`. 
- `type`, optional: `int` or `float`. By default, uses type of `default`. 

### `ParameterBool`
This class is used to define `bool` type parameters.

#### Attributes
- `default`, optional: Can be `True` or `False`. If nothing is given, `default=False`.

### `ParameterColor`
This class is used to create a color selector. The value is always a `ndarray` of shape `(3, )`.

#### Attributes
- `default`, optional: An array-like element of length 3. If nothing is given, `default=(0, 0, 0)`.

### `ParameterNDArray`
This class is used to create an array of 1 or 2 dimensions. 

#### Attributes
- `default`: An array-like element of length of shape `(N, )` or `(N, M)`. Both the `dtype` and `shape` of the array will be taken from the default value, which is why it's non-optional.

### `ParameterOptions`
This class defines a combobox to choose between different values. 

#### Attributes
- `options`: `list` of possible values.
- `default`: Original default value must be in `options`.

### `ParameterPath`
This class defines a path selector with a button to search manually. 

#### Attributes
- `path_type`, optional: 
  - `"open"`: Open files (**Default**).
  - `"open_dir"`: Open directory. 
  - `"save"`: Save file.
- `suffixes`, optional: Dictionary mapping `suffix: str -> description: str`, to add as filters to the path selection.

### `ParameterCurrent`
This class defines a convenient way of passing a specific `Element` as a separated input to an extension.

## Creating by `dict` descriptors
To define parameters by a `dict` descriptor, `ParameterBase.create_from_dict` can be used instead of the direct class initialized of the other instances, as long as the `type` is set directly as an argument. 

For example, both `type=int` and `type=float` create instances of `ParameterNumeric`.

The descriptors follow the following `type -> Class` mapping:

- `bool` → `ParameterBool`
- `"bool"` → `ParameterBool`
- `int` → `ParameterNumeric`
- `"int"` → `ParameterNumeric`
- `float` → `ParameterNumeric`
- `"float"` → `ParameterNumeric`
- `list` → `ParameterOptions`
- `"list"` → `ParameterOptions`
- `"options"` → `ParameterOptions`
- `numpy.ndarray` → `ParameterNDArray`
- `"ndarray"` → `ParameterNDArray`
- `"array"` → `ParameterNDArray`
- `open3d.visualization.gui.Color` → `ParameterColor`
- `"color"` → `ParameterColor`
- `"current"` → `ParameterCurrentElement`
- `pathlib.Path` → `ParameterPath`
- `"path"` → `ParameterPath`


