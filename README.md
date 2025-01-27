# pyShapeDetector

Python 3 module to detect geometrical primitives in pointclouds.

## Dependencies

- `numpy`
- `Open3D`
- `Scipy`
- `sklearn`
- `TODO` ...

### Optional Dependencies
- `mapbox_earcut`: for proper triangulation of concave plane boundaries
- `h5_files`: for importing pointclouds stored in `.h5` files
- `e57_files`: for importing pointclouds stored in `.e57` files
- `shapely`: can be used to glue concave planes
- `pyperclip`: to copy information of selected elements to clipboard

### Extra
- `UnityPrimitiveLoader`: C# extension for the Unity Editor to create game objects for primitive descriptors. 
- If optional dependency `pyperclip` is installed, copying elements from the editor also copy their names to the clipboard. On Linux, `xclip` must be installed to assure proper behavior. 

## Installation

To install from GitHub source, first clone repo using `git`:

    $ git clone https://github.com/evbernardes/pyShapeDetector.git

Then, in the `pyShapeDetector` repository that you cloned, simply run:

    $ pip install .

To make an editable installation, run:

    $ pip install -e .

## Documentation and Usage

Simple example for detecting a sphere in a pointcloud using classical RANSAC:

``` python
>>> from pyShapeDetector.methods import RANSAC_Classic
>>> from pyShapeDetector.primitives import Sphere
>>> from pyShapeDetector.geometry import PointCloud
>>>
>>> pcd = PointCloud.read_point_cloud('data/1spheres.pcd')
>>> pcd.estimate_normals()
>>>
>>> detector = RANSAC_Classic()
>>> detector.add(Sphere)
>>> detector.options.num_iterations = 15
>>> detector.options.threshold_angle_degrees = 15
>>> detector.options.inliers_min = 100
>>>
>>> # pre-compute normals, otherwise angles will be ignored by method 
>>> pcd.compute_normals()
>>>
>>> shape, inliers, metrics = detector.fit(pcd, debug=True)
```

For more examples, see the `examples` directory.

## Editor
A Graphical User Interface is also available. 
For more information, see: [Editor](https://github.com/evbernardes/pyShapeDetector/tree/main/pyShapeDetector/editor/README.md).

## Tests

`TODO`

## Acknowledgements

`TODO`

