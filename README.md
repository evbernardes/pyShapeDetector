# pyShapeDetector

Python 3 module to detect geometrical primitives in pointclouds.

## Dependencies

- `Open3D`
- `TODO` ...



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
>>> from open3d.io import read_point_cloud as i3d
>>> from pyShapeDetector.methods import RANSAC_Classic
>>> from pyShapeDetector.primitives import Sphere
>>> pcd = read_point_cloud('data/1spheres.pcd')
>>> pcd.estimate_normals()
>>> detector = RANSAC_Classic(Sphere, num_iterations=15,
                              threshold_angle=30,
                              threshold_angle=15,
                              inliers_min=100)

>>> shape, inliers, metrics = detector.fit(pcd.points, debug=True, normals=normals)
```

For more examples, see the `examples` directory.

## Tests

`TODO`

## Acknowledgements

`TODO`

## License

`TODO`

