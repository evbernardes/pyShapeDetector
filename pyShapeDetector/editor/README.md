# Editor

This submodule defines a Graphical Editor that can be used to manually select elements and apply functions to them.

## Supported element types
The Editor currently supports:
- `Primitive`: Every kind of finite primitive with non-zero surface area, including:
  - `PlaneBounded`
  - `PlaneTriangulated`
  - `Cylinder`
  - `Sphere`
  - `Cone`
- `PointCloud`: Both `Open3D`'s legacy pointclouds and instances of `pyShapeDetector.Numpy_Geometry.PointCloud`.
- `TriangleMesh`: Both `Open3D`'s legacy triangle meshes and instances of `pyShapeDetector.Numpy_Geometry.TriangleMesh`.

## Selecting elements
When selecting an element, first you must set it as the `current` element, by either pressing `[CTRL] + [LEFT MOUSE BUTTON]` on it, or by using the keyboard keys `[<-]` and `[->]`.
The current element can be seen as the one with a bounding box around it.

You can then select it by pressing `[SPACE]`. When the current element is selected, it's bounding box's color changes to reflect that.

Most functions are applied on the multiple `selected` elements, and sometimes they take the `current` element as a special input.

## Import/Export
When importing and exporting elements, the following extensions are supported:
- `Primitives`:
  - `.tar` - Primitive / inliers bundled descriptors
  - `.json` - Primitive descriptors (without inlier information)
- `PointCloud`:
  - `.pcd` - Point Cloud Data files
  - `.ply` - Polygon files
  - `.xyz` - ASCII point cloud files
  - `.xyzn` - ASCII point cloud with normals
  - `.xyzrgb` - ASCII point cloud files with colors
  - `.pts` - 3D Points files (.pts)
  - `.h5` (**Requires `h5py` dependency**) - Point Cloud in Hierarchical Data Format
  - `.e57` (**Requires `pye57` dependency**) - Point Cloud in ASTM E57 file format
- `TriangleMesh`:
  - `.stl` - Stereolithography files
  - `.fbx` - Autodesk Filmbox files
  - `.obj` - Wavefront OBJ files
  - `.off` - Object file format
  - `.gltf` - OpenGL transfer files
  - `.glb` - OpenGL binary transfer files

## Scene files
The scene can be fully saved and load as a `.sdscene` file, which is a specialized tar file which bundles all elements and some extra information on the scene.

## Extensions
Extra functionality can be created by defining extensions as Python `dict`s. 
For more information, see: [Extension](https://github.com/evbernardes/pyShapeDetector/tree/main/pyShapeDetector/editor/extensions)
For examples extension descriptors, see: [Default Extensions](https://github.com/evbernardes/pyShapeDetector/tree/main/pyShapeDetector/editor/extensions/default_extensions)
