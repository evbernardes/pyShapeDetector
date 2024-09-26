# UnityPrimitiveLoader

CSharp extension for the Unity Editor to create game objects for primitive descriptors. 

## Installation
- Copy the `UnityPrimitiveLoader` directory to a `assers/Editor` inside of your Unity project.

- Unity will compile it for you.

## Use
After compilation, the extension can be accessed via `Tools/Load pyShapeDetector shapes` on the Unity editor window.

## Available primitives:
- `Sphere`: A regular "Sphere" GameObject is created and translated/rotated accordingly.
- `Cylinder`: Similar to `Sphere`.
- `PlaneRectangular`: Similar to `Sphere` and `Cylinder`, but two child elements are created: `front` and a rotated version called `back`.
- `PlaneTriangulated`: A new `Mesh` is created, added as `front` child element, and a mirrowed version `back` is added.
- `PlaneBounded`: `earcut.net` creates a triangulation which is then fed to the `PlaneTriangulated algorithm.`

## Primitives not available:
- `Plane`: Not available, as they are not bounded.
- `Cone`: TODO

## Dependencies
[earcut.net](https://github.com/oberbichler/earcut.net) is used to create triangulated meshes for `PlaneBounded` types. For simplicity, a copy of this dependency is already included in this directory.

