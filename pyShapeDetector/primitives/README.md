# Geometrical primitives

 This submodule defines classes that implement different geometrical primitives.

 ## Available classes
 - `PrimitiveBase`: Base Class, other primitives inherit from it.
 - `Plane`: Defined by a unit `normal` vector (3 elements) and a `distance`.
 - `Sphere`: Defined by a `center` point (3 elements) and the `radius`.
 - `Cylinder`: Defined by a point at the cylinder's `base` (3 elements), a `vector` from base to top (3 elements) and a `radius`.

## Misc
- `Template`: Blueprint to start implementation of new primitive.
