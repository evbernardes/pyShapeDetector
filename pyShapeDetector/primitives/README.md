# Geometrical primitives

 This submodule defines classes that implement different geometrical primitives.

 ## Available surface primitives
 - `Plane`: Defined by a unit `normal` vector (3 values) and a `distance`.
 - `PlaneBounded`: A `Plane` with added points defining its borders. Can also contain holes.
 - `PlaneTriangulated`: A `Plane` with its triangles pre-defiend.
 - `Sphere`: Defined by a `center` point (3 values) and the `radius`.
 - `Cylinder`: Defined by a point at the cylinder's `base` (3 values), a `vector` from base to top (3 values) and a `radius`.
 - `Cone`: defined by an `appex` (3 values), a `vector` from the appex to the end of the cone (3 values) and a `half_angle`.

## Helper primitives
- `Line`: does not define a surface and cannot be used for fitting purposes (for now). But can be used as a helper to plane intersections.

## Misc
- `PrimitiveBase`: Base Class, other primitives inherit from it.
- `Template`: Blueprint to start implementation of new primitive.
