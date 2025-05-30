#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:32:38 2024

@author: ebernardes
"""
import copy
import warnings
import numpy as np
from pathlib import Path
from typing import Union, TYPE_CHECKING
from open3d.geometry import TriangleMesh as open3d_TriangleMesh
from open3d import io

from pyShapeDetector.utility import mesh_to_obj_description
from .numpy_geometry import link_to_open3d_geometry, Numpy_Geometry
from .lineset import LineSet
from .axis_aligned_bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from pyShapeDetector.primitives import Line


@link_to_open3d_geometry(open3d_TriangleMesh)
class TriangleMesh(Numpy_Geometry):
    """
    TriangleMesh class that uses Open3D.geometry.TriangleMesh internally.

    Almost every method and property are automatically copied and decorated.

    Extra Attributes
    ----------------
    surface_area
    center
    volume
    curvature
    has_curvature
    colors_cielab

    Extra Methods
    -------------
    create_arrow_from_points
    add_reverse_triangles
    get_triangle_points
    get_triangle_sides
    get_triangle_perimeters
    get_triangle_surface_areas
    get_triangle_circumradius
    get_triangle_lines
    get_triangle_LineSet
    get_triangle_boundary_indexes
    _get_sliceplane
    clean_crop
    _get_obj_vertices_triangles
    get_obj_description
    get_loop_indexes_from_boundary_indexes
    get_fused_mesh
    triangulate_earclipping
    fuse_vertices_triangles
    write_triangle_mesh
    read_triangle_mesh
    """

    @property
    def surface_area(self) -> float:
        return self.get_surface_area()

    @property
    def center(self) -> np.ndarray[float]:
        return self.get_center()

    @classmethod
    def create_arrow_from_points(
        cls,
        from_point: np.ndarray[float],
        to_point: np.ndarray[float],
        radius: float = 0.01,
        arrow_head_length: float = 0.05,
        arrow_head_radius: float = 0.02,
    ) -> "TriangleMesh":
        """
        Create an arrow mesh defined by a cylinder (shaft) and a cone (head).

        Parameters
        ----------
        from_point : list or np.array of length 3
            The start point of the vector.
        to_point : list or np.array of length 3
            The end point of the vector.
        radius : float, optional
            Radius of the arrow shaft. Default: 0.01.
        arrow_head_length : float, optional
            Length of the cone (arrowhead). If set to None, this length will
            be set to 1% of the total length. Default: 0.05.
        arrow_head_radius : float, optional
            Radius of the cone's base (arrowhead size). If set to None,
            this radius will be set to twice the shaft radius. Default: 0.02.

        Returns
        -------
            TriangleMesh
        """

        from pyShapeDetector.primitives import Cylinder, Cone

        from_point = np.array(from_point)
        to_point = np.array(to_point)

        if arrow_head_radius is None:
            radius_ratio = 2
            arrow_head_radius = radius_ratio * radius

        # Compute vector direction and length
        vector = to_point - from_point
        total_length = np.linalg.norm(vector)

        if arrow_head_length is None:
            length_ratio = 0.1
            arrow_head_length = length_ratio * total_length
        else:
            arrow_head_length = min(arrow_head_length, total_length)

        cylinder_length = total_length - arrow_head_length
        direction_vector = vector / np.linalg.norm(vector)

        # Cylinder: Shaft of the arrow
        cylinder = Cylinder.from_base_vector_radius(
            from_point, direction_vector * cylinder_length, radius
        )

        # Cone: Arrowhead
        cone = Cone.from_appex_vector_radius(
            to_point, -direction_vector * arrow_head_length, arrow_head_radius
        )

        mesh = cylinder.get_mesh() + cone.get_mesh()
        return cls(mesh.vertices, mesh.triangles)

    def add_reverse_triangles(self):
        """Add necessary triangles so that each triangle has a
        back-side counterpart.

        """
        triangles = self.triangles
        if len(triangles) == 0:
            warnings.warn("TrianglMesh has no triangles, nothing to reverse.")
            return

        triangles_sorted = np.sort(triangles, axis=1)

        triangles_set = set(tuple(t) for t in triangles_sorted)
        one_side = np.vstack(list(triangles_set))
        self.triangles = np.vstack([one_side, one_side[:, ::-1]])

    def get_triangle_points(self) -> np.ndarray:
        """Get positions of each triangle points.

        Returns
        -------
        np.array
            Points.
        """
        return self.vertices[self.triangles]

    def get_triangle_sides(self) -> np.ndarray:
        """Get side lengths of each triangle.

        Returns
        -------
        np.array
            Side lenghts of each triangle.
        """
        triangle_points = self.get_triangle_points()
        triangle_points_wrap = np.concatenate(
            [triangle_points, triangle_points[:, 0:1, :]], axis=1
        )
        triangle_points_diff = np.diff(triangle_points_wrap, axis=1)
        return np.linalg.norm(triangle_points_diff, axis=2)

    def get_triangle_perimeters(self) -> np.ndarray:
        """Get perimeter of each triangle.

        Returns
        -------
        np.array
            Perimeters defined by triangles.
        """
        return self.get_triangle_sides().sum(axis=1)

    def get_triangle_surface_areas(self) -> np.ndarray:
        """Get surface area of each triangle.

        Returns
        -------
        np.array
            Surface areas defined by triangles.
        """

        sides = self.get_triangle_sides()
        s = sides.sum(axis=1) / 2
        return np.sqrt(s * np.prod(s[:, np.newaxis] - sides, axis=1))

    def get_triangle_circumradius(self) -> np.ndarray:
        """Get circumradius of each triangle.

        Returns
        -------
        np.array
            Perimeters defined by triangles.
        """

        sides = self.get_triangle_sides()
        perimeters = sides.sum(axis=1)
        return sides.prod(axis=1) / np.sqrt(
            perimeters * (perimeters[:, np.newaxis] - 2 * sides).prod(axis=1)
        )

    def get_triangle_lines(self) -> list["Line"]:
        """Get pyShapeDetector.primitives.Line instances for every line in every
        triangle.

        Returns
        -------
        list of Line instances
            Three lines for each triangle.
        """
        from pyShapeDetector.primitives import Line

        triangle_points = self.get_triangle_points()
        lines = []
        for p1, p2, p3 in triangle_points:
            lines.append(Line.from_two_points(p1, p2))
            lines.append(Line.from_two_points(p2, p3))
            lines.append(Line.from_two_points(p3, p1))
        return lines

    def get_triangle_LineSet(self) -> LineSet:
        """Get a Open3D.geomery.LineSet instance containing every line in every
        triangle.

        Returns
        -------
        Open3d.geometry.LineSet
            Three lines for each triangle.
        """

        lineset = LineSet()
        lineset.points = self.vertices
        lineset.lines = self.triangles[:, [0, 1, 1, 2, 2, 0]].reshape(
            (len(self.triangles) * 3, 2)
        )

        return lineset

    def get_triangle_boundary_indexes(self) -> list[tuple]:
        """Get tuples defining edge lines in boundary of mesh.

        Edges are detected as lines which only belong to a single triangle.

        Returns
        -------
        list of tuples
            Each tuple contains the indexes of the two vertices defining each edge
        """
        # vertices, triangles = _get_vertices_triangles(mesh_or_vertices, triangles)

        # get tuples containing all possible lines
        lines = []
        for triangle in self.triangles:
            for i in range(3):
                line = [triangle[i], triangle[(i + 1) % 3]]
                line.sort()
                lines.append(tuple(line))

        occurences = {}
        lines.sort()
        while len(lines) > 0:
            line = lines.pop()
            count = 1

            while len(lines) > 0:
                if lines[-1] != line:
                    break

                count += 1
                line = lines.pop()

            occurences[line] = count

        # should be 1, except if triangles are doubled.
        min_value = min(occurences.values())
        boundary_indexes = [k for k, v in occurences.items() if v == min_value]

        return boundary_indexes

    def _get_sliceplane(self, axis, value, direction):
        # axis can be 0,1,2 (which corresponds to x,y,z)
        # value where the plane is on that axis
        # direction can be True or False (True means remove everything that is
        # greater, False means less
        # than)

        new_vertices = self.vertices.tolist()
        new_triangles = []

        # (a, b) -> c
        # c refers to index of new vertex that sits at the intersection between a,b
        # and the boundingbox edge
        # a is always inside and b is always outside
        intersection_edges = dict()

        # find axes to compute
        axes_compute = [0, 1, 2]
        # remove axis that the plane is on
        axes_compute.remove(axis)

        def _compute_intersection(vertex_in_index, vertex_out_index):
            vertex_in = self.vertices[vertex_in_index]
            vertex_out = self.vertices[vertex_out_index]
            if (vertex_in_index, vertex_out_index) in intersection_edges:
                intersection_index = intersection_edges[
                    (vertex_in_index, vertex_out_index)
                ]
                intersection = new_vertices[intersection_index]
            else:
                intersection = [None, None, None]
                intersection[axis] = value
                const_1 = (value - vertex_in[axis]) / (
                    vertex_out[axis] - vertex_in[axis]
                )
                c = axes_compute[0]
                intersection[c] = (
                    const_1 * (vertex_out[c] - vertex_in[c])
                ) + vertex_in[c]
                c = axes_compute[1]
                intersection[c] = (
                    const_1 * (vertex_out[c] - vertex_in[c])
                ) + vertex_in[c]
                assert None not in intersection
                # save new vertice and remember that this intersection already added an edge
                new_vertices.append(intersection)
                intersection_index = len(new_vertices) - 1
                intersection_edges[
                    (vertex_in_index, vertex_out_index)
                ] = intersection_index

            return intersection_index

        for t in self.triangles:
            v1, v2, v3 = t
            if direction:
                v1_out = self.vertices[v1][axis] > value
                v2_out = self.vertices[v2][axis] > value
                v3_out = self.vertices[v3][axis] > value
            else:
                v1_out = self.vertices[v1][axis] < value
                v2_out = self.vertices[v2][axis] < value
                v3_out = self.vertices[v3][axis] < value

            bool_sum = sum([v1_out, v2_out, v3_out])
            # print(f"{v1_out=}, {v2_out=}, {v3_out=}, {bool_sum=}")

            if bool_sum == 0:  # triangle completely inside --> add and continue
                new_triangles.append(t)
            elif bool_sum == 3:  # triangle completely outside --> skip
                continue
            elif (
                bool_sum == 2
            ):  # two vertices outside, add triangle using both intersections
                vertex_in_index = v1 if (not v1_out) else (v2 if (not v2_out) else v3)
                vertex_out_1_index = v1 if v1_out else (v2 if v2_out else v3)
                vertex_out_2_index = v3 if v3_out else (v2 if v2_out else v1)
                assert sum(
                    [vertex_in_index, vertex_out_1_index, vertex_out_2_index]
                ) == sum([v1, v2, v3])

                # add new triangle
                new_triangles.append(
                    [
                        vertex_in_index,
                        _compute_intersection(vertex_in_index, vertex_out_1_index),
                        _compute_intersection(vertex_in_index, vertex_out_2_index),
                    ]
                )

            elif bool_sum == 1:  # one vertice outside, add three triangles
                vertex_out_index = v1 if v1_out else (v2 if v2_out else v3)
                vertex_in_1_index = v1 if (not v1_out) else (v2 if (not v2_out) else v3)
                vertex_in_2_index = v3 if (not v3_out) else (v2 if (not v2_out) else v1)
                assert sum(
                    [vertex_out_index, vertex_in_1_index, vertex_in_2_index]
                ) == sum([v1, v2, v3])

                new_triangles.append(
                    [
                        vertex_in_1_index,
                        _compute_intersection(vertex_in_1_index, vertex_out_index),
                        vertex_in_2_index,
                    ]
                )
                new_triangles.append(
                    [
                        _compute_intersection(vertex_in_1_index, vertex_out_index),
                        _compute_intersection(vertex_in_2_index, vertex_out_index),
                        vertex_in_2_index,
                    ]
                )

            else:
                assert False

        # TODO remap indices and remove unused
        return TriangleMesh(new_vertices, new_triangles)

    def clean_crop(
        self, axis_aligned_bounding_box: AxisAlignedBoundingBox
    ) -> "TriangleMesh":
        """Crops mesh by slicing facets instead of completely removing them, as
        seen on [1].

        Parameters
        ----------
        axis_aligned_bounding_box: AxisAlignedBoundingBox
            Bounding box defining region of mesh to be saved.

        Returns
        -------
        TriangleMesh
            Cropped mesh.

        References
        ----------
        [1] https://stackoverflow.com/questions/75082217/crop-function-that-slices-triangles-instead-of-removing-them-open3d
        """

        min_bound = axis_aligned_bounding_box.min_bound
        max_bound = axis_aligned_bounding_box.max_bound

        # mesh = sliceplane(mesh, 0, min_x, False)
        mesh_sliced = copy.copy(self)
        for i in range(3):
            min_, max_ = sorted([min_bound[i], max_bound[i]])
            mesh_sliced = self._get_sliceplane(i, max_, True)
            mesh_sliced = mesh_sliced._sliceplane(i, min_, False)
        return mesh_sliced

    def _get_obj_vertices_triangles(self):
        obj_content = []

        # Write vertices to obj_content
        for vertex in self.vertices:
            obj_content.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}")

        for normal in self.triangle_normals:
            obj_content.append(f"vn {normal[0]} {normal[1]} {normal[2]}")

        # Write triangles to obj_content (OBJ format uses 1-based indexing)
        for triangle in self.triangles:
            obj_content.append(
                f"f {triangle[0] + 1} {triangle[1] + 1} {triangle[2] + 1}"
            )
        return "\n".join(obj_content)

    def get_obj_description(
        self, shading: str = "off", mtl: str = "Material", **kwargs
    ) -> str:
        """
        Converts the TriangleMesh to an OBJ file content string.

        Parameters
        ----------
        shading: str or int
            Shading across polygons is enabled by smoothing groups. Default: "off".
        mtl: str
            Specifies the material name for the element following it.
            The material name matches a named material definition in an external
            .mtl file. Default: "Material"

        Returns
        -------
        str
            The content of the OBJ file as a string.
        """
        return mesh_to_obj_description(
            "TriangleMesh", self, shading=shading, mtl=mtl, **kwargs
        )

    @staticmethod
    def get_loop_indexes_from_boundary_indexes(
        boundary_indexes: list[tuple]
    ) -> list[list]:
        """Detect loops in list of tuples.

        See: get_triangle_boundary_indexes

        Parameters
        ----------
        boundary_indexes : list of tuples
            List of tuples, each tuple containing the indexes of two points in the
            triangle, defining an edge line.
            Can be the output of get_triangle_boundary_indexes.

        Returns
        -------
        list of lists
            All detected loops.
        """

        # separating (and ordering) all loops
        def find_tuple_index(lst, value):
            return next((index for index, tup in enumerate(lst) if value in tup), None)

        loop_indexes = []
        boundary_indexes = copy.copy(boundary_indexes)
        while len(boundary_indexes) > 0:
            boundary = []
            edge = boundary_indexes.pop()
            boundary.append(edge)

            while True:
                edge = boundary[-1]

                index = find_tuple_index(boundary_indexes, edge[1])
                if index is None:
                    index = find_tuple_index(boundary_indexes, edge[0])

                if index is None:
                    break

                edge = boundary_indexes.pop(index)
                if edge[1] == boundary[-1][1]:
                    edge = edge[::-1]

                boundary.append(edge)

            if boundary[-1] == boundary[0]:
                break

            # boundaries.append(boundary)
            loop_indexes.append([p[0] for p in boundary])

        return loop_indexes

    @classmethod
    def get_fused_mesh(cls, meshes: list["TriangleMesh"]) -> "TriangleMesh":
        """Fuse TriangleMesh instances into single mesh.

        Parameters
        ----------
        meshes : list of meshes
            TriangleMesh instances to be fused

        Returns
        -------
        TriangleMesh
            Single triangle mesh
        """
        vertices_list = [mesh.vertices for mesh in meshes]
        triangles_list = [mesh.triangles for mesh in meshes]
        vertices, triangles = cls.fuse_vertices_triangles(vertices_list, triangles_list)
        return cls(vertices, np.vstack(triangles))

    @classmethod
    def triangulate_earclipping(cls, polygon: np.ndarray) -> np.ndarray:
        """
        Shamelessly copied from tripy:
            https://github.com/linuxlewis/tripy/blob/master/tripy.py

        Simple earclipping algorithm for a given polygon p.
        polygon is expected to be an array of 2-tuples of the cartesian points of the polygon

        For a polygon with n points it will return n-2 triangles.
        The triangles are returned as an array of 3-tuples where each item in the tuple is a 2-tuple of the cartesian point.

        e.g
        >>> polygon = [(0,1), (-1, 0), (0, -1), (1, 0)]
        >>> triangles = tripy.earclip(polygon)
        >>> triangles
        [((1, 0), (0, 1), (-1, 0)), ((1, 0), (-1, 0), (0, -1))]

        Implementation Reference:
            - https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf

        Returns
        -------
        Numpy array
            Defines triangles
        """
        import math
        import sys
        from collections import namedtuple

        warnings.warn(
            "TriangleMesh.triangulate_earclipping is deprecated and should not be used.",
            DeprecationWarning,
        )

        polygon = np.array(polygon)

        if not len(polygon.shape) == 2 or not polygon.shape[-1] == 2:
            raise ValueError(f"Array of shape (N, 2) expected, got {polygon.shape}.")

        original_polygon = copy.copy(polygon)

        Point = namedtuple("Point", ["x", "y"])

        EPSILON = math.sqrt(sys.float_info.epsilon)

        def _is_clockwise(polygon):
            s = 0
            polygon_count = len(polygon)
            for i in range(polygon_count):
                point = polygon[i]
                point2 = polygon[(i + 1) % polygon_count]
                s += (point2.x - point.x) * (point2.y + point.y)
            return s > 0

        def _triangle_sum(x1, y1, x2, y2, x3, y3):
            return x1 * (y3 - y2) + x2 * (y1 - y3) + x3 * (y2 - y1)

        def _is_convex(prev, point, next):
            return _triangle_sum(prev.x, prev.y, point.x, point.y, next.x, next.y) < 0

        def _contains_no_points(p1, p2, p3, polygon):
            for pn in polygon:
                if pn in (p1, p2, p3):
                    continue
                elif _is_point_inside(pn, p1, p2, p3):
                    return False
            return True

        def _is_ear(p1, p2, p3, polygon):
            ear = (
                _contains_no_points(p1, p2, p3, polygon)
                and _is_convex(p1, p2, p3)
                and _triangle_area(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y) > 0
            )
            return ear

        def _is_point_inside(p, a, b, c):
            area = _triangle_area(a.x, a.y, b.x, b.y, c.x, c.y)
            area1 = _triangle_area(p.x, p.y, b.x, b.y, c.x, c.y)
            area2 = _triangle_area(p.x, p.y, a.x, a.y, c.x, c.y)
            area3 = _triangle_area(p.x, p.y, a.x, a.y, b.x, b.y)
            areadiff = abs(area - sum([area1, area2, area3])) < EPSILON
            return areadiff

        def _triangle_area(x1, y1, x2, y2, x3, y3):
            return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

        ear_vertex = []
        triangles = []

        polygon = [Point(*point) for point in polygon]

        if _is_clockwise(polygon):
            polygon.reverse()

        point_count = len(polygon)
        for i in range(point_count):
            prev_index = i - 1
            prev_point = polygon[prev_index]
            point = polygon[i]
            next_index = (i + 1) % point_count
            next_point = polygon[next_index]

            if _is_ear(prev_point, point, next_point, polygon):
                ear_vertex.append(point)

        while ear_vertex and point_count >= 3:
            ear = ear_vertex.pop(0)
            i = polygon.index(ear)
            prev_index = i - 1
            prev_point = polygon[prev_index]
            next_index = (i + 1) % point_count
            next_point = polygon[next_index]

            polygon.remove(ear)
            point_count -= 1
            triangles.append(
                (
                    (prev_point.x, prev_point.y),
                    (ear.x, ear.y),
                    (next_point.x, next_point.y),
                )
            )
            if point_count > 3:
                prev_prev_point = polygon[prev_index - 1]
                next_next_index = (i + 1) % point_count
                next_next_point = polygon[next_next_index]

                groups = [
                    (prev_prev_point, prev_point, next_point, polygon),
                    (prev_point, next_point, next_next_point, polygon),
                ]
                for group in groups:
                    p = group[1]
                    if _is_ear(*group):
                        if p not in ear_vertex:
                            ear_vertex.append(p)
                    elif p in ear_vertex:
                        ear_vertex.remove(p)

        triangles = np.array(
            [[np.where(original_polygon == p)[0][0] for p in t] for t in triangles]
        )

        return triangles

    @staticmethod
    def fuse_vertices_triangles(
        vertices_list: list[np.ndarray[float]], triangles_list: list[np.ndarray[int]]
    ) -> tuple[np.ndarray[float], np.ndarray[int]]:
        if len(vertices_list) != len(triangles_list):
            raise ValueError(
                f"{len(vertices_list)} vertices lists and "
                f"{len(triangles_list)} triangles lists, should be equal."
            )

        vertices = np.vstack(vertices_list)
        triangles = []
        L = 0
        # for i in range(1, len(vertices_list)):
        for i in range(len(vertices_list)):
            triangles.append(np.array(triangles_list[i]) + L)
            L += len(vertices_list[i])
        return vertices, np.vstack(triangles)

    def write_triangle_mesh(self, filepath: Union[Path, str], **options):
        """Write triangle mesh to file.

        Internal call to Open3D.io.write_triangle_mesh

        Parameters
        ----------
        filepath : string or instance of pathlib.Path
            File to be loaded

        See: Open3D.io.write_triangle_mesh
        """

        filepath = Path(filepath)
        if filepath.suffix == ".stl":
            self.compute_vertex_normals()

        io.write_triangle_mesh(filepath.as_posix(), self.as_open3d, **options)

    @classmethod
    def read_triangle_mesh(cls, filepath: Union[Path, str]) -> "TriangleMesh":
        """Read file to triangle mesh.

        Internal call to Open3D.io.read_triangle_mesh.

        Parameters
        ----------
        filepath : string or instance of pathlib.Path
            File to be loaded

        Returns
        -------
        PointCloud
            Loaded point cloud.

        See: Open3D.io.read_triangle_mesh
        """
        if isinstance(filepath, Path):
            filename = filepath.as_posix()
        else:
            filename = filepath
            filepath = Path(filename)

        return TriangleMesh(io.read_triangle_mesh(filename))

    # def alphashape_2d(projections, alpha):
    #     """ Compute the alpha shape (concave hull) of a set of 2D points. If the number
    #     of points in the input is three or less, the convex hull is returned to the
    #     user.

    #     Parameters
    #     ----------
    #     projections : array_like, shape (N, 2)
    #         Points corresponding to the 2D projections in the plane.

    #     Returns
    #     -------
    #     vertices : np.array_like, shape (N, 2)
    #         Boundary points in computed shape.

    #     triangles : np.array shape (N, 3)
    #         Indices of each triangle.
    #     """
    #     projections = np.asarray(projections)
    #     if projections.shape[1] != 2:
    #         raise ValueError("Input points must be 2D.")

    #     # If given a triangle for input, or an alpha value of zero or less,
    #     # return the convex hull.
    #     if len(projections) < 4 or (alpha is not None and not callable(
    #             alpha) and alpha <= 0):

    #         convex_hull = ConvexHull(projections)
    #         return convex_hull.points, convex_hull.simplices

    #     # Determine alpha parameter if one is not given
    #     if alpha is None:
    #         try:
    #             from optimizealpha import optimizealpha
    #         except ImportError:
    #             from .optimizealpha import optimizealpha
    #         alpha = optimizealpha(projections)

    #     vertices = np.array(projections)

    #     # Create a set to hold unique edges of simplices that pass the radius
    #     # filtering
    #     edges = set()

    #     # Create a set to hold unique edges of perimeter simplices.
    #     # Whenever a simplex is found that passes the radius filter, its edges
    #     # will be inspected to see if they already exist in the `edges` set.  If an
    #     # edge does not already exist there, it will be added to both the `edges`
    #     # set and the `permimeter_edges` set.  If it does already exist there, it
    #     # will be removed from the `perimeter_edges` set if found there.  This is
    #     # taking advantage of the property of perimeter edges that each edge can
    #     # only exist once.
    #     perimeter_edges = set()

    #     for point_indices, circumradius in alphasimplices(vertices):
    #         if callable(alpha):
    #             resolved_alpha = alpha(point_indices, circumradius)
    #         else:
    #             resolved_alpha = alpha

    #         # Radius filter
    #         if circumradius < 1.0 / resolved_alpha:
    #             for edge in itertools.combinations(
    #                     # point_indices, r=coords.shape[-1]):
    #                     point_indices, 3):
    #                 if all([e not in edges for e in itertools.combinations(
    #                         edge, r=len(edge))]):
    #                     edges.add(edge)
    #                     perimeter_edges.add(edge)
    #                 else:
    #                     perimeter_edges -= set(itertools.combinations(
    #                         edge, r=len(edge)))

    #     triangles = np.array(list(perimeter_edges))
    #     return vertices, triangles

    # def polygonize_alpha_shape(vertices, edges):
    #     # Create the resulting polygon from the edge points
    #     m = MultiLineString([vertices[np.array(edge)] for edge in edges])
    #     triangles = list(polygonize(m))
    #     result = unary_union(triangles)
    #     return result
