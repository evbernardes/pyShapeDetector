#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:57:08 2023

@author: ebernardes
"""
import warnings
import numpy as np
from pyShapeDetector.geometry import TriangleMesh
# from skspatial.objects.cylinder import Cylinder as skcylinder

from pyShapeDetector import utility
from pyShapeDetector.geometry import PointCloud
from .primitivebase import Primitive
from .plane import Plane
from .planebounded import PlaneBounded


class Cylinder(Primitive):
    """
    Cylinder primitive.

    Attributes
    ----------
    dimensions
    is_bounded
    fit_n_min
    model_args_n
    name
    model
    equation
    surface_area
    center
    volume
    canonical
    color
    mesh
    inliers
    inliers_flattened
    has_inliers
    inlier_mean
    inlier_median
    inlier_points
    inlier_points_flattened
    inlier_normals
    inlier_colors
    metrics
    axis_spherical
    axis_cylindrical
    aabb
    obb

    base
    top
    center
    center_projection
    vector
    length
    height
    axis
    radius
    rotation_from_axis

    Methods
    -------
    __init__
    __repr__
    __eq__
    random
    fit
    get_signed_distances
    get_distances
    get_normals
    get_angles_cos
    get_angles
    get_residuals
    flatten_points
    flatten_PointCloud
    set_inliers
    add_inliers
    closest_inliers
    inliers_average_dist
    get_axis_aligned_bounding_box
    get_oriented_bounding_box
    sample_points_uniformly
    sample_points_density
    sample_PointCloud_uniformly
    sample_PointCloud_density
    get_mesh
    get_cropped_mesh
    is_similar_to
    __copy_atributes__
    __copy__
    copy
    translate
    rotate
    transform
    align
    save
    load
    get_obj_description
    fuse
    group_similar_shapes
    fuse_shape_groups
    fuse_similar_shapes

    from_base_vector_radius
    from_center_half_vector_radius
    from_base_top_radius
    closest_to_line
    get_orthogonal_component
    project_to_plane
    cuts
    """

    _fit_n_min = 6
    _model_args_n = 7
    _name = "cylinder"
    _translatable = [0, 1, 2]
    _rotatable = [3, 4, 5]
    _color = np.array([1.0, 0.0, 0.0])
    _dimensions = 2
    _is_bounded = True

    @property
    def equation(self):
        def sig(x):
            return "-" if x < 0 else "+"

        delta = [
            f"{p} {sig(-a)} {abs(a)}" for p, a in zip(("x", "y", "z"), self.center)
        ]
        A = " + ".join([f"({p})**2" for p in delta])
        B = " + ".join([f"{e} * ({p})" for e, p in zip(self.axis, delta)])
        return A + " + [" + B + "]**2" + f" = {self.radius ** 2}"

    @property
    def surface_area(self):
        """Surface area of primitive"""
        return 2 * np.pi * self.radius * self.height

    @property
    def volume(self):
        """Volume of primitive."""
        return np.pi * (self.radius**2) * self.height

    @property
    def canonical(self):
        """Return canonical form for testing."""
        if self.vector[-1] >= 0:
            return self

        return Cylinder(list(self.base) + list(-self.vector) + [self.radius])

    @property
    def base(self):
        """Point at the base of the cylinder."""
        return np.array(self.model[:3])
        # return self.center + self.vector / 2

    @property
    def top(self):
        """Point at the top of the cylinder."""
        return self.base + self.vector

    @property
    def center(self):
        """Center point of the cylinder."""
        return self.base + self.vector / 2
        # return np.array(self.model[:3])

    @property
    def center_projection(self):
        """Center point of the cylinder."""
        return self.center - self.axis.dot(self.center)
        # return np.array(self.model[:3])

    @property
    def vector(self):
        """Vector from base point to top point."""
        return np.array(self.model[3:6])

    @property
    def length(self):
        """Height/length of cylinder."""
        return np.linalg.norm(self.vector)

    @property
    def height(self):
        """Height/length of cylinder."""
        return self.length

    @property
    def axis(self):
        """Unit vector defining axis of cylinder."""
        return self.vector / self.height

    @property
    def radius(self):
        """Radius of the cylinder."""
        return self.model[6]

    @property
    def rotation_from_axis(self):
        """Rotation matrix that aligns z-axis with cylinder axis."""
        return utility.get_rotation_from_axis([0, 0, 1], self.axis)

    @classmethod
    def from_base_vector_radius(cls, base, vector, radius):
        """Creates cylinder from center base point, vector and radius as
        separated arguments.

        Parameters
        ----------
        base : 3 x 1 array
            Center point at the base of the cylinder.
        vector : 3 x 1 array
            Vector from base center point to top center point.
        radius : float
            Radius of the cylinder.

        Returns
        -------
        Cone
            Generated shape.
        """
        return cls(list(base) + list(vector) + [radius])

    @staticmethod
    def fuse(
        shapes: list["Cylinder"],
        detector=None,
        ignore_extra_data=False,
        **extra_options,
    ):
        """Find weigthed average of shapes, where the weight is the fitness
        metric.

        If a detector is given, use it to compute the metrics of the resulting
        average shapes.

        When fusing Cylinders, find base and top such that it constructs a full
        cylinder containing all the other instances.

        Parameters
        ----------
        shapes : list
            Grouped shapes. All shapes must be of the same type.
        detector : instance of some Detector, optional
            Used to recompute metrics. Default: None.
        ignore_extra_data : boolean, optional
            If True, ignore everything and only fuse model. Default: False.

        Returns
        -------
        Cylinder
            Averaged Cylinder instance.
        """
        try:
            fitness = [shape.metrics["fitness"] for shape in shapes]
        except Exception:
            fitness = [1] * len(shapes)

        points = []
        # axes = []
        axis = np.array([0.0, 0.0, 0.0])
        center = np.array([0.0, 0.0, 0.0])
        radii = []
        for weight, cylinder in zip(fitness, shapes):
            radii.append(cylinder.radius)
            points.append(cylinder.base)
            points.append(cylinder.top)
            if cylinder.axis.dot(axis) < 0:
                axis += -weight * cylinder.axis / sum(fitness)
            else:
                axis += weight * cylinder.axis / sum(fitness)
            center += cylinder.center / sum(fitness)
        # axis = np.average(axes, axis=0, weights=fitness)
        axis /= np.linalg.norm(axis)
        radius = np.average(radii, weights=fitness)
        points = np.asarray(points)

        projections = axis.dot((points - center).T)
        base = center + axis * min(projections)
        top = center + axis * max(projections)

        shape = Cylinder.from_base_top_radius(base, top, radius)

        if not ignore_extra_data:
            pcd = PointCloud.fuse_pointclouds([shape.inliers for shape in shapes])
            shape.set_inliers(pcd)
            shape.color = np.average([s.color for s in shapes], axis=0, weights=fitness)

            if detector is not None:
                num_points = sum([shape.metrics["num_points"] for shape in shapes])
                num_inliers = len(pcd.points)
                distances, angles = shape.get_residuals(pcd.points, pcd.normals)
                shape.metrics = detector.get_metrics(
                    num_points, num_inliers, distances, angles
                )

        return shape

    @staticmethod
    def fit(points, normals=None):
        """Gives cylinder that fits the input points.

        If normals are given: first calculate cylinder axis using normals as
        explained in [1] and then use least squares to calculate center point
        and radius.

        If normals are not given, uses Scikit Spatial, which is slower and not
        recommended.

        References
        ----------
        [1]: http://dx.doi.org/10.1016/j.cag.2014.09.027

        Parameters
        ----------
        points : N x 3 array
            N input points

        Returns
        -------
        Cylinder
            Fitted cylinder.
        """
        points = np.asarray(points)

        num_points = len(points)

        if num_points < 6:
            raise ValueError("A minimun of 6 points are needed to fit a cylinder")

        if normals is None:
            raise NotImplementedError(
                "Fitting of cylinder without normals has not been implemented."
            )
            # # if no normals, use scikit spatial, slower
            # warnings.warn('Cylinder fitting works much quicker if normals '
            #               'are given.')
            # solution = skcylinder.best_fit(points)

            # base = list(solution.point)
            # # center = list(solution.point + solution.vector/2)
            # vector = list(solution.vector)
            # radius = solution.radius

        # Reference for axis estimation with normals:
        # http://dx.doi.org/10.1016/j.cag.2014.09.027
        normals = np.asarray(normals)
        if len(normals) != num_points:
            raise ValueError("Different number of points and normals")

        eigval, eigvec = np.linalg.eig(normals.T @ normals)
        idx = eigval == min(eigval)
        if sum(idx) != 1:  # no well defined minimum eigenvalue
            return None

        axis = eigvec.T[idx][0]

        # Reference for the rest:
        # Was revealed to me in a dream
        axis_neg_squared_skew = np.eye(3) - axis[np.newaxis].T * axis
        points_skew = (axis_neg_squared_skew @ points.T).T
        b = sum(points_skew.T * points_skew.T)
        a = np.c_[2 * points_skew, np.ones(num_points)]
        X = np.linalg.lstsq(a, b, rcond=None)[0]

        point = X[:3]
        radius = np.sqrt(X[3] + point.dot(axis_neg_squared_skew @ point))

        # find point in base of cylinder
        proj = points.dot(axis)
        idx = np.where(proj == max(proj))[0][0]

        # point = list(point)
        height = max(proj) - min(proj)
        vector = axis * height
        center = -np.cross(axis, np.cross(axis, point)) + np.median(proj) * axis
        base = center - vector / 2

        # base = list(base)
        # center = list(center)
        # vector = list(vector)

        # return Cylinder(center+vector+[radius])
        return Cylinder.from_base_vector_radius(base, vector, radius)

    @utility.accept_one_or_multiple_elements(3)
    def get_signed_distances(self, points):
        """Gives the minimum distance between each point to the cylinder.

        Parameters
        ----------
        points : N x 3 array
            N input points

        Returns
        -------
        distances
            Nx1 array distances.
        """
        points = np.asarray(points)
        distances = np.linalg.norm(np.cross(self.axis, points - self.base), axis=1)

        return distances - self.radius

    @utility.accept_one_or_multiple_elements(3)
    def get_normals(self, points):
        """Gives, for each input point, the normal vector of the point closest
        to the cylinder.

        Parameters
        ----------
        points : N x 3 array
            N input points

        Returns
        -------
        normals
            Nx3 array containing normal vectors.
        """
        normals = self.get_orthogonal_component(points)
        normals /= np.linalg.norm(normals, axis=1)[..., np.newaxis]
        return normals

    def get_mesh(self, **options):
        """Returns mesh defined by the cylinder model.

        Parameters
        ----------
        resolution : int, optional
            Resolution parameter for mesh. Default: 30
        closed : bool, optional
            If True, does not remove top and bottom of cylinder

        Returns
        -------
        TriangleMesh
            Mesh corresponding to the cylinder.
        """

        resolution = options.get("resolution", 30)
        closed = options.get("closed", False)

        mesh = TriangleMesh.create_cylinder(
            radius=self.radius, height=self.height, resolution=resolution
        )
        # radius=self.radius, height=self.height, resolution=100, split=100)

        # first and second points are the central points defining top / base
        triangles = np.asarray(mesh.triangles)

        if not closed:
            triangles = np.array([t for t in triangles if 0 not in t and 1 not in t])
            # triangles = np.vstack([triangles, triangles[:, ::-1]])
            mesh.triangles = triangles

        mesh.rotate(self.rotation_from_axis)
        mesh.translate(self.center)

        return mesh

    @classmethod
    def from_center_half_vector_radius(cls, center, half_vector, radius):
        """Creates cylinder from center point, half vector and radius as
        separated arguments.

        Parameters
        ----------
        center : 3 x 1 array
            Center point of the cylinder.
        half_vector : 3 x 1 array
            Vector from center point to top center point.
        radius : float
            Radius of the cylinder.

        Returns
        -------
        Cone
            Generated shape.
        """
        half_vector = np.asarray(half_vector)
        base = np.asarray(center) - half_vector
        return cls.from_base_vector_radius(base, 2 * half_vector, radius)

    @classmethod
    def from_base_top_radius(cls, base, top, radius):
        """Creates cylinder from center base point, center top point and
        radius as separated arguments.

        Parameters
        ----------
        base : 3 x 1 array
            Center point at the base of the cylinder.
        top : 3 x 1 array
            Center point at the top of the cylinder.
        radius : float
            Radius of the cylinder.

        Returns
        -------
        Cone
            Generated shape.
        """
        vector = np.array(top) - np.array(base)
        return cls.from_base_vector_radius(base, vector, radius)

    def closest_to_line(self, points):
        """Returns points in cylinder axis that are the closest to the input
        points.

        Parameters
        ----------
        points : N x 3 array
            N input points

        Returns
        -------
        points_closest: N x 3 array
            N points in cylinder line
        """
        points = np.asarray(points)
        projection = (points - self.base).dot(self.axis)
        return self.base + projection[..., np.newaxis] * self.axis

    def get_orthogonal_component(self, points):
        """Removes the axis-aligned components of points.

        Parameters
        ----------
        points : N x 3 array
            N input points

        Returns
        -------
        points_orthogonal: N x 3 array
            N points
        """
        points = np.asarray(points)
        delta = points - self.base
        return -np.cross(self.axis, np.cross(self.axis, delta))

    def project_to_plane(self, plane, resolution=30):
        """Projects cylinder into a plane, creating an elliptical plane.

        Parameters
        ----------
        plane : Plane
            Plane instance.
        resolution : int, optional
            Number of points defining elliptical plane.

        Returns
        -------
        PlaneBounded
            Fitted elliptical plane.
        """
        cos_theta = np.dot(self.axis, plane.normal)
        if np.abs(np.dot(self.axis, plane.normal)) < 1e-7:
            warnings.warn("Plane normal and cylinder axis cannot be orthogonal.")
            return None

        random_axis_in_plane = np.cross(np.random.random(3), plane.normal)
        random_axis_in_plane /= np.linalg.norm(random_axis_in_plane)
        vx = np.cross(self.axis, plane.normal)
        if np.linalg.norm(vx) < 1e-7:
            vx = random_axis_in_plane
        vy = np.cross(plane.normal, vx)
        vx *= self.radius / np.linalg.norm(vx)
        vy *= self.radius / np.linalg.norm(vy) / cos_theta

        dist = plane.get_signed_distances(self.center)
        cos_theta = np.dot(self.axis, plane.normal)
        center = self.center + self.axis * dist / cos_theta

        return PlaneBounded.create_ellipse(center, vx, vy, resolution)

        # Older wrong circular implementation
        # circle = PlaneBounded.create_circle(
        #     self.center, self.axis, self.radius, resolution)

        # random_axis = np.random.random(3)
        # random_axis /= np.linalg.norm(random_axis)
        # vx = np.cross(random_axis, plane.normal)
        # vy = np.cross(plane.normal, vx)

        # dist = plane.get_distances(self.center)
        # cos_theta = np.dot(self.axis, plane.normal)

        # points = circle.bounds
        # points += self.axis * dist / cos_theta

        # if not np.isclose(cos_theta, 1):
        #     rot_axis = np.cross(self.axis, plane.normal)
        #     rot_axis /= np.linalg.norm(rot_axis)
        #     rot = Rotation.from_rotvec(np.arccos(cos_theta) * rot_axis)
        #     center = points.mean(axis=0)
        #     points = center + rot.apply(points - center)

        # return PlaneBounded.create_ellipse(center, vx, vy)

    def cuts(self, plane, total_cut=False, eps=0):
        """Returns true if cylinder cuts through plane.

        Parameters
        ----------
        plane : Plane
            Plane instance.
        total_cut : boolean, optional
            When True, only accepts cuts when either the top of the bottom
            completely cuts the plane. Default: False.
        eps : float, optional
            Adds some backlash to top and bottom of cylinder. Default: 0.

        Returns
        -------
        Bool
            True if cylinder cuts plane.
        """
        if not isinstance(plane, Plane):
            warnings.warn(
                "'cuts_shape' is only known to work with planes, "
                "trying it anyway with {plane.name}"
            )

        top = PlaneBounded.create_circle(
            self.top + eps * self.axis, self.axis, self.radius
        )
        base = PlaneBounded.create_circle(
            self.base - eps * self.axis, self.axis, self.radius
        )
        center = np.sign(np.dot(self.center, plane.normal))

        # test_passes = False

        def test(center, circle, total_cut):
            sign_circle = np.sign(plane.get_signed_distances(circle.bounds))
            if not total_cut and not np.all(sign_circle == sign_circle[0]):
                return True
            sign_circle = [s for s in sign_circle if s != 0]
            if np.all(sign_circle == sign_circle[0]) and center != sign_circle[0]:
                return True
            return False

        return test(center, base, total_cut) or test(center, top, total_cut)

        # def point_in_poly(hull, point):
        #     for simplex in hull.simplices:
        #         x, y = hull.points[simplex, 0], hull.points[simplex, 1]
        #         p1, p2 = (x[0], y[0]), (x[1], y[1])
        #         # Check if the point is on the left side of all edges
        #         if (point[0] - p1[0]) * (p2[1] - p1[1]) - (point[1] - p1[1]) * (p2[0] - p1[0]) > 0:
        #             return False
        #     return True

        # rot = plane.utility.get_rotation_from_axis([0, 0, 1], plane.normal)
        # projected_circle = self.project_to_plane(plane)
        # projected_points = (rot @ projected_circle.bounds.T).T[:, :2]
        # hull = ConvexHull((rot @ plane.bounds.T).T[:, :2])

        # return np.all([point_in_poly(hull, p) for p in projected_points])
