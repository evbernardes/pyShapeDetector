#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:15:09 2024

@author: ebernardes
"""
from warnings import warn
from itertools import permutations, product
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
# from scipy.spatial.transform import Rotation
from open3d.geometry import TriangleMesh
from open3d.utility import Vector3iVector, Vector3dVector

from pyShapeDetector.utility import get_triangle_surface_areas
from .plane import Plane
# from alphashape import alphashape

class PlaneBounded(Plane):
    """
    PlaneBounded primitive.

    Attributes
    ----------
    fit_n_min 
    model_args_n
    name
    model
    equation
    surface_area 
    volume
    canonical
    color
    inlier_points
    inlier_points_flattened
    inlier_normals
    inlier_colors
    inlier_PointCloud
    inlier_PointCloud_flattened
    metrics
    axis_spherical
    axis_cylindrical

    normal
    dist
    centroid
    holes

    plane
    bounds
    bounds_indices
    bounds_projections
    bound_lines

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
    add_inliers
    closest_inliers
    inliers_average_dist
    inliers_bounding_box
    sample_points_uniformly
    sample_points_density
    get_mesh
    get_cropped_mesh
    is_similar_to
    copy
    translate
    rotate
    align
    save
    load
    check_bbox_intersection
    check_inlier_distance

    from_normal_dist
    from_normal_point
    bound_lines_meshes
    add_holes
    remove_hole
    intersect
    closest_bounds
    get_bounded_plane
    get_projections
    get_points_from_projections
    get_mesh_alphashape
    get_square_plane
    get_square_mesh

    contains_projections
    set_bounds
    add_bound_points
    create_circle
    create_ellipse
    create_box
    intersection_bounds

    """

    _name = 'bounded plane'
    _bounds_indices = np.array([])
    _bounds = np.array([])
    _bounds_projections = np.array([])

    @property
    def model(self):
        """ Return model. """
        return self._plane.model

    @property
    def surface_area(self):
        """ Surface area of bounded plane. """
        return self.get_mesh().get_surface_area()

    @property
    def canonical(self):
        """ Return canonical form for testing. """
        model = self.model
        if self.dist >= 0:
            model = -model
        canonical_plane = PlaneBounded(list(-self.model), self.bounds)
        canonical_plane._holes = self._holes
        return canonical_plane
    
    @property
    def plane(self):
        """ Return internal plane without bounds. """
        return self._plane

    @property
    def bounds(self):
        return self._bounds

    @property
    def bounds_indices(self):
        """ Indices of points corresponding to bounds. """
        # TODO: should take into consideration added bounds
        return self._bounds_indices

    @property
    def bounds_projections(self):
        return self._bounds_projections

    @property
    def bound_lines(self):
        """ Lines defining bounds. """
        from .line import Line
        return Line.from_bounds(self.bounds)

    def __init__(self, planemodel, bounds=None, rmse_max=1e-3,
                 method='convex', alpha=None):
        """
        Parameters
        ----------
        planemodel : Plane or list of 4 values
            Shape defining plane
        bounds : array_like, shape (N, 3)
            Points defining bounds
        method : string, optional
            "convex" for convex hull, or "alpha" for alpha shapes.
            Default: "convex"
        alpha : float, optional
            Alpha parameter for alpha shapes algorithm. If equal to None,
            calculates the optimal alpha. Default: None

        Raises
        ------
        ValueError
            If number of parameters is incompatible with the model of the 
        """

        if isinstance(planemodel, PlaneBounded):
            self._plane = planemodel.unbounded
        elif isinstance(planemodel, Plane):
            self._plane = planemodel
        else:
            self._plane = Plane(planemodel)
        # self._model = self._plane.model

        if bounds is None:
            warn('No input bounds, returning square plane')
            self = self._plane.get_square_plane(1)

        else:
            bounds = np.asarray(bounds)
            if bounds.shape[1] != 3:
                raise ValueError('Expected shape of bounds is (N, 3), got '
                                 f'{bounds.shape}')

            distances = self._plane.get_distances(bounds)
            rmse = np.sqrt(sum(distances * distances)) / len(distances)
            if rmse_max is not None and rmse_max > 1e-3:
                raise ValueError('Boundary points are not close enough to '
                                 f'plane: rmse={rmse}, expected less than '
                                 f'{rmse_max}.')

            self.set_bounds(bounds, flatten=True)

            if self._bounds is None:
                self = None

        self._holes = []

    @classmethod
    def random(cls, scale=1):
        """ Generates a random shape.

        Parameters
        ----------
        scale : float, optional
            scaling factor for random model values.

        Returns
        -------
        Primitive
            Random shape.
        """
        plane = Plane(np.random.random(4) * scale)
        return plane.get_square_plane(np.random.random() * scale)

    @staticmethod
    def fit(points, normals=None):
        """ Gives plane that fits the input points. If the numb
        points : N x 3 arrayer of points is
        higher than the `3`, the fitted shape will return a least squares 
        estimation.

        Reference:
            https://www.ilikebigbits.com/2015_03_04_plane_from_points.html

        Parameters
        ----------
        points : N x 3 array
            N input points 
        normals : N x 3 array
            N normal vectors

        Returns
        -------
        PlaneBounded
            Fitted plane.
        """

        plane = Plane.fit(points, normals)
        return plane.get_bounded_plane(points)

    def get_mesh(self, resolution=None):
        """ Flatten points and creates a simplified mesh of the plane defined
        by the points at the borders.

        Parameters
        ----------
        points : N x 3 array
            Points corresponding to the fitted shape.

        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        # points = self.bounds
        # projections = self.bounds_projections
        # print(f'has {len(self._fusion_intersections)} intersections')
        if len(self._fusion_intersections) == 0:
            points = self.bounds
            projections = self.bounds_projections
            idx_intersections_sorted = []
        else:
            points = np.vstack([self.bounds, self._fusion_intersections])
            projections = self.get_projections(points)

            angles = projections - projections.mean(axis=0)
            angles = np.arctan2(*angles.T) + np.pi
            idx = np.argsort(angles)

            points = points[idx]
            projections = projections[idx]

            idx_intersections = list(
                range(len(points) - len(self._fusion_intersections), len(points)))
            idx_intersections_sorted = [
                np.where(i == idx)[0][0] for i in idx_intersections]
            # print(idx_intersections)

        # holes = self._holes
        # has_holes = len(holes) != 0
        # if has_holes:
        #     labels = []
        #     projections_holes = []
        #     points_holes = []
        #     labels_holes = []
        #     for i in range(len(holes)):
        #         hole = holes[i]
        #         projections_holes.append(hole.bounds_projections)
        #         points_holes.append(self.flatten_points(
        #             hole.bounds))
        #         labels += [i+1] * len(hole.bounds_projections)
        #     labels = np.array(
        #         [0] * len(projections) + labels)
        #     projections = np.vstack([projections]+projections_holes)
        #     points = np.vstack([points]+points_holes)

        # # is_points = np.asarray(is_points)
        # # A = dict(vertices=plane.projections, holes=[circle.projections])
        # # triangles = tr.triangulate(A)
        # triangles = Delaunay(projections).simplices
        # if has_holes:
        #     for i in range(len(holes)):
        #         triangles = triangles[
        #             ~np.all(labels[triangles] == i+1, axis=1)]

        holes = self._holes
        has_holes = len(holes) != 0
        # has_holes = False
        if has_holes:
            points_holes = [self.flatten_points(hole.bounds) for hole in holes]
            points = np.vstack([points]+points_holes)
            projections = self.get_projections(points)

        triangles = Delaunay(projections).simplices

        for hole in self._holes:
            inside_hole = np.array(
                [hole.contains_projections(p).all() for p in points[triangles]])
            triangles = triangles[~inside_hole]

        # print(len(triangles))
        # for i in idx_intersections_sorted:
        #     select = []
        #     for triangle in triangles:
        #         test = {i-1, i, (i+1) % len(points)}.intersection(triangle)
        #         select.append(len(test) == 3)
        #     select = np.array(select)
        #     triangles = triangles[~select]

        areas = get_triangle_surface_areas(points, triangles)
        triangles = triangles[areas > 0]

        # needed to make plane visible from both sides
        triangles = np.vstack([triangles, triangles[:, ::-1]])
        # print(len(triangles))

        mesh = TriangleMesh()
        mesh.vertices = Vector3dVector(points)
        mesh.triangles = Vector3iVector(triangles)

        if len(self.inlier_colors) > 0:
            # vertex_colors = []
            # for p in self.bounds:
            #     diff = np.linalg.norm(self.inlier_points - p, axis=1)
            #     i = np.where(diff == min(diff))[0][0]
            #     vertex_colors.append(self.inlier_colors[i])
            # mesh.vertex_colors = Vector3dVector(vertex_colors)
            mesh.paint_uniform_color(np.median(self.inlier_colors, axis=0))
        return mesh

    def copy(self, copy_holes=True):
        """ Returns copy of plane

        Parameters
        ----------
        copy_holes: boolean, optional
            If True, also copy holes. Default: True.

        Returns
        -------
        Primitive
            Copied primitive
        """
        shape = PlaneBounded(self.model.copy(), bounds=None)
        shape._bounds_indices = self._bounds_indices.copy()
        shape._bounds = self._bounds.copy()
        shape._bounds_projections = self._bounds_projections.copy()
        shape._inlier_points = self._inlier_points.copy()
        shape._inlier_normals = self._inlier_normals.copy()
        shape._inlier_colors = self._inlier_colors.copy()
        shape._fusion_intersections = self._fusion_intersections.copy()
        shape._metrics = self._metrics.copy()
        if copy_holes:
            holes = [h.copy(copy_holes=False) for h in self._holes]
            shape._holes = holes
        return shape

    def translate(self, translation):
        """ Translate the shape.

        Parameters
        ----------
        translation : 1 x 3 array
            Translation vector.
        """
        Plane.translate(self._plane, translation)
        self._bounds = self._bounds + translation
        for hole in self.holes:
            hole._translate_points(translation)

    def rotate(self, rotation, is_hole=False):
        """ Rotate the shape.

        Parameters
        ----------
        rotation : 3 x 3 rotation matrix or scipy.spatial.transform.Rotation
            Rotation matrix.
        """
        if not hasattr(self, '_rotatable'):
            raise NotImplementedError('Shapes of type {shape.name} do not '
                                      'have an implemented _rotatable '
                                      'attribute')
        rotation = self._parse_rotation(rotation)
        Plane.rotate(self._plane, rotation)

        self._rotate_points_normals(rotation)

        bounds = rotation.apply(self._bounds)
        self.set_bounds(bounds, flatten=True)

        if not is_hole and len(self.holes) > 0:
            for hole in self.holes:
                hole.rotate(rotation, is_hole=True)

    def contains_projections(self, points):
        """ For each point in points, check if its projection on the plane lies
        inside of the plane's bounds. 

        Parameters
        ----------
        points : N x 3 array
            N input points 

        Returns
        -------
        array of booleans
            True for points whose projection lies in plane's bounds
        """
        inside = np.array([True] * len(points))
        projections = self.get_projections(points)
        for i in range(len(points)):
            point = projections[i]
            for j in range(1, len(self.bounds_projections)):
                p1 = self.bounds_projections[j-1]
                p2 = self.bounds_projections[j]
                if (point[0] - p1[0]) * (p2[1] - p1[1]) - (point[1] - p1[1]) * (p2[0] - p1[0]) > 0:
                    inside[i] = False
                    continue
        return inside

    def set_bounds(self, points, flatten=True, method='convex', alpha=None):
        """ Flatten points according to plane model, get projections of 
        flattened points in the model and compute its boundary using either 
        the convex hull or alpha shapes.

        Parameters
        ----------
        plane : Plane
            Plane model
        points : array_like, shape (N, 3)
            Points corresponding to the fitted shape.
        flatten : bool, optional
            If False, does not flatten points
        method : string, optional
            "convex" for convex hull, or "alpha" for alpha shapes.
            Default: "convex"
        alpha : float, optional
            Alpha parameter for alpha shapes algorithm. If equal to None,
            calculates the optimal alpha. Default: None

        Returns
        -------
        bounds : array_like, shape (M, 3)
            Boundary points in plane, where M is lower or equal to N.
        projections : array_like, shape (M, 2)
            projections of boundary points in plane, where M is lower or 
            equal to N.
        """
        method = method.lower()
        if method == 'convex':
            pass
        elif method == 'alpha':
            raise NotImplementedError("Alpha shapes not implemented yet.")
        else:
            raise ValueError(
                f"method can be 'convex' or 'alpha', got {method}.")

        if flatten:
            points = self.flatten_points(points)
        if np.any(np.isnan(points)):
            raise ValueError('NaN found in points')

        projections = self.get_projections(points)

        if method == 'convex':
            chull = ConvexHull(projections)
            self._bounds_indices = chull.vertices
            self._bounds = points[chull.vertices]
            self._bounds_projections = projections[chull.vertices]

    def add_bound_points(self, new_bound_points, flatten=True, method='convex',
                         alpha=None):
        """ Add points to current bounds.

        Parameters
        ----------
        new_bound_points : N x 3 np.array
            New points to be added.
        flatten : bool, optional
            If False, does not flatten points
        method : string, optional
            "convex" for convex hull, or "alpha" for alpha shapes.
            Default: "convex"
        alpha : float, optional
            Alpha parameter for alpha shapes algorithm. If equal to None,
            calculates the optimal alpha. Default: None

        """
        points = np.vstack([self.bounds, new_bound_points])
        self.set_bounds(points, flatten, method, alpha)

    @staticmethod
    def create_circle(center, normal, radius, resolution=30):
        """ Creates circular plane.

        Parameters
        ----------
        center : 3 x 1 array
            Center of circle.
        normal : 3 x 1 array
            Normal vector of plane.
        radius : float
            Radius of circle.
        resolution : int, optional
            Number of points defining circular plane.

        Returns
        -------
        PlaneBounded
            Fitted circular plane.
        """

        def normalize(x): return x / np.linalg.norm(x)

        center = np.array(center)
        normal = normalize(np.array(normal))

        random_axis = normalize(np.random.random(3))
        ex = normalize(np.cross(random_axis, normal))
        ey = normalize(np.cross(normal, ex))

        theta = np.linspace(-np.pi, np.pi, resolution + 1)[None].T
        points = center + (np.cos(theta) * ex + np.sin(theta) * ey) * radius

        plane = Plane.from_normal_point(normal, center)
        return plane.get_bounded_plane(points)

    @staticmethod
    def create_ellipse(center, vx, vy, resolution=30):
        """ Creates elliptical plane from two vectors. The input vectors are 
        interpreted as the two axes of the ellipse, multiplied by their radius.

        They must be orthogonal.

        Parameters
        ----------
        center : 3 x 1 array
            Center of circle.
        vx : 3 x 1 array
            First axis and multiplied by its semiradius.
        vy : 3 x 1 array
            Second axis and multiplied by its semiradius.
        resolution : int, optional
            Number of points defining circular plane.

        Returns
        -------
        PlaneBounded
            Fitted circular plane.
        """
        if np.dot(vx, vy) > 1e-5:
            raise ValueError('Axes must be orthogonal.')

        center = np.array(center)
        vx = np.array(vx)
        vy = np.array(vy)
        normal = np.cross(vx, vy)
        normal /= np.linalg.norm(normal)

        theta = np.linspace(-np.pi, np.pi, resolution+1)[None].T
        points = center + np.cos(theta) * vx + np.sin(theta) * vy

        plane = Plane.from_normal_point(normal, center)
        return plane.get_bounded_plane(points)

    @staticmethod
    def create_box(center=[0, 0, 0], dimensions=[1, 1, 1]):
        """ Gives list of planes that create, together, a closed box.

        Reference:
            https://www.ilikebigbits.com/2015_03_04_plane_from_points.html

        Parameters
        ----------
        center : array
            Center point of box
        dimensions : array
            Dimensions of box

        Returns
        -------
        box : list
            List of planes.
        """

        vectors = np.eye(3) * np.array(dimensions) / 2
        center = np.array(center)
        planes = []

        for i, j in permutations([0, 1, 2], 2):
            k = 3 - i - j
            sign = int((i-j) * (j-k) * (k-i) / 2)
            v1, v2, v3 = vectors[[i, j, k]]

            points = center + np.array([
                + v1 + v2,
                + v1 - v2,
                - v1 - v2,
                - v1 + v2,
            ])

            plane = Plane.from_normal_point(v3, center + sign * v3)
            planes.append(plane.get_bounded_plane(points))

        return planes

    def intersection_bounds(self, other, within_segment=True, eps=1e-3):
        """ Calculates intersection point between bounding lines.

        Parameters
        ----------
        other : instance of PlaneBoundeds
            Other line to instersect.
        within_segment : boolean, optional
            If set to False, will suppose lines are infinite. Default: True.
        eps : float, optional
            Acceptable slack added to intervals in order to check if point
            lies on both lines, if 'within_segment' is True. Default: 1e-3.

        Returns
        -------
        point or None
            1x3 array containing point.
        """
        if not isinstance(other, PlaneBounded):
            raise ValueError("'other' must be an instance of PlaneBounded.")

        if len(self.bounds) == 0 or len(other.bounds) == 0:
            raise ValueError("Both planes must have bounds.")

        lines = self.bound_lines
        lines_other = other.bound_lines

        points = []
        for l1, l2 in product(lines, lines_other):
            p = l1.point_from_intersection(
                l2, within_segment=within_segment, eps=eps)
            if p is not None:
                points.append(p)

        if len(points) == 0:
            return np.array([])
        else:
            return np.vstack(points)      
        
        
