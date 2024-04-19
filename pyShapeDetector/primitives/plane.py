#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:42:59 2023

@author: ebernardes
"""
import warnings
from itertools import permutations, product
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation
from open3d.geometry import TriangleMesh, AxisAlignedBoundingBox
from open3d.utility import Vector3iVector, Vector3dVector

from pyShapeDetector.utility import get_rotation_from_axis, get_triangle_surface_areas
from .primitivebase import Primitive
from alphashape import alphashape
# from .line import Line

def _get_rectangular_vertices(v1, v2):
    if abs(v1.dot(v2)) > 1e-8:
        raise RuntimeError("Vectors are not orthogonal.")
        
    return np.array([
        -v1 - v2,
        +v1 - v2,
        +v1 + v2,
        -v1 + v2]) / 2
    

class Plane(Primitive):
    """
    Plane primitive.

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
    mesh
    has_inliers
    inlier_points
    inlier_points_flattened
    inlier_normals
    inlier_colors
    inlier_PointCloud
    inlier_PointCloud_flattened
    metrics
    axis_spherical
    axis_cylindrical
    bbox
    bbox_bounds
    inlier_bbox
    inlier_bbox_bounds

    is_convex
    normal
    dist
    centroid
    holes
    is_hole

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
    get_inliers_axis_aligned_bounding_box
    get_axis_aligned_bounding_box
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
    align
    save
    load
    check_bbox_intersection
    check_inlier_distance
    fuse

    from_normal_dist
    from_normal_point
    add_holes
    remove_hole
    intersect
    get_unbounded_plane
    get_bounded_plane
    get_triangulated_plane
    get_projections
    get_points_from_projections
    get_mesh_alphashape
    get_square_plane
    get_rectangular_vectors_from_inliers
    get_rectangular_plane
    get_square_mesh
    get_rectangular_mesh
    create_circle
    create_ellipse
    create_box
    """

    _fit_n_min = 3
    _model_args_n = 4
    _name = 'plane'
    _holes = []
    _rotatable = [0, 1, 2]
    _translatable = []
    _fusion_intersections = np.array([])
    _color = np.array([0, 0, 1])
    _is_hole = False
    _convex = None

    @property
    def equation(self):
        n = self.normal
        d = self.dist
        equation = ''
        equation += f'{n[0]} * x '
        equation += '-' if n[1] < 0 else '+'
        equation += f' {abs(n[1])} * y '
        equation += '-' if n[2] < 0 else '+'
        equation += f' {abs(n[2])} * z '
        equation += '-' if d < 0 else '+'
        equation += f' {abs(d)} = 0'

        return equation

    @property
    def surface_area(self):
        """ For unbounded plane, returns NaN and gives warning """
        warnings.warn('For unbounded planes, the surface area is undefined')
        return float('nan')

    @property
    def volume(self):
        """ Volume of plane, which is zero. """
        return 0

    @property
    def canonical(self):
        """ Return canonical form for testing. """
        shape = self.copy()
        if np.sign(self.dist) < 0:
            shape._model = -self._model
        return shape
    
    @property
    def is_convex(self):
        return self._convex

    @property
    def normal(self):
        """ Normal vector defining point. """
        return np.array(self.model[:3])

    @property
    def dist(self):
        """ Distance to origin. """
        return self.model[3]

    @property
    def centroid(self):
        """ A point in the plane. """
        return -self.normal * self.dist

    @property
    def holes(self):
        """ Existing holes in plane. """
        return self._holes
    
    @property
    def is_hole(self):
        return self._is_hole

    def __init__(self, model, decimals=None):
        """
        Parameters
        ----------
        model : Primitive or list of 4 values
            Shape defining plane
        decimals : int, optional
            Number of decimal places to round to (default: 0). If
            decimals is negative, it specifies the number of positions to
            the left of the decimal point. Default: None.

        Raises
        ------
        ValueError
            If number of parameters is incompatible with the model of the 
            primitive.
        """
        if isinstance(model, Primitive):
            shape = model
            model = model.model
            Primitive.__copy_atributes__(self, shape)
        else:
            model = np.array(model)

        norm = np.linalg.norm(model[:3])
        super().__init__(model / norm, decimals)
        self._holes = []

    @staticmethod
    def fit(points, normals=None):
        """ Gives plane that fits the input points. If the number of points is
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
        Plane
            Fitted plane.
        """
        points = np.asarray(points)
        num_points = len(points)

        if num_points < 3:
            raise ValueError('A minimun of 3 points are needed to fit a '
                             'plane')

        # if simplest case, the result is direct
        elif num_points == 3:
            p0, p1, p2 = points

            e1 = p1 - p0
            e2 = p2 - p0
            abc = np.cross(e1, e2)
            centroid = p0

        # for more points, find the plane such that the summed squared distance
        # from the plane to all points is minimized.
        else:
            centroid = sum(points) / num_points
            x, y, z = np.asarray(points - centroid).T
            xx = x.dot(x)
            yy = y.dot(y)
            zz = z.dot(z)
            xy = x.dot(y)
            yz = y.dot(z)
            xz = z.dot(x)

            det_x = yy * zz - yz * yz
            det_y = xx * zz - xz * xz
            det_z = xx * yy - xy * xy

            if (det_x > det_y and det_x > det_z):
                abc = np.array([det_x,
                                xz * yz - xy * zz,
                                xy * yz - xz * yy])
            elif (det_y > det_z):
                abc = np.array([xz * yz - xy * zz,
                                det_y,
                                xy * xz - yz * xx])
            else:
                abc = np.array([xy * yz - xz * yy,
                                xy * xz - yz * xx,
                                det_z])

        norm = np.linalg.norm(abc)
        if norm == 0.0:
            return None

        return Plane.from_normal_point(abc / norm, centroid)

    def get_signed_distances(self, points):
        """ Gives the minimum distance between each point to the model. 

        Actual implementation depends on the type of primitive.

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
        return points.dot(self.normal) + self.dist

    def get_normals(self, points):
        """ Gives, for each input point, the normal vector of the point closest 
        to the primitive. 

        Actual implementation depends on the type of primitive.

        Parameters
        ----------
        points : N x 3 array
            N input points 

        Returns
        -------
        normals
            Nx3 array containing normal vectors.
        """
        return np.tile(self.normal, (len(points), 1))
    
    def get_axis_aligned_bounding_box(self, slack=0):
        """ Returns an axis-aligned bounding box of the primitive.
        
        Parameters
        ----------
        slack : float, optional
            Expand bounding box in all directions, useful for testing purposes.
            Default: 0.
        
        See: open3d.geometry.get_axis_aligned_bounding_box
            
        Returns
        -------
        open3d.geometry.AxisAlignedBoundingBox
        """
        if slack < 0:
            raise ValueError("Slack must be non-negative.")
            
        warnings.warn("Unbounded planes have infinite axis aligned bounding boxes.")
        
        eps = 1e-3
        if np.linalg.norm(self.normal - [1, 0, 0]) < eps:
            idx = 0
        elif np.linalg.norm(self.normal - [0, 1, 0]) < eps:
            idx = 1
        elif np.linalg.norm(self.normal - [0, 0, 1]) < eps:
            idx = 2
        else:
            idx = -1
        
        centroid = self.centroid
        expand = np.array(
            [slack if n == idx else np.inf for n in range(3)])
        return AxisAlignedBoundingBox(centroid - expand, 
                                      centroid + expand)

    def get_mesh(self, **options):
        """ Flatten inliers points and creates a simplified mesh of the plane. If the
        shape has pre-defined inlier points, use them to find borders.
        Otherwise, return square mesh.

        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        from .planebounded import PlaneBounded
        
        if not self.has_inliers:
            warnings.warn('No inlier points, returning square plane...')
            return self.get_square_mesh()

        bounded_plane = PlaneBounded(self.model, self.inlier_points_flattened)
        # bounded_plane.__copy_atributes__(self)
        mesh = bounded_plane.get_mesh()

        return mesh

    def translate(self, translation):
        """ Translate the shape.

        Parameters
        ----------
        translation : 1 x 3 array
            Translation vector.
        """
        super().translate(translation)

        centroid = self.centroid + translation
        self._model = Plane.from_normal_point(
            self.normal, centroid).model
            
        for hole in self.holes:
            hole._translate_points(translation)

    def rotate(self, rotation):
        """ Rotate the shape.

        Parameters
        ----------
        rotation : 3 x 3 rotation matrix or scipy.spatial.transform.Rotation
            Rotation matrix.
        """

        centroid = rotation.apply(self.centroid)
        normal = rotation.apply(self.normal)
        
        rotation = Primitive._parse_rotation(rotation)
        
        # for everything else
        super().rotate(rotation)
        
        # only for model
        self._model = Plane.from_normal_point(normal, centroid).model
        
        if not self.is_hole and len(self.holes) > 0:
            for hole in self.holes:
                hole.rotate(rotation)

    def __copy_atributes__(self, shape_original):
        super().__copy_atributes__(shape_original)
        self._model = shape_original._model.copy()
        self._fusion_intersections = shape_original._fusion_intersections.copy()
        self._is_hole = shape_original._is_hole
        if not shape_original.is_hole:
            holes = [h.copy() for h in shape_original._holes]
            self._holes = holes

    # def bound_lines_meshes(self, radius=0.001, color=(0, 0, 0)):
    #     lines = self.bound_lines
    #     meshes = [line.get_mesh(radius=radius) for line in lines]
    #     [mesh.paint_uniform_color(color) for mesh in meshes]
    #     return meshes

    @classmethod
    def from_normal_dist(cls, normal, dist):
        """ Creates plane from normal vector and distance to origin.

        Parameters
        ----------            
        normal : 3 x 1 array
            Normal vector defining plane.
        radius : float
            Distance to origin.

        Returns
        -------
        Cone
            Generated shape.
        """
        return cls(list(normal)+[dist])

    @classmethod
    def from_normal_point(cls, normal, point):
        """ Creates plane from normal vector and point in plane.

        Parameters
        ----------            
        normal : 3 x 1 array
            Normal vector defining plane.
        point : 3 x 1 array
            Point in plane.

        Returns Creates plane from normal vector and point in plane.
        -------
        Cone
            Generated shape.
        """
        return cls.from_normal_dist(normal, -np.dot(normal, point))

    def add_holes(self, holes, remove_points=True):
        """ Add one or more holes to plane.

        Parameters
        ----------            
        holes : PlaneBounded or list of PlaneBounded instances
            Planes defining holes.
        remove_points : boolean, optional
            If True, removes points of plane inside holes and points of holes
            outside of plane. Default: True.
        """
        from .planebounded import PlaneBounded
        if remove_points and not isinstance(self, PlaneBounded):
            warnings.warn('Option remove_points only works if plane is bounded.')
            remove_points = False

        if not isinstance(holes, list):
            holes = [holes]

        fixed_holes = []
        for hole in holes:
            if not isinstance(hole, PlaneBounded):
                raise ValueError("Holes must be instances of PlaneBounded, got"
                                 f" {hole}.")

            cos_theta = np.dot(hole.normal, self.normal)
            if abs(cos_theta) < 1 - 1e-5:
                raise ValueError("Plane and hole must be aligned.")

            model = hole.model
            if cos_theta < 0:
                model = -model

            inside1 = self.contains_projections(hole.bounds)
            if sum(inside1) < 1:
                warnings.warn('shape does not contain hole')
                continue

            if remove_points:
                # inside1 = self.contains_projections(hole.bounds)
                # if sum(inside1) < 1:
                #     print('shape does not contain hole')
                #     continue
                if sum(~inside1) > 0:
                    intersections = []
                    for l1, l2 in product(hole.bound_lines, self.bound_lines):
                        if (point := l1.point_from_intersection(l2)) is not None:
                            intersections.append(point)
                    bounds = np.vstack([hole.bounds[inside1]]+intersections)
                else:
                    bounds = hole.bounds

                inside2 = hole.contains_projections(self.bounds)
                hole = PlaneBounded(model, bounds, convex=hole.is_convex)
                self._bounds = self._bounds[~inside2]
                self._bounds_projections = self._bounds_projections[~inside2]
            else:
                hole = PlaneBounded(model, hole.bounds, convex=hole.is_convex)
            hole._is_hole = True
            fixed_holes.append(hole)
        self._holes += fixed_holes

    def remove_hole(self, idx):
        """ Remove hole according to index.

        Parameters
        ----------            
        idx : int
            Index of hole to be removed.
        """
        self._holes.pop(idx)

    def intersect(self, other_plane, separated=False, intersect_parallel=False,
                  eps_angle=np.deg2rad(0.9), eps_distance=1e-2):
        """ Calculate the line defining the intersection with another planes.

        If separated is True, give two colinear lines, each one fitting the
        projection of one of the planes.

        Parameters
        ----------
        other_plane : instance of Plane of PlaneBounded
            Second plane
        separated : boolean, optional
            If separated is True, give two colinear lines, each one fitting the
            projection of one of the planes.
            Default: False.
        intersect_parallel : boolean, optional
            If True, try to intersect parallel planes too. Default: False.
        eps_angle : float, optional
            Minimal angle (in radians) between normals necessary for detecting
            whether planes are parallel. Default: 0.0017453292519943296
        eps_distance : float, optional
            When planes are parallel, eps_distance is used to detect if the 
            planes are close enough to each other in the dimension aligned
            with their axes. Default: 1e-2.

        Returns
        -------
        Line or None
        """
        from .planebounded import PlaneBounded
        if not isinstance(other_plane, PlaneBounded):
            raise ValueError("Only intersection with other instances of "
                             "PlaneBounded is implemented.")
        from .line import Line

        if not separated:
            return Line.from_plane_intersection(
                self, other_plane, intersect_parallel=intersect_parallel, eps_angle=eps_angle, eps_distance=eps_distance)

        line = Line.from_plane_intersection(
            self, other_plane, intersect_parallel=intersect_parallel, eps_angle=eps_angle, eps_distance=eps_distance, fit_bounds=False)

        line1 = line.get_line_fitted_to_projections(self.bounds)
        line2 = line.get_line_fitted_to_projections(other_plane.bounds)

        return line1, line2
    
    def get_unbounded_plane(self):
        """ Gives unbounded version of plane

        Returns
        -------
        Bounded
            unbounded version of plane.
        """
        plane = Plane(self)
        plane.__copy_atributes__(self)
        return plane

    def get_bounded_plane(self, bounds):
        """ Gives bounded version of plane, using input points to define 
        border.

        If points is None, then use vertices or inliers, if they exist.

        Parameters
        ----------
        points : array_like, shape (N, 3).
            Bound points.

        Returns
        -------
        PlaneBounded
            Bounded version of plane.
        """
        from .planebounded import PlaneBounded

        plane = PlaneBounded(self.model, bounds)
        Plane.__copy_atributes__(plane, self)
        return plane
    
    def get_triangulated_plane(self, vertices, triangles):
        """ Gives bounded version of plane, using input points to define 
        border.

        Parameters
        ----------
        vertices : array_like, shape (N, 3).
            Vertices for triangles.
        vertices : array_like, shape (N, 3).
            Vertices for triangles.

        Returns
        -------
        PlaneBounded
            Bounded version of plane.
        """
        from .planetriangulated import PlaneTriangulated

        plane = PlaneTriangulated(self.model, vertices, triangles)
        Plane.__copy_atributes__(plane, self)
        return plane

    def get_projections(self, points):
        """ Get 2D projections of points in plane.

        See: get_points_from_projections

        Parameters
        ----------
        points : array_like, shape (N, 3)
            Points corresponding to the fitted shape.

        Returns
        -------
        projections : array_like, shape (N, 2)
            2D projections of boundary points in plane
        """
        points = np.asarray(points)
        if points.shape[1] != 3:
            raise ValueError("Input points must be 3D.")
        rot = get_rotation_from_axis([0, 0, 1], self.normal)
        return (rot @ points.T).T[:, :2]

    def get_points_from_projections(self, projections):
        """ Get 3D points from 2D projections in plane.

        See: get_projections

        Parameters
        ----------
        projections : array_like, shape (N, 2)
            2D projections of boundary points in plane

        Returns
        -------
        projections : array_like, shape (N, 3)
            Points corresponding to the fitted shape.
        """
        projections = np.asarray(projections)
        if projections.shape[1] != 2:
            raise ValueError("Input points must be 2D.")
        N = len(projections)

        rot = get_rotation_from_axis([0, 0, 1], self.normal)
        proj_z = (rot @ self.centroid)[2]
        projections_3D = np.vstack([projections.T, np.repeat(proj_z, N)]).T

        return (rot.T @ projections_3D.T).T

    def get_mesh_alphashape(self, points, alpha=None):
        """ Flatten input points and creates a simplified mesh of the plane 
        using alpha shapes. 

        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        from pyShapeDetector.utility import alphashape_2d, new_TriangleMesh
        projections = self.get_projections(points)
        vertices_2d, triangles = alphashape_2d(projections, alpha)
        vertices = self.get_points_from_projections(vertices_2d)
        triangles = np.vstack([triangles, triangles[:, ::-1]])
        return new_TriangleMesh(vertices, triangles)
    
    def get_rectangular_vectors_from_inliers(self, return_center=False):
        """ Gives vectors defining a rectangle that roughly contains the plane.
        
        Parameters
        ----------
        return_center : boolean, optional
            If True, return tuple containing both vectors and calculated center.

        Returns
        -------
        numpy.array of shape (2, 3)
            Two non unit vectors
        """
        points = self.inlier_points
        # center = np.median(points, axis=0)
        center = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
        delta = points - center
        cov_matrix = np.cov(delta, rowvar=False)
        eigval, eigvec = np.linalg.eig(cov_matrix)
        v1, v2, _ = eigvec

        projs = delta.dot(eigvec)
        # V1 = (max(projs[:, 0]) - min(projs[:, 0])) * v1
        # V2 = (max(projs[:, 1]) - min(projs[:, 1])) * v2
        V1 = 2 * max(abs(projs[:, 0])) * v1
        V2 = 2 * max(abs(projs[:, 1])) * v2
        
        if return_center:
            return np.array([V1, V2]), center
        return np.array([V1, V2])
    
    def get_rectangular_plane(self, vectors, center=None):
        """ Gives rectangular plane defined two vectors and its center.
        
        Vectors v1 and v2 should not be unit, and instead have lengths equivalent
        to widths of rectangle.
        
        Parameters
        ----------
        vectors : arraylike of shape (2, 3)
            The two orthogonal unit vectors defining the rectangle plane.
        center : arraylike of length 3, optional
            Center of rectangle. If not given, either inliers or centroid are 
            used.

        Returns
        -------
        PlaneBounded
            Rectangular plane
        """
        if center is None:
            if self.has_inliers:
                points = self.inlier_points
                # center = np.median(self.inlier_points, axis=0)
                center = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
            else:
                center = self.centroid
                
        V1, V2 = vectors
        vertices = center + _get_rectangular_vertices(V1, V2)
        return self.get_bounded_plane(vertices)
    
    def get_square_plane(self, length, center=None):
        """ Gives square plane defined by four points.

        Parameters
        ----------
        length : float, optional
            Length of the sides of the square
        center : arraylike of length 3, optional
            Center of rectangle. If not given, either inliers or centroid are 
            used.

        Returns
        -------
        PlaneBounded
            Square plane
        """
        if center is None:
            if self.has_inliers:
                center = np.median(self.inlier_points, axis=0)
            else:
                center = self.centroid
        
        def normalized(x): return x / np.linalg.norm(x)

        if np.isclose(self.normal[1], 1, atol=1e-7):
            v1 = normalized(np.cross(self.normal, [0, 0, 1]))
            v2 = normalized(np.cross(v1, self.normal))
        else:
            v1 = normalized(np.cross([0, 1, 0], self.normal))
            v2 = normalized(np.cross(self.normal, v1))

        vertices = center + _get_rectangular_vertices(v1, v2)

        return self.get_bounded_plane(vertices)

    def get_square_mesh(self, length, center=None):
        """ Gives a square mesh that fits the plane model.   

        Parameters
        ----------
        length : float
            Length of the sides of the square
        center : arraylike of length 3, optional
            Center of rectangle. If not given, either inliers or centroid are 
            used.

        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        return self.get_square_plane(length, center).get_mesh()
    
    def get_rectangular_mesh(self, v1, v2, center=None):
        """ Gives a rectangular mesh that fits the plane model.

        Parameters
        ----------
        v1 : arraylike of length 3
            First vector defining one of the directions, must be orthogonal
            to v2.
        v2 : arraylike of length 3
            Second vector defining one of the directions, must be orthogonal
            to v1.
        center : arraylike of length 3, optional
            Center of rectangle. If not given, either inliers or centroid are 
            used.

        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        return self.get_rectangular_plane(v1, v2, center).get_mesh()
    
    @classmethod
    def create_circle(cls, center, normal, radius, resolution=30):
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
        Plane
            Circular plane.
        """

        def normalize(x): return x / np.linalg.norm(x)

        center = np.array(center)
        normal = normalize(np.array(normal))

        random_axis = normalize(np.random.random(3))
        ex = normalize(np.cross(random_axis, normal))
        ey = normalize(np.cross(normal, ex))

        theta = np.linspace(-np.pi, np.pi, resolution + 1)[None].T
        points = center + (np.cos(theta) * ex + np.sin(theta) * ey) * radius

        plane_unbounded = Plane.from_normal_point(normal, center)
        plane_unbounded.set_inliers(points)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cls(plane_unbounded)

    @classmethod
    def create_ellipse(cls, center, vx, vy, resolution=30):
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
        Plane
            Elliptical plane.
        """
        center = np.array(center)
        vx = np.array(vx)
        vy = np.array(vy)

        if np.dot(vx, vy) > 1e-5:
            raise ValueError('Axes must be orthogonal.')

        normal = np.cross(vx, vy)
        normal = normal / np.linalg.norm(normal)

        theta = np.linspace(-np.pi, np.pi, resolution+1)[None].T
        points = center + np.cos(theta) * vx + np.sin(theta) * vy

        plane_unbounded = Plane.from_normal_point(normal, center)
        plane_unbounded.set_inliers(points)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cls(plane_unbounded)
    
    @classmethod
    def create_box(cls, center=[0, 0, 0], dimensions=[1, 1, 1]):
        """ Gives list of planes that create, together, a closed box.

        Parameters
        ----------
        center : array
            Center point of box
        dimensions : array
            Dimensions of box

        Returns
        -------
        box : list
            List of PlaneBounded instances.
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

            plane_unbounded = Plane.from_normal_point(v3, center + sign * v3)
            plane_unbounded.set_inliers(points)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                planes.append(cls(plane_unbounded))

        return planes