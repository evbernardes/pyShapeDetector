#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:15:09 2024

@author: ebernardes
"""
import warnings
from itertools import permutations, product, combinations
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
# from scipy.spatial.transform import Rotation
from open3d.geometry import TriangleMesh, AxisAlignedBoundingBox
from open3d.utility import Vector3iVector, Vector3dVector

from pyShapeDetector.utility import (
    get_triangle_surface_areas, 
    fuse_vertices_triangles, 
    planes_ressample_and_triangulate)
from .primitivebase import Primitive
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
    mesh
    has_inliers
    inlier_points
    inlier_points_flattened
    inlier_normals
    inlier_colors
    inlier_PointCloud
    inlier_PointCloud_flattened
    metricsd
    axis_spherical
    axis_cylindrical
    bbox
    bbox_bounds
    inlier_bbox
    inlier_bbox_bounds

    normal
    dist
    centroid
    holes

    is_convex
    plane
    bounds
    bounds_indices
    bounds_projections
    bound_lines
    vertices
    triangles
    bounds_or_vertices

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
    closest_bounds
    get_bounded_plane
    get_projections
    get_points_from_projections
    get_mesh_alphashape
    get_square_plane
    get_square_mesh

    contains_projections
    bound_lines_meshes
    set_bounds
    set_vertices_triangles
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
    _vertices = np.array([])
    _triangles = np.array([])
    _convex = True
    _rotatable = []  # no automatic rotation

    @property
    def model(self):
        """ Return model. """
        return self._plane.model

    @property
    def surface_area(self):
        """ Surface area of bounded plane. """
        
        def shoelace(projections):
            # Reference:
            # https://en.wikipedia.org/wiki/Shoelace_formula
            i = np.arange(len(projections))
            x, y = projections.T
            return np.abs(np.sum(x[i-1] * y[i] - x[i] * y[i-1]) * 0.5)
        
        def test(projections):
            return np.linalg.norm(np.cross(*np.diff(projections, axis=0))) * 0.5
        
        if self.is_convex:
            surface_area = shoelace(self.bounds_projections)
            for hole in self.holes:
                surface_area -= shoelace(hole.bounds_projections)
        else:
            triangle_projections = self.get_projections(self.vertices)[self.triangles]
            diff = np.diff(triangle_projections, axis=1)
            areas = abs(np.cross(diff[:,0,:], diff[:,1,:])) * 0.5
            surface_area = sum(areas)
            
        return surface_area
    
    @property
    def canonical(self):
        """ Return canonical form for testing. """
        shape = self.copy()
        if np.sign(self.dist) < 0:
            shape.plane._model = -self.plane._model
        return shape
    
    @property
    def is_convex(self):
        return self._convex
        
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

    @property
    def vertices(self):
        return self._vertices

    @property
    def triangles(self):
        return self._triangles
    
    @property
    def bounds_or_vertices(self):
        if self.is_convex:
            points = self.bounds
        else:
            points = self.vertices
            
        if len(points) == 0:
            raise RuntimeError("PlaneBounded instance has no bounds or vertices.")
            
        return points
    
    @property
    def bounds_or_vertices_or_inliers(self):
        try:
            points = self.bounds_or_vertices
        except RuntimeError:
            points = self.inlier_points
            
        if len(points) == 0:
            raise RuntimeError("PlaneBounded instance has no bounds, vertices or inliers.")
            
        return points

    def __init__(self, planemodel, bounds=None, vertices=None, triangles=None,
                 decimals=None):
        """
        Parameters
        ----------
        planemodel : Plane or list of 4 values
            Shape defining plane
        bounds : array_like, shape (N, 3), optional
            Points defining bounds. 
        vertices : array_like, shape (N, 3)
            Vertices of plane triangulation.
        triangles : array_like, shape (N, 3)
            Vertices of plane triangulation.
        decimals : int, optional
            Number of decimal places to round to (default: 0). If
            decimals is negative, it specifies the number of positions to
            the left of the decimal point. Default: None.

        Raises
        ------
        ValueError
            If number of parameters is incompatible with the model of the 
        """
        if bounds is not None:
            if vertices is None and triangles is None:
                self._convex = True
            else:
                raise ValueError("If 'bounds' are given, 'vertices' and "
                                 "'triangles' should not be given.")
        elif vertices is None and triangles is None:
            warnings.warn('No input bounds or vertices/triagles, returning square plane')
        elif vertices is None or triangles is None:
            raise ValueError("Either 'vertices' and 'triangles' should be given, or one of them.")
        else:
            self._convex = False

        if isinstance(planemodel, PlaneBounded):
            self._plane = planemodel._plane
        elif isinstance(planemodel, Plane):
            self._plane = planemodel
        else:
            self._plane = Plane(planemodel, decimals=decimals)

        if self._convex:
            if bounds is None:
                # warnings.warn('No input bounds, returning square plane')
                self = self._plane.get_square_plane(1)

            else:
                self.set_bounds(bounds, flatten=True)
                if self._bounds is None:
                    self = None
        else:
            self.set_vertices_triangles(vertices, triangles, flatten=True)

        self._decimals = decimals
        self._holes = []

    @classmethod
    def random(cls, scale=1, decimals=16):
        """ Generates a random shape.

        Parameters
        ----------
        scale : float, optional
            scaling factor for random model values.
        decimals : int, optional
            Number of decimal places to round to (default: 0). If
            decimals is negative, it specifies the number of positions to
            the left of the decimal point.

        Returns
        -------
        Primitive
            Random shape.
        """
        model = np.random.random(4) * scale
        plane = Plane(model, decimals=decimals)
        length = np.random.random() * scale
        return plane.get_square_plane(np.round(length, decimals))

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

    def get_mesh(self, **options):
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
        if not self.is_convex:
            points = self.vertices
            triangles = self.triangles
        else:
            if len(self._fusion_intersections) == 0:
                points = self.bounds
                projections = self.bounds_projections
                # idx_intersections_sorted = []
            else:
                points = np.vstack([self.bounds, self._fusion_intersections])
                projections = self.get_projections(points)
    
                angles = projections - projections.mean(axis=0)
                angles = np.arctan2(*angles.T) + np.pi
                idx = np.argsort(angles)
    
                points = points[idx]
                projections = projections[idx]
    
                # idx_intersections = list(
                    # range(len(points) - len(self._fusion_intersections), len(points)))
                # idx_intersections_sorted = [
                    # np.where(i == idx)[0][0] for i in idx_intersections]
    
            holes = self._holes
            has_holes = len(holes) != 0
            if has_holes:
                points_holes = [self.flatten_points(hole.bounds) for hole in holes]
                points = np.vstack([points]+points_holes)
                projections = self.get_projections(points)
            
            triangles = Delaunay(projections).simplices
    
            for hole in self._holes:
                inside_hole = np.array(
                    [hole.contains_projections(p).all() for p in points[triangles]])
                triangles = triangles[~inside_hole]
    
            areas = get_triangle_surface_areas(points, triangles)
            triangles = triangles[areas > 0]
    
            # needed to make plane visible from both sides
            # triangles = np.vstack([triangles, triangles[:, ::-1]])

        mesh = TriangleMesh()
        mesh.vertices = Vector3dVector(points)
        mesh.triangles = Vector3iVector(triangles)

        return mesh

    def __copy__(self):
        """ Method for compatibility with copy module """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shape = Plane.__copy__(self)
        
        # Copying attributes particular to bounded planes
        shape._convex = self._convex
        shape._vertices = self._vertices.copy()
        shape._triangles = self._triangles.copy()
        shape._bounds_indices = self._bounds_indices.copy()
        shape._bounds = self._bounds.copy()
        shape._bounds_projections = self._bounds_projections.copy()
        shape._fusion_intersections = self._fusion_intersections.copy()
        return shape

    def translate(self, translation):
        """ Translate the shape.

        Parameters
        ----------
        translation : 1 x 3 array
            Translation vector.
        """
        Primitive.translate(self, translation)
        Plane.translate(self._plane, translation)
        
        for hole in self.holes:
            hole._translate_points(translation)
            
        if self.is_convex:
            self._bounds = self._bounds + translation
        else:
            self._vertices = self._vertices + translation

    def rotate(self, rotation):
        """ Rotate the shape.

        Parameters
        ----------
        rotation : 3 x 3 rotation matrix or scipy.spatial.transform.Rotation
            Rotation matrix.
        """
        rotation = self._parse_rotation(rotation)
        
        Primitive.rotate(self, rotation)
        Plane.rotate(self._plane, rotation)

        if self.is_convex:
            self._bounds = rotation.apply(self._bounds)
        else:
            self._vertices = rotation.apply(self._vertices)        
                
    def get_inliers_axis_aligned_bounding_box(self, slack=0, num_sample=15):
        """ If the shape includes inlier points, returns the minimum and 
        maximum bounds of their bounding box.
        
        If no inlier points, use bounds or vertices.
        
        Parameters
        ----------
        slack : float, optional
            Expand bounding box in all directions, useful for testing purposes.
            Default: 0.
        num_sample : int, optional
            If no inliers, bounds or vertices found, sample mesh instead.
            Default: 15.
            
        Returns
        -------
        tuple of two 3 x 1 arrays
            Minimum and maximum bounds of inlier points bounding box.
        """
        
        if len(self.bounds) > 0:
            points = self.bounds
        elif len(self.vertices) > 0:
            points = self.vertices
        else:
            return Primitive.inliers_bounding_box(slack=0, num_sample=15)
        
        slack = abs(slack)
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        return AxisAlignedBoundingBox(min_bound - slack, max_bound + slack)
    
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
        points = self.bounds_or_vertices
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        return AxisAlignedBoundingBox(min_bound - slack, max_bound + slack)
                
    @staticmethod
    def fuse(shapes, detector=None, ignore_extra_data=False, line_intersection_eps=1e-3,
             **extra_options):
        """ Find weigthed average of shapes, where the weight is the fitness
        metric.
        
        If a detector is given, use it to compute the metrics of the resulting
        average shapes.
        
        Parameters
        ----------
        shapes : list
            Grouped shapes. All shapes must be of the same type.
        detector : instance of some Detector, optional
            Used to recompute metrics. Default: None.
        ignore_extra_data : boolean, optional
            If True, ignore everything and only fuse model. Default: False.
        line_intersection_eps : float, optional
            Distance for detection of intersection between planes. Default: 0.001.
        force_concave : boolean, optional.
            If True, the fused plane will be concave regardless of inputs.
            Default: True.
        ressample_density : float, optional
            Default: 1.5
        ressample_radius_ratio : float, optional
            Default: 1.2
            
        Returns
        -------
        PlaneBounded
            Averaged PlaneBounded instance.    
        """
        force_concave = extra_options.get('force_concave', True)
        ressample_density = extra_options.get('ressample_density', 1.5)
        ressample_radius_ratio = extra_options.get('ressample_radius_ratio', 1.2)
        
        if len(shapes) == 1:
            return shapes[0]
        elif isinstance(shapes, Primitive):
            return shapes
        
        shape = Plane.fuse(shapes, detector, ignore_extra_data)
        
        is_convex = np.array([shape.is_convex for shape in shapes])
        all_convex = is_convex.all()
        if not all_convex and is_convex.any():
            force_concave = True
            # raise ValueError("If 'force_concave' is False, PlaneBounded "
            #                  "instances should either all be convex or "
            #                  "all non convex.")
        
        if not ignore_extra_data:
            if force_concave:
                vertices, triangles = planes_ressample_and_triangulate(
                    shapes, ressample_density, ressample_radius_ratio, double_triangles=True)
                shape.set_vertices_triangles(vertices, triangles)
                
            elif not all_convex:
                vertices = [shape.vertices for shape in shapes]
                triangles = [shape.triangles for shape in shapes]
                vertices, triangles = fuse_vertices_triangles(vertices, triangles)                                       
                shape.set_vertices_triangles(vertices, triangles)
                
            else:
                bounds = np.vstack([shape.bounds for shape in shapes])
                shape.set_bounds(bounds)
            
                intersections = []
                for plane1, plane2 in combinations(shapes, 2):
                    points = plane1.intersection_bounds(plane2, True, eps=line_intersection_eps)
                    if len(points) > 0:
                        intersections.append(points)
                
                # temporary hack, saving intersections for mesh generation
                if len(intersections) > 0:
                    shape._fusion_intersections = np.vstack(intersections)
        
        return shape

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
    
    def bound_lines_meshes(self, radius=0.001, color=(0, 0, 0)):
        lines = self.bound_lines
        meshes = [line.get_mesh(radius=radius) for line in lines]
        [mesh.paint_uniform_color(color) for mesh in meshes]
        return meshes

    def set_bounds(self, bounds, flatten=True):
        """ Flatten points according to plane model, get projections of 
        flattened points in the model and compute its boundary using either 
        the convex hull or alpha shapes.

        Parameters
        ----------
        plane : Plane
            Plane model
        bounds : array_like, shape (N, 3)
            Points corresponding to the fitted shape.
        flatten : bool, optional
            If False, does not flatten points

        """
        # method = method.lower()
        # if method == 'convex':
        #     pass
        # elif method == 'alpha':
        #     raise NotImplementedError("Alpha shapes not implemented yet.")
        # else:
        #     raise ValueError(
        #         f"method can be 'convex' or 'alpha', got {method}.")
        self._mesh = None
        bounds = np.asarray(bounds)
        
        if bounds.shape[1] != 3 :
            raise ValueError("Invalid shape of 'bounds' array.")
        
        if flatten:
            bounds = self.flatten_points(bounds)
        if np.any(np.isnan(bounds)):
            raise ValueError('NaN found in points')
            
        self._vertices = np.array([])
        self._triangles = np.array([])
        self._convex = True

        projections = self.get_projections(bounds)

        # if method == 'convex':
        chull = ConvexHull(projections)
        self._bounds_indices = chull.vertices
        self._bounds = bounds[chull.vertices]
        self._bounds_projections = projections[chull.vertices]
        
    def set_vertices_triangles(self, vertices, triangles, flatten=True):
        """ Flatten points according to plane model, get projections of 
        flattened points in the model and set desired vertices and triangles.

        Parameters
        ----------
        plane : Plane
            Plane model
        vertices : array_like, shape (N, 3)
            Vertices of plane triangulation.
        triangles : array_like, shape (N, 3)
            Vertices of plane triangulation.
        flatten : bool, optional
            If False, does not flatten points

        """
        # method = method.lower()
        # if method == 'convex':
        #     pass
        # elif method == 'alpha':
        #     raise NotImplementedError("Alpha shapes not implemented yet.")
        # else:
        #     raise ValueError(
        #         f"method can be 'convex' or 'alpha', got {method}.")

        # if np.
        self._mesh = None
        vertices = np.asarray(vertices)
        triangles = np.asarray(triangles)
        
        if vertices.shape[1] != 3 or triangles.shape[1] != 3:
            raise ValueError("Invalid shape of 'vertices' and/or 'triangles' "
                             "array.")
        
        if not all([x.is_integer() for x in triangles.flatten()]):
            raise ValueError("All elements of 'triangles' must be integers.")
        
        if (triangles >= len(vertices)).any() or (triangles < 0).any():
            raise ValueError("Each element in 'triangles' should be an integer"
                             " between 0 and len(vertices) - 1.")

        if flatten:
            vertices = self.flatten_points(vertices)
        if np.any(np.isnan(vertices)):
            raise ValueError('NaN found in points')
            
        self._bounds = np.array([])
        self._bounds_indices = np.array([])
        self._vertices = vertices
        self._triangles = triangles
        self._convex = False

    # def add_bound_points(self, new_bound_points, flatten=True, method='convex',
                         # alpha=None):
    # method : string, optional
    #     "convex" for convex hull, or "alpha" for alpha shapes.
    #     Default: "convex"
    # alpha : float, optional
    #     Alpha parameter for alpha shapes algorithm. If equal to None,
    #     calculates the optimal alpha. Default: None
    def add_bound_points(self, new_bound_points, flatten=True):
        """ Add points to current bounds.

        Parameters
        ----------
        new_bound_points : N x 3 np.array
            New points to be added.
        flatten : bool, optional
            If False, does not flatten points

        """
        if not self.is_convex:
            warnings.warn("Cannot add bounds to concave plane.")
        else:
            if flatten:
                new_bound_points = self.flatten_points(new_bound_points)
            bounds = np.vstack([self.bounds, new_bound_points])
            self.set_bounds(bounds, flatten=False)
            # self.set_bounds(points, flatten, method, alpha)

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

        self._metrics = self._metrics.copy()
        self._color = self._color.copy()
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
        
        
