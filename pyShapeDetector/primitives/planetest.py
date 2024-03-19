#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:42:59 2023

@author: ebernardes
"""
from warnings import warn
from itertools import permutations, product, combinations
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation
from open3d.geometry import TriangleMesh, AxisAlignedBoundingBox
from open3d.utility import Vector3iVector, Vector3dVector

from pyShapeDetector.utility import (
    get_rotation_from_axis,
    get_triangle_surface_areas, 
    fuse_vertices_triangles, 
    planes_ressample_and_triangulate)

from .primitivebase import Primitive
from alphashape import alphashape
# from .line import Line

class PlaneTest(Primitive):
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

    is_unbounded
    normal
    dist
    centroid
    holes
    is_hole

    is_convex
    plane
    bounds
    bounds_indices
    bounds_projections
    bound_lines
    vertices
    triangles
    bounds_or_vertices
    bounds_or_vertices_or_inliers

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
    set_vertices_triangles
    add_bound_points
    create_circle
    create_ellipse
    create_box
    intersection_bounds

    """

    _fit_n_min = 3
    _model_args_n = 4
    _name = 'plane test'
    _holes = []
    # _rotatable = [0, 1, 2]
    _translatable = []
    _fusion_intersections = np.array([])
    _color = np.array([0, 0, 1])
    _is_hole = False

    #from Plane Bounded:
    _bounds_indices = np.array([])
    _bounds = np.array([])
    _bounds_projections = np.array([])
    _vertices = np.array([])
    _triangles = np.array([])
    _convex = True
    _rotatable = []  # no automatic rotation

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
        """ Surface area of plane. """

        if not self.has_inliers and (len(self.bounds) == len(self.vertices) == 0):
            # For unbounded plane, returns NaN and gives warning 
            warn('For unbounded planes, the surface area is undefined')
            return float('nan')
        
        def shoelace(projections):
            # Reference:
            # https://en.wikipedia.org/wiki/Shoelace_formula
            i = np.arange(len(projections))
            x, y = projections.T
            return np.abs(np.sum(x[i-1] * y[i] - x[i] * y[i-1]) * 0.5)
        
        # def test(projections):
            # return np.linalg.norm(np.cross(*np.diff(projections, axis=0))) * 0.5
        
        if self.is_convex:
            surface_area = shoelace(self.bounds_projections)
            for hole in self.holes:
                surface_area -= shoelace(hole.bounds_projections)
        else:
            triangle_projections = self.get_projections(self.vertices)[self.triangles]
            # areas = [shoelace(points) for points in triangle_projections]
            # areas = [test(points) for points in triangle_projections]
            diff = np.diff(triangle_projections, axis=1)
            areas = abs(np.cross(diff[:,0,:], diff[:,1,:])) * 0.5
            surface_area = sum(areas)
            
        return surface_area

    # @property
    # def surface_area(self):
    #     """ For unbounded plane, returns NaN and gives warning """
    #     warn('For unbounded planes, the surface area is undefined')
    #     return float('nan')

    @property
    def volume(self):
        """ Volume of plane, which is zero. """
        return 0

    @property
    def canonical(self):
        """ Return canonical form for testing. """
        shape = self.copy()
        if np.sign(self.dist) < 0:
            self._model = -self._model
        return shape
    
    @property
    def is_unbounded(self):
        if self.is_convex and len(self.bounds) > 0:
            return True
        
        if self.is_convex is False and len(self.vertices) > 0:
            return True
        
        if self.has_inliers:
            warn('Supposing plane is bounded because it has inliers')
            return True
        
        return False

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
    
    # attributes from Plane Bounded:
    @property
    def is_convex(self):
        return self._convex
        
    # @property
    # def plane(self):
    #     """ Return internal plane without bounds. """
    #     return self._plane

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
            raise RuntimeError("Plane instance has no bounds or vertices.")
            
        return points
    
    @property
    def bounds_or_vertices_or_inliers(self):
        if self.is_convex:
            points = self.bounds
            
        elif self.is_convex is False:
            points = self.vertices

        else:
            points = self.inlier_points
            
        if len(points) == 0:
            raise RuntimeError("Plane instance has no bounds, vertices or inliers.")
            
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
            pass
            # warn('No input bounds or vertices/triagles, returning square plane')

        elif vertices is None or triangles is None:
            raise ValueError("Either 'vertices' and 'triangles' should be given, or one of them.")
        else:
            self._convex = False

        if isinstance(planemodel, PlaneTest):
            planemodel = planemodel._model

        # self._plane = PlaneTest(planemodel, decimals=decimals)
        model = np.array(planemodel)
        norm = np.linalg.norm(model[:3])
        model = Primitive._parse_model_decimals(model / norm, decimals)
        Primitive.__init__(self, model)
        self._decimals = decimals
        self._holes = []

        if self._convex:
            self.set_bounds(bounds, flatten=True)
        elif self._convex is None:
            pass
        else:
            self.set_vertices_triangles(vertices, triangles, flatten=True)

    # def __init__(self, model, decimals=None):
    #     """
    #     Parameters
    #     ----------
    #     model : list or tuple
    #         Parameters defining the shape model 
    #     decimals : int, optional
    #         Number of decimal places to round to (default: 0). If
    #         decimals is negative, it specifies the number of positions to
    #         the left of the decimal point. Default: None.

    #     Raises
    #     ------
    #     ValueError
    #         If number of parameters is incompatible with the model of the 
    #         primitive.
    #     """
    #     model = np.array(model)
    #     norm = np.linalg.norm(model[:3])
    #     model = Primitive._parse_model_decimals(model / norm, decimals)
    #     Primitive.__init__(self, model)
    #     self._decimals = decimals
    #     self._holes = []

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

        plane = PlaneTest.from_normal_point(abc / norm, centroid)
        plane.set_bounds(points)
        return plane

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
        if slack < 0:
            raise ValueError("Slack must be non-negative.")
        
        if self.has_inliers:
            return Primitive.inliers_bounding_box(slack=slack, num_sample=num_sample)
        
        warn("Inliers not found for plane, using bounds/vertices.")
        points = self.bounds_or_vertices

        # if len(self.bounds) > 0:
        #     points = self.bounds
        #     warn("Inliers not found for convex plane, using bounds.")
        # elif len(self.vertices) > 0:
        #     points = self.vertices
        #     warn("Inliers not found for concave plane, using vertices.")
        # else:
        #     return RuntimeError('No inliers, bounds or vertices found.')
        
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

        
        try:
            points = self.bounds_or_vertices_or_inliers
            min_bound = np.min(points, axis=0)
            max_bound = np.max(points, axis=0)
            return AxisAlignedBoundingBox(min_bound - slack, max_bound + slack)
        
        except RuntimeError:
            warn("Unbounded planes have infinite axis aligned bounding boxes.")
            
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
            expand = np.array([slack if n == idx else np.inf for n in range(3)])
            return AxisAlignedBoundingBox(centroid - expand, centroid + expand)

    def get_mesh(self, **options):
        """ Flatten inliers points and creates a simplified mesh of the plane. If the
        shape has pre-defined inlier points, use them to find borders.
        Otherwise, return square mesh.

        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        if self.is_convex is False:
            points = self.vertices
            triangles = self.triangles

        elif self.is_convex is None:
            warn('No inlier bounds or vertices, trying convex plane from '
                 'inliers...')
            
            if not self.has_inliers:
                warn('No inlier points, returning square plane...')
                return self.get_square_mesh()

            bounded_plane = PlaneTest(self.model, 
                                      bounds=self.inlier_points_flattened)
            
            bounded_plane._holes = self._holes
            mesh = bounded_plane.get_mesh()

            return mesh

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
        shape = Primitive.__copy__(self)
        shape._model = self.model.copy()
        shape._fusion_intersections = self._fusion_intersections.copy()
        shape._is_hole = self._is_hole
        if not self.is_hole:
            holes = [h.copy() for h in self._holes]
            shape._holes = holes

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
        centroid = self.centroid + translation
        self._model = PlaneTest.from_normal_point(
            self.normal, centroid).model
            
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
        rotation = Primitive._parse_rotation(rotation)
        
        # for everything else
        Primitive.rotate(self, rotation)
        
        # only for model
        normal = rotation.apply(self.normal)
        centroid = rotation.apply(self.centroid)
        self._model = PlaneTest.from_normal_point(normal, centroid).model
        
        if not self.is_hole and len(self.holes) > 0:
            for hole in self.holes:
                hole.rotate(rotation)

        if self.is_convex:
            self._bounds = rotation.apply(self._bounds)
        else:
            self._vertices = rotation.apply(self._vertices) 

    def bound_lines_meshes(self, radius=0.001, color=(0, 0, 0)):
        lines = self.bound_lines
        meshes = [line.get_mesh(radius=radius) for line in lines]
        [mesh.paint_uniform_color(color) for mesh in meshes]
        return meshes
    
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
        Plane
            Averaged Plane instance.    
        """
        force_concave = extra_options.get('force_concave', True)
        ressample_density = extra_options.get('ressample_density', 1.5)
        ressample_radius_ratio = extra_options.get('ressample_radius_ratio', 1.2)
        
        if len(shapes) == 1:
            return shapes[0]
        elif isinstance(shapes, Primitive):
            return shapes
        
        shape = PlaneTest.fuse(shapes, detector, ignore_extra_data)
        
        is_convex = np.array([shape.is_convex for shape in shapes])
        all_convex = is_convex.all()
        if not all_convex and is_convex.any():
            force_concave = True
        
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
        holes : Plane or list of Plane instances
            Planes defining holes.
        remove_points : boolean, optional
            If True, removes points of plane inside holes and points of holes
            outside of plane. Default: True.
        """
        if remove_points and self.is_unbounded:
            warn('Option remove_points only works if plane is bounded.')
            remove_points = False

        if not isinstance(holes, list):
            holes = [holes]

        fixed_holes = []
        for hole in holes:
            if not isinstance(hole, PlaneTest) and not self.is_convex:
                raise ValueError("Holes must be convex instances of Plane, got"
                                 f" {hole}.")

            cos_theta = np.dot(hole.normal, self.normal)
            if abs(cos_theta) < 1 - 1e-5:
                raise ValueError("Plane and hole must be aligned.")

            model = hole.model
            if cos_theta < 0:
                model = -model

            if remove_points:
                inside1 = self.contains_projections(hole.bounds)
                if sum(inside1) < 1:
                    print('shape does not contain hole')
                    continue
                elif sum(~inside1) > 0:
                    intersections = []
                    for l1, l2 in product(hole.bound_lines, self.bound_lines):
                        if (point := l1.point_from_intersection(l2)) is not None:
                            intersections.append(point)
                    bounds = np.vstack([hole.bounds[inside1]]+intersections)
                else:
                    bounds = hole.bounds
                # if sum(inside1) < 1:
                    # print('shape does not contain hole')
                    # continue

                inside2 = hole.contains_projections(self.bounds)
                # print(inside2)
                # hole = PlaneTest(model, hole.bounds[inside1])
                hole = PlaneTest(model, bounds=bounds)
                self._bounds = self._bounds[~inside2]
                self._bounds_projections = self._bounds_projections[~inside2]
            else:
                hole = PlaneTest(model, bounds=hole.bounds)
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
        other_plane : instance of Plane
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
        if not isinstance(other_plane, PlaneTest) or other_plane.is_unbounded:
            raise ValueError("Only intersection with other bounded instances of "
                             "Plane is implemented.")
        from .line import Line

        if not separated:
            return Line.from_plane_intersection(
                self, other_plane, intersect_parallel=intersect_parallel, eps_angle=eps_angle, eps_distance=eps_distance)

        line = Line.from_plane_intersection(
            self, other_plane, intersect_parallel=intersect_parallel, eps_angle=eps_angle, eps_distance=eps_distance, fit_bounds=False)

        line1 = line.get_line_fitted_to_projections(self.bounds)
        line2 = line.get_line_fitted_to_projections(other_plane.bounds)

        return line1, line2

    def closest_bounds(self, other_plane, n=1):
        """ Returns n pairs of closest bound points with a second plane.

        Parameters
        ----------            
        other_plane : Plane
            Another plane.
        n : int, optional
            Number of pairs. Default=1.

        Returns
        -------
        closest_points : np.array
            Pairs of points.
        distances : np.array
            Distances for each pair.
        """
        if not isinstance(other_plane, PlaneTest) or other_plane.is_unbounded:
            raise ValueError("Only intersection with other bounded instances of "
                             "Plane is implemented.")

        from pyShapeDetector.utility import find_closest_points

        closest_points, distances = find_closest_points(
            self.bounds, other_plane.bounds, n)

        return closest_points, distances

    def get_bounded_plane(self, points=None):
        """ Gives bounded version of plane, using input points to define 
        border.

        Parameters
        ----------
        points : array_like, shape (N, 3)
            Parameters defining the shape model

        Returns
        -------
        Plane
            Bounded version of plane.
        """
        if points is None:
            points = self.inlier_points
        if len(points) == 0:
            raise ValueError('if no points are given, shape must have inlier points')
        bounds = self.flatten_points(points)
        bounded_plane = PlaneTest(self, bounds=bounds)
        bounded_plane._holes = self._holes
        return bounded_plane

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

    def get_square_plane(self, length=1):
        """ Gives square plane defined by four points.

        Parameters
        ----------
        length : float, optional
            Length of the sides of the square

        Returns
        -------
        Plane
            Square plane
        """
        # center = self.centroid
        # center = np.mean(np.asarray(pcd.points), axis=0)
        # bb = pcd.get_axis_aligned_bounding_box()
        # half_length = max(bb.max_bound - bb.min_bound) / 2

        def normalized(x): return x / np.linalg.norm(x)

        if np.isclose(self.normal[1], 1, atol=1e-7):
            v1 = normalized(np.cross(self.normal, [0, 0, 1]))
            v2 = normalized(np.cross(v1, self.normal))
        else:
            v1 = normalized(np.cross([0, 1, 0], self.normal))
            v2 = normalized(np.cross(self.normal, v1))

        centroid = self.centroid
        vertices = np.vstack([
            centroid + (+ v1 + v2) * length / 2,
            centroid + (+ v1 - v2) * length / 2,
            centroid + (- v1 + v2) * length / 2,
            centroid + (- v1 - v2) * length / 2])

        plane = PlaneTest(self.model, bounds=vertices, decimals=self._decimals)
        plane._holes = self._holes
        return plane

        # triangles = Vector3iVector(np.array([
        #     [0, 1, 2],
        #     [2, 1, 0],
        #     [0, 2, 3],
        #     [3, 2, 0]]))
        # vertices = Vector3dVector(vertices)

        # return TriangleMesh(vertices, triangles)

    def get_square_mesh(self, length=1):
        """ Gives a square mesh that fits the plane model.   

        Parameters
        ----------
        length : float, optional
            Length of the sides of the square

        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        return self.get_square_plane(length).get_mesh()
    
    ### Plane Bounded methods
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
            warn("Cannot add bounds to concave plane.")
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
        Plane
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

        plane = PlaneTest.from_normal_point(normal, center)
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
        Plane
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

        plane = PlaneTest.from_normal_point(normal, center)
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

            plane = PlaneTest.from_normal_point(v3, center + sign * v3)
            planes.append(plane.get_bounded_plane(points))

        return planes

    def intersection_bounds(self, other_plane, within_segment=True, eps=1e-3):
        """ Calculates intersection point between bounding lines.

        Parameters
        ----------
        other_plane : instance of PlaneTest
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
        if not isinstance(other_plane, PlaneTest) or other_plane.is_unbounded:
            raise ValueError("Only intersection with other bounded instances of "
                             "Plane is implemented.")

        if len(self.bounds_or_vertices) == 0 or len(other_plane.bounds_or_vertices) == 0:
            raise ValueError("Both planes must have bounds or vertices.")

        lines = self.bound_lines
        lines_other = other_plane.bound_lines

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

