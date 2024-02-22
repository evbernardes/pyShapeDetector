#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:42:59 2023

@author: ebernardes
"""
import warnings
from abc import ABC, abstractmethod
import numpy as np
from open3d.geometry import PointCloud, AxisAlignedBoundingBox
from open3d.utility import Vector3dVector
from scipy.spatial.transform import Rotation
from pyShapeDetector.utility import clean_crop, get_rotation_from_axis
    
class Primitive(ABC):
    """
    Base class used to represent a geometrical primitive.
    
    To define a primitive, inherit from this class and define at least the 
    following internal attributes:
        `_fit_n_min`
        `_model_args_n`
        `_name`
    And the following methods:
        `get_distances`
        `get_normals`
        
    The method `get_mesh` can also optionally be implemented to return a : 3 x 3 array
    TriangleMesh instance.
    
    The properties `surface_area` and `volume` can also be implemented.
    
    When multiple set of parameters can define the same surface, it might be
    useful to implement the property `canonical` to return the canonical form 
    (useful for testing).
    
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
    fuse
    """
    _inlier_points = np.asarray([])
    _inlier_normals = np.asarray([])
    _inlier_colors = np.asarray([])
    # _inlier_indices = np.asarray([])
    _metrics = {}
    
    @property
    def fit_n_min(self):
        """ Minimum number of points necessary to fit a model."""
        return self._fit_n_min
    
    @property
    def model_args_n(self):
        """ Number of parameters in the model. """
        return self._model_args_n
    
    @property
    def name(self):
        """ Name of primitive. """
        return self._name
    
    @property
    def model(self):
        return self._model
    
    @property
    def equation(self):
        """ Equation that defines the primitive."""
        raise NotImplementedError(f'Equation not implemented for {self.name} '
                                  'primitives.')
        
    @property
    def surface_area(self):
        """ Surface area of primitive. """
        
        raise NotImplementedError('Surface area not implemented for '
                                  f'{self.name} primitives.')
    
    @property
    def volume(self):
        """ Volume of primitive. """
        raise NotImplementedError(f'Volume not implemented for {self.name} '
                                  'primitives.')
        
    @property
    def canonical(self):
        """ Return canonical form for testing."""
        return self
    
    def _get_axis_or_vector_or_normal(self):
        if hasattr(self, 'axis'):
            return self.axis
        if hasattr(self, 'vector'):
            return self.vector
        if hasattr(self, 'normal'):
            return self.normal
        raise RuntimeError(f'Primitives of type {self.name} do not have an '
                           'axis.')
        
    @property
    def color(self):
        seed = int(str(abs(hash(self.name)))[:9])
        np.random.seed(seed)
        return np.random.random(3)
        
    @property
    def inlier_points(self):
        """ Convenience attribute that can be set to save inlier points """
        return self._inlier_points
        
    @property
    def inlier_points_flattened(self):
        """ Convenience attribute that can be set to save inlier points """
        return self.flatten_points(self.inlier_points)
        
    @property
    def inlier_normals(self):
        """ Convenience attribute that can be set to save inlier normals """
        return self._inlier_normals
        
    @property
    def inlier_colors(self):
        """ Convenience attribute that can be set to save inlier colors """
        return self._inlier_colors
    
    @property
    def inlier_PointCloud(self):
        """ Creates Open3D.geometry.PointCloud object from inlier points. """
        pcd = PointCloud()
        pcd.points = Vector3dVector(self.inlier_points)
        pcd.normals = Vector3dVector(self.inlier_normals)
        pcd.colors = Vector3dVector(self.inlier_colors)
        return pcd
    
    @property
    def inlier_PointCloud_flattened(self):
        """ Creates Open3D.geometry.PointCloud object from inlier points. """
        pcd = PointCloud()
        pcd.points = Vector3dVector(self.inlier_points_flattened)
        pcd.normals = Vector3dVector(self.inlier_normals)
        pcd.colors = Vector3dVector(self.inlier_colors)
        return pcd
        
    @property
    def metrics(self):
        """ Convenience attribute that can be set to save shape metrics """
        return self._metrics
    
    @metrics.setter
    def metrics(self, metrics):
        if type(metrics) != dict:
            raise ValueError('metrics should be a dict')
        self._metrics = metrics
    
    @property
    def axis_spherical(self):
        """ Get axis in spherical coordinates, if the primitive has an axis. """
        x, y, z = self._get_axis_or_vector_or_normal()
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arccos(z/r)
        return np.array([r, theta, phi])
    
    @property
    def axis_cylindrical(self):
        """ Get axis in cylindrical coordinates, if the primitive has an axis. """
        x, y, z = self._get_axis_or_vector_or_normal()
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return np.array([rho, theta, z])
    
    def __repr__(self):
        round_ = lambda x:round(x, 5)
        params = list(map(round_, self.model))
        return type(self).__name__+'('+str(params)+')'
    
    def __eq__(self, other_shape):
        return self.is_similar_to(other_shape, rtol=1e-05, atol=1e-08)
    
    def __init__(self, model):
        """
        Parameters
        ----------
        model : list or tuple
            Parameters defining the shape model            
                        
        Raises
        ------
        ValueError
            If number of parameters is incompatible with the model of the 
            primitive.
        """
        model = np.array(model)
        if len(model) != self.model_args_n:
            raise ValueError(f'{self.name.capitalize()} primitives take '
                             f'{self.model_args_n} elements, got {model}')
        self._model = model
        
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
        return cls(np.random.random(cls._model_args_n) * scale)
    
    @staticmethod
    @abstractmethod
    def fit(points, normals=None):
        """ Gives shape that fits the input points. If the number of points is
        higher than the `_fit_n_min`, the fitted shape will return some kind of
        estimation. 
        
        Moreover, some primitives do not need the normal vectors to fit, while
        others (like cylinders) might benefit from it.
        
        Actual implementation depends on the type of primitive, m
        
        Parameters
        ----------
        points : N x 3 array
            N input points 
        normals : N x 3 array
            N normal vectors
        
        Returns
        -------
        Primitive
            Fitted shape.
        """
        pass
        
    @abstractmethod
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
        pass
    
    def get_distances(self, points):
        """ Gives the absolute value of the minimum distance between each point 
        to the model. 
        
        Parameters
        ----------
        points : N x 3 array
            N input points 
        
        Returns
        -------
        distances
            Nx1 array distances.
        """
        return abs(self.get_signed_distances(points))
    
    @abstractmethod
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
        pass
    
    def get_angles_cos(self, points, normals):
        """ Gives the absolute value of cosines of the angles between the input 
        normal vectors and the calculated normal vectors from the input points.
        
        Parameters
        ----------
        points : N x 3 array
            N input points 
        normals : N x 3 array
            N normal vectors
            
        Raises
        ------
        ValueError
            If number of points and normals are not equal
        
        Returns
        -------
        angles_cos or None
            Nx1 array with the absolute value of the cosines of the angles, or
            `None` if `normals` is `None`.
            
        """
        if len(normals) != len(points):
            raise ValueError('Number of points and normals should be equal.')
        
        if normals is None:
            return None
        normals = np.asarray(normals)
        normals_from_points = self.get_normals(points)
        angles_cos = np.clip(
            np.sum(normals * normals_from_points, axis=1), -1, 1)
        return np.abs(angles_cos)
    
    def get_angles(self, points, normals):
        """ Gives the angles between the input normal vectors and the 
        calculated normal vectors from the input points.
        
        Parameters
        ----------
        points : N x 3 array
            N input points 
        normals : N x 3 array
            N normal vectors
            
        Raises
        ------
        ValueError
            If number of points and normals are not equal
        
        Returns
        -------
        angles or None
            Nx1 array with the angles, or `None` if `normals` is `None`.
            
        """
        if normals is None:
            return None
        
        return np.arccos(
            self.get_angles_cos(points, normals))
    
    def get_residuals(self, points, normals):
        """ Convenience function returning both distances and angles.
        
        Parameters
        ----------
        points : N x 3 array
            N input points 
        normals : N x 3 array
            N normal vectors
            
        Raises
        ------
        ValueError
            If number of points and normals are not equal
        
        Returns
        -------
        tuple
            Tuple containing distances and angles.
            
        """
        return self.get_distances(points), \
            self.get_angles(points, normals)
    
    def flatten_points(self, points):
        """ Stick each point in input to the closest point in shape's surface.
        
        Parameters
        ----------
        points : N x 3 array
            N input points
        
        Returns
        -------
        points_flattened : N x 3 array
            N points on the surface
            
        """
        if len(points) == 0:
            return points
    
        points = np.asarray(points)        
        difference = self.get_signed_distances(points)[..., np.newaxis] * self.get_normals(points)
        points_flattened = points - difference
        return points_flattened
    
    def flatten_PointCloud(self, pcd):
        """ Return new pointcloud with flattened points.
        
        Parameters
        ----------
        pcd : Open3D.geometry.PointCloud
            Input pointcloud
        
        Returns
        -------
        Open3D.geometry.PointCloud
            Pointcloud with points flattened
            
        """
        pcd_flattened = PointCloud()
        pcd_flattened.points = Vector3dVector(self.flatten_points(pcd.points))
        pcd_flattened.colors = pcd.colors
        return pcd_flattened
    
    def add_inliers(self, points, normals=None, colors=None):
        """ Add inlier points to shape.
        
        Parameters
        ----------
        points : N x 3 array
            Inlier points.
        normals : optional, N x 3 array
            Inlier point normals.
        colors : optional, N x 3 array
            Colors of inlier points.
        """
            
        points = np.asarray(points)
        if points.shape == (3, ):
            points = np.reshape(points, (1,3))
        elif points.shape[1] != 3:
            raise ValueError('Invalid shape for input points, must be a single'
                             ' point or an array of shape (N, 3), got '
                             f'{points.shape}')
        self._inlier_points = points
        
        if normals is not None:
            normals = np.asarray(normals)
            if normals.shape == (3, ):
                normals = np.reshape(normals, (1,3))
            elif normals.shape[1] != 3:
                raise ValueError('Invalid shape for input normals, must be a single'
                                 ' point or an array of shape (N, 3), got '
                                 f'{normals.shape}')
            self._inlier_normals = normals
        
        if colors is not None:
            colors = np.asarray(colors)
            if colors.shape == (3, ):
                colors = np.reshape(colors, (1,3))
            elif normals.shape[1] != 3:
                raise ValueError('Invalid shape for input colors, must be a single'
                                 ' point or an array of shape (N, 3), got '
                                 f'{colors.shape}')
            self._inlier_colors = colors
            
    def closest_inliers(self, other_shape, n=1):
        """ Returns n pairs of closest inlier points with a second shape.
        
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
        if not isinstance(other_shape, Primitive):
            raise ValueError("other_shape must be a Primitive.")
            
        from pyShapeDetector.utility import find_closest_points
        
        closest_points, distances = find_closest_points(
            self.inlier_points, other_shape.inlier_points, n)
        
        return closest_points, distances
    
    def inliers_average_dist(self, k=15, leaf_size=40):
        """ Calculates the K nearest neighbors of the inlier points and returns 
        the average distance between them.
        
        Parameters
        ----------
        k : positive int, default = 15
            Number of neighbors.
        
        leaf_size : positive int, default=40
            Number of points at which to switch to brute-force. Changing
            leaf_size will not affect the results of a query, but can
            significantly impact the speed of a query and the memory required
            to store the constructed tree.  The amount of memory needed to
            store the tree scales as approximately n_samples / leaf_size.
            For a specified ``leaf_size``, a leaf node is guaranteed to
            satisfy ``leaf_size <= n_points <= 2 * leaf_size``, except in
            the case that ``n_samples < leaf_size``.
        
        Returns
        -------
        float
            Average nearest dist.
        """        
        from pyShapeDetector.utility import average_nearest_dist
        return average_nearest_dist(self.inlier_points, k, leaf_size)
    
    def inliers_bounding_box(self, slack=0):
        """ If the shape includes inlier points, returns the minimum and 
        maximum bounds of their bounding box.
        
        If 'slack' parameter is given, use it expand bounding box in all
        directions (useful for testing purposes).
        
        Returns
        -------
        tuple of two 3 x 1 arrays
            Minimum and maximum bounds of inlier points bounding box.
        """
        if len(self.inlier_points) == 0:
            return None, None
        
        slack = abs(slack)
        min_bound = np.min(self.inlier_points, axis=0)
        max_bound = np.max(self.inlier_points, axis=0)
        return np.vstack([min_bound - slack, max_bound + slack])
            
    def sample_points_uniformly(self, number_of_points=100, 
                            use_triangle_normal=False):
        """ Sample points from the mesh generated by the shape and then
        returns pointcloud with sampled points from the mesh.
        
        Parameters
        ----------
        number_of_points : int, optional
            Number of points that should be uniformly sampled. Default = 100.
        use_triangle_normal : bool, optional
            If True assigns the triangle normals instead of the interpolated 
            vertex normals to the returned points. The triangle normals will 
            be computed and added to the mesh if necessary. Default = False.
        
        Returns
        -------
        open3d.geometry.PointCloud
            Sampled pointcloud from shape.
        """
        mesh = self.get_mesh()
        return mesh.sample_points_uniformly(number_of_points, use_triangle_normal)
    
    def sample_points_density(self, density=1, 
                            use_triangle_normal=False):
        """ Sample points from the mesh generated by the shape and then
        returns pointcloud with sampled points from the mesh.
        
        Parameters
        ----------
        density: float, optional
            Ratio between points and surface area. Default: 1.
        use_triangle_normal : bool, optional
            If True assigns the triangle normals instead of the interpolated 
            vertex normals to the returned points. The triangle normals will 
            be computed and added to the mesh if necessary. Default = False.
        
        Returns
        -------
        open3d.geometry.PointCloud
            Sampled pointcloud from shape.
        """
        # mesh = self.get_mesh()
        number_of_points = int(density * self.surface_area)
        return self.sample_points_uniformly(number_of_points, use_triangle_normal)
    
    def get_mesh(self, resolution=30):
        """ Creates mesh of the shape.
        
        Parameters
        ----------
        resolution : int, optional
            Resolution parameter for mesh. Default: 30
        
        Returns
        -------
        TriangleMesh
            Mesh corresponding to the shape.
        """
        raise NotImplementedError('The mesh generating function for '
                                  f'primitives of type {self.name} has not '
                                  'been implemented.')
        
    def get_cropped_mesh(self, points=None, eps=1E-3):
        """ Creates mesh of the shape and crops it according to points.
        
        Parameters
        ----------
        points : N x 3 array, optional
            N input points. If points are not given, tries to use inlier points
            of shape.
        eps : float, optional
            Small value for cropping.
        
        Returns
        -------
        TriangleMesh
            Mesh corresponding to the shape. Default: 1E-3
        """
        
        if points is None:
            points = self.inlier_points
        
        if len(points) == 0:
            raise ValueError('No points given, and no inlier points.')
            
        mesh = self.get_mesh()
        points_flattened = self.flatten_points(points)
        pcd = PointCloud(Vector3dVector(points_flattened))
        bb = pcd.get_axis_aligned_bounding_box()
        bb = AxisAlignedBoundingBox(bb.min_bound - [eps]*3, 
                                    bb.max_bound + [eps]*3)
        # return mesh.crop(bb)
        return clean_crop(mesh, bb)
    
    def is_similar_to(self, other_shape, rtol=1e-02, atol=1e-02):
        """ Check if shapes represent same model.
        
        Parameters
        ----------
        other_shape : Primitive
            Primitive to compare
        
        Returns
        -------
        Bool
            True if shapes are similar.
        """
        if not isinstance(self, type(other_shape)):
            return False
        
        compare = np.isclose(self.canonical.model, other_shape.canonical.model,
                             rtol=rtol, atol=atol)
        return compare.all()
    
    def copy(self):
        """ Returns copy of shape 
        
        Returns
        -------
        Primitive
            Copied primitive
        """
        shape = type(self)(self.model.copy())
        shape._inlier_points = self._inlier_points.copy()
        shape._inlier_normals = self._inlier_normals.copy()
        shape._inlier_colors = self._inlier_colors.copy()
        shape._metrics = self._metrics.copy()
        return shape
    
    def _translate_points(self, translation):
        """ Internal helper function for translation"""
        if len(self._inlier_points) > 0:
            self._inlier_points = self._inlier_points + translation
    
    def translate(self, translation):
        """ Translate the shape.
        
        Parameters
        ----------
        translation : 1 x 3 array
            Translation vector.
        """
        if not hasattr(self, '_translatable'):
            raise NotImplementedError('Shapes of type {shape.name} do not '
                                      'have an implemented _translatable '
                                      'attribute')
        self._model[self._translatable] += translation        
        self._translate_points(translation)
        
    @staticmethod
    def _parse_rotation(rotation):
        """ Internal helper function for rotation"""
        if not isinstance(rotation, Rotation):
            rotation = Rotation.from_matrix(rotation)
            
        if isinstance(rotation, Rotation):
            try:
                length = len(rotation)
                if length[0] != 1:
                    raise ValueError('Rotation input should contain a single '
                                     ' rotation but has len(rotation) instead')
                rotation = rotation[0]
        
            except TypeError:
                pass

        return rotation
    
    def _rotate_points_normals(self, rotation):
        """ Internal helper function for rotation"""
        if len(self._inlier_points) > 0:
            self._inlier_points = rotation.apply(self._inlier_points)
        if len(self._inlier_normals) > 0:
            self._inlier_normals = rotation.apply(self._inlier_normals)
        
    def rotate(self, rotation):
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
            
        rotation = Primitive._parse_rotation(rotation)
            
        self._model[self._rotatable] = rotation.apply(
            self.model[self._rotatable])
        self._model[self._translatable] = rotation.apply(
            self.model[self._translatable])
        
        self._rotate_points_normals(rotation)
        
    
    def align(self, axis, possible_attributes=['axis', 'vector', 'normal']):
        """ Returns aligned 
        
        Parameters
        ----------
        axis : 3 x 1 array
            Axis to which the shape should be aligned.
        possible_attributes : list of strings, optional
            Attribute that should be aligned to axis. If shape has any of 
            those, it will be aligned. Otherwise, nothing is done. 
            Default: ['axis', 'vector', 'normal']
        """
        for attr in possible_attributes:
            if hasattr(self, attr):
                axis_original = getattr(self, attr)
                rotation = get_rotation_from_axis(axis_original, axis)
                self.rotate(rotation)
                break
    
    def save(self, path):
        """ Saves shape to JSON file.
        
        Parameters
        ----------
        path : string of pathlib.Path
            File destination.
        """
        import json
        from pathlib import Path
        path = Path(path)
        if path.exists():
            path.unlink()
        
        f = open(path, 'w')
        data = {
            'name': self.name,
            'model': self.model.tolist(),
            'inlier_points': self.inlier_points.tolist(),
            'inlier_normals': self.inlier_normals.tolist(),
            'inlier_colors': self.inlier_colors.tolist()}
        if self.name == 'bounded plane':
            data['bounds'] = self.bounds.tolist()
            data['_fusion_intersections'] = self._fusion_intersections.tolist()
            data['hole_bounds'] = [h.bounds.tolist() for h in self.holes]
        json.dump(data, f)
        f.close()
    
    @staticmethod
    def load(path):
        """ Laods shape from JSON file.
        
        Parameters
        ----------
        path : string of pathlib.Path
            File destination.
            
        Returns
        -------
        Primitive
            Loadded shape.
        """
        import json
        from pyShapeDetector.primitives import dict_primitives
        
        f = open(path, 'r')
        data = json.load(f)
        name = data['name']
        model = data['model']
        primitive = dict_primitives[name]
        if name == 'bounded plane':
            shape = primitive(model, np.array(data['bounds']))
            shape._fusion_intersections = np.array(data['_fusion_intersections'])
            
            hole_bounds = data['hole_bounds']
            holes = [primitive(shape.model, bounds) for bounds in hole_bounds]
            shape.add_holes(holes)
            
        else:
            shape = primitive(model)

        shape._inlier_points = np.array(data['inlier_points'])
        shape._inlier_normals = np.array(data['inlier_normals'])
        shape._inlier_colors = np.array(data['inlier_colors'])

        f.close()
        return shape
    
    def check_bbox_intersection(self, other_shape, distance):
        
        if distance is None:
            return True
        
        if distance <= 0:
            raise ValueError("Distance must be positive.")
        
        bb1 = self.inliers_bounding_box(slack=distance/2)
        bb2 = other_shape.inliers_bounding_box(slack=distance/2)
        
        test_order = bb2[1] - bb1[0] >= 0
        if test_order.all():
            pass
        elif (~test_order).all():
            bb1, bb2 = bb2, bb1
        else:
            return False
        
        # test_intersect = (bb1.max_bound + atol) - (bb2.min_bound - atol) >= 0
        test_intersect = bb1[1] - bb2[0] >= 0
        return test_intersect.all()
    
    def check_inlier_distance(self, other_shape, distance):
        
        if distance is None:
            return True
        
        if distance <= 0:
            raise ValueError("Distance must be positive.")
        
        _, dist = self.closest_inliers(other_shape)
        return dist[0] <= distance
    
    @staticmethod
    def fuse(shapes, detector=None, ignore_extra_data=False, line_intersection_eps=None):
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
            
        Returns
        -------
        Primitive
            Averaged shape.    
        """
        if len(shapes) == 1:
            return shapes[0]
        elif isinstance(shapes, Primitive):
            return shapes
        
        primitive = type(shapes[0])
        for shape in shapes[1:]:
            if not isinstance(shape, primitive):
                raise ValueError('Shapes in input must all have the same type.')
                
        try:
            fitness = [shape.metrics['fitness'] for shape in shapes]
        except:
            fitness = [1] * len(shapes)
            
        model = np.vstack([shape.model for shape in shapes])
        model = np.average(model, axis=0, weights=fitness)
        
        # Catch warning in case shape is a PlaneBounded
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shape = primitive(model)
        
        if not ignore_extra_data:
            points = np.vstack([shape.inlier_points for shape in shapes])
            normals = np.vstack([shape.inlier_normals for shape in shapes])
            colors = np.vstack([shape.inlier_colors for shape in shapes])
            if points.shape[1] == 0:
                points = None
            if normals.shape[1] == 0 or len(normals) < len(points):
                normals = None
            if colors.shape[1] == 0 or len(colors) < len(points):
                colors = None
            shape.add_inliers(points, normals, colors)
            
            if detector is not None:
                num_points = sum([shape.metrics['num_points'] for shape in shapes])
                num_inliers = len(points)
                distances, angles = shape.get_residuals(points, normals)
                shape.metrics = detector.get_metrics(
                    num_points, num_inliers, distances, angles)
                
        return shape
        
        
        
        
        
        
