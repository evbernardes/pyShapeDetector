#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:42:59 2023

@author: ebernardes
"""
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from open3d.geometry import TriangleMesh, PointCloud
from open3d.utility import Vector3iVector, Vector3dVector

from .primitivebase import Primitive
    
class Plane(Primitive):
    """
    Plane primitive.
    
    Attributes
    ----------
    _fit_n_min : int
        Minimum number of points necessary to fit a model.
    _model_args_n : str
        Number of parameters in the model.
    name : str
        Name of primitive.
    equation : str
        Equation that defines the primitive.
    normal : 3 x 1 array
        Normal vector defining plane.
    canonical : Plane
        Return canonical form for testing.
    
    Methods
    ------- 
    
    get_surface_area(points):
        Gives the surface area of Plane. 
        
    get_volume():
        Gives the volume of model. 
    
    def get_signed_distances(points):
        Gives the minimum distance between each point to the model. 
    
    get_distances(points)
        Gives the minimum distance between each point to the model. 
        
    get_normals(points)
        Gives, for each input point, the normal vector of the point closest 
        to the primitive. 
        
    fit(points, normals=None):
        Gives plane that fits the input points.
    
    get_angles_cos(self, points, normals):
        Gives the absolute value of cosines of the angles between the input 
        normal vectors and the calculated normal vectors from the input points.
    
    get_rotation_from_axis(axis, axis_origin=[0, 0, 1])
        Rotation matrix that transforms `axis_origin` in `axis`.
        
    flatten_points(points):
        Stick each point in input to the closest point in shape's surface.
        
    get_angles_cos(points, normals):
        Gives the absolute value of cosines of the angles between the input 
        normal vectors and the calculated normal vectors from the input points.
        
    get_angles(points, normals):
        Gives the angles between the input normal vectors and the 
        calculated normal vectors from the input points.
        
    get_residuals(points, normals):
        Convenience function returning both distances and angles.
    
    get_mesh(points): TriangleMesh
        Flatten points and creates a simplified mesh of the plane defined
        by the points at the borders.
    
    """
    
    _fit_n_min = 3
    _model_args_n = 4
    name = 'plane'
    
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
        norm = np.linalg.norm(model[:3])
        model /= norm
        
        Primitive.__init__(self, model)
    
    @property
    def canonical(self):
        """ Return canonical form for testing. """
        model = self.model
        if self.model[-1] >= 0:
            model = -model
        return Plane(list(-self.model))
    
    @property
    def equation(self):
        n = self.normal
        d = self.model[-1]
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
    def normal(self):
        """ Normal vector defining point. """
        return np.array(self.model[:3])
    
    def get_surface_area(self, points):
        """ Gives the surface area of plane, needs points. 
        
        Parameters
        ----------
        points, optional : 3 x N array
            N input points 
        
        Returns
        -------
        float
            surface area.
        """
        return self.get_mesh(points).get_surface_area()
    
    def get_volume(self):
        """ Gives the volume of plane, which is zero. 
        
        Returns
        -------
        float
            volume.
        """
        return 0
    
    def get_signed_distances(self, points):
        """ Gives the minimum distance between each point to the model. 
        
        Actual implementation depends on the type of primitive.
        
        Parameters
        ----------
        points : 3 x N array
            N input points 
        
        Returns
        -------
        distances
            Nx1 array distances.
        """
        points = np.asarray(points)
        return points.dot(self.normal) + self.model[3]
    
    def get_normals(self, points):
        """ Gives, for each input point, the normal vector of the point closest 
        to the primitive. 
        
        Actual implementation depends on the type of primitive.
        
        Parameters
        ----------
        points : 3 x N array
            N input points 
        
        Returns
        -------
        normals
            Nx3 array containing normal vectors.
        """
        return np.tile(self.normal, (len(points), 1))
    
    def get_mesh(self, points=None):
        """ Flatten points and creates a simplified mesh of the plane defined
        by the points at the borders.      

        Parameters
        ----------
        points : 3 x N array
            Points corresponding to the fitted shape.
        
        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        if points is None:
            return self.get_square_mesh()
        points = np.asarray(points)
        points_flat = self.flatten_points(points)
        
        rot = self.get_rotation_from_axis(self.normal)
        projection = (rot @ points_flat.T).T[:, :2]
        
        chull = ConvexHull(projection)
        borders = projection[chull.vertices]
        
        triangles = Delaunay(borders).simplices
        
        # needed to make plane visible from both sides
        triangles = np.vstack([triangles, triangles[:, ::-1]]) 
        
        mesh = TriangleMesh()
        # mesh.vertices = Vector3dVector(points_flat[chull.vertices])
        mesh.vertices = Vector3dVector(points[chull.vertices])
        mesh.triangles = Vector3iVector(triangles)
        
        return mesh
    
    def get_square_mesh(self, half_length=10):
        center = -self.model[-1] * self.normal
        # center = np.mean(np.asarray(pcd.points), axis=0)
        # bb = pcd.get_axis_aligned_bounding_box()
        # half_length = max(bb.max_bound - bb.min_bound) / 2
        
        if list(self.normal) == [0, 0, 1]: 
            v1 = np.cross([0, 1, 0], self.normal)
        else:
            v1 = np.cross([0, 0, 1], self.normal)
            
        v2 = np.cross(v1, self.normal)
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        vertices = np.vstack([
            center + v1 * half_length,
            center + v2 * half_length,
            center - v1 * half_length,
            center - v2 * half_length])

        triangles = Vector3iVector(np.array([
            [0, 1, 2], 
            [2, 1, 0],
            [0, 2, 3],
            [3, 2, 0]]))
        vertices = Vector3dVector(vertices)

        return TriangleMesh(vertices, triangles)
    
    @staticmethod
    def fit(points, normals=None):
        """ Gives plane that fits the input points. If the number of points is
        higher than the `3`, the fitted shape will return a least squares 
        estimation.
        
        Reference:
            https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
        
        Parameters
        ----------
        points : 3 x N array
            N input points 
        normals : 3 x N array
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
        
        abc = abc / norm
        return Plane([abc[0], abc[1], abc[2], -abc.dot(centroid)]) 
        
    
class PlaneBounded(Plane):
    """
    Plane primitive.
    
    Attributes
    ----------
    _fit_n_min : int
        Minimum number of points necessary to fit a model.
    _model_args_n : str
        Number of parameters in the model.
    name : str
        Name of primitive.
    equation : str
        Equation that defines the primitive.
    normal : 3 x 1 array
        Normal vector defining plane.
    canonical : Plane
        Return canonical form for testing.
    
    Methods
    ------- 
    
    get_surface_area(points):
        Gives the surface area of Plane. 
        
    get_volume():
        Gives the volume of model. 
    
    def get_signed_distances(points):
        Gives the minimum distance between each point to the model. 
    
    get_distances(points)
        Gives the minimum distance between each point to the model. 
        
    get_normals(points)
        Gives, for each input point, the normal vector of the point closest 
        to the primitive. 
        
    fit(points, normals=None):
        Gives plane that fits the input points.
    
    get_angles_cos(self, points, normals):
        Gives the absolute value of cosines of the angles between the input 
        normal vectors and the calculated normal vectors from the input points.
    
    get_rotation_from_axis(axis, axis_origin=[0, 0, 1])
        Rotation matrix that transforms `axis_origin` in `axis`.
        
    flatten_points(points):
        Stick each point in input to the closest point in shape's surface.
        
    get_angles_cos(points, normals):
        Gives the absolute value of cosines of the angles between the input 
        normal vectors and the calculated normal vectors from the input points.
        
    get_angles(points, normals):
        Gives the angles between the input normal vectors and the 
        calculated normal vectors from the input points.
        
    get_residuals(points, normals):
        Convenience function returning both distances and angles.
    
    get_mesh(points): TriangleMesh
        Flatten points and creates a simplified mesh of the plane defined
        by the points at the borders.
    
    """
    
    _fit_n_min = 3
    _model_args_n = 4
    name = 'plane'
    
    def __init__(self, plane, bounds):
        """
        Parameters
        ----------
        plane : Plane
            Shape defining plane
        bounds : array_like, shape (N, 3)
            Points defining bounds
                        
        Raises
        ------
        ValueError
            If number of parameters is incompatible with the model of the 
            primitive.
        """
        bounds = np.asarray(bounds)
        if bounds.shape[1] != 3:
            raise ValueError('Expected shape of bounds is (N, 3), got '
                             f'{bounds.shape}')
        distances = plane.get_distances(bounds)
        rmse = np.sqrt(sum(distances * distances)) / len(distances)
        if  rmse > 1e-7:
            raise ValueError('Boundary points are not close enough to plane: '
                             f'rmse={rmse}, expected less than 1e-7.')
        self.model = plane.model
        self.bounds = plane.flatten_points(bounds)
    
    @property
    def canonical(self):
        """ Return canonical form for testing. """
        model = self.model
        if self.model[-1] >= 0:
            model = -model
        return PlaneBounded(list(-self.model), self.bounds)
    
    @property
    def unbounded(self):
        """ Return regular unbounded plane. """
        return Plane(self.model)
    
    def get_surface_area(self, points):
        """ Gives the surface area of plane, needs points. 
        
        Parameters
        ----------
        points, optional : 3 x N array
            N input points 
        
        Returns
        -------
        float
            surface area.
        """
        return self.get_mesh(points).get_surface_area()
    
    def get_mesh(self):
        """ Flatten points and creates a simplified mesh of the plane defined
        by the points at the borders.      

        Parameters
        ----------
        points : 3 x N array
            Points corresponding to the fitted shape.
        
        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        points = self.bounds
        
        rot = self.get_rotation_from_axis(self.normal)
        projection = (rot @ points.T).T[:, :2]
        
        chull = ConvexHull(projection)
        borders = projection[chull.vertices]
        
        triangles = Delaunay(borders).simplices
        
        # needed to make plane visible from both sides
        triangles = np.vstack([triangles, triangles[:, ::-1]]) 
        
        mesh = TriangleMesh()
        # mesh.vertices = Vector3dVector(points_flat[chull.vertices])
        mesh.vertices = Vector3dVector(points[chull.vertices])
        mesh.triangles = Vector3iVector(triangles)
        
        return mesh
    
    def get_bounds(self, points):
        points = np.asarray(points)
        points_flat = self.flatten_points(points)
        
        rot = self.get_rotation_from_axis(self.normal)
        projection = (rot @ points_flat.T).T[:, :2]
        
        chull = ConvexHull(projection)
        return points[chull.vertices]
    
    def get_mesh(self, points=None):
        """ Flatten points and creates a simplified mesh of the plane defined
        by the points at the borders.      

        Parameters
        ----------
        points : 3 x N array
            Points corresponding to the fitted shape.
        
        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        if points is None:
            return self.get_square_mesh()
        points = np.asarray(points)
        points_flat = self.flatten_points(points)
        
        rot = self.get_rotation_from_axis(self.normal)
        projection = (rot @ points_flat.T).T[:, :2]
        
        chull = ConvexHull(projection)
        borders = projection[chull.vertices]
        
        triangles = Delaunay(borders).simplices
        
        # needed to make plane visible from both sides
        triangles = np.vstack([triangles, triangles[:, ::-1]]) 
        
        mesh = TriangleMesh()
        # mesh.vertices = Vector3dVector(points_flat[chull.vertices])
        mesh.vertices = Vector3dVector(points[chull.vertices])
        mesh.triangles = Vector3iVector(triangles)
        
        return mesh
    
    @staticmethod
    def fit(points, normals=None):
        """ Gives plane that fits the input points. If the number of points is
        higher than the `3`, the fitted shape will return a least squares 
        estimation.
        
        Reference:
            https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
        
        Parameters
        ----------
        points : 3 x N array
            N input points 
        normals : 3 x N array
            N normal vectors
        
        Returns
        -------
        PlaneBounded
            Fitted plane.
        """
        
        plane = Plane.fit(points, normals)
        
        return Plane([abc[0], abc[1], abc[2], -abc.dot(centroid)]) 

            
        
        
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    