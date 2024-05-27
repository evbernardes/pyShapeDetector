#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions that act on meshes.

Created on Wed Dec  6 14:59:38 2023

@author: ebernardes
"""
import warnings
import copy
import open3d as o3d
import numpy as np
from itertools import product
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from open3d.geometry import TriangleMesh
from open3d.utility import Vector3dVector, Vector3iVector
from open3d.visualization import draw_geometries_with_key_callbacks
from .helpers_primitives import fuse_shape_groups

def _get_vertices_triangles(mesh_or_vertices, triangles=None):
    """ Helper function to deal with inputs. """
    if isinstance(mesh_or_vertices, TriangleMesh):
        if triangles is not None:
            raise ValueError("Input is either a single mesh, or an array of "
                             "vertices and one array of triangles.")
        vertices = np.asarray(mesh_or_vertices.vertices)
        triangles = np.asarray(mesh_or_vertices.triangles)
    else:
        if triangles is None:
            raise ValueError("If vertices are given as first input instead of "
                             "a mesh, then triangles must be given.")
        vertices = np.asarray(mesh_or_vertices)
        triangles = np.asarray(triangles)
    return vertices, triangles

# def get_rectangular_grid(vectors, center, grid_width, return_perimeter = False, grid_type = "hexagonal"):
#     """ Gives rectangular grid defined two vectors and its center.
    
#     Vectors v1 and v2 should not be unit, and instead have lengths equivalent
#     to widths of rectangle.
    
#     Available grid types: "regular" and "hexagonal".
    
#     If `return_perimeter` is set to True, also return the expected perimeter
#     of each triangle.
    
#     Parameters
#     ----------
#     vectors : arraylike of shape (2, 3)
#         The two orthogonal unit vectors defining the rectangle plane.
#     center : arraylike of length 3, optional
#         Center of rectangle. If not given, either inliers or centroid are 
#         used.
#     grid_width : float
#         Distance between two points in first dimension (and also second 
#         dimension for regular grids).
#     return_perimeter : boolean, optional
#         If True, return tuple containing both grid and calculated perimeter.
#         Default: False.
#     grid_type : str, optional
#         Type of grid, can be "hexagonal" or "regular". Default: "hexagonal".

#     See also:
#         select_grid_points

#     Returns
#     -------
#     numpy.array
#         Grid points
        
#     float
#         Perimeter of one triangle
#     """
#     if grid_type not in ["regular", "hexagonal"]:
#         raise ValueError("Possible grid types are 'regular' and 'hexagonal', "
#                          f"got '{grid_type}'.")
#     eps = 1e-8
#     lengths = np.linalg.norm(vectors, axis=1)
    
#     # get unit vectors
#     vx, vy = vectors / lengths[:, np.newaxis]

#     n = []
#     for length in lengths:
#         ratio = length / grid_width
#         n.append(int(np.floor(ratio)) + int(ratio % 1 > eps))
    
#     def get_range(length, width):
#         array = np.arange(stop=length, step=width) - length / 2
#         assert abs(width - (array[1] - array[0])) < eps
        
#         # adding last point if needed
#         if (length / width) % 1 > eps:
#             # array = np.hstack([array, array[-1] + grid_width]) 
#             array = np.hstack([array, length / 2])
            
#         return array
    
#     if grid_type == "regular":
#         array_x = get_range(lengths[0], grid_width)
#         array_y = get_range(lengths[1], grid_width)
        
#         x_ = vx * array_x[np.newaxis].T
#         y_ = vy * array_y[np.newaxis].T
#         # grid_lines = [px + py for px, py in product(x_, y_)]
#         grid_lines = [x_ + py for py in y_]
#         grid = np.vstack(grid_lines)

#         perimeter = (2 + np.sqrt(2)) * grid_width
        
#     elif grid_type == "hexagonal":
#         h = grid_width * np.sqrt(3) / 2
        
#         array_x = get_range(lengths[0], grid_width)
#         array_y = get_range(lengths[1], h)
        
#         x_ = vx * array_x[np.newaxis].T
#         y_ = vy * array_y[np.newaxis].T
#         # grid_lines = [px + py for px, py in product(x_, y_)]
#         grid_lines = [x_ + py for py in y_]
#         for i in range(len(grid_lines)):
#             if i % 2 == 1:
#                 grid_lines[i] += vx * grid_width / 2
#                 grid_lines[i] = np.vstack([-vx * lengths[0]/2, grid_lines[i][:-1] ])

#         perimeter = 3 * grid_width
    
#     grid = np.vstack(grid_lines)
    
#     if return_perimeter:
#         return center + grid, perimeter
#     return center + grid

# def select_grid_points(grid, inlier_points, max_distance, cores=6):
#     """
#     Select points from grid that are close enough to at least some point in the
#     inlier_points points array.
    
#     Attention: Consider flattening both "grid" and "inlier_points" to the 
#     desired surface.
    
#     See: get_rectangular_grid

#     Parameters
#     ----------
#     grid : numpy array
#         Array of grid points.
#     inlier_points : numpy array
#         Array of points (most likely, inlier points).
#     max_distance : float
#         Max distance allowed between grid points and inlier points.

#     Returns
#     -------
#     numpy array
#         Selected points.
#     """
#     import multiprocessing
#     from .helpers_internal import parallelize
    
#     if cores > (cpu_count := multiprocessing.cpu_count()):
#         warnings.warn(f'Only {cpu_count} available, {cores} required.'
#                        ' limiting to max availability.')
#         cores = cpu_count
        
#     max_distance_squared = max_distance * max_distance
    
#     @parallelize(6)
#     def select_points(grid):
#         dist_squared = cdist(inlier_points, grid, 'sqeuclidean')
#         return np.any(dist_squared <= max_distance_squared, axis=0)
    
#     selected_idxs = select_points(grid)
#     return grid[selected_idxs]

# def remove_big_triangles(vertices, triangles, radius_ratio):
#     vertices = np.asarray(vertices)
#     triangles = np.asarray(triangles)
    
#     from pyShapeDetector.utility import get_triangle_circumradius
    
#     circumradiuses = get_triangle_circumradius(vertices, triangles)
#     mean_circumradiuses = np.mean(circumradiuses)
#     return triangles[circumradiuses < radius_ratio * mean_circumradiuses]

def planes_ressample_and_triangulate(planes, density, radius_ratio=None, double_triangles=False):
    from pyShapeDetector.primitives import PlaneBounded
    
    if isinstance(planes, PlaneBounded):
        planes = [planes]
    else:
        for plane in planes:
            if not isinstance(plane, PlaneBounded):
                raise ValueError("Input must one or more instances of "
                                 "PlaneBounded.")
                
                
    vertices = []
    for p in planes:
        N = int(np.ceil(density * len(p.inlier_points)))
    #     print(N)
        vertices.append(p.sample_points_uniformly(N))
    vertices = np.vstack(vertices)
    
    # number_of_points = int(self.surface_area * density)
    # vertices = [p.sample_points_uniformly(int(len(p.inlier_points) * density)) for p in planes]
    # vertices = [p.sample_points_density(density).points for p in planes]
    # vertices = np.vstack(vertices)
    
    # from pyShapeDetector.utility import fuse_shape_groups
    plane_fused = PlaneBounded.fuse(planes, ignore_extra_data=True, force_concave=False)

    projections = plane_fused.get_projections(vertices)
    triangles = Delaunay(projections).simplices
    
    if radius_ratio is not None:
        triangles = remove_big_triangles(vertices, triangles, radius_ratio)
        
    if double_triangles:
        triangles = np.vstack([triangles, triangles[:, ::-1]])

    return vertices, triangles

def planes_ressample_and_triangulate_gui(planes, translation_ratio = 0.055,
                                         double_triangles=False):
    
    meshes = [plane.get_mesh() for plane in planes]

    plane_fused = fuse_shape_groups(
        [planes], detector=None, line_intersection_eps=1e-3)[0]
    
    density = 0
    all_points = plane_fused.inlier_points_flattened
    all_projections = plane_fused.get_projections(all_points)
    all_triangles = Delaunay(all_projections).simplices
    
    # all_points = np.vstack(
    #     [p.sample_points_uniformly(density * len(p.inlier_points)) for p in planes])
    # all_projections = planes[0].get_projections(all_points)
    # all_triangles = Delaunay(all_projections).simplices
    # vertices = all_points
    
    # perimeters = util.get_triangle_perimeters(all_points, triangles)
    circumradius = TriangleMesh(all_points, all_triangles).get_triangle_circumradius()
    # mean_circumradius = np.mean(circumradius)
    
    global data
    # viss = 0
    
    window_name = """ 
        (W): increase cutout limit. 
        (S): decrease cutout limit. 
        (A): add point density.
        (D): reduce point density.
        """
    
    data = {
        'original_vertices': all_points,
        'original_triangles': all_triangles,
        'density': density,
        'mean_circumradius': np.mean(circumradius),
        'circumradius': circumradius,
        'vertices': all_points,
        'triangles': all_triangles,
        'radius_ratio': 1,
        'color': (1, 0, 0),
        'translation': translation_ratio * plane_fused.normal,
        'plotted': False}
    
    triangles = data['triangles'][
        data['circumradius'] < data['radius_ratio'] * data['mean_circumradius']]
    data['current'] = TriangleMesh(all_points, triangles)
    data['current'].paint_uniform_color(data['color'])
    data['current'].translate(data['translation'])
    data['lines'] = TriangleMesh(data['vertices']+data['translation'], data['triangles']).get_triangle_LineSet()
    
    def update_geometries(vis):
        global data
        print(f"density: {data['density']}, radius_ratio: {data['radius_ratio']}")
        
        triangles = data['triangles'][
            data['circumradius'] < data['radius_ratio'] * data['mean_circumradius']]
        
        data['old'] = data['current']
        data['current'] = TriangleMesh(data['vertices'], triangles)
        data['current'].paint_uniform_color(data['color'])
        data['current'].translate(data['translation'])
        
        vis.add_geometry(data['current'])
        vis.remove_geometry(data['old'])
        
        return False
    
    def ratio_increase(vis):
        global data
        data['radius_ratio'] += 0.1
        # data['radius_ratio'] *= 2
        update_geometries(vis)
    
    def ratio_decrease(vis):
        global data
        data['radius_ratio'] -= 0.1
        # data['radius_ratio'] /= 2
        data['radius_ratio'] = max(data['radius_ratio'], 0)
        update_geometries(vis)
        
    def update_density(vis):
        global data
        
        if data['density'] == 0:
            data['vertices'] = data['original_vertices']
            data['triangles'] = data['original_triangles']
        else:
            data['vertices'], data['triangles'] = planes_ressample_and_triangulate(
                planes, data['density'])
            
        data['circumradius'] = TriangleMesh(
            data['vertices'], data['triangles']).get_triangle_circumradius
        data['mean_circumradius'] = np.mean(circumradius)
        
        vis.remove_geometry(data['lines'])
        data['lines'] = TriangleMesh(data['vertices']+data['translation'], data['triangles']).get_triangle_LineSet()
        vis.add_geometry(data['lines'])
        
    def density_increase(vis):
        global data
        data['density'] += 0.1
        update_density(vis)
        update_geometries(vis)
    
    def density_decrease(vis):
        global data
        data['density'] -= 0.1
        data['density'] = max(data['density'], 0.0)
        update_density(vis)
        update_geometries(vis)
    
    key_to_callback = {}
    key_to_callback[ord("W")] = ratio_increase
    key_to_callback[ord("S")] = ratio_decrease
    key_to_callback[ord("A")] = density_decrease
    key_to_callback[ord("D")] = density_increase
    # key_to_callback[ord("D")] = remove_last
    # key_to_callback[ord("R")] = load_render_option
    # key_to_callback[ord(",")] = capture_depth
    # key_to_callback[ord(".")] = capture_image
    
    draw_geometries_with_key_callbacks(
        meshes+[data['current'], data['lines']], 
        # [data['current'], data['lines']], 
        key_to_callback,
        window_name = window_name,
        # mesh_show_wireframe=True
        )
    
    triangles = data['triangles'][
        data['circumradius'] < data['radius_ratio'] * data['mean_circumradius']]
    
    if double_triangles:
        triangles = np.vstack([triangles, triangles[:, ::-1]])
    
    out_data = {
        'density': data['density'],
        'radius_ratio': data['radius_ratio'],
        'vertices': data['vertices'],
        'triangles': triangles}
    
    return out_data

def alphashape_2d(projections, alpha):
    """ Compute the alpha shape (concave hull) of a set of 2D points. If the number
    of points in the input is three or less, the convex hull is returned to the
    user.

    Parameters
    ----------
    projections : array_like, shape (N, 2)
        Points corresponding to the 2D projections in the plane.
    
    Returns
    -------
    vertices : np.array_like, shape (N, 2)
        Boundary points in computed shape.
        
    triangles : np.array shape (N, 3)
        Indices of each triangle.
    """
    projections = np.asarray(projections)
    if projections.shape[1] != 2:
        raise ValueError("Input points must be 2D.")

    # If given a triangle for input, or an alpha value of zero or less,
    # return the convex hull.
    if len(projections) < 4 or (alpha is not None and not callable(
            alpha) and alpha <= 0):
        
        convex_hull = ConvexHull(projections)
        return convex_hull.points, convex_hull.simplices

    # Determine alpha parameter if one is not given
    if alpha is None:
        try:
            from optimizealpha import optimizealpha
        except ImportError:
            from .optimizealpha import optimizealpha
        alpha = optimizealpha(projections)

    vertices = np.array(projections)

    # Create a set to hold unique edges of simplices that pass the radius
    # filtering
    edges = set()

    # Create a set to hold unique edges of perimeter simplices.
    # Whenever a simplex is found that passes the radius filter, its edges
    # will be inspected to see if they already exist in the `edges` set.  If an
    # edge does not already exist there, it will be added to both the `edges`
    # set and the `permimeter_edges` set.  If it does already exist there, it
    # will be removed from the `perimeter_edges` set if found there.  This is
    # taking advantage of the property of perimeter edges that each edge can
    # only exist once.
    perimeter_edges = set()

    for point_indices, circumradius in alphasimplices(vertices):
        if callable(alpha):
            resolved_alpha = alpha(point_indices, circumradius)
        else:
            resolved_alpha = alpha

        # Radius filter
        if circumradius < 1.0 / resolved_alpha:
            for edge in itertools.combinations(
                    # point_indices, r=coords.shape[-1]):
                    point_indices, 3):
                if all([e not in edges for e in itertools.combinations(
                        edge, r=len(edge))]):
                    edges.add(edge)
                    perimeter_edges.add(edge)
                else:
                    perimeter_edges -= set(itertools.combinations(
                        edge, r=len(edge)))
    
    triangles = np.array(list(perimeter_edges))
    return vertices, triangles


def polygonize_alpha_shape(vertices, edges):
    # Create the resulting polygon from the edge points
    m = MultiLineString([vertices[np.array(edge)] for edge in edges])
    triangles = list(polygonize(m))
    result = unary_union(triangles)
    return result   
