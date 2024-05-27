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

def get_rectangular_grid(vectors, center, grid_width, return_perimeter = False, grid_type = "hexagonal"):
    """ Gives rectangular grid defined two vectors and its center.
    
    Vectors v1 and v2 should not be unit, and instead have lengths equivalent
    to widths of rectangle.
    
    Available grid types: "regular" and "hexagonal".
    
    If `return_perimeter` is set to True, also return the expected perimeter
    of each triangle.
    
    Parameters
    ----------
    vectors : arraylike of shape (2, 3)
        The two orthogonal unit vectors defining the rectangle plane.
    center : arraylike of length 3, optional
        Center of rectangle. If not given, either inliers or centroid are 
        used.
    grid_width : float
        Distance between two points in first dimension (and also second 
        dimension for regular grids).
    return_perimeter : boolean, optional
        If True, return tuple containing both grid and calculated perimeter.
        Default: False.
    grid_type : str, optional
        Type of grid, can be "hexagonal" or "regular". Default: "hexagonal".

    See also:
        select_grid_points

    Returns
    -------
    numpy.array
        Grid points
        
    float
        Perimeter of one triangle
    """
    if grid_type not in ["regular", "hexagonal"]:
        raise ValueError("Possible grid types are 'regular' and 'hexagonal', "
                         f"got '{grid_type}'.")
    eps = 1e-8
    lengths = np.linalg.norm(vectors, axis=1)
    
    # get unit vectors
    vx, vy = vectors / lengths[:, np.newaxis]

    n = []
    for length in lengths:
        ratio = length / grid_width
        n.append(int(np.floor(ratio)) + int(ratio % 1 > eps))
    
    def get_range(length, width):
        array = np.arange(stop=length, step=width) - length / 2
        assert abs(width - (array[1] - array[0])) < eps
        
        # adding last point if needed
        if (length / width) % 1 > eps:
            # array = np.hstack([array, array[-1] + grid_width]) 
            array = np.hstack([array, length / 2])
            
        return array
    
    if grid_type == "regular":
        array_x = get_range(lengths[0], grid_width)
        array_y = get_range(lengths[1], grid_width)
        
        x_ = vx * array_x[np.newaxis].T
        y_ = vy * array_y[np.newaxis].T
        # grid_lines = [px + py for px, py in product(x_, y_)]
        grid_lines = [x_ + py for py in y_]
        grid = np.vstack(grid_lines)

        perimeter = (2 + np.sqrt(2)) * grid_width
        
    elif grid_type == "hexagonal":
        h = grid_width * np.sqrt(3) / 2
        
        array_x = get_range(lengths[0], grid_width)
        array_y = get_range(lengths[1], h)
        
        x_ = vx * array_x[np.newaxis].T
        y_ = vy * array_y[np.newaxis].T
        # grid_lines = [px + py for px, py in product(x_, y_)]
        grid_lines = [x_ + py for py in y_]
        for i in range(len(grid_lines)):
            if i % 2 == 1:
                grid_lines[i] += vx * grid_width / 2
                grid_lines[i] = np.vstack([-vx * lengths[0]/2, grid_lines[i][:-1] ])

        perimeter = 3 * grid_width
    
    grid = np.vstack(grid_lines)
    
    if return_perimeter:
        return center + grid, perimeter
    return center + grid

def select_grid_points(grid, inlier_points, max_distance, cores=6):
    """
    Select points from grid that are close enough to at least some point in the
    inlier_points points array.
    
    Attention: Consider flattening both "grid" and "inlier_points" to the 
    desired surface.
    
    See: get_rectangular_grid

    Parameters
    ----------
    grid : numpy array
        Array of grid points.
    inlier_points : numpy array
        Array of points (most likely, inlier points).
    max_distance : float
        Max distance allowed between grid points and inlier points.

    Returns
    -------
    numpy array
        Selected points.
    """
    import multiprocessing
    from .helpers_internal import parallelize
    
    if cores > (cpu_count := multiprocessing.cpu_count()):
        warnings.warn(f'Only {cpu_count} available, {cores} required.'
                       ' limiting to max availability.')
        cores = cpu_count
        
    max_distance_squared = max_distance * max_distance
    
    @parallelize(6)
    def select_points(grid):
        dist_squared = cdist(inlier_points, grid, 'sqeuclidean')
        return np.any(dist_squared <= max_distance_squared, axis=0)
    
    selected_idxs = select_points(grid)
    return grid[selected_idxs]

# def clean_crop(self, axis_aligned_bounding_box):
#     """ Crops mesh by slicing facets instead of completely removing them, as
#     seen on [1].
    
#     Parameters
#     ----------
#     axis_aligned_bounding_box: AxisAlignedBoundingBox
#         Bounding box defining region of mesh to be saved.
    
#     Returns
#     -------
#     TriangleMesh
#         Cropped mesh.
        
#     References
#     ----------
#     [1] https://stackoverflow.com/questions/75082217/crop-function-that-slices-triangles-instead-of-removing-them-open3d
#     """
    
#     min_bound = axis_aligned_bounding_box.min_bound
#     max_bound = axis_aligned_bounding_box.max_bound

#     # mesh = sliceplane(mesh, 0, min_x, False)
#     mesh_sliced = copy.copy(self)
#     for i in range(3):
#         min_, max_ = sorted([min_bound[i], max_bound[i]])
#         mesh_sliced = _sliceplane(mesh_sliced, i, max_, True)
#         mesh_sliced = _sliceplane(mesh_sliced, i, min_, False)
#     return mesh_sliced

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

# def triangulate_earclipping(polygon):
#     """
#     Shamelessly copied from tripy:
#         https://github.com/linuxlewis/tripy/blob/master/tripy.py
    
#     Simple earclipping algorithm for a given polygon p.
#     polygon is expected to be an array of 2-tuples of the cartesian points of the polygon

#     For a polygon with n points it will return n-2 triangles.
#     The triangles are returned as an array of 3-tuples where each item in the tuple is a 2-tuple of the cartesian point.

#     e.g
#     >>> polygon = [(0,1), (-1, 0), (0, -1), (1, 0)]
#     >>> triangles = tripy.earclip(polygon)
#     >>> triangles
#     [((1, 0), (0, 1), (-1, 0)), ((1, 0), (-1, 0), (0, -1))]

#     Implementation Reference:
#         - https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
#     """
#     import math
#     import sys
#     from collections import namedtuple

#     polygon = np.array(polygon)
    
#     if not len(polygon.shape) == 2 or not polygon.shape[-1] == 2:
#         raise ValueError(f"Array of shape (N, 2) expected, got {polygon.shape}.")
    
#     original_polygon = copy.copy(polygon)

#     Point = namedtuple('Point', ['x', 'y'])

#     EPSILON = math.sqrt(sys.float_info.epsilon)
    
#     def _is_clockwise(polygon):
#         s = 0
#         polygon_count = len(polygon)
#         for i in range(polygon_count):
#             point = polygon[i]
#             point2 = polygon[(i + 1) % polygon_count]
#             s += (point2.x - point.x) * (point2.y + point.y)
#         return s > 0

#     def _triangle_sum(x1, y1, x2, y2, x3, y3):
#         return x1 * (y3 - y2) + x2 * (y1 - y3) + x3 * (y2 - y1)

#     def _is_convex(prev, point, next):
#         return _triangle_sum(prev.x, prev.y, point.x, point.y, next.x, next.y) < 0
    
#     def _contains_no_points(p1, p2, p3, polygon):
#         for pn in polygon:
#             if pn in (p1, p2, p3):
#                 continue
#             elif _is_point_inside(pn, p1, p2, p3):
#                 return False
#         return True

#     def _is_ear(p1, p2, p3, polygon):
#         ear = _contains_no_points(p1, p2, p3, polygon) and \
#             _is_convex(p1, p2, p3) and \
#             _triangle_area(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y) > 0
#         return ear

#     def _is_point_inside(p, a, b, c):
#         area = _triangle_area(a.x, a.y, b.x, b.y, c.x, c.y)
#         area1 = _triangle_area(p.x, p.y, b.x, b.y, c.x, c.y)
#         area2 = _triangle_area(p.x, p.y, a.x, a.y, c.x, c.y)
#         area3 = _triangle_area(p.x, p.y, a.x, a.y, b.x, b.y)
#         areadiff = abs(area - sum([area1, area2, area3])) < EPSILON
#         return areadiff

#     def _triangle_area(x1, y1, x2, y2, x3, y3):
#         return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

    
#     ear_vertex = []
#     triangles = []

#     polygon = [Point(*point) for point in polygon]

#     if _is_clockwise(polygon):
#         polygon.reverse()

#     point_count = len(polygon)
#     for i in range(point_count):
#         prev_index = i - 1
#         prev_point = polygon[prev_index]
#         point = polygon[i]
#         next_index = (i + 1) % point_count
#         next_point = polygon[next_index]

#         if _is_ear(prev_point, point, next_point, polygon):
#             ear_vertex.append(point)

#     while ear_vertex and point_count >= 3:
#         ear = ear_vertex.pop(0)
#         i = polygon.index(ear)
#         prev_index = i - 1
#         prev_point = polygon[prev_index]
#         next_index = (i + 1) % point_count
#         next_point = polygon[next_index]

#         polygon.remove(ear)
#         point_count -= 1
#         triangles.append(((prev_point.x, prev_point.y), (ear.x, ear.y), (next_point.x, next_point.y)))
#         if point_count > 3:
#             prev_prev_point = polygon[prev_index - 1]
#             next_next_index = (i + 1) % point_count
#             next_next_point = polygon[next_next_index]

#             groups = [
#                 (prev_prev_point, prev_point, next_point, polygon),
#                 (prev_point, next_point, next_next_point, polygon),
#             ]
#             for group in groups:
#                 p = group[1]
#                 if _is_ear(*group):
#                     if p not in ear_vertex:
#                         ear_vertex.append(p)
#                 elif p in ear_vertex:
#                     ear_vertex.remove(p)
                    
#     triangles = np.array(
#         [[np.where(original_polygon == p)[0][0] for p in t] for t in triangles])
                    
#     return triangles

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
