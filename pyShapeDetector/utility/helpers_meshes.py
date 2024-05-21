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

def get_triangle_points(mesh_or_vertices, triangles=None):
    """ Get positions of each triangle poits..
    
    Parameters
    ----------
    mesh_or_vertices : instance of Open3D.geometry.TriangleMesh or numpy.array
        If mesh_or_vertices is TriangleMesh, use it to get both vertices and
        triangles.
    triangles : np.array, optional
        If mesh_or_vertices is an array of vertices, triangles is the array
        of triangles. Should be set to None if mesh_or_vertices is a 
        TriangleMesh. Default: None.
        
    Returns
    -------
    np.array
        Points.
    """
    vertices, triangles = _get_vertices_triangles(mesh_or_vertices, triangles)
    return vertices[triangles]

def get_triangle_sides(mesh_or_vertices, triangles=None):
    """ Get side lengths of each triangle.
    
    Parameters
    ----------
    mesh_or_vertices : instance of Open3D.geometry.TriangleMesh or numpy.array
        If mesh_or_vertices is TriangleMesh, use it to get both vertices and
        triangles.
    triangles : np.array, optional
        If mesh_or_vertices is an array of vertices, triangles is the array
        of triangles. Should be set to None if mesh_or_vertices is a 
        TriangleMesh. Default: None.
        
    Returns
    -------
    np.array
        Side lenghts of each triangle.
    """
    vertices, triangles = _get_vertices_triangles(mesh_or_vertices, triangles)
    triangle_points = vertices[triangles]
    triangle_points_wrap = np.concatenate([triangle_points, 
                                           triangle_points[:, 0:1, :]], axis=1)
    triangle_points_diff = np.diff(triangle_points_wrap, axis=1)
    return np.linalg.norm(triangle_points_diff, axis=2)

def get_triangle_lines(mesh_or_vertices, triangles=None):
    """ Get pyShapeDetector.primitives.Line instances for every line in every 
    triangle.
    
    Parameters
    ----------
    mesh_or_vertices : instance of Open3D.geometry.TriangleMesh or numpy.array
        If mesh_or_vertices is TriangleMesh, use it to get both vertices and
        triangles.
    triangles : np.array, optional
        If mesh_or_vertices is an array of vertices, triangles is the array
        of triangles. Should be set to None if mesh_or_vertices is a 
        TriangleMesh. Default: None.
        
    Returns
    -------
    list of Line instances
        Three lines for each triangle.
    """
    from pyShapeDetector.primitives import Line
    vertices, triangles = _get_vertices_triangles(mesh_or_vertices, triangles)
    triangle_points = vertices[triangles]
    lines = []
    for p1, p2, p3 in triangle_points:
        lines.append(Line.from_two_points(p1, p2))
        lines.append(Line.from_two_points(p2, p3))
        lines.append(Line.from_two_points(p3, p1))
    return lines

def get_triangle_LineSet(mesh_or_vertices, triangles=None):
    """ Get a Open3D.geomery.LineSet instance containing every line in every 
    triangle.
    
    Parameters
    ----------
    mesh_or_vertices : instance of Open3D.geometry.TriangleMesh or numpy.array
        If mesh_or_vertices is TriangleMesh, use it to get both vertices and
        triangles.
    triangles : np.array, optional
        If mesh_or_vertices is an array of vertices, triangles is the array
        of triangles. Should be set to None if mesh_or_vertices is a 
        TriangleMesh. Default: None.
        
    Returns
    -------
    Open3d.geometry.LineSet
        Three lines for each triangle.
    """
    # from pyShapeDetector.primitives import Line
    from open3d.geometry import LineSet
    from open3d.utility import Vector2iVector
    vertices, triangles = _get_vertices_triangles(mesh_or_vertices, triangles) 
    lineset = LineSet()
    lineset.points = Vector3dVector(vertices)
    lineset.lines = Vector2iVector(
        triangles[:, [0, 1, 1, 2, 2, 0]].reshape((len(triangles) * 3, 2)))
    return lineset

def get_triangle_boundary_indexes(mesh_or_vertices, triangles=None):
    """ Get tuples defining edge lines in boundary of mesh. 
    
    Edges are detected as lines which only belong to a single triangle.
    
    Parameters
    ----------
    mesh_or_vertices : instance of Open3D.geometry.TriangleMesh or numpy.array
        If mesh_or_vertices is TriangleMesh, use it to get both vertices and
        triangles.
    triangles : np.array, optional
        If mesh_or_vertices is an array of vertices, triangles is the array
        of triangles. Should be set to None if mesh_or_vertices is a 
        TriangleMesh. Default: None.
        
    Returns
    -------
    list of tuples
        Each tuple contains the indexes of the two vertices defining each edge
    """
    vertices, triangles = _get_vertices_triangles(mesh_or_vertices, triangles)
    
    # get tuples containing all possible lines
    lines = []
    for triangle in triangles:
        for i in range(3):
            line = [triangle[i], triangle[(i+1)%3]]
            line.sort()
            lines.append(tuple(line))

    occurences = {}
    lines.sort()
    while(len(lines) > 0):
        line = lines.pop()
        count = 1
        
        while(len(lines) > 0):            
            if lines[-1] != line:
                break
            
            count += 1
            line = lines.pop()
            
        occurences[line] = count

    # should be 1, except if triangles are doubled.
    min_value = min(occurences.values())
    boundary_indexes = [k for k, v in occurences.items() if v == min_value]
    
    return boundary_indexes

def get_loop_indexes_from_boundary_indexes(boundary_indexes):
    """ Detect loops in list of tuples.
    
    See: utility.get_triangle_boundary_indexes
    
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
    while(len(boundary_indexes) > 0):
        boundary = []
        edge = boundary_indexes.pop()
        boundary.append(edge)
        
        while(True):        
            edge = boundary[-1]
            
            index = find_tuple_index(boundary_indexes, edge[1])
            if index is None:
                index =  find_tuple_index(boundary_indexes, edge[0])
                
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

def simplify_loop_with_angle(vertices, loop_indexes, angle_colinear, colinear_recursive=True):
    """ For each consecutive line in boundary points, simplify it if they are
    almost colinear.
    
    For example, defining:
        line1 = (vertices[loop_indexes[0]], vertices[loop_indexes[1]])
        line2 = (vertices[loop_indexes[1]], vertices[loop_indexes[2]])
    If angle(line1, line2) < angle_colinear, then loop_indexes[1] is removed
    from loop_indexes.
    
    Parameters
    ----------
    vertices : array_like of shape (N, 3)
        List of all points.
    loop_indexes : list
        Ordered indices defining which points in `vertices` define the loop.
    angle_colinear : float, optional
        Small angle value for assuming two lines are colinear
    colinear_recursive : boolean, optional
        If False, only try to simplify loop once. If True, try to simplify
        it until no more simplification is possible. Default: True.
            
    Returns
    -------
    list
        Simplified loop.
    """
    if angle_colinear < 0:
        raise ValueError("angle_colinear must be a positive value, "
                         f"got {angle_colinear}")
    
    from pyShapeDetector.primitives import Line
    
    cos_angle_colinear = np.cos(angle_colinear)
    vertices = np.array(vertices)
    loop_indexes = np.array(loop_indexes)
    
    count = -1
    while count != 0:
        bounds = vertices[loop_indexes]
        lines = Line.from_bounds(bounds)
        keep = []
        N = len(lines)
        for i in range(N):
            line1 = lines[i]
            line2 = lines[(i + 1) % len(lines)]
            # for bigger angle, smaller dot product/cossine
            keep.append(line1.axis.dot(line2.axis) < cos_angle_colinear)
        loop_indexes = loop_indexes[keep]
        if colinear_recursive:
            count = N - sum(keep)
        else:
            count = 0
        # print(f"N = {N}, keep = {sum(keep)}")
        
    return list(loop_indexes)
    
def get_triangle_perimeters(mesh_or_vertices, triangles=None):
    """ Get perimeter of each triangle.
    
    Parameters
    ----------
    mesh_or_vertices : instance of Open3D.geometry.TriangleMesh or numpy.array
        If mesh_or_vertices is TriangleMesh, use it to get both vertices and
        triangles.
    triangles : np.array, optional
        If mesh_or_vertices is an array of vertices, triangles is the array
        of triangles. Should be set to None if mesh_or_vertices is a 
        TriangleMesh. Default: None.
        
    Returns
    -------
    np.array
        Perimeters defined by triangles.
    """
    sides = get_triangle_sides(mesh_or_vertices, triangles)
    return sides.sum(axis=1)

def get_triangle_surface_areas(mesh_or_vertices, triangles=None):
    """ Get surface area of each triangle.
    
    Parameters
    ----------
    mesh_or_vertices : instance of Open3D.geometry.TriangleMesh or numpy.array
        If mesh_or_vertices is TriangleMesh, use it to get both vertices and
        triangles.
    triangles : np.array, optional
        If mesh_or_vertices is an array of vertices, triangles is the array
        of triangles. Should be set to None if mesh_or_vertices is a 
        TriangleMesh. Default: None.
        
    Returns
    -------
    np.array
        Surface areas defined by triangles.
    """
    
    sides = get_triangle_sides(mesh_or_vertices, triangles)
    s = sides.sum(axis=1) / 2
    return np.sqrt(s * np.prod(s[:, np.newaxis] - sides, axis=1))

def get_triangle_circumradius(mesh_or_vertices, triangles=None):
    """ Fuse TriangleMesh instances into single mesh.
    
    Parameters
    ----------
    mesh_or_vertices : instance of Open3D.geometry.TriangleMesh or numpy.array
        If mesh_or_vertices is TriangleMesh, use it to get both vertices and
        triangles.
    triangles : np.array, optional
        If mesh_or_vertices is an array of vertices, triangles is the array
        of triangles. Should be set to None if mesh_or_vertices is a 
        TriangleMesh. Default: None.
        
    Returns
    -------
    np.array
        Perimeters defined by triangles.
    """
    
    sides = get_triangle_sides(mesh_or_vertices, triangles)
    perimeters = sides.sum(axis=1)
    return sides.prod(axis=1) / np.sqrt(
        perimeters * (perimeters[:, np.newaxis] - 2 * sides).prod(axis=1))

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

def new_TriangleMesh(vertices, triangles, double_triangles=False):
    """ Creates Open3d.geometry.TriangleMesh instance from vertices and triangles.

    Parameters
    ----------
    vertices : numpy.array
        Vertices of the mesh.
    triangles : np.array
        Array containing indices defining the triangles of the mesh.
    double_triangles : boolean, optional
        If True, double the amount of triangles to get a visible mesh from
        both sides. Default: False.
        
    Returns
    -------
    Open3d.geometry.TriangleMesh
        Mesh defined by the inputs.
    """
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(vertices)
    if double_triangles:
        mesh.triangles = Vector3iVector(np.vstack([triangles, triangles[:, ::-1]]))
    else:
        mesh.triangles = Vector3iVector(triangles)
    return mesh

def fuse_vertices_triangles(vertices_list, triangles_list):
    if len(vertices_list) != len(triangles_list):
        raise ValueError(f"{len(vertices_list)} vertices lists and "
                         f"{len(triangles_list)} triangles lists, should be equal.")
        
    vertices = np.vstack(vertices_list)
    triangles = []
    L = 0
    # for i in range(1, len(vertices_list)):
    for i in range(len(vertices_list)):
        triangles.append(np.array(triangles_list[i]) + L)
        L += len(vertices_list[i])
    return vertices, np.vstack(triangles)

def fuse_meshes(meshes):
    """ Fuse TriangleMesh instances into single mesh.
    
    Parameters
    ----------
    meshes : list of meshes
        TriangleMesh instances to be fused
        
    Returns
    -------
    TriangleMesh
        Single triangle mesh
    """
    vertices_list = [np.array(mesh.vertices) for mesh in meshes]
    triangles_list = [np.array(mesh.triangles) for mesh in meshes]
    vertices, triangles = fuse_vertices_triangles(vertices_list, triangles_list)
    return new_TriangleMesh(vertices, triangles)  

def paint_by_type(elements, shapes):
    """ Paint each pointcloud/mesh with a color according to shape type.
    
    Parameters
    ----------
    elements : list of meshes
        Elements to be painted
    shapes : list of shapes
        Shapes definying color of element.
    """
    for element, shape in zip(elements, shapes):
        element.paint_uniform_color(shape.color)

def _sliceplane(mesh, axis, value, direction):
    # axis can be 0,1,2 (which corresponds to x,y,z)
    # value where the plane is on that axis
    # direction can be True or False (True means remove everything that is
    # greater, False means less
    # than)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    new_vertices = list(vertices)
    new_triangles = []

    # (a, b) -> c
    # c refers to index of new vertex that sits at the intersection between a,b
    # and the boundingbox edge
    # a is always inside and b is always outside
    intersection_edges = dict()

    # find axes to compute
    axes_compute = [0,1,2]
    # remove axis that the plane is on
    axes_compute.remove(axis)

    def compute_intersection(vertex_in_index, vertex_out_index):
        vertex_in = vertices[vertex_in_index]
        vertex_out = vertices[vertex_out_index]
        if (vertex_in_index, vertex_out_index) in intersection_edges:
            intersection_index = intersection_edges[(vertex_in_index, vertex_out_index)]
            intersection = new_vertices[intersection_index]
        else:
            intersection = [None, None, None]
            intersection[axis] = value
            const_1 = (value - vertex_in[axis])/(vertex_out[axis] - vertex_in[axis])
            c = axes_compute[0]
            intersection[c] = (const_1 * (vertex_out[c] - vertex_in[c])) + vertex_in[c]
            c = axes_compute[1]
            intersection[c] = (const_1 * (vertex_out[c] - vertex_in[c])) + vertex_in[c]
            assert not (None in intersection)
            # save new vertice and remember that this intersection already added an edge
            new_vertices.append(intersection)
            intersection_index = len(new_vertices) - 1
            intersection_edges[(vertex_in_index, vertex_out_index)] = intersection_index

        return intersection_index

    for t in triangles:
        v1, v2, v3 = t
        if direction:
            v1_out = vertices[v1][axis] > value
            v2_out = vertices[v2][axis] > value
            v3_out = vertices[v3][axis] > value
        else: 
            v1_out = vertices[v1][axis] < value
            v2_out = vertices[v2][axis] < value
            v3_out = vertices[v3][axis] < value

        bool_sum = sum([v1_out, v2_out, v3_out])
        # print(f"{v1_out=}, {v2_out=}, {v3_out=}, {bool_sum=}")

        if bool_sum == 0:
            # triangle completely inside --> add and continue
            new_triangles.append(t)
        elif bool_sum == 3:
            # triangle completely outside --> skip
            continue
        elif bool_sum == 2:
            # two vertices outside 
            # add triangle using both intersections
            vertex_in_index = v1 if (not v1_out) else (v2 if (not v2_out) else v3)
            vertex_out_1_index = v1 if v1_out else (v2 if v2_out else v3)
            vertex_out_2_index = v3 if v3_out else (v2 if v2_out else v1)
            # print(f"{vertex_in_index=}, {vertex_out_1_index=}, {vertex_out_2_index=}")
            # small sanity check if indices sum matches
            assert sum([vertex_in_index, vertex_out_1_index, vertex_out_2_index]) == sum([v1,v2,v3])

            # add new triangle 
            new_triangles.append([vertex_in_index, compute_intersection(vertex_in_index, vertex_out_1_index), 
                compute_intersection(vertex_in_index, vertex_out_2_index)])

        elif bool_sum == 1:
            # one vertice outside
            # add three triangles
            vertex_out_index = v1 if v1_out else (v2 if v2_out else v3)
            vertex_in_1_index = v1 if (not v1_out) else (v2 if (not v2_out) else v3)
            vertex_in_2_index = v3 if (not v3_out) else (v2 if (not v2_out) else v1)
            # print(f"{vertex_out_index=}, {vertex_in_1_index=}, {vertex_in_2_index=}")
            # small sanity check if outdices sum matches
            assert sum([vertex_out_index, vertex_in_1_index, vertex_in_2_index]) == sum([v1,v2,v3])

            new_triangles.append([vertex_in_1_index, compute_intersection(vertex_in_1_index, vertex_out_index), vertex_in_2_index])
            new_triangles.append([compute_intersection(vertex_in_1_index, vertex_out_index), 
                compute_intersection(vertex_in_2_index, vertex_out_index), vertex_in_2_index])

        else:
            assert False

    # TODO remap indices and remove unused 
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(new_vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(new_triangles))
    return mesh

def clean_crop(mesh, axis_aligned_bounding_box):
    """ Crops mesh by slicing facets instead of completely removing them, as
    seen on [1].
    
    Parameters
    ----------
    mesh: TriangleMesh
        Mesh to be cropped
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
    mesh_sliced = copy.copy(mesh)
    for i in range(3):
        min_, max_ = sorted([min_bound[i], max_bound[i]])
        mesh_sliced = _sliceplane(mesh_sliced, i, max_, True)
        mesh_sliced = _sliceplane(mesh_sliced, i, min_, False)
    return mesh_sliced

def remove_big_triangles(vertices, triangles, radius_ratio):
    vertices = np.asarray(vertices)
    triangles = np.asarray(triangles)
    
    from pyShapeDetector.utility import get_triangle_circumradius
    
    circumradiuses = get_triangle_circumradius(vertices, triangles)
    mean_circumradiuses = np.mean(circumradiuses)
    return triangles[circumradiuses < radius_ratio * mean_circumradiuses]

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
    circumradius = get_triangle_circumradius(all_points, all_triangles)
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
    data['current'] = new_TriangleMesh(all_points, triangles, True)
    data['current'].paint_uniform_color(data['color'])
    data['current'].translate(data['translation'])
    data['lines'] = get_triangle_LineSet(data['vertices']+data['translation'], data['triangles'])
    
    def update_geometries(vis):
        global data
        print(f"density: {data['density']}, radius_ratio: {data['radius_ratio']}")
        
        triangles = data['triangles'][
            data['circumradius'] < data['radius_ratio'] * data['mean_circumradius']]
        
        data['old'] = data['current']
        data['current'] = new_TriangleMesh(data['vertices'], triangles, True)
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
            
        data['circumradius'] = get_triangle_circumradius(
            data['vertices'], data['triangles'])
        data['mean_circumradius'] = np.mean(circumradius)
        
        vis.remove_geometry(data['lines'])
        data['lines'] = get_triangle_LineSet(data['vertices']+data['translation'], data['triangles'])
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

def triangulate_earclipping(polygon):
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
    """
    import math
    import sys
    from collections import namedtuple

    polygon = np.array(polygon)
    
    if not len(polygon.shape) == 2 or not polygon.shape[-1] == 2:
        raise ValueError(f"Array of shape (N, 2) expected, got {polygon.shape}.")
    
    original_polygon = copy.copy(polygon)

    Point = namedtuple('Point', ['x', 'y'])

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
        ear = _contains_no_points(p1, p2, p3, polygon) and \
            _is_convex(p1, p2, p3) and \
            _triangle_area(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y) > 0
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
        triangles.append(((prev_point.x, prev_point.y), (ear.x, ear.y), (next_point.x, next_point.y)))
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
        [[np.where(original_polygon == p)[0][0] for p in t] for t in triangles])
                    
    return triangles