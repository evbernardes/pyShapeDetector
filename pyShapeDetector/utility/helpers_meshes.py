#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions that act on meshes.

Created on Wed Dec  6 14:59:38 2023

@author: ebernardes
"""

import copy
import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay
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

def new_TriangleMesh(vertices, triangles, double_triangles=False):
    """ Fuse TriangleMesh instances into single mesh.
    
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
    mesh = TriangleMesh()

    mesh.vertices = Vector3dVector(
        np.vstack([mesh.vertices for mesh in meshes]))

    triangles = [np.vstack(meshes[0].triangles)]
    for i in range(1, len(meshes)):
        triangles.append(
            np.array(meshes[i].triangles) + len(meshes[i-1].vertices) - 1
            )
        
    mesh.triangles = Vector3iVector(np.vstack(triangles))
    return mesh
            
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

def planes_ressample_and_triangulate(planes, density, radius_ratio=None):
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
        vertices.append(p.sample_points_uniformly(N).points)
    vertices = np.vstack(vertices)
    
    # number_of_points = int(self.surface_area * density)
    # vertices = [p.sample_points_uniformly(int(len(p.inlier_points) * density)).points for p in planes]
    # vertices = [p.sample_points_density(density).points for p in planes]
    # vertices = np.vstack(vertices)
    
    from pyShapeDetector.utility import fuse_shape_groups
    plane_fused = fuse_shape_groups([planes], ignore_extra_data=True)[0]

    projections = plane_fused.get_projections(vertices)
    triangles = Delaunay(projections).simplices
    
    if radius_ratio is not None:
        triangles = remove_big_triangles(vertices, triangles, radius_ratio)

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
    #     [p.sample_points_uniformly(density * len(p.inlier_points)).points for p in planes])
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