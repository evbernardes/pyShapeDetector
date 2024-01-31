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
from open3d.geometry import TriangleMesh
from open3d.utility import Vector3dVector, Vector3iVector

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
