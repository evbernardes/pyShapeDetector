# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:35:12 2023

@author: EB000036
"""
import matplotlib.pyplot as plt
import copy
import numpy as np
import random
import open3d as o3d
from open3d.geometry import TriangleMesh
from open3d.utility import Vector3dVector, Vector3iVector
from open3d.visualization import draw_geometries
from pyShapeDetector.primitives import Plane, Cylinder, Sphere
import yaml
from pathlib import Path
from scipy.spatial.transform import Rotation
import time
from sklearn.neighbors import KDTree

color_red = [0.87843137, 0.16078431, 0.11372549]    # E0291D
color_blue = [0.11372549, 0.14509804, 0.87843137]   # 1D25E0
color_yellow = [1., 0.90196078, 0.]                 # FFE600
color_gray = [0.6, 0.6, 0.6]

def normalise(v):
    return v / np.linalg.norm(v)

def draw(objs):
    if not isinstance(objs, list):
        objs = [objs]
        
    for i in range(len(objs)):
        if isinstance(objs[i], o3d.cuda.pybind.t.geometry.PointCloud):
            objs[i] = objs[i].to_legacy()
            
    draw_geometries(objs)
    
def draw_two_colomns(objs_left, objs_right, dist=5,
                     lookat=None, up=None, front=None, zoom=None):
    
    if type(objs_left) != list:
        objs_left = [objs_left]
    objs_left = copy.deepcopy(objs_left)
    if type(objs_right) != list:
        objs_right = [objs_right]
    objs_right = copy.deepcopy(objs_right)
    
    try:
        translate = 0.5 * dist * np.cross(up, front)
    except:
        translate = np.array([0, 0.5 * dist, 0])
        
    for i in range(len(objs_left)):
        objs_left[i].translate(-translate)
    for i in range(len(objs_right)):
        objs_right[i].translate(translate)
        
    if zoom is None:
        draw_geometries(objs_right+objs_left)
    else:
        draw_geometries(objs_right+objs_left,
                        lookat=lookat,
                        up=up,
                        front=front,
                        zoom=zoom
                        )
    
    
def create_square_plane(normal, center, size=1.0):
    " Creates a square plane"
    # center = np.mean(np.asarray(points), axis=0)

    if list(normal) == [0, 0, 1]: 
        v1 = normalise(np.cross([0, 1, 0], normal))
    else:
        v1 = normalise(np.cross([0, 0, 1], normal))
    v2 = normalise(np.cross(v1, normal))

    vertices = np.vstack([
        center + v1 * size,
        center + v2 * size,
        center - v1 * size,
        center - v2 * size])

    triangles = Vector3iVector(np.array([
        [0, 1, 2], 
        [2, 1, 0],
        [0, 2, 3],
        [3, 2, 0]]))
    vertices = Vector3dVector(vertices)

    return TriangleMesh(vertices, triangles)

def get_random_spheres(num_spheres, translate_lim, radius_lim, num_points=1000, noise=0):
    pcds = []
    models = []
    for idx in range(num_spheres):
        
        radius = random.uniform(*radius_lim)
        mesh = TriangleMesh.create_sphere(radius=radius)
        
        translation = translate_lim[0] + \
            (translate_lim[1]-translate_lim[0]) * np.random.random(3)
            
        mesh.translate(translation)
            
        pcds.append(mesh.sample_points_uniformly(num_points))
        models.append(np.array(translation+[radius]))
    return pcds, models

def get_random_planes(num_planes, translate_lim, size_lim, num_points=1000, noise=0):
    pcds = []
    models = []
    for idx in range(num_planes):
        
        normal = normalise(np.random.random(3))
        dist = random.uniform(*translate_lim)
        model = list(normal) + [dist]
        center = normalise(np.random.random(3)) * dist
        
        mesh = create_square_plane(normal, center, size=random.uniform(*size_lim))
            
        pcds.append(mesh.sample_points_uniformly(num_points))
        models.append(model)
    return pcds, models

def get_random_cylinders(num_cylinders, translate_lim, radius_lim, height_lim, num_points=1000, noise=0):
    pcds = []
    models = []
    for idx in range(num_cylinders):
        
        height = random.uniform(*height_lim)
        axis = normalise(np.random.random(3))
        radius = random.uniform(*radius_lim)
        center = translate_lim[0] + \
            (translate_lim[1]-translate_lim[0]) * np.random.random(3)
        model = list(center) + list(axis * height) + [radius]
        shape = Cylinder(model)
        pcd = shape.get_mesh().sample_points_uniformly(num_points)

        pcds.append(pcd)
        models.append(shape.model)
    return pcds, models

def center_pointcloud(source):
    points = source.points - np.mean(source.points, axis=0)
    source.points = o3d.utility.Vector3dVector(points)

def parse_transformation(data, scale=True):
    
    if isinstance(data, Path):
        data = yaml.safe_load(data.read_text())
        
    transformation = np.eye(4)
          
    if data['rotation_format'] == 'quaternion_xyzw':
        transformation[:3, :3] = Rotation.from_quat(data['rotation']).as_matrix()
    elif data['rotation_format'] == 'rotation_matrix':
        transformation[:3, :3] = data['rotation']
    
    if scale and 'scale' in data:
        transformation[:3, :3] *= data['scale']
    transformation[:3, -1] = data['translation']
    return transformation

def draw_registration_result_2(source, template, 
                             transformation1=np.eye(4), transformation2=np.eye(4),
                             window_name='Open3D', camera = {
                                 'lookat': [0, +10, -10],
                                 'up': [0, 0.75, 1],
                                 'front': [0, 1, -0.75],
                                 'zoom': 1,
                                 'space': [10, 0, 0],
                                 }):

    source_temp1 = copy.deepcopy(source)
    source_temp2 = copy.deepcopy(source)
    template_temp1 = copy.deepcopy(template)
    template_temp2 = copy.deepcopy(template)
    
    transformation_source_temp1 = copy.deepcopy(transformation1)
    transformation_source_temp2 = copy.deepcopy(transformation2)
    
    transformation_source_temp1[:3, -1] -= camera['space'] 
    transformation_source_temp2[:3, -1] += camera['space']
    
    transformation_template_temp1 = np.eye(4)
    transformation_template_temp1[:3, -1] -= camera['space'] 
    transformation_template_temp2 = np.eye(4)
    transformation_template_temp2[:3, -1] += camera['space'] 
    
    source_temp1.transform(transformation_source_temp1)
    source_temp2.transform(transformation_source_temp2)
    template_temp1.transform(transformation_template_temp1)
    template_temp2.transform(transformation_template_temp2)
    
    source_temp1.paint_uniform_color([1, 0.706, 0])
    source_temp2.paint_uniform_color([0.706, 1, 0])
    template_temp1.paint_uniform_color([0, 0.651, 0.929])
    template_temp2.paint_uniform_color([0, 0.651, 0.929])
    
    o3d.visualization.draw_geometries([source_temp1, source_temp2, template_temp1, template_temp2],
                                      window_name=window_name,
                                      lookat = camera['lookat'],
                                      up = camera['up'],
                                      front = camera['front'],
                                      zoom = camera['zoom'])
    
# def draw_registration_result_2(source, template, transformation=np.eye(4), window_name='Open3D'):

#     source_temp = copy.deepcopy(source)
#     template_temp = copy.deepcopy(template)
#     source_temp.transform(transformation)
    
#     source_temp.paint_uniform_color([1, 0.706, 0])
#     template_temp.paint_uniform_color([0, 0.651, 0.929])
    
#     if isinstance(template_temp, o3d.cpu.pybind.geometry.TriangleMesh):
#         template_temp.compute_vertex_normals()
    
#     o3d.visualization.draw_geometries([source_temp, template_temp],
#                                       window_name=window_name)
    
def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def load_dataset(path_source, path_target):
    # print(":: Loading raw source data, CAD model and ground truth data.")

    path_data = path_source.with_suffix('.yaml')
    data = yaml.safe_load(path_data.read_text())

    source = o3d.io.read_point_cloud(path_source.as_posix())
    source.scale(data['scale'], [0, 0, 0])

    target_mesh = o3d.io.read_triangle_mesh(
        (path_target / data['model']).as_posix())

    target = target_mesh.sample_points_uniformly(
        number_of_points=data['sampling_density_ratio'] * len(source.points))
    
    transformation = parse_transformation(data, scale=False)
    
    return source, target, data, transformation

# def prepare_dataset(voxel_size):
#     print(":: Load two point clouds and disturb initial pose.")

#     demo_icp_pcds = o3d.data.DemoICPPointClouds()
#     source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
#     target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
#     trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
#                              [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
#     source.transform(trans_init)
#     # draw_registration_result(source, target, np.identity(4))

#     source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
#     target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
#     # draw_registration_result(source_down, target_down, np.identity(4))
#     return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_ransac_global_registration(source_down, target_down, source_fpfh,
                                       target_fpfh, voxel_size, 
                                       distance_voxel_ratio = 1.5):
    
    distance_threshold = voxel_size * distance_voxel_ratio
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, result_global, voxel_size, 
                        distance_voxel_ratio = 0.4):
    
    distance_threshold = voxel_size * distance_voxel_ratio
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_global.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size, 
                                     distance_voxel_ratio = 0.5):
    
    distance_threshold = voxel_size * distance_voxel_ratio
    # print(":: Apply fast global registration with distance threshold %.3f" \
            # % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def execute_ransac_and_icp(source, target, voxel_size = 0.5):
    
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    
    # Global registration with RANSAC
    result = execute_ransac_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    
    # Local refinement with ICP
    result = refine_registration(source, target, result, voxel_size)
    
    return result, [source_down, target_down, source_fpfh, target_fpfh]

def get_transformation_results(path_source, path_cads):
    source, target, data, transformation = load_dataset(path_source, 
                                                        path_cads)
    
    #%% Downsample and prepare
    start = time.time()
    voxel_size = 0.5
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    time_preprocess = (time.time() - start)
    
    start = time.time()
    # Global registration with RANSAC
    result_ransac = execute_ransac_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    time_ransac = (time.time() - start)
    
    # Local refinement with ICP
    start = time.time()
    result_ransacicp = refine_registration(source, target, result_ransac, voxel_size)
    time_icp = (time.time() - start)
    
    # Global registration with FGR
    start = time.time()
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                   source_fpfh, target_fpfh,
                                                   voxel_size)
    time_fast = (time.time() - start)
    
    results = {
        'source': source,
        'target': target,
        'transformation_ransacicp': result_ransacicp.transformation,
        'transformation_fast': result_fast.transformation,
        'transformation_groundtruth': transformation,
        'time_preprocess': time_preprocess,
        'time_ransac': time_ransac,
        'time_icp': time_icp,
        'time_fast': time_fast,
        }
    
    return results


