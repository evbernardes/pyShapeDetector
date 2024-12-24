#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:59:02 2024

@author: ebernardes
"""
import numpy as np
import warnings
import shutil
import time
import uuid
import tempfile
import tarfile
import multiprocessing
from importlib.util import find_spec
from pathlib import Path

if has_pye57 := find_spec("pye57") is not None:
    import pye57


def _format_int(i, num_digits):
    return "0" * (num_digits - len(str(i))) + str(i)


def _create_uuid_metadata():
    guid = uuid.uuid4().hex
    metadata_content = f"""fileFormatVersion: 2
        guid: {guid}
        timeCreated: {int(time.time())}
        licenseType: Free
        DefaultImporter:
          externalObjects: {{}}
          userData:
          assetBundleName:
          assetBundleVariant:
        """
    return metadata_content


def _color_to_mat_description(color):
    mat_content = f"""Shader "Standard"
    {{
        Properties
        {{
            _Color ("Color", Color) = ({color[0]}, {color[1]}, {color[2]}, 1)
        }}
        SubShader
        {{
            Pass
            {{
                CGPROGRAM
                #pragma vertex vert
                #pragma fragment frag
                ENDCG
            }}
        }}
    }}
    """
    return mat_content


def mesh_to_obj_description(name, mesh, shading="off", mtl="Material", **kwargs):
    obj_content = []
    obj_content.append("o " + name)
    obj_content.append(mesh._get_obj_vertices_triangles())
    obj_content.append("g " + name)
    obj_content.append("usemtl " + str(mtl))
    obj_content.append("s " + str(shading))
    # for key, value in kwargs.items():
    #     obj_content.append(f"{key} {value}")
    return "\n".join(obj_content)


def write_obj(
    path,
    element,
    create_material=True,
    create_metadata=True,
    shading="off",
    mtl="Material",
):
    path = Path(path)

    with open(path.with_suffix(".obj"), "w") as f:
        f.write(element.get_obj_description(shading=shading, mtl=mtl))

    if create_metadata:
        with open(path.with_suffix(".obj.meta"), "w") as f:
            f.write(_create_uuid_metadata())

    if create_material:
        with open(path.with_suffix(".mat"), "w") as f:
            if hasattr(element, "color"):
                color = element.color
            elif hasattr(element, "vertex_colors"):
                color = element.vertex_colors.mean(axis=0)
            else:
                color = (0.0, 0.0, 0.0)
            f.write(_color_to_mat_description(color))

        # Generate GUID and create metadata for the material
        if create_metadata:
            with open(path.with_suffix(".mat").with_suffix(".mat.meta"), "w") as f:
                f.write(_create_uuid_metadata())


def create_unity_package(
    path,
    elements,
    create_material=True,
    create_metadata=True,
    shading="off",
    mtl="Material",
):
    path = Path(path)

    if path.is_dir():
        raise RuntimeError("Path seems to be a directory.")

    if not path.parent.exists():
        raise RuntimeError("Path directory does not exist.")

    if path.suffix != ".unitypackage":
        raise RuntimeError(
            f"Only '.unitypackage' extension is supported, got '{path.suffix}'."
        )

    num = len(elements)
    num_digits = len(str(len(elements)))
    written = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        for i, element in enumerate(elements):
            if not hasattr(element, "get_obj_description"):
                warnings.warn(f"Element {i} from {num} cannot be transformed to obj.")
                continue

            write_obj(
                temp_dir / f"element_{_format_int(i, num_digits)}",
                element,
                create_material=create_material,
                create_metadata=create_metadata,
                shading=shading,
                mtl=mtl,
            )
            written += 1

        if written > 0:
            with tarfile.open(path.with_suffix(".tar"), "w:gz") as tar:
                for file in temp_dir.glob("*"):
                    print(file)
                    print(file.name)
                    tar.add(file, arcname=file.name)
        else:
            warnings.warn("No elements could be transformed, no Unity Package created.")


def check_existance(outdir, remove_dir):
    if outdir.exists() and remove_dir:
        shutil.rmtree(outdir)

    if not outdir.exists():
        outdir.mkdir()


def save_ask(outdir, name=""):
    if name == "":
        val = input(f"Save at {outdir}? (y)es, (N)o, (o)ther name. ").lower()
    else:
        val = input(f"Save {name} at {outdir}? (y)es, (N)o, (o)ther name. ").lower()

    if val == "y":
        return outdir
    elif val == "o":
        return outdir.parent / input("Enter dir:\n")
    return None


def save_elements(
    outdir, elems, start=None, order_func=None, reverse=True, remove_dir=True
):
    check_existance(outdir, remove_dir)
    from pyShapeDetector.geometry import PointCloud

    single = False
    if not isinstance(elems, (list, tuple)):
        elems = [elems]
        single = True

    num_digits = len(str(len(elems)))

    for elem in elems[1:]:
        if not isinstance(elem, type(elems[0])):
            raise ValueError("Elements must all have same size")

    if order_func is not None:
        sorted_arg = np.argsort([order_func(elem) for elem in elems])

        if reverse:
            sorted_arg = sorted_arg[::-1]

        elems = np.array(elems)[sorted_arg].tolist()

    from pyShapeDetector.primitives import Primitive

    if len(elems) == 0:
        print(f"No elements saved to {outdir}.\n")
        return

    if isinstance(elems[0], (PointCloud, PointCloud.__open3d_class__)):
        if start is None:
            start = "pcd"
        extension = "ply"
    elif isinstance(elems[0], Primitive):
        if start is None:
            start = "shape"
        extension = "tar"
    else:
        raise ValueError(f"Not implemented for elements of type {type(elems[0])}.")

    for i, elem in enumerate(elems):
        if single:
            out_path = outdir / f"{start}.{extension}"
        else:
            # i_corrected = "0" * (num_digits - len(str(i))) + str(i)
            i_corrected = _format_int(i, num_digits)
            out_path = outdir / f"{start}_{i_corrected}.{extension}"

        if isinstance(elem, PointCloud):
            elem.write_point_cloud(out_path)
        elif isinstance(elem, Primitive):
            elem.save(out_path)

    print(f"All elements saved to {outdir}.\n")


def ask_and_save(
    outdir, elems, start=None, order_func=None, reverse=True, remove_dir=True
):
    outdir = save_ask(outdir)
    if outdir is not None:
        save_elements(outdir, elems, start, order_func, reverse, remove_dir)


def load_pointcloud_from_e57(filepath, fuse=False, downsample=1):
    from pyShapeDetector.geometry import PointCloud

    filepath = Path(filepath)

    if not has_pye57:
        raise ImportError("Module pye57 not installed.")

    e57 = pye57.E57(filepath.as_posix())

    N = len(e57.data3d)
    manager = multiprocessing.Manager()
    scans = manager.dict()

    # Worker function for multiprocessing
    def _worker(scans, filepath, idx, downsample):
        # Creating e57 object again to avoid parallel errors
        e57 = pye57.E57(filepath.as_posix())

        data = e57.read_scan(idx, transform=True, colors=True)

        points = np.array(
            [
                data["cartesianX"][::downsample],
                data["cartesianY"][::downsample],
                data["cartesianZ"][::downsample],
            ]
        ).T

        colors = (
            np.array(
                [
                    data["colorRed"][::downsample],
                    data["colorGreen"][::downsample],
                    data["colorBlue"][::downsample],
                ]
            ).T
            / 255
        )

        scans[idx] = (points, colors)

    # Create processes and queues
    processes = [
        multiprocessing.Process(target=_worker, args=(scans, filepath, idx, downsample))
        for idx in range(N)
    ]

    # Start all processes
    for process in processes:
        process.start()

    # Wait until all processes are finished
    for process in processes:
        process.join()

    # pcd_full = PointCloud()
    pcds = []

    # for scan in scans.values():
    for i in range(len(scans)):
        points, colors = scans.get(i)
        pcd = PointCloud()
        pcd.points = points
        pcd.colors = colors
        pcds.append(pcd)

    if fuse:
        return PointCloud.fuse_pointclouds(pcds)
    else:
        return pcds
