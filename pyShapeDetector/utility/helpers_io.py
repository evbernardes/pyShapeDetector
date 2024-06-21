#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:59:02 2024

@author: ebernardes
"""
import numpy as np
import shutil


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
        extension = "json"
    else:
        raise ValueError(f"Not implemented for elements of type {type(elems[0])}.")

    for i, elem in enumerate(elems):
        if single:
            out_path = outdir / f"{start}.{extension}"
        else:
            i_corrected = "0" * (num_digits - len(str(i))) + str(i)
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
