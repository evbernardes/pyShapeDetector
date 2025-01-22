#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:27:03 2024

@author: ebernardes
"""
import copy
import numpy as np
from typing import Union, TYPE_CHECKING
from open3d.geometry import OrientedBoundingBox as open3d_OrientedBoundingBox

from pyShapeDetector.utility import _set_and_check_3d_array
from .numpy_geometry import link_to_open3d_geometry, Numpy_Geometry
from .axis_aligned_bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from pyShapeDetector.primitives import PlaneBounded
    from pyShapeDetector.geometry import LineSet


@link_to_open3d_geometry(open3d_OrientedBoundingBox)
class OrientedBoundingBox(Numpy_Geometry):
    """
    OrientedBoundingBox class that uses Open3D.geometry.OrientedBoundingBox internally.

    Almost every method and property are automatically copied and decorated.

    Extra Methods
    -------------
    from_multiple_elements
    contains_points
    expanded
    split
    as_planes
    as_lineset
    """

    @classmethod
    def from_multiple_elements(
        cls, elements: list[Numpy_Geometry]
    ) -> "OrientedBoundingBox":
        """
        Gets minimal oriented bounding box from all elements.

        Each element in list must have a "get_oriented_bounding_box"
        method implemented.

        Parameters
        ----------

        elements : list
            List containing Open3D geometries, Numpy Geometries or primitives

        Returns
        -------
        OrientedBoundingBox
        """
        if isinstance(elements, list):
            bboxes = [element.get_oriented_bounding_box() for element in elements]
            if len(bboxes) == 1:
                return bboxes[0]
            points = np.vstack([bbox.get_box_points() for bbox in bboxes])
            return cls.create_from_points(points)
        else:
            return cls(elements.get_oriented_bounding_box())

    def contains_points(
        self, points: np.ndarray, inclusive: bool = True, eps: float = 1e-5
    ) -> np.ndarray[bool]:
        """
        Check which points are inside of the bounding box.

        Parameters
        ----------

        points : N x 3 array
            N input points
        inclusive : bool, optional
            If True, includes points exactly in the border. Default: True.

        Returns
        -------
        Numpy array
            Boolean values
        """
        points = _set_and_check_3d_array(points, name="points")
        points = (points - self.center) @ self.R
        extent = self.extent / 2 + eps
        aabb = AxisAlignedBoundingBox(-extent, +extent)

        return aabb.contains_points(points, inclusive=inclusive)

    def expanded(self, slack: float = 0) -> "OrientedBoundingBox":
        """Return expanded version with bounds expanded in all directions.

        Parameters
        ----------
        slack : float, optional
            Expand bounding box in all directions, useful for testing purposes.
            Default: 0.

        Returns
        -------
        AxisAlignedBoundingBox
        """
        obb = OrientedBoundingBox(
            center=self.center, R=self.R, extent=self.extent + slack
        )
        obb.color = self.color
        return obb

    def split(
        self, num_boxes: int, dim: Union[None, int] = None
    ) -> list["OrientedBoundingBox"]:
        """Separates bounding boxes into multiple sub-boxes.

        Parameters
        ----------
        num_boxes : int
            Number of sub-boxes.
        dim : int, optional
            Dimension that should be divided. If not given, will be chosen as the
            largest dimension. Default: None.

        Returns
        -------
        list
            Divided boxes
        """
        aabb = AxisAlignedBoundingBox(
            self.center - self.extent / 2, self.center + self.extent / 2
        )
        aabb.color = (0.0, 0.0, 1.0)

        bboxes = []
        for bbox in aabb.split(num_boxes, dim=dim):
            center = (bbox.max_bound + bbox.min_bound) / 2
            extent = bbox.max_bound - bbox.min_bound
            new_bbox = OrientedBoundingBox(center=center, R=np.eye(3), extent=extent)
            center = aabb.get_center()
            new_bbox.rotate(self.R, center=self.center)
            bboxes.append(new_bbox)

        return bboxes

    def as_planes(self) -> list["PlaneBounded"]:
        """
        Get the bounded planes for the faces of the bounding box.

        Returns
        -------
            list of planes
        """
        from pyShapeDetector.primitives import PlaneBounded

        # Face vertices (relative to the center and extent)
        face_offsets = [
            [+1, +1, +1],
            [+1, +1, -1],
            [+1, -1, +1],
            [+1, -1, -1],
            [-1, +1, +1],
            [-1, +1, -1],
            [-1, -1, +1],
            [-1, -1, -1],
        ]

        center = self.center
        half_extent = self.extent / 2.0
        axes = self.R.T  # Local axes directions

        vertices = center + [
            np.dot(offset * half_extent, axes) for offset in face_offsets
        ]

        # Define faces (each with 4 vertices)
        faces_indices = [
            [0, 1, 3, 2],  # Front face
            [4, 5, 7, 6],  # Back face
            [0, 1, 5, 4],  # Top face
            [2, 3, 7, 6],  # Bottom face
            [0, 2, 6, 4],  # Left face
            [1, 3, 7, 5],  # Right face
        ]

        planes = []

        for indices in faces_indices:
            plane_face = PlaneBounded.fit(vertices[indices])
            planes.append(plane_face)

        return planes

    def as_lineset(self) -> "LineSet":
        """
        Convert bounding box to lineset instance.

        Returns
        -------
            LineSet
        """
        from pyShapeDetector.geometry import LineSet

        center = self.center
        half_extent = self.extent / 2.0
        axes = self.R.T  # Local axes directions

        # Face vertices (relative to the center and extent)
        face_offsets = [
            [+1, +1, +1],
            [+1, +1, -1],
            [+1, -1, +1],
            [+1, -1, -1],
            [-1, +1, +1],
            [-1, +1, -1],
            [-1, -1, +1],
            [-1, -1, -1],
        ]

        # Compute the vertices of the bounding box
        vertices = center + [
            np.dot(offset * half_extent, axes) for offset in face_offsets
        ]

        # Define edges (pairs of vertices)
        lines = [
            [0, 1],
            [1, 3],
            [3, 2],
            [2, 0],  # Front face edges
            [4, 5],
            [5, 7],
            [7, 6],
            [6, 4],  # Back face edges
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],  # Connecting edges between front and back
        ]

        lineset = LineSet(vertices, lines)
        lineset.color = self.color
        return lineset
