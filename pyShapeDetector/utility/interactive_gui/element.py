import warnings
from typing import Union
from abc import ABC, abstractmethod
import numpy as np

from open3d.geometry import Geometry as Open3d_Geometry

from pyShapeDetector.primitives import Primitive

from pyShapeDetector.geometry import (
    PointCloud,
    TriangleMesh,
    OrientedBoundingBox,
    AxisAlignedBoundingBox,
    LineSet,
    Numpy_Geometry,
    equivalent_classes_dict,
)

line_elements = (LineSet, AxisAlignedBoundingBox, OrientedBoundingBox)

from .editor_app import Editor
from .settings import Settings


from .helpers import (
    # extract_element_colors,
    # set_element_colors,
    get_painted_element,
    # get_distance_checker,
)


class Element(ABC):
    @property
    def name(self):
        return str(id(self))

    @property
    def raw(self):
        return self._raw

    @property
    def drawable(self):
        return self._drawable

    @property
    def selected(self):
        return self._selected

    @selected.setter
    def selected(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"'selected' should be a boolean, got {value}.")
        self._selected = value

    @property
    def current(self):
        return self._current

    @property
    def distance_checker(self):
        return self._distance_checker

    @property
    def color(self):
        return self._color

    @staticmethod
    @abstractmethod
    def _parse_raw(raw):
        pass

    @abstractmethod
    def _get_drawable(self):
        pass

    @abstractmethod
    def _get_distance_checker(self):
        pass

    def _get_bbox(self):
        if self.raw is None or isinstance(self.raw, LineSet):
            return None

        BBOX_expand = self._editor_instance._get_preference("BBOX_expand")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                bbox_original = self.raw.get_oriented_bounding_box()
                bbox = OrientedBoundingBox(bbox_original).expanded(BBOX_expand)
            except Exception:
                bbox_original = self.raw.get_axis_aligned_bounding_box()
                bbox = AxisAlignedBoundingBox(bbox_original).expanded(BBOX_expand)

        if self.selected:
            bbox.color = self._editor_instance._get_preference("color_BBOX_selected")
        else:
            bbox.color = self._editor_instance._get_preference("color_BBOX_unselected")

        return bbox

    def _extract_color(self):
        drawable = self.drawable
        if hasattr(drawable, "vertex_colors"):
            self._color = np.asarray(drawable.vertex_colors).copy()
        elif hasattr(drawable, "mesh"):
            self._color = np.asarray(drawable.mesh.vertex_colors).copy()
        elif hasattr(drawable, "color"):
            self._color = np.asarray(drawable.color).copy()
        elif hasattr(drawable, "colors"):
            self._color = np.asarray(drawable.colors).copy()
        else:
            warnings.warn("Could not get color from element {self}.")
        # self._color = np.array([0, 0, 0])

    def _set_drawable_color(self, input_color):
        if input_color is None:
            return

        from open3d.utility import Vector3dVector

        color = np.clip(input_color, 0, 1)

        if hasattr(self.drawable, "vertex_colors"):
            if color.ndim == 2:
                self._drawable.vertex_colors = Vector3dVector(color)
            else:
                self._drawable.paint_uniform_color(color)

        elif hasattr(self.drawable, "colors"):
            if color.ndim == 2:
                self._drawable.colors = Vector3dVector(color)
            else:
                self._drawable.paint_uniform_color(color)

        elif hasattr(self.drawable, "color"):
            self._drawable.color = color

    def paint_element(self):
        paint_random = self._editor_instance._get_preference("paint_random")
        paint_selected = self._editor_instance._get_preference("paint_random")
        brightness = self._editor_instance._get_preference("random_color_brightness")

        if paint_random:
            self._drawable = get_painted_element(
                self._drawable,
                "random",
                brightness,
            )

        elif paint_selected and self.selected:
            color = self._editor_instance._settings.get_element_color(
                True, self.current
            )
            self._drawable = get_painted_element(self._drawable, color)

    def add_to_scene(self):
        # drawable = self.drawable
        if isinstance(self.raw, line_elements):
            material = self._editor_instance.material_line
        else:
            material = self._editor_instance.material_regular
        self._editor_instance._scene.scene.add_geometry(
            self.name, self.drawable, material
        )

    def remove_from_scene(self):
        self._editor_instance._scene.scene.remove_geometry(self.name)

    def update_on_scene(self):
        self.remove_from_scene()
        self.add_to_scene()

    def update(self, is_current: bool, update_scene: bool = True):
        paint_random = self._editor_instance._get_preference("paint_random")
        paint_selected = self._editor_instance._get_preference("paint_random")
        brightness = self._editor_instance._get_preference("random_color_brightness")

        if paint_selected and self.selected:
            color = self._editor_instance._settings.get_element_color(True, is_current)
            self._editor_instance.print_debug(
                f"[Element.update] Painting drawable to color: {color}.",
                require_verbose=True,
            )

        else:
            highlight_offset = brightness * (int(self.selected) + int(is_current))
            color = self.color + highlight_offset

        self._set_drawable_color(color)

        if update_scene:
            self._editor_instance.print_debug(
                "[Element.update] Updating geometry on scene.",
                require_verbose=True,
            )
            self.update_on_scene()

    def __init__(
        self,
        editor_instance: Editor,
        raw,
        selected: bool = False,
        current: bool = False,
    ):
        self._editor_instance = editor_instance
        self._raw = self._parse_raw(raw)
        self._selected = selected
        self._current = current
        # self._drawable = self._get_open3d(raw)
        # self._distance_checker = self._get_distance_checker(raw)
        # self._color = self._extract_color(self._drawable)

        self._get_drawable()
        self._get_distance_checker()
        self._extract_color()

        self.paint_element()

        # if self._editor_instance._get_preference("paint_random"):
        #     self.print_debug(
        #         f"[_insert_elements] Randomly painting element.",
        #         require_verbose=True,
        #     )

    @staticmethod
    def get_from_type(
        editor_instance: Editor, raw, selected: bool = False, current: bool = False
    ):
        if isinstance(raw, Primitive):
            element_class = ElementPrimitive
        elif TriangleMesh.is_instance_or_open3d(raw):
            element_class = ElementTriangleMesh
        elif PointCloud.is_instance_or_open3d(raw):
            element_class = ElementPointCloud
        elif isinstance(raw, [Numpy_Geometry, Open3d_Geometry]):
            element_class = ElementGeometry
        else:
            raise TypeError("Expected primitive or geometry, got {type(raw)}.")

        return element_class(editor_instance, raw, selected, current)


class ElementPrimitive(Element):
    @staticmethod
    def _parse_raw(raw: Primitive):
        if isinstance(raw, Primitive):
            return raw
        else:
            raise ValueError("Expected Primitive instance, got {raw}.")

    def _get_drawable(self):
        self._drawable = self.raw.mesh

    def _get_distance_checker(self):
        self._distance_checker = self.raw


class ElementGeometry(Element):
    @staticmethod
    def _parse_raw(raw: Union[Numpy_Geometry, Open3d_Geometry]):
        if isinstance(raw, Numpy_Geometry):
            return raw
        elif raw.__class__ in equivalent_classes_dict:
            return equivalent_classes_dict[raw.__class__](raw)
        elif isinstance(raw, Open3d_Geometry):
            raise NotImplementedError(
                "Not yet implemented for Open3D geometry of type {raw.__class__}."
            )
        else:
            raise TypeError("Expected Numpy Geometry or Open3D geometry, got {raw}.")

    def _get_drawable(self):
        self._drawable = self.raw.as_open3d

    def _get_distance_checker(self):
        self._distance_checker = None


class ElementPointCloud(ElementGeometry):
    @staticmethod
    def _parse_raw(raw: PointCloud):
        if PointCloud.is_instance_or_open3d(raw):
            return PointCloud(raw)
        else:
            raise ValueError(f"Expected PointCloud instance, got {raw}.")

    def _get_distance_checker(self):
        number_points_distance = self._editor_instance._get_preference(
            "number_points_distance"
        )
        if len(self.raw.points) > number_points_distance:
            ratio = int(len(self.raw.points) / number_points_distance)
            pcd = self.raw.uniform_down_sample(ratio)
        else:
            pcd = self.raw
        self._distance_checker = PointCloud(pcd)


class ElementTriangleMesh(ElementGeometry):
    @staticmethod
    def _parse_raw(raw: TriangleMesh):
        if TriangleMesh.is_instance_or_open3d(raw):
            return TriangleMesh(raw)
        else:
            raise ValueError("Expected TriangleMesh instance, got {raw}.")

    def _get_drawable(self):
        mesh_show_back_face = self._editor_instance._get_preference(
            "mesh_show_back_face"
        )
        mesh = self.raw
        if mesh_show_back_face:
            mesh = mesh.copy()
            mesh.add_reverse_triangles()
        return mesh.as_open3d

    def _get_distance_checker(self):
        number_points_distance = self._editor_instance._get_preference(
            "number_points_distance"
        )
        self._distance_checker = PointCloud(
            self.raw.sample_points_uniformly(number_points_distance)
        )
