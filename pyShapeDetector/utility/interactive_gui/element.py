import warnings
import copy
from typing import Union
from abc import ABC, abstractmethod
import numpy as np

from .editor_app import Editor

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


class Element(ABC):
    """

    Abstract class for encapsulation of elements.

    The inherited classes are:

        ElementPrimitive:
        - For instances of pyShapeDetector.primitive.Primitive

        ElementGeometry:
        - For instances of pyShapeDetector.geometry.Numpy_Geometry

        ElementPointCloud
        - For instances of pyShapeDetector.geometry.Numpy_Geometry.PointCloud

        ElementTriangleMesh
        - For instances of pyShapeDetector.geometry.Numpy_Geometry.TriangleMesh

    Attributes
    ----------
       `name`: Strind ID of object, to add to 3DScene
       `raw`: Either a Primitive instance or a Numpy Geometry
       `drawable`: An instance of Open3D Geometry that can be added to the 3DScene
       `selected`: A flag indicating whether the element is selected or not
       `current`: A flag indicating whether the element is the current one
       `distance_checker`: Either a simplified PointCloud or a Primitive, to
       detect distances to the screen when clicking.
       `brightness`: Color brightness
       `color_original`: The original color when created
       `color`: The current color, either equal to original or a random one
       `is_color_fixed`: Flag indicating whether the color and brightness can be updated

    Methods
    -------
       `_parse_raw`: check if raw element is compatible
       `_get_drawable`: set Open3D geometry
       `_get_distance_checker`: gets element for distance checking
       `_get_bbox`: gets element's bounding box
       `_extract_drawable_color`: return current color of drawable, used at init
       `_update_drawable_color`: sets drawable for new color
       `_get_dimmed_color`: get dimmed version of desired color
       `add_to_scene`: adds element to 3DScene
       `remove_from_scene`: removes elemnt from 3DScene
       `update_on_scene`: updates element on 3DScene
       `update`: updates element

    Static Methods
    --------------
       `get_from_type`: gets an instance of appropriate type

    """

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
    def brightness(self):
        return self._brightness

    @property
    def color_original(self):
        return self._color_original

    @property
    def color(self):
        return self._color

    @property
    def is_color_fixed(self):
        return self._is_color_fixed

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

    def _extract_drawable_color(self):
        drawable = self.drawable
        if hasattr(drawable, "vertex_colors"):
            return np.asarray(drawable.vertex_colors).copy()
        elif hasattr(drawable, "mesh"):
            return np.asarray(drawable.mesh.vertex_colors).copy()
        elif hasattr(drawable, "color"):
            return np.asarray(drawable.color).copy()
        elif hasattr(drawable, "colors"):
            return np.asarray(drawable.colors).copy()
        # else:
        warnings.warn("Could not get color from element {self}.")
        return self._color_original
        # self._color = np.array([0, 0, 0])

    def _update_drawable_color(self, color_input):
        from open3d.utility import Vector3dVector

        color = self._get_dimmed_color(color_input)

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

    def _get_dimmed_color(self, color):
        highlight_ratio = (
            self._editor_instance._get_preference("highlight_ratio") * self._brightness
        )

        brightness = self._brightness
        if self.current:
            brightness += highlight_ratio
        if self.selected:
            brightness += highlight_ratio

        return np.clip(color * brightness, 0, 1)

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
        if self.is_color_fixed:
            return

        paint_selected = self._editor_instance._get_preference("paint_selected")
        self._current = is_current

        if paint_selected and self.selected:
            color = self._editor_instance._settings.get_element_color(True, is_current)
        else:
            color = self._color

        self._update_drawable_color(color)

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
        is_color_fixed: bool = False,
        brightness: float = 1,
    ):
        self._editor_instance = editor_instance
        self._raw = self._parse_raw(raw)
        self._selected = selected
        self._current = current
        self._is_color_fixed = is_color_fixed
        self._brightness = brightness

        self._get_drawable()
        self._get_distance_checker()
        self._color_original = self._extract_drawable_color()  # saving original color

        if editor_instance._get_preference("paint_random"):
            self._color = np.random.random(3)
            self._brightness = editor_instance._get_preference(
                "random_color_brightness"
            )
        else:
            self._color = self.color_original
            self._brightness = editor_instance._get_preference(
                "original_color_brightness"
            )

        self._update_drawable_color(self._color)

    @staticmethod
    def get_from_type(
        editor_instance: Editor,
        raw,
        selected: bool = False,
        current: bool = False,
        is_color_fixed: bool = False,
    ):
        if isinstance(raw, Primitive):
            element_class = ElementPrimitive
        elif TriangleMesh.is_instance_or_open3d(raw):
            element_class = ElementTriangleMesh
        elif PointCloud.is_instance_or_open3d(raw):
            element_class = ElementPointCloud
        elif isinstance(raw, (Numpy_Geometry, Open3d_Geometry)):
            element_class = ElementGeometry
        else:
            raise TypeError("Expected primitive or geometry, got {type(raw)}.")

        return element_class(editor_instance, raw, selected, current, is_color_fixed)


class ElementPrimitive(Element):
    @staticmethod
    def _parse_raw(raw: Primitive):
        if isinstance(raw, Primitive):
            return raw
        else:
            raise ValueError("Expected Primitive instance, got {raw}.")

    def _get_drawable(self):
        self._drawable = self.raw.copy().mesh.as_open3d

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
        self._drawable = copy.copy(self.raw).as_open3d

    def _get_distance_checker(self):
        self._distance_checker = None


class ElementPointCloud(ElementGeometry):
    @staticmethod
    def _parse_raw(raw: PointCloud):
        if PointCloud.is_instance_or_open3d(raw):
            pcd = PointCloud(raw)
            if not pcd.has_normals():
                pcd.estimate_normals()
            return pcd
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
        mesh = copy.copy(self.raw)
        if mesh_show_back_face:
            mesh.add_reverse_triangles()
        return mesh.as_open3d

    def _get_distance_checker(self):
        number_points_distance = self._editor_instance._get_preference(
            "number_points_distance"
        )
        self._distance_checker = PointCloud(
            self.raw.sample_points_uniformly(number_points_distance)
        )
