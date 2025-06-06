import warnings
import traceback
import copy
from typing import Union, TypeVar, Generic, TYPE_CHECKING
from abc import ABC, abstractmethod
import numpy as np

from open3d.visualization.rendering import Scene, Open3DScene
from open3d.geometry import Geometry as Open3d_Geometry
from open3d.core import Tensor
from open3d.t.geometry import PointCloud as TensorPointCloud
from open3d.utility import Vector3dVector

from pyShapeDetector.primitives import Primitive, Plane
from pyShapeDetector import geometry

if TYPE_CHECKING:
    from pyShapeDetector.editor.settings import Settings

ELEMENT_TYPE = Union[Primitive, geometry.Numpy_Geometry]
T = TypeVar("T", Primitive, geometry.Numpy_Geometry)

ELEMENT_LINE_CLASSES = (
    geometry.LineSet,
    geometry.AxisAlignedBoundingBox,
    geometry.OrientedBoundingBox,
)

DEFAULT_POINTCLOUD_COLOR = (0.3, 0.3, 0.3)


def _is_point_in_convex_region(screen_point, screen_plane, points):
    """Check if mouse-click was done inside of the element actual region."""

    points = np.asarray(points)
    if len(points) < 3:
        return False

    plane_bounded = screen_plane.get_bounded_plane(points, convex=True)
    return plane_bounded.contains_projections(screen_point)


class Element(ABC, Generic[T]):
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
       `is_selected`: A flag indicating whether the element is selected or not
       `current`: A flag indicating whether the element is the current one
       `distance_checker`: Either a simplified PointCloud or a Primitive, to
       detect distances to the screen when clicking.
       `brightness`: Color brightness
       `color_original`: The original color when created
       `color`: The current color, either equal to original or a random one
       `is_color_fixed`: Flag indicating whether the color and brightness can be updated
       `is_hidden`: Flat indicating whether it should be shown.

    Methods
    -------
       `_parse_raw`: check if raw element is compatible
       `_set_extra_elements`: set drawable Open3D geometry and other elements if needed
       `_delete_extra_elements`: remove extra elements from memory
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

    _drawable: Open3d_Geometry
    _color_original: np.ndarray
    _color: np.ndarray

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.raw.__class__.__name__}]"

    @property
    def name(self) -> str:
        return str(id(self))

    @property
    def raw(self) -> T:
        return self._raw

    @property
    def drawable(self) -> Union[geometry.TriangleMesh, geometry.PointCloud]:
        return self._drawable

    @property
    def is_selected(self) -> bool:
        return self._is_selected

    @is_selected.setter
    def is_selected(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"'selected' should be a boolean, got {value}.")
        self._is_selected = value

    @property
    def current(self) -> bool:
        return self._current

    @property
    def brightness(self) -> float:
        return self._brightness

    @property
    def color_original(self) -> np.ndarray:
        return self._color_original

    @property
    def color(self) -> np.ndarray:
        return self._color

    @property
    def is_color_fixed(self) -> bool:
        return self._is_color_fixed

    @property
    def scene(self) -> Union[Open3DScene, None]:
        return self._scene

    @property
    def is_hidden(self) -> bool:
        return self._is_hidden

    @is_hidden.setter
    def is_hidden(self, value: bool):
        self._is_hidden = value
        if self.scene is None:
            # app did not start yet
            return
        if value:
            self.remove_from_scene()
        else:
            self.add_to_scene()

    @staticmethod
    @abstractmethod
    def _parse_raw(raw):
        pass

    @abstractmethod
    def _set_extra_elements(self):
        pass

    def _set_distance_checker(self):
        pass

    def _delete_extra_elements(self):
        del self._drawable
        del self._color_original
        del self._color

    @abstractmethod
    def _get_distance_to_screen_point(
        self, screen_point: np.ndarray, screen_plane: Plane
    ):
        pass

    def _get_bbox(self):
        eps = 1e-5

        if self.raw is None or isinstance(self.raw, ELEMENT_LINE_CLASSES):
            return None

        bbox_expand_ratio = self._settings.get_setting("bbox_expand_percentage") / 100.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                bbox = self.raw.get_oriented_bounding_box()
            except Exception:
                warnings.warn(
                    "Could not get oriented bounding box, getting axis aligned box instead."
                )
                bbox = self.raw.get_axis_aligned_bounding_box()
                traceback.print_exc()

            try:
                if bbox.volume() < eps:
                    bbox = bbox.expanded(eps)
                    self._settings.print_debug(
                        "Bounding box with zero volume found, correcting...",
                        require_verbose=True,
                    )

                bbox_expand = bbox_expand_ratio * (bbox.volume() ** (1 / 3))
                bbox = bbox.expanded(bbox_expand)
            except Exception:
                warnings.warn("Could not extend bounding box.")
                traceback.print_exc()

        if self.is_selected:
            bbox.color = self._settings.get_setting("color_bbox_selected")
        else:
            bbox.color = self._settings.get_setting("color_bbox_unselected")

        self._settings.print_debug(
            f"Final printed {bbox.__class__.__name__}: volume={bbox.volume()}, extend={bbox.extent}",
            require_verbose=True,
        )

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
        warnings.warn(f"Could not get color from element {self}.")
        return self._color_original
        # self._color = np.array([0, 0, 0])

    def _update_drawable_color(self, color_input):
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
            self._settings.get_setting("highlight_ratio") * self._brightness
        )

        brightness = self._brightness
        if self.current:
            brightness += highlight_ratio
        if self.is_selected:
            brightness += highlight_ratio

        return np.clip(color * brightness, 0, 1)

    def add_to_scene(self, scene: Union[Open3DScene, None] = None):
        if scene is None:
            # app did not start
            pass

        elif isinstance(scene, Open3DScene):
            self._scene = scene
        else:
            raise TypeError(f"Expected Open3DScene, got {scene}.")

        if self.scene is None:
            raise RuntimeError("No scene was set for element!")

        if isinstance(self.raw, ELEMENT_LINE_CLASSES):
            material = self._settings.get_material("line")
        else:
            material = self._settings.get_material("regular")

        if not self.is_hidden:
            self.scene.add_geometry(self.name, self.drawable, material)

    def remove_from_scene(self):
        self.scene.remove_geometry(self.name)

    def update_on_scene(self, reset: bool = False):
        self.remove_from_scene()

        if not self.is_hidden:
            self.add_to_scene()

    def transform(self, transformation_matrix):
        self.raw.transform(transformation_matrix)
        self.drawable.transform(transformation_matrix)
        self.update_on_scene()

    def update(self, is_current: bool, update_scene: bool = True, reset: bool = False):
        if self.is_color_fixed:
            return

        paint_selected = self._settings.get_setting("paint_selected")
        self._current = is_current

        if paint_selected and self.is_selected:
            color = self._settings.get_element_color(True, is_current)
        else:
            color = self._color

        self._update_drawable_color(color)

        if update_scene:
            self._settings.print_debug(
                "[Element.update] Updating geometry on scene.",
                require_verbose=True,
            )
            self.update_on_scene(reset=reset)

    def _init_colors(self):
        self._color_original = self._extract_drawable_color()  # saving original color

        if self._settings.get_setting("paint_random"):
            self._color = np.random.random(3)
            self._brightness = self._settings.get_setting("random_color_brightness")
        else:
            self._color = self.color_original
            self._brightness = self._settings.get_setting("original_color_brightness")

    def __init__(
        self,
        settings: "Settings",
        raw,
        is_selected: bool = False,
        current: bool = False,
        is_color_fixed: bool = False,
        brightness: float = 1,
        is_hidden: bool = False,
    ):
        self._settings: Settings = settings
        self._raw = self._parse_raw(raw)
        self._is_selected: bool = is_selected
        self._current: bool = current
        self._is_color_fixed: bool = is_color_fixed
        self._brightness: float = brightness
        self._is_hidden: bool = is_hidden
        self._scene: Union[Open3DScene, None] = None

        self._set_extra_elements()
        self._init_colors()

        self._update_drawable_color(self._color)

    @staticmethod
    def get_from_type(
        settings: "Settings",
        raw,
        is_selected: bool = False,
        current: bool = False,
        is_color_fixed: bool = False,
    ) -> "Element":
        if isinstance(raw, Primitive):
            element_class = ElementPrimitive
        elif geometry.TriangleMesh.is_instance_or_open3d(raw):
            element_class = ElementTriangleMesh
        elif geometry.PointCloud.is_instance_or_open3d(raw):
            if len(raw.points) == 0:
                warnings.warn("Tried to create an empty PointCloud, ignoring.")
                return None
            element_class = ElementPointCloud
        elif isinstance(raw, (geometry.Numpy_Geometry, Open3d_Geometry)):
            element_class = ElementGeometry
        else:
            raise TypeError(f"Expected primitive or geometry, got {type(raw)}.")

        return element_class(settings, raw, is_selected, current, is_color_fixed)


class ElementPrimitive(Element[Primitive]):
    def _get_distance_to_screen_point(
        self, screen_point: np.ndarray, screen_plane: Plane
    ):
        points = self.raw.mesh.vertices
        if not _is_point_in_convex_region(screen_point, screen_plane, points):
            return np.inf

        return self.raw.get_distances(screen_point)

    @staticmethod
    def _parse_raw(raw: Primitive):
        if isinstance(raw, Primitive):
            return raw
        else:
            raise ValueError("Expected Primitive instance, got {raw}.")

    def _set_extra_elements(self):
        # self._drawable = self.raw.copy().mesh.as_open3d
        self._drawable = copy.copy(self.raw.mesh).as_open3d


class ElementGeometry(Element[geometry.Numpy_Geometry], Generic[T]):
    def _delete_extra_elements(self):
        super()._delete_extra_elements()
        del self._distance_checker

    @staticmethod
    def _parse_raw(raw: Union[geometry.Numpy_Geometry, Open3d_Geometry]):
        if isinstance(raw, geometry.Numpy_Geometry):
            return raw
        elif raw.__class__ in geometry.equivalent_classes_dict:
            return geometry.equivalent_classes_dict[raw.__class__](raw)
        elif isinstance(raw, Open3d_Geometry):
            raise NotImplementedError(
                "Not yet implemented for Open3D geometry of type {raw.__class__}."
            )
        else:
            raise TypeError("Expected Numpy Geometry or Open3D geometry, got {raw}.")

    def _set_extra_elements(self):
        self._drawable = copy.copy(self.raw).as_open3d
        self._set_distance_checker()

    def _set_distance_checker(self):
        if isinstance(self.raw, ELEMENT_LINE_CLASSES):
            self._distance_checker = None
            return

        try:
            bbox = self.raw.get_oriented_bounding_box()
            self._distance_checker = geometry.PointCloud(bbox.points)
        except Exception:
            self._distance_checker = None
            print(f"Could not get distance checker for {self}.")
            # traceback.print_exc()

    def _get_distance_to_screen_point(
        self, screen_point: np.ndarray, screen_plane: Plane
    ):
        if self._distance_checker is None:
            return np.inf

        if not _is_point_in_convex_region(
            screen_point, screen_plane, self._distance_checker.points
        ):
            return np.inf

        return self._distance_checker.get_distances(screen_point)


class ElementPointCloud(ElementGeometry[geometry.PointCloud]):
    @staticmethod
    def _parse_raw(raw: geometry.PointCloud):
        if geometry.PointCloud.is_instance_or_open3d(raw):
            pcd = geometry.PointCloud(raw)
            if not pcd.has_normals():
                pcd.estimate_normals()
            return pcd
        else:
            raise ValueError(f"Expected PointCloud instance, got {raw}.")

    def _extract_drawable_color(self):
        PCD_use_Tensor = self._settings.get_setting("PCD_use_Tensor")
        if not PCD_use_Tensor:
            return super()._extract_drawable_color()

        try:
            return self.drawable.point.colors.numpy()
        except KeyError:
            return np.empty((0, 3))

    def _update_drawable_color(self, color_input):
        PCD_use_Tensor = self._settings.get_setting("PCD_use_Tensor")
        if not PCD_use_Tensor:
            super()._update_drawable_color(color_input)
            return

        drawable = self.drawable
        color = self._get_dimmed_color(color_input)

        # drawable.point.colors = color

        self._settings.print_debug(
            f"Painting Tensor-based PointCloud. Input color shape: {color.shape}. "
            f"Prevous color shape: {drawable.point.colors.shape}.",
            require_verbose=True,
        )

        try:
            if color.shape == (3,):
                drawable.paint_uniform_color(color.astype("float32"))
                self._colors_updated = False
            elif color.shape == drawable.point.colors.shape:
                drawable.point.colors = Tensor(color.astype("float32"))
                self._colors_updated = False
            else:
                warnings.warn("Could not paint Tensor-based drawable PointCloud.")
        except Exception:
            warnings.warn(f"Could not update color of PointCloud element {self.raw}.")
            traceback.print_exc()

    def _set_distance_checker(self):
        """Creates small pointcloud for distance checking"""
        number_points_distance = self._settings.get_setting("number_points_distance")
        if len(self.raw.points) > number_points_distance:
            ratio = int(len(self.raw.points) / number_points_distance)
            pcd = self.raw.uniform_down_sample(ratio)
        else:
            pcd = self.raw
        self._distance_checker = geometry.PointCloud(pcd)

    def _set_extra_elements(self):
        settings = self._settings
        drawable = copy.deepcopy(self.raw).as_open3d
        downsample = settings.get_setting("PCD_downsample_when_drawing")

        PCD_use_Tensor = settings.get_setting("PCD_use_Tensor")

        if downsample:
            max_points = settings.get_setting("PCD_max_points")
            if len(drawable.points) > max_points:
                downsample_mode = settings.get_setting("PCD_downsample_mode")
                if downsample_mode == "Uniform":
                    ratio = int(len(drawable.points) / max_points)
                    drawable = drawable.uniform_down_sample(ratio)
                elif downsample_mode == "Voxel":
                    voxel_ratio = settings.get_setting("PCD_downsample_voxel_ratio")
                    PointCloud_density = settings.get_setting("PointCloud_density")
                    drawable = drawable.voxel_down_sample(
                        voxel_ratio * PointCloud_density
                    )

        has_color = len(drawable.colors) > 0
        if not has_color:
            print(
                "PointCloud does not have color, painting it with "
                f"default color {DEFAULT_POINTCLOUD_COLOR}."
            )
            drawable.paint_uniform_color(DEFAULT_POINTCLOUD_COLOR)

        if PCD_use_Tensor:
            self._drawable = TensorPointCloud.from_legacy(drawable)
        else:
            self._drawable = drawable

        # self._drawable = TensorPointCloud.from_legacy(self.raw.as_open3d)
        self._colors_updated = True
        self._normals_updated = True
        self._points_updated = True

        self._set_distance_checker()

    def update_on_scene(self, reset: bool = False):
        if reset or not self._settings.get_setting("PCD_use_Tensor"):
            super().update_on_scene()
            return

        if not self._colors_updated:
            self.scene.scene.update_geometry(
                self.name, self.drawable, Scene.UPDATE_COLORS_FLAG
            )
            self._colors_updated = True
        if not self._normals_updated:
            self.scene.scene.update_geometry(
                self.name, self.drawable, Scene.UPDATE_NORMALS_FLAG
            )
            self._normals_updated = True
        if not self._points_updated:
            self.scene.scene.update_geometry(
                self.name, self.drawable, Scene.UPDATE_POINTS_FLAG
            )
            self._points_updated = True

    def transform(self, transformation_matrix):
        self._points_updated = False
        # self._normals_updated = False
        super().transform(transformation_matrix)


class ElementTriangleMesh(ElementGeometry[geometry.TriangleMesh]):
    @staticmethod
    def _parse_raw(raw: geometry.TriangleMesh):
        if geometry.TriangleMesh.is_instance_or_open3d(raw):
            return geometry.TriangleMesh(raw)
        else:
            raise ValueError("Expected TriangleMesh instance, got {raw}.")

    def _set_distance_checker(self):
        number_points_distance = self._settings.get_setting("number_points_distance")
        self._distance_checker = geometry.PointCloud(
            self.raw.sample_points_uniformly(number_points_distance)
        )

    def _set_extra_elements(self):
        mesh_show_back_face = self._settings.get_setting("mesh_show_back_face")
        mesh = copy.copy(self.raw)
        if mesh_show_back_face:
            mesh.add_reverse_triangles()
        self._drawable = mesh.as_open3d

        self._set_distance_checker()
