import time
import copy
import inspect
import traceback
import signal
import sys
import warnings
import inspect
import itertools
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from open3d.visualization import gui, rendering

from open3d import geometry as 
from pyShapeDetector.geometry import PointCloud, Open3D_Geometry
from pyShapeDetector.primitives import Primitive

# from open3d import geometry as Open3d_Geometry

# from .editor_app import Editor
from .settings import Settings


from .helpers import (
    extract_element_colors,
    set_element_colors,
    get_painted_element,
    get_distance_checker,
)


class Element(ABC):
    @property
    def raw(self):
        return self._raw

    @property
    def drawable(self):
        return self._drawable

    @property
    def selected(self):
        return self._selected

    @property
    def current(self):
        return self._current

    @property
    def distance_checker(self):
        return self._distance_checker

    @property
    def color(self):
        return self._color

    @abstractmethod
    @staticmethod
    def _parse_raw(raw):
        pass

    @abstractmethod
    @staticmethod
    def _get_open3d(raw):
        pass

    @abstractmethod
    @staticmethod
    def _get_distance_checker(raw):
        pass

    @abstractmethod
    @staticmethod
    def _extract_color(raw):
        pass

    def paint_element(self):
        if self._settings["paint_random"]:
            self._drawable = get_painted_element(
                self._drawable, "random", self._settings["random_color_brightness"]
            )

        elif self._settings["paint_selected"] and self.selected:
            color = self._settings.get_element_color(True, self.current)
            self._drawable = get_painted_element(self._drawable, color)

    def __init__(
        self, settings: Settings, raw, selected: bool = False, current: bool = False
    ):
        self._settings = settings
        self._raw = self._parse_raw(raw)
        self._selected = selected
        self._current = current
        self._drawable = self._get_open3d(raw)
        self._distance_checker = self._get_distance_checker(raw)
        self._color = self._extract_color(self._drawable)

        self.paint_element()

        # if self._settings["paint_random"]:
        #     self.print_debug(
        #         f"[_insert_elements] Randomly painting element.",
        #         require_verbose=True,
        #     )


class ElementPointCloud(Element):
    @staticmethod
    def _parse_raw(raw):
        if PointCloud.is_instance_or_open3d(raw):
            return PointCloud(raw)
        else:
            raise ValueError("Expected PointCloud instance, got {raw}.")


class ElementPrimitive(Element):
    @staticmethod
    def _parse_raw(raw):
        if isinstance(raw, Primitive):
            return raw
        else:
            raise ValueError("Expected Primitive instance, got {raw}.")


class ElementGeometry(Element):
    @staticmethod
    def _parse_raw(raw):
        if isinstance(raw, Open3d_Geometry):
            return raw
        else:

