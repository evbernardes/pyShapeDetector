import warnings
import copy
import traceback
import numpy as np
from typing import List, Union, TYPE_CHECKING
from open3d.visualization.rendering import Open3DScene
from .element import Element, ELEMENT_TYPE

if TYPE_CHECKING:
    from .settings import Settings

import psutil

process = psutil.Process()


class ElementContainer(list):
    """
    Class implementing a specialized list for containing Element instances.

    Attributes
    ----------
    previous_index
    current_index
    current_element
    raw
    drawable
    is_selected
    selected_indices
    is_hidden
    hidden_indices
    unhidden_indices

    Methods
    -------
    add_to_scene
    remove_from_scene
    get_closest_unhidden_index
    insert_multiple
    pop_multiple
    get_distances_to_point
    update_indices
    toggle_indices
    update_current_index
    scene
    """

    @property
    def previous_index(self) -> int:
        return self._previous_index

    @property
    def current_index(self) -> int:
        return self._current_index

    @current_index.setter
    def current_index(self, new_index: int):
        if not isinstance(new_index, (int, np.integer)) or new_index < 0:
            raise ValueError(f"Index has to be positive integer, got {new_index}.")
        # self._previous_index = self._current_index

        if len(self) == 0:
            warnings.warn(
                f"Tried setting index to {new_index}, but there are "
                f"no elements in the container."
            )
            self._current_index = None

        if new_index > len(self) - 1:
            warnings.warn(
                f"Tried setting index to {new_index}, but there are "
                f"only {len(self)} elements in the container."
            )
            self._current_index = len(self) - 1

        self._current_index = int(new_index)

    @property
    def current_element(self) -> Element:
        num_elems = len(self)
        if self._current_index in range(num_elems):
            return self[self._current_index]
        else:
            warnings.warn(
                f"Tried to update index {self._current_index}, but {num_elems} elements present."
            )
            return None

    @property
    def is_current_selected(self) -> bool:
        if self.current_element is None:
            return None
        return self.current_element.is_selected

    @property
    def scene(self) -> Union[Open3DScene, None]:
        return self._scene

    def __repr__(self):
        return f"ElementContainer({len(self)} elements)"

    def __init__(
        self,
        settings: "Settings",
        elements: list[ELEMENT_TYPE] = [],
        is_color_fixed: bool = False,
    ):
        super().__init__(elements)
        self._previous_index = None
        self._current_index = None
        self._settings = settings
        self._is_color_fixed = is_color_fixed
        self._scene = None

    @property
    def raw(self) -> list[ELEMENT_TYPE]:
        return [element.raw for element in self]

    @property
    def drawable(self) -> list[ELEMENT_TYPE]:
        return [element.drawable for element in self]

    @property
    def is_selected(self) -> List[bool]:
        return [element.is_selected for element in self]

    @is_selected.setter
    def is_selected(self, values: Union[List[bool], bool]):
        if isinstance(values, bool):
            values = [values] * len(self)

        elif len(values) != len(self):
            raise ValueError(
                "Length of input expected to be the same as the "
                f"current number of elements ({len(self)}), "
                f"got {len(values)}."
            )

        values = copy.deepcopy(values)

        for elem, value in zip(self, values):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"Expected boolean, got {type(value)}")

            elem._selected = value

    @property
    def is_hidden(self) -> List[bool]:
        return [element.is_hidden for element in self]

    @is_hidden.setter
    def is_hidden(self, values: Union[List[bool], bool]):
        if isinstance(values, bool):
            values = [values] * len(self)

        elif len(values) != len(self):
            raise ValueError(
                "Length of input expected to be the same as the "
                f"current number of elements ({len(self)}), "
                f"got {len(values)}."
            )

        values = copy.deepcopy(values)

        for elem, value in zip(self, values):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"Expected boolean, got {type(value)}")

            elem._is_hidden = value

    @property
    def selected_indices(self) -> list[int]:
        if len(self) == 0:
            return []
        return np.where(self.is_selected)[0].tolist()

    @property
    def hidden_indices(self) -> list[int]:
        if len(self) == 0:
            return []
        return np.where(self.is_hidden)[0].tolist()

    @property
    def unhidden_indices(self) -> list[int]:
        if len(self) == 0:
            return []
        return np.where(~np.array(self.is_hidden))[0].tolist()

    def add_to_scene(self, scene: Open3DScene, startup: bool = False):
        if self._scene is not None and self._scene != scene:
            warnings.warn("Remove from current scene before adding to another.")
            return

        if isinstance(scene, Open3DScene):
            self._scene = scene
        else:
            raise TypeError(f"Expected Open3DScene, got {scene}.")

        for elem in self:
            if startup:
                elem.update(is_current=False, update_scene=False)
            elem.add_to_scene(scene)

    def remove_from_scene(self):
        if self._scene is None:
            warnings.warn("Cannot remove from scene: currently not in any scene.")
            return None

        for elem in self:
            elem.remove_from_scene()
        self._scene = None

    def get_closest_unhidden_index(self, index: Union[int, None] = None):
        if index is None:
            index = self.current_index
        elif not isinstance(index, int) or index < 0:
            raise ValueError("Expected positive integer index, got {index}.")
        if index in self.unhidden_indices:
            return index

        try:
            position = np.argmin(abs(np.array(self.unhidden_indices) - index))
            return self.unhidden_indices[position]
        except Exception:
            warnings.warn(
                f"Could not get closest unhidden index to index {index}. "
                f"There are currently {len(self.unhidden_indices)} unhidden indexes."
            )
            traceback.print_exc()
            return None

    def insert_multiple(
        self,
        elements_new: Union[ELEMENT_TYPE, list[ELEMENT_TYPE]],
        indices=None,
        is_selected=False,
        to_gui=False,
    ):
        if not isinstance(elements_new, (tuple, list)):
            elements_new = [elements_new]
        if isinstance(elements_new, tuple):
            elements_new = list(elements_new)

        if indices is None:
            indices = range(len(self), len(self) + len(elements_new))

        if isinstance(is_selected, bool):
            is_selected = [is_selected] * len(indices)

        if self.current_index is None:
            self._current_index = 0
            self._previous_index = 0

        idx_new = self.current_index

        self._settings.print_debug(
            f"Adding {len(elements_new)} elements to the existing {len(self)}",
            require_verbose=True,
        )

        failed_indices = []
        tested_indices = []

        for i, idx in enumerate(indices):
            idx -= len([failed for failed in failed_indices if failed < idx])

            if idx_new > idx:
                idx_new += 1

            is_current = (self.scene is not None) and (self.current_index == idx)

            if isinstance(elements_new[i], Element):
                elem = elements_new[i]
                if elem.is_color_fixed != self._is_color_fixed:
                    warnings.warn(
                        "Changing element to a container with a different color type."
                    )
                    elem._is_color_fixed = self._is_color_fixed
            else:
                elem = Element.get_from_type(
                    self._settings,
                    elements_new[i],
                    is_selected[i],
                    is_current,
                    self._is_color_fixed,
                )
                if elem is None:
                    warnings.warn(f"Could not insert element {elements_new[i]}.")
                    continue

            self._settings.print_debug(
                f"Added {elem.raw} at index {idx}.",
                require_verbose=True,
            )
            self.insert(idx, elem)
            tested_indices.append(idx)

        if self.scene is not None:
            for idx in tested_indices:
                self._settings.print_debug(
                    f"Adding element of index = {idx} to scene.", require_verbose=True
                )
                self[idx]._scene = self.scene
                # Updating vis explicitly in order not to remove it
                self.update_indices(idx, update_gui=False)
                if to_gui:
                    self[idx].add_to_scene()

            self._settings.print_debug(
                f"Added {len(tested_indices)} elements to scene."
            )

            self._settings.print_debug(
                f"{len(self)} now.",
                require_verbose=True,
            )

            idx_new = max(min(idx_new, len(self) - 1), 0)
            self.update_current_index(idx_new, update_old=self.scene is not None)
            self._previous_index = self.current_index

        self._settings.print_debug(f"Used memory: {process.memory_info().rss}")

    def pop_multiple(self, indices: list[int], from_gui: bool = False):
        # update_old = self.i in indices
        # idx_new = self.i
        elements_popped = ElementContainer(
            self._settings, is_color_fixed=self._is_color_fixed
        )

        for n, i in enumerate(indices):
            try:
                elem = self.pop(i - n)
                if from_gui:
                    elem.remove_from_scene()
                elements_popped.append(elem.raw)
                del elem._drawable
                del elem._distance_checker
                del elem._color_original
                del elem._color
            except Exception:
                print(f"Could not remove index {i}!")

        idx_new = self.current_index - sum(
            [idx < self.current_index for idx in indices]
        )
        self._settings.print_debug(f"popped: {indices}", require_verbose=True)
        self._settings.print_debug(
            f"old index: {self.current_index}, new index: {idx_new}",
        )

        if len(self) == 0:
            warnings.warn("No elements left after popping, not updating.")
            self.current_index = 0
            self._previous_index = 0
        else:
            idx_new = max(min(idx_new, len(self) - 1), 0)
            self.update_current_index(idx_new, update_old=False)
            self._previous_index = self.current_index

        return elements_popped

    def get_distances_to_point(
        self, screen_point: np.ndarray, screen_vector: np.ndarray
    ):
        """Called by Mouse callback to get distances to point when clicked."""
        from pyShapeDetector.primitives import Plane

        screen_plane = Plane.from_normal_point(screen_vector, screen_point)

        def _is_point_in_convex_region(elem, point=screen_point, plane=screen_plane):
            """Check if mouse-click was done inside of the element actual region."""

            if hasattr(elem, "points"):
                boundary_points = elem.points
            elif hasattr(elem, "vertices"):
                boundary_points = elem.vertices
            elif hasattr(elem, "mesh"):
                boundary_points = elem.mesh.vertices
            else:
                return False

            boundary_points = np.asarray(boundary_points)
            if len(boundary_points) < 3:
                return False

            plane_bounded = plane.get_bounded_plane(boundary_points, convex=True)
            return plane_bounded.contains_projections(point)

        def _distance_to_point(
            distance_checker,
            point: np.ndarray = screen_point,
            plane: Plane = screen_plane,
        ):
            """Check if mouse-click was done inside of the element actual region."""

            if not _is_point_in_convex_region(distance_checker, point, plane):
                return np.inf

            try:
                return distance_checker.get_distances(point)
            except AttributeError:
                warnings.warn(
                    f"Element of type {type(distance_checker)} "
                    "found in distance elements, should not happen."
                )
                return np.inf

        distances = []
        for i, elem in enumerate(self):
            if i == self.current_index or elem.is_hidden:
                # for selecting smaller objects closer to bigger ones,
                # ignores currently selected one
                distances.append(np.inf)
            else:
                distances.append(_distance_to_point(elem.distance_checker))
        return distances

    def update_indices(self, indices, update_gui=True):
        num_elems = len(self)

        if indices is None:
            indices = range(num_elems)

        if not isinstance(indices, (list, range)):
            indices = [indices]

        if len(indices) == 0:
            return

        if num_elems == 0 or max(indices) >= num_elems:
            warnings.warn(
                f"Tried to update index {indices}, but {num_elems} elements present."
            )
            return

        for idx in indices:
            elem = self[idx]
            elem._selected = elem.is_selected and not elem.is_hidden
            is_current = (self.scene is not None) and (idx == self.current_index)

            elem.update(is_current, update_gui)

    def toggle_indices(
        self,
        indices_or_slice: Union[range, list[int], np.ndarray[int, int]],
        to_value=None,
    ):
        if isinstance(indices_or_slice, (range, list, np.ndarray)):
            indices = indices_or_slice

        elif indices_or_slice is None:
            # if indices are not given, update everything
            indices = self.unhidden_indices

        elif isinstance(indices_or_slice, slice):
            start, stop, stride = indices_or_slice.indices(len(self))
            indices = range(start, stop, stride)

        else:
            warnings.warn(
                "Invalid input to toggle_indices, expected index list/array, "
                f"range or slice, got {indices_or_slice}."
            )
            return

        for idx in indices:
            elem = self[idx]
            if to_value is None:
                is_selected = not self._hotkeys._is_lshift_pressed
            else:
                is_selected = to_value
            elem.is_selected = is_selected

        self.update_indices(indices)

    def update_current_index(
        self, idx: Union[None, int] = None, update_old: bool = True
    ):
        if idx is not None:
            self._previous_index = self.current_index
            self._current_index = idx

            self._settings.print_debug(
                f"Updating index, from {self._previous_index} to {self.current_index}",
                require_verbose=True,
            )
        else:
            self._settings.print_debug(
                f"Updating current index: {self.current_index}",
                require_verbose=True,
            )

        if self.current_index >= len(self):
            warnings.warn(
                f"Index error, tried accessing {self.current_index} out of "
                f"{len(self)} elements. Getting last one."
            )
            idx = len(self) - 1

        self.update_indices(self.current_index)
        if update_old:
            self.update_indices(self._previous_index)
