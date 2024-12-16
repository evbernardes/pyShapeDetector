import warnings
import copy

# from typing import List
import numpy as np
# from .element import Element


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
    selected
    selected_indices

    Methods
    -------
    insert_multiple
    pop_multiple
    get_distances_to_point
    update_all

    """

    @property
    def previous_index(self):
        return self._previous_index

    @property
    def current_index(self):
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
    def current_element(self):
        num_elems = len(self)
        if self._current_index in range(num_elems):
            return self[self._current_index]
        else:
            warnings.warn(
                f"Tried to update index {self._current_index}, but {num_elems} elements present."
            )
            return None

    @property
    def is_current_selected(self):
        if self.current_element is None:
            return None
        return self.current_element.selected

    @is_current_selected.setter
    def is_current_selected(self, boolean_value: bool):
        if not isinstance(boolean_value, bool):
            raise RuntimeError(f"Expected boolean, got {type(boolean_value)}.")
        if self.current_element is None:
            raise RuntimeError(
                "Error setting selected, current element does not exist."
            )
        self.current_element.selected = boolean_value

    def __repr__(self):
        return f"ElementContainer({len(self)} elements)"

    def __init__(self, editor_instance, elements=[], is_color_fixed=False):
        super().__init__(elements)
        self._previous_index = None
        self._current_index = None
        self._editor_instance = editor_instance
        self._is_color_fixed = is_color_fixed

    @property
    def raw(self):
        return [element.raw for element in self]

    @property
    def drawable(self):
        return [element.drawable for element in self]

    @property
    def is_selected(self):
        return [element.selected for element in self]

    @is_selected.setter
    def is_selected(self, values):
        if isinstance(values, bool):
            values = [values] * len(self)

        elif len(values) != len(self):
            raise ValueError(
                "Length of input expected to be the same as the "
                f"current number of elements ({len(self.elements)}), "
                f"got {len(values)}."
            )

        values = copy.deepcopy(values)

        for elem, value in zip(self, values):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"Expected boolean, got {type(value)}")

            elem._selected = value and self.select_filter(elem)

    @property
    def selected_indices(self):
        return np.where(self.is_selected)[0].tolist()

    def insert_multiple(self, elements_new, indices=None, selected=False, to_gui=False):
        from .element import Element

        if not isinstance(elements_new, (tuple, list)):
            elements_new = [elements_new]
        if isinstance(elements_new, tuple):
            elements_new = list(elements_new)

        if indices is None:
            indices = range(len(self), len(self) + len(elements_new))

        if isinstance(selected, bool):
            selected = [selected] * len(indices)

        if self.current_index is None:
            self._current_index = 0
            self._previous_index = 0

        idx_new = self.current_index

        self._editor_instance.print_debug(
            f"Adding {len(elements_new)} elements to the existing {len(self)}",
            require_verbose=True,
        )

        for i, idx in enumerate(indices):
            if idx_new > idx:
                idx_new += 1

            is_current = self._editor_instance._started and (self.current_index == idx)

            if isinstance(elements_new[i], Element):
                elem = elements_new[i]
                if elem.is_color_fixed != self._is_color_fixed:
                    warnings.warn(
                        "Changing element to a container with a different color type."
                    )
                    elem._is_color_fixed = self._is_color_fixed
            else:
                elem = Element.get_from_type(
                    self._editor_instance,
                    elements_new[i],
                    selected[i],
                    is_current,
                    self._is_color_fixed,
                )

            self._editor_instance.print_debug(
                f"Added {elem.raw} at index {idx}.", require_verbose=True
            )
            self.insert(idx, elem)

        if self._editor_instance._started:
            for idx in indices:
                # Updating vis explicitly in order not to remove it
                self._editor_instance._update_elements(idx, update_gui=False)
                if to_gui:
                    self[idx].add_to_scene()

            self._editor_instance.print_debug(f"{len(self)} now.", require_verbose=True)

            idx_new = max(min(idx_new, len(self) - 1), 0)
            self._editor_instance._update_current_idx(
                idx_new, update_old=self._editor_instance._started
            )
            self._previous_index = self.current_index

    def pop_multiple(self, indices, from_gui=False):
        # update_old = self.i in indices
        # idx_new = self.i
        elements_popped = ElementContainer(
            self._editor_instance, is_color_fixed=self._is_color_fixed
        )

        for n, i in enumerate(indices):
            elem = self.pop(i - n)
            if from_gui:
                elem.remove_from_scene()
            elements_popped.append(elem.raw)
            del elem._drawable

        idx_new = self.current_index - sum(
            [idx < self.current_index for idx in indices]
        )
        self._editor_instance.print_debug(
            f"popped: {indices}",
        )
        self._editor_instance.print_debug(
            f"old index: {self.current_index}, new index: {idx_new}"
        )

        if len(self) == 0:
            warnings.warn("No elements left after popping, not updating.")
            self.current_index = 0
            self._previous_index = 0
        else:
            idx_new = max(min(idx_new, len(self) - 1), 0)
            self._editor_instance._update_current_idx(idx_new, update_old=False)
            self._previous_index = self.current_index

        return elements_popped

    def get_distances_to_point(self, screen_point, screen_vector):
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

        def _distance_to_point(elem, point=screen_point, plane=screen_plane):
            """Check if mouse-click was done inside of the element actual region."""

            if not _is_point_in_convex_region(elem, point, plane):
                return np.inf

            try:
                return elem.get_distances(point)
            except AttributeError:
                warnings.warn(
                    f"Element of type {type(elem)} "
                    "found in distance elements, should not happen."
                )
                return np.inf

        distances = []
        for i, elem in enumerate(self):
            if i == self.current_index:
                # for selecting smaller objects closer to bigger ones,
                # ignores currently selected one
                distances.append(np.inf)
            else:
                distances.append(_distance_to_point(elem.distance_checker))
        return distances

    def update_all(self):
        for i, elem in enumerate(self):
            elem._current = self.current_index == 0
            elem.update()
