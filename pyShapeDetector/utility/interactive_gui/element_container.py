import warnings
import copy
from typing import List
import numpy as np
# from .element import Element


class ElementContainer(list):
    # class ElementContainer:
    # @property
    # def elements(self):
    #     return self._elements

    # def __getitem__(self, index):
    #     return self.elements[index]

    # def __setitem__(self, index, value):
    #     from .element import Element

    #     if not isinstance(value, Element):
    #         raise TypeError("Can only directly add instances of Element to container.")
    #     self.elements[index] = value

    def __repr__(self):
        return f"ElementContainer({len(self)} elements)"

    def __init__(self, editor_instance, elements=[], is_color_fixed=False):
        super().__init__(elements)
        self._editor_instance = editor_instance
        self._is_color_fixed = is_color_fixed
        # self._elements = []

    @property
    def raw(self):
        return [element.raw for element in self]

    @property
    def selected(self):
        return [element.selected for element in self]

    @selected.setter
    def selected(self, values):
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
        return np.where(self.selected)[0].tolist()

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

        idx_new = self._editor_instance.i

        self._editor_instance.print_debug(
            f"Adding {len(elements_new)} elements to the existing {len(self)}",
            require_verbose=True,
        )

        for i, idx in enumerate(indices):
            if idx_new > idx:
                idx_new += 1

            is_current = self._editor_instance._started and (
                self._editor_instance.i == idx
            )

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
            self._editor_instance.i_old = self._editor_instance.i

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
            elements_popped.append(elem)

        idx_new = self._editor_instance.i - sum(
            [idx < self._editor_instance.i for idx in indices]
        )
        self._editor_instance.print_debug(
            f"popped: {indices}",
        )
        self._editor_instance.print_debug(
            f"old index: {self._editor_instance.i}, new index: {idx_new}"
        )

        if len(self) == 0:
            warnings.warn("No elements left after popping, not updating.")
            self._editor_instance.i = 0
            self._editor_instance.i_old = 0
        else:
            idx_new = max(min(idx_new, len(self) - 1), 0)
            self._editor_instance._update_current_idx(idx_new, update_old=False)
            self._editor_instance.i_old = self._editor_instance.i

        return elements_popped

    def get_distances_to_point(self, screen_point, screen_vector):
        """Called by Mouse callback to get distances to point when clicked."""
        from pyShapeDetector.geometry import PointCloud
        from pyShapeDetector.primitives import Primitive, Plane, Sphere

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
            if i == self._editor_instance.i:
                # for selecting smaller objects closer to bigger ones,
                # ignores currently selected one
                distances.append(np.inf)
            else:
                distances.append(_distance_to_point(elem.distance_checker))
        return distances

    def update_all(self):
        for i, elem in enumerate(self):
            elem._current = self._editor_instance.i == 0
            elem.update()
