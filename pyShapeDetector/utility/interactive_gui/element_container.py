import warnings
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

    def __init__(self, editor_instance, elements=[]):
        super().__init__(elements)
        self._editor_instance = editor_instance
        # self._elements = []

    def raw(self):
        return [element.raw for element in self]

    def selected(self):
        return [element.selected for element in self]

    def selected_indices(self):
        return np.where(self.selected)[0].tolist()

    def insert_multiple(self, elements_new, indices=None, selected=False, to_gui=False):
        from .element import Element

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
            else:
                elem = Element.get_from_type(
                    self._editor_instance, elements_new[i], selected[i], is_current
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
        elements_popped = []
        for n, i in enumerate(indices):
            elem = self.pop(i - n)
            if from_gui:
                elem.remove_from_scene()
            elements_popped.append(elem.raw)

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
