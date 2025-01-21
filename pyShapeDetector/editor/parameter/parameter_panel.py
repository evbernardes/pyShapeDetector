#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-12 10:06:33

@author: evbernardes
"""
from typing import Union
from open3d.visualization import gui
from .parameter import ParameterBase


class ParameterPanel:
    def _get_panel(
        self,
        separation_width: float,
        separation_height: float,
        title: Union[str, None],
    ) -> gui.Vert:
        panel = gui.Vert(
            separation_width,
            gui.Margins(
                separation_width, separation_width, separation_width, separation_width
            ),
        )

        if title is not None:
            h = gui.Horiz()
            h.add_stretch()
            h.add_child(gui.Label(title))
            h.add_stretch()
            panel.add_child(h)

        subpanels = {}

        for param in self.parameters.values():
            element = param.get_gui_widget(separation_width)
            if param.subpanel is None:
                panel.add_child(element)
                panel.add_fixed(separation_height)
                continue

            if param.subpanel not in subpanels:
                subpanel = gui.CollapsableVert(
                    param.subpanel,
                    0.25 * separation_width,
                    gui.Margins(separation_width, 0, 0, 0),
                )
                subpanels[param.subpanel] = subpanel
                panel.add_child(subpanel)
                subpanel.set_is_open(False)
                panel.add_fixed(separation_height)

            subpanels[param.subpanel].add_child(element)
        return panel

    @property
    def panel(self) -> gui.Vert:
        return self._panel

    @property
    def parameters(self) -> dict[str, ParameterBase]:
        return self._parameters

    @property
    def has_limit_setters(self) -> bool:
        return self._has_limit_setters

    def __init__(
        self,
        parameters: dict[str, ParameterBase],
        separation_width: float,
        separation_height: float,
        title: Union[str, None] = None,
    ):
        self._parameters = parameters
        self._panel = self._get_panel(separation_width, separation_height, title)

        self._has_limit_setters = False
        for parameter in parameters.values():
            if getattr(parameter, "limit_setter", None) is not None:
                self._has_limit_setters = True
                break
