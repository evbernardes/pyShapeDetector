from open3d.visualization import gui


class MenuFunctions:
    def __init__(self, app_instance, functions, name="Menu Functions"):
        self._app_instance = app_instance
        self._functions = functions
        self._name = name
        self._selected_func = None

    # def _on_clicked(self, func):
    #     self._app_instance._apply_function_to_elements()

    def _create_panel(self, window):
        em = window.theme.font_size
        separation_height = int(round(0.5 * em))

        # _panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        _panel_collapsable = gui.CollapsableVert(
            self._name, 0.25 * em, gui.Margins(em, 0, 0, 0)
        )

        for func in self._functions:
            name_pretty = func.__name__.replace("_", " ").capitalize()

            button = gui.Button(name_pretty)
            button.set_on_clicked(
                lambda: self._app_instance._apply_function_to_elements(func),
            )

            # element = gui.VGrid(2, 0.25 * em)
            # element.add_child(label)
            # element.add_child(color_selector)

            _panel_collapsable.add_child(button)
            _panel_collapsable.add_fixed(separation_height)

        _panel_collapsable.visible = False
        self._app_instance._main_panel.add_child(_panel_collapsable)

        # self._panel = _panel
        self._panel = _panel_collapsable
        # window.add_child(self._panel)

    def _create_menu(self, id):
        self.menu_id = id
        window = self._app_instance.window
        self._create_panel(window)

        menubar = self._app_instance._menubar
        functions_menu = gui.Menu()
        functions_menu.add_item(self._name, id)
        functions_menu.set_checked(id, False)
        menubar.add_menu("Functions", functions_menu)
        window.set_on_menu_item_activated(id, self._on_menu_toggle)
        menubar.set_checked(id, False)

    def _on_menu_toggle(self):
        window = self._app_instance.window
        menubar = self._app_instance._menubar
        self._panel.visible = not self._panel.visible
        menubar.set_checked(self.menu_id, self._panel.visible)
        window.set_needs_layout()

    def _on_layout(self, content_rect, layout_context):
        r = content_rect
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height,
        )
        self._panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)
