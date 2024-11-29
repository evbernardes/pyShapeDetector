from open3d.visualization import gui


class MenuFunctions:
    def __init__(self, app_instance, functions, name="Menu Functions"):
        self._app_instance = app_instance
        self._functions = functions
        self._name = name
        self._selected_func = None

    def _create_menu(self, id):
        self.menu_id = id
        app_instance = self._app_instance
        window = app_instance.window
        menubar = app_instance._menubar

        menu = gui.Menu()
        menubar.add_menu(self._name, menu)

        functions_dict = {id * 100 + i: func for i, func in enumerate(self._functions)}

        for i, func in functions_dict.items():
            name_function = func.__name__.replace("_", " ").capitalize()

            self._app_instance.print_debug(f"Linking {i} to {func}.")
            menu.add_item(name_function, i)
            menu.set_checked(i, False)

            _on_click = lambda f=func: app_instance._apply_function_to_elements(f)

            window.set_on_menu_item_activated(i, _on_click)

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
