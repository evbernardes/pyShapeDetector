from open3d.visualization import gui
from .interactive_gui import AppWindow


class MenuFunctions:
    def __init__(self, app_instance: AppWindow, extensions, name):
        self._app_instance = app_instance
        self._extensions = extensions
        self._name = name
        self._selected_func = None

    def _create_menu(self, id):
        self.menu_id = id
        app_instance = self._app_instance
        window = app_instance.window
        menubar = app_instance._menubar

        menu = gui.Menu()
        menubar.add_menu(self._name, menu)

        extensions_dict = {id * 100 + i: ext for i, ext in enumerate(self._extensions)}

        for i, extension in extensions_dict.items():
            menu.add_item(extension.name, i)
            menu.set_checked(i, False)

            _on_click = lambda f=extension: app_instance._apply_function_to_elements(f)

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
