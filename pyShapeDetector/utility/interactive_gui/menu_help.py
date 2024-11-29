from open3d.visualization import gui


class MenuHelp:
    def __init__(self, app_instance, name="Help"):
        self._app_instance = app_instance
        self._name = name

    def _create_panel(self, window):
        em = window.theme.font_size

        _panel_collapsable = gui.CollapsableVert("Help", em, gui.Margins(0, 0, 0, 0))

        dlg_layout = gui.Vert(em, gui.Margins(0, 0, 0, 0))
        for line in self._app_instance._instructions.split("\n"):
            dlg_layout.add_child(gui.Label(line))

        _panel_collapsable.add_child(dlg_layout)

        _panel_collapsable.visible = False
        self._app_instance._main_panel.add_child(_panel_collapsable)
        self._panel = _panel_collapsable

    def _create_menu(self, id):
        self.menu_id = id
        window = self._app_instance.window
        menubar = self._app_instance._menubar

        self._create_panel(window)

        menu = gui.Menu()
        menubar.add_menu(self._name, menu)

        menu.add_item("Help (H)", id)
        menu.set_checked(id, False)

        window.set_on_menu_item_activated(id, self._on_menu_toggle)

    def _on_menu_toggle(self):
        window = self._app_instance.window
        menubar = self._app_instance._menubar
        self._panel.visible = not self._panel.visible
        menubar.set_checked(self.menu_id, self._panel.visible)
        window.set_needs_layout()
