from open3d.visualization import gui


class MenuHelp:
    def __init__(self, app_instance):
        self._app_instance = app_instance

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
        self._create_panel(window)
        menubar = self._app_instance._menubar

        help_menu = gui.Menu()
        help_menu.add_item("Help (H)", id)
        help_menu.set_checked(id, False)
        menubar.add_menu("Help", help_menu)

        window.set_on_menu_item_activated(id, self._on_menu_toggle)
        menubar.set_checked(id, False)

    # def _on_about_ok(self):
    #     self._app_instance.window.close_dialog()

    def _on_menu_toggle(self):
        window = self._app_instance.window
        menubar = self._app_instance._menubar
        self._panel.visible = not self._panel.visible
        menubar.set_checked(self.menu_id, self._panel.visible)
        window.set_needs_layout()
        # # menubar = self._app_instance._menubar
        # # self._panel.visible = not self._panel.visible
        # # menubar.set_checked(self.menu_id, self._panel.visible)
        # window = self._app_instance.window
        # em = window.theme.font_size
        # dlg = gui.Dialog("About")

        # # Add the text
        # dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        # # dlg_layout.add_child(gui.Label("Open3D GUI Example"))
        # dlg_layout.add_child(gui.Label(self._app_instance._instructions))

        # # Add the Ok button. We need to define a callback function to handle
        # # the click.
        # ok = gui.Button("OK")
        # ok.set_on_clicked(self._on_about_ok)

        # # We want the Ok button to be an the right side, so we need to add
        # # a stretch item to the layout, otherwise the button will be the size
        # # of the entire row. A stretch item takes up as much space as it can,
        # # which forces the button to be its minimum size.
        # h = gui.Horiz()
        # h.add_stretch()
        # h.add_child(ok)
        # h.add_stretch()
        # dlg_layout.add_child(h)

        # dlg.add_child(dlg_layout)
        # window.show_dialog(dlg)
