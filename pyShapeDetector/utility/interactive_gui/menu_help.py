from open3d.visualization import gui
from .interactive_gui import AppWindow


class MenuHelp:
    def __init__(self, app_instance: AppWindow, name="Help"):
        self._name = name
        self._app_instance = app_instance

    def _create_panel(self):
        window = self._app_instance._window
        em = window.theme.font_size

        _panel_collapsable = gui.CollapsableVert("Help", em, gui.Margins(0, 0, 0, 0))

        dlg_layout = gui.Vert(em, gui.Margins(0, 0, 0, 0))
        text = gui.Label(self._app_instance._hotkeys.get_instructions())
        dlg_layout.add_child(text)

        _panel_collapsable.add_child(dlg_layout)

        _panel_collapsable.visible = False
        self._app_instance._main_panel.add_child(_panel_collapsable)
        self._panel = _panel_collapsable
        self._text = text

    def _create_menu(self):
        app_instance = self._app_instance

        self._create_panel()

        self._help_id = app_instance._add_menu_item(
            self._name, "Help (H)", self._on_help_toggle
        )
        app_instance._add_menu_item(self._name, "About", self._on_menu_about)

    def _on_help_toggle(self):
        app_instance = self._app_instance
        window = app_instance._window
        menubar = app_instance._menubar
        self._panel.visible = not self._panel.visible

        if self._panel.visible:
            self._text.text = app_instance._hotkeys.get_instructions()

        menubar.set_checked(self._help_id, self._panel.visible)
        window.set_needs_layout()

    def _on_menu_about(self):
        window = self._app_instance._window
        em = window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Shape Detector"))
        dlg_layout.add_child(
            gui.Label(
                "Developed by Evandro Bernardes\nVrije Universiteit Brussel (VUB)"
            )
        )
        dlg_layout.add_child(
            gui.Label(
                "More information at\nhttps://github.com/evbernardes/pyShapeDetector"
            )
        )

        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        window.show_dialog(dlg)

    def _on_about_ok(self):
        self._app_instance._window.close_dialog()
