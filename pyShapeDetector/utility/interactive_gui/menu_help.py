from open3d.visualization import gui
from .editor_app import Editor
from .binding import Binding


class MenuHelp:
    def __init__(self, editor_instance: Editor, menu="Help"):
        self._menu = menu
        self._editor_instance = editor_instance

    def _create_panel(self):
        window = self._editor_instance._window
        em = window.theme.font_size

        _panel_collapsable = gui.CollapsableVert(self.menu, em, gui.Margins(0, 0, 0, 0))

        dlg_layout = gui.Vert(em, gui.Margins(0, 0, 0, 0))
        text = gui.Label("")
        dlg_layout.add_child(text)

        _panel_collapsable.add_child(dlg_layout)

        _panel_collapsable.visible = False
        self._editor_instance._right_side_panel.add_child(_panel_collapsable)
        self._panel = _panel_collapsable
        self._text = text

    def _create_menu(self):
        editor_instance = self._editor_instance

        self._create_panel()

        self._hotkeys_binding = editor_instance._internal_functions._dict[
            "Show Hotkeys"
        ]
        self._hotkeys_binding._menu = self._menu
        self._hotkeys_binding.add_to_menu(self._editor_instance)

        self._info_binding = editor_instance._internal_functions._dict["Show Info"]
        self._info_binding._menu = self._menu
        self._info_binding.add_to_menu(self._editor_instance)

        self._about_binding = Binding(
            description="About",
            menu=self._menu,
            callback=self._on_menu_about,
            creates_window=True,
        )
        self._about_binding.add_to_menu(self._editor_instance)

    def _on_help_toggle(self):
        editor_instance = self._editor_instance
        window = editor_instance._window
        menubar = editor_instance._menubar
        self._panel.visible = not self._panel.visible

        if self._panel.visible:
            self._text.text = (
                "\n\n".join(
                    [
                        f"({binding.key_instruction}):\n- {binding.description}"
                        for binding in self._editor_instance._hotkeys._bindings_map.values()
                    ]
                )
                + "\n\n(Ctrl):"
                + "\n- Set current with mouse"
                + "\n\n(Ctrl + Shift):"
                + "\n- Set current with mouse and toggle"
            )

        menubar.set_checked(self._hotkeys_binding.menu_id, self._panel.visible)
        window.set_needs_layout()

    def _on_info_toggle(self):
        editor_instance = self._editor_instance
        window = editor_instance._window
        menubar = editor_instance._menubar

        editor_instance._info.visible = not editor_instance._info.visible

        menubar.set_checked(self._info_binding.menu_id, editor_instance._info.visible)
        window.set_needs_layout()

    def _on_menu_about(self):
        window = self._editor_instance._window
        em = window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))

        title = gui.Horiz()
        title.add_stretch()
        title.add_child(gui.Label("Shape Detector"))
        title.add_stretch()

        dlg_layout.add_child(title)
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

        title = gui.Horiz()
        title.add_stretch()
        title.add_child(ok)
        title.add_stretch()
        dlg_layout.add_child(title)

        dlg.add_child(dlg_layout)
        window.show_dialog(dlg)

    def _on_about_ok(self):
        self._editor_instance._window.close_dialog()
