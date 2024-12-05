from open3d.visualization import gui
from .editor_app import Editor


class MenuHelp:
    def __init__(self, editor_instance: Editor, name="Help"):
        self._name = name
        self._editor_instance = editor_instance

    def _create_panel(self):
        window = self._editor_instance._window
        em = window.theme.font_size

        _panel_collapsable = gui.CollapsableVert("Help", em, gui.Margins(0, 0, 0, 0))

        dlg_layout = gui.Vert(em, gui.Margins(0, 0, 0, 0))
        # text = gui.Label(self._editor_instance._hotkeys.get_instructions())
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

        help_binding = (
            editor_instance._internal_functions.find_binding_from_description(
                "Show Help"
            )
        )
        self._help_id = editor_instance._add_menu_item(
            self._name, help_binding.description_and_instruction, self._on_help_toggle
        )

        info_binding = (
            editor_instance._internal_functions.find_binding_from_description(
                "Show Info"
            )
        )
        self._info_id = editor_instance._add_menu_item(
            self._name, info_binding.description_and_instruction, self._on_info_toggle
        )

        editor_instance._add_menu_item(self._name, "About", self._on_menu_about)

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
                + f"\n\n(Ctrl):"
                + "\n- Set current with mouse"
                + f"\n\n(Ctrl + Shift):"
                + "\n- Set current with mouse and toggle"
            )

        menubar.set_checked(self._help_id, self._panel.visible)
        window.set_needs_layout()

    def _on_info_toggle(self):
        editor_instance = self._editor_instance
        window = editor_instance._window
        menubar = editor_instance._menubar

        editor_instance._info.visible = not editor_instance._info.visible

        menubar.set_checked(self._info_id, editor_instance._info.visible)
        window.set_needs_layout()

    def _on_menu_about(self):
        window = self._editor_instance._window
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
        self._editor_instance._window.close_dialog()
