from open3d.visualization import gui
from .editor_app import Editor
from .binding import Binding


class MenuShow:
    def __init__(self, editor_instance: Editor, menu="Show"):
        self._menu = menu
        self._editor_instance = editor_instance

        self._bindings = {
            "ground": editor_instance._internal_functions._dict["Show Ground Plane"],
            "axes": editor_instance._internal_functions._dict["Show Global Axes"],
            "hotkeys": editor_instance._internal_functions._dict["Show Hotkeys"],
            "info": editor_instance._internal_functions._dict["Show Info"],
            "about": Binding(
                description="About",
                callback=self._on_menu_about,
                creates_window=True,
            ),
        }

    def _create_hotkeys_panel(self):
        window = self._editor_instance._window
        em = window.theme.font_size

        hotkeys_panel_collapsable = gui.CollapsableVert(
            self._menu, em, gui.Margins(0, 0, 0, 0)
        )

        dlg_layout = gui.Vert(em, gui.Margins(0, 0, 0, 0))
        hotkeys_panel_text = gui.Label("")
        dlg_layout.add_child(hotkeys_panel_text)

        hotkeys_panel_collapsable.add_child(dlg_layout)

        hotkeys_panel_collapsable.visible = False
        self._editor_instance._right_side_panel.add_child(hotkeys_panel_collapsable)
        self._hotkeys_panel = hotkeys_panel_collapsable
        self._hotkeys_panel_text = hotkeys_panel_text

    def _create_menu(self):
        self._create_hotkeys_panel()

        for binding in self._bindings.values():
            binding._menu = self._menu
            binding.add_to_menu(self._editor_instance)

        editor_instance = self._editor_instance
        menubar = editor_instance._menubar
        menubar.set_checked(
            self._bindings["ground"].menu_id, editor_instance._ground_plane_visible
        )
        menubar.set_checked(
            self._bindings["axes"].menu_id, editor_instance._global_axes_visible
        )
        menubar.set_checked(
            self._bindings["info"].menu_id, editor_instance._info.visible
        )

    def _on_hotkeys_toggle(self):
        editor_instance = self._editor_instance
        window = editor_instance._window
        menubar = editor_instance._menubar
        self._hotkeys_panel.visible = not self._hotkeys_panel.visible

        if self._hotkeys_panel.visible:
            self._hotkeys_panel_text.text = (
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

        menubar.set_checked(
            self._bindings["hotkeys"].menu_id, self._hotkeys_panel.visible
        )
        window.set_needs_layout()

    def _on_ground_plane_toggle(self):
        editor_instance = self._editor_instance
        window = editor_instance._window
        menubar = editor_instance._menubar

        # editor_instance._info.visible = not editor_instance._info.visible
        ground_plane_visible = not editor_instance._ground_plane_visible

        editor_instance._scene.scene.show_ground_plane(
            ground_plane_visible, editor_instance._ground_plane
        )

        menubar.set_checked(self._bindings["ground"].menu_id, ground_plane_visible)
        window.set_needs_layout()

        editor_instance._ground_plane_visible = ground_plane_visible

    def _on_global_axes_toggle(self):
        editor_instance = self._editor_instance
        window = editor_instance._window
        menubar = editor_instance._menubar

        # editor_instance._info.visible = not editor_instance._info.visible
        global_axes_visible = not editor_instance._global_axes_visible

        editor_instance._scene.scene.show_axes(global_axes_visible)

        menubar.set_checked(self._bindings["axes"].menu_id, global_axes_visible)
        window.set_needs_layout()

        editor_instance._global_axes_visible = global_axes_visible

    def _on_info_toggle(self):
        editor_instance = self._editor_instance
        window = editor_instance._window
        menubar = editor_instance._menubar

        editor_instance._info.visible = not editor_instance._info.visible

        menubar.set_checked(
            self._bindings["info"].menu_id, editor_instance._info.visible
        )
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
                "Developed by:\n\n\tEvandro Bernardes"
                "\n\nAt:\n\n\tVrije Universiteit Brussel (VUB)"
            )
        )

        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        button_stretch = gui.Horiz()
        button_stretch.add_stretch()
        button_stretch.add_child(ok)
        button_stretch.add_stretch()

        dlg_layout.add_child(button_stretch)

        dlg.add_child(dlg_layout)
        window.show_dialog(dlg)

        def _on_key_event(event):
            if event.type == gui.KeyEvent.DOWN and event.key == gui.KeyName.ENTER:
                self._on_about_ok()
                return True
            return False

        window.set_on_key(_on_key_event)

    def _on_about_ok(self):
        self._editor_instance._close_dialog()
