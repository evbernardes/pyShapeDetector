from itertools import count
from typing import TYPE_CHECKING
from open3d.visualization import gui
from .binding import Binding

if TYPE_CHECKING:
    from .editor_app import Editor

# Generating large numbers for ids so it's on the right of the menubar
MENU_SHOW_ID_GENERATOR = count(1000, 1)


class MenuShow:
    @property
    def bindings(self) -> list[Binding]:
        return list(self._dict.values())

    def __init__(self, editor_instance: "Editor", menu="Show"):
        self._menu = menu
        self._editor_instance = editor_instance

        self._bindings = [
            Binding(
                key=gui.KeyName.G,
                menu_id=next(MENU_SHOW_ID_GENERATOR),
                lctrl=False,
                lshift=False,
                description="Show Ground Plane",
                callback=self._cb_show_ground_plane,
                menu=self._menu,
            ),
            Binding(
                key=gui.KeyName.A,
                menu_id=next(MENU_SHOW_ID_GENERATOR),
                lctrl=False,
                lshift=False,
                description="Show Global Axes",
                callback=self._cb_show_global_axes,
                menu=self._menu,
            ),
            Binding(
                key=gui.KeyName.H,
                menu_id=next(MENU_SHOW_ID_GENERATOR),
                lctrl=False,
                lshift=False,
                description="Show Help",
                callback=self._cb_show_help,
                menu=self._menu,
            ),
            Binding(
                key=gui.KeyName.I,
                menu_id=next(MENU_SHOW_ID_GENERATOR),
                lctrl=False,
                lshift=False,
                description="Show Info",
                callback=self._cb_show_info,
                menu=self._menu,
            ),
            Binding(
                description="About",
                menu_id=next(MENU_SHOW_ID_GENERATOR),
                callback=self._cb_about,
                creates_window=True,
                menu=self._menu,
            ),
        ]

        self._dict = {binding.description: binding for binding in self._bindings}

    def _create_help_panel(self):
        window = self._editor_instance._main_window
        em = window.theme.font_size
        dlg_layout = gui.Vert(em, gui.Margins(0, 0, 0, 0))

        help_panel_collapsable = gui.CollapsableVert(
            "Help", em, gui.Margins(0, 0, 0, 0)
        )

        help_panel_collapsable.add_child(dlg_layout)
        help_panel_collapsable.visible = False

        self._editor_instance._right_side_panel.add_child(help_panel_collapsable)
        self._help_panel = help_panel_collapsable
        self._help_panel_text = gui.Label("")

        dlg_layout.add_child(self._help_panel_text)

    def _create_menu(self):
        self._create_help_panel()

        editor_instance = self._editor_instance
        menubar = editor_instance.app.menubar
        menubar.set_checked(
            self._dict["Show Ground Plane"].menu_id,
            editor_instance._ground_plane_visible,
        )
        menubar.set_checked(
            self._dict["Show Global Axes"].menu_id,
            editor_instance._global_axes_visible,
        )
        menubar.set_checked(
            self._dict["Show Info"].menu_id,
            editor_instance._info_visible,
        )

    def _cb_show_help(self):
        editor_instance = self._editor_instance
        window = editor_instance._main_window
        menubar = editor_instance.app.menubar
        self._help_panel.visible = not self._help_panel.visible

        if self._help_panel.visible:
            self._help_panel_text.text = (
                self._editor_instance._hotkeys.help_text
                + "\n\n(Ctrl):"
                + "\n- Set current with mouse"
                + "\n\n(Ctrl + Shift):"
                + "\n- Set current with mouse and toggle"
            )

        menubar.set_checked(self._dict["hotkeys"].menu_id, self._help_panel.visible)
        window.set_needs_layout()

    def _cb_show_ground_plane(self):
        editor_instance = self._editor_instance
        window = editor_instance._main_window
        menubar = editor_instance.app.menubar

        # editor_instance._info.visible = not editor_instance._info.visible
        ground_plane_visible = not editor_instance._ground_plane_visible

        editor_instance._scene.scene.show_ground_plane(
            ground_plane_visible, editor_instance._ground_plane
        )

        menubar.set_checked(
            self._dict["Show Ground Plane"].menu_id,
            ground_plane_visible,
        )
        window.set_needs_layout()

        editor_instance._ground_plane_visible = ground_plane_visible

    def _cb_show_global_axes(self):
        editor_instance = self._editor_instance
        window = editor_instance._main_window
        menubar = editor_instance.app.menubar

        # editor_instance._info.visible = not editor_instance._info.visible
        global_axes_visible = not editor_instance._global_axes_visible

        editor_instance._scene.scene.show_axes(global_axes_visible)

        menubar.set_checked(
            self._dict["Show Global Axes"].menu_id,
            global_axes_visible,
        )
        window.set_needs_layout()

        editor_instance._global_axes_visible = global_axes_visible

    def _cb_show_info(self):
        editor_instance = self._editor_instance
        menubar = editor_instance.app.menubar

        editor_instance._info_visible = not editor_instance._info_visible

        menubar.set_checked(
            self._dict["Show Info"].menu_id,
            editor_instance._info_visible,
        )
        editor_instance._update_info()

    def _cb_about(self):
        self._editor_instance._create_simple_dialog(
            content_text=(
                "Developed by:\n\n\tEvandro Bernardes"
                "\n\nAt:\n\n\tVrije Universiteit Brussel (VUB)"
            ),
            create_button=True,
            button_text="Ok",
            button_callback=None,
            title_text="Shape Detector",
        )
