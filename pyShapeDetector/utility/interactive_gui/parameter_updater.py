import copy
import numpy as np
from open3d.visualization import gui
# from .interactive_gui import AppWindow


class ParameterUpdater:
    def __init__(self, window, parameters):
        self._parameters = copy.deepcopy(parameters)
        self._window = window
        self._interrupted = False
        # self._app_instance = app_instance

    def _update_element(self, name, value):
        parameter = self._parameters[name]
        parameter["default"] = parameter["type"](value)

    def _on_accept(self):
        self._window.close_dialog()

    def _on_cancel(self):
        self._interrupted = True
        self._window.close_dialog()

    def _create_dialog(self):
        em = self._window.theme.font_size
        separation_height = int(round(0.5 * em))
        button_separation_width = 2 * separation_height

        dlg = gui.Dialog("Parameter selection")
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))

        label = gui.Label("Enter parameters:")
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(label)
        h.add_stretch()
        dlg_layout.add_child(label)

        for key, value in self._parameters.items():
            name_pretty = key.replace("_", " ").capitalize()
            label = gui.Label(name_pretty)

            value_type = value["type"]
            value_default = value["default"]
            limits = value["limits"]

            _callback = lambda new_value=key: self._update_element(key, new_value)

            # value = getattr(self, key)
            # try:
            #     cb = getattr(self, "_cb_" + key)  # callback function
            # except AttributeError:
            #     continue

            if value_type is bool:
                element = gui.Checkbox(name_pretty + "?")
                element.checked = value_default
                element.set_on_checked(_callback)

            elif value_type is int and limits is not None:
                slider = gui.Slider(gui.Slider.INT)
                slider.set_limits(*limits)
                slider.int_value = value_default
                slider.set_on_value_changed(_callback)

                element = gui.VGrid(2, 0.25 * em)
                element.add_child(label)
                element.add_child(slider)

            elif value_type is float and limits is not None:
                slider = gui.Slider(gui.Slider.DOUBLE)
                slider.set_limits(*limits)
                slider.double_value = value_default
                slider.set_on_value_changed(_callback)

                element = gui.VGrid(2, 0.25 * em)
                element.add_child(label)
                element.add_child(slider)

            else:
                # Text field for general inputs
                text_edit = gui.TextEdit()
                text_edit.placeholder_text = str(value_default)
                text_edit.set_on_value_changed(_callback)

                element = gui.VGrid(2, 0.25 * em)
                element.add_child(label)
                element.add_child(text_edit)
                # layout.add_child(text_field)

            dlg_layout.add_child(element)
            dlg_layout.add_fixed(separation_height)

        accept = gui.Button("Accept")
        accept.set_on_clicked(self._on_accept)
        cancel = gui.Button("Cancel")
        cancel.set_on_clicked(self._on_cancel)
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(accept)
        h.add_fixed(button_separation_width)
        h.add_child(cancel)
        h.add_stretch()
        dlg_layout.add_child(h)
        dlg.add_child(dlg_layout)

        self._window.show_dialog(dlg)

    @staticmethod
    def update_descriptor_values(window, function_descriptor):
        parameters = function_descriptor["parameters"]
        if len(parameters) == 0:
            return

        try:
            parameter_updater = ParameterUpdater(window, parameters)
            parameter_updater._create_dialog()
        except:
            window.close_dialog()
            return
        if parameter_updater._interrupted:
            raise KeyboardInterrupt

        for key, value in parameter_updater._parameters.items():
            parameters[key]["default"] = value["default"]

    # def _on_layout(self, content_rect, layout_context):
    #     r = content_rect
    #     width = 17 * layout_context.theme.font_size
    #     height = min(
    #         r.height,
    #         self._panel.calc_preferred_size(
    #             layout_context, gui.Widget.Constraints()
    #         ).height,
    #     )
    #     self._panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)
