try:
    from ipywidgets import (
        HTML,
        Image,
        AppLayout,
        VBox,
        HBox,
        Button,
        Label,
        Layout,
        Tab,
        Accordion,
    )
    from ipyevents import Event
    from typing import Optional
    import klayout.db as db
    import klayout.lay as lay
except ImportError as e:
    print(
        "You need install jupyter notebook plugin with `pip install gdsfactory[full]`"
    )
    raise e


class LayoutViewer:
    def __init__(
        self,
        filepath: str,
        layer_properties: Optional[str],
        hide_unused_layers: bool = True,
        with_layer_selector: bool = True,
    ):
        filepath = str(filepath)
        layer_properties = str(layer_properties)
        self.hide_unused_layers = hide_unused_layers
        self.filepath = filepath
        self.layer_properties = layer_properties

        self.layout_view = lay.LayoutView()
        self.load_layout(filepath, layer_properties)

        if self.hide_unused_layers:
            self.layout_view.remove_unused_layers()
            self.layout_view.reload_layout(self.layout_view.current_layer_list)

        pixel_buffer = self.layout_view.get_pixels_with_options(800, 600)
        png_data = pixel_buffer.to_png_data()
        self.image = Image(value=png_data, format="png")
        scroll_event = Event(source=self.image, watched_events=["wheel"])
        scroll_event.on_dom_event(self.on_scroll)
        self.wheel_info = HTML("Waiting for a scroll...")
        self.mouse_info = HTML("Waiting for a mouse event...")
        # self.layout_view.on_image_updated_event = lambda: self.refresh
        mouse_event = Event(
            source=self.image, watched_events=["mousedown", "mouseup", "mousemove"]
        )
        mouse_event.on_dom_event(self.on_mouse_down)

        if with_layer_selector:
            layer_selector_tabs = self.layer_selector_tabs = self.build_layer_selector(
                max_height=pixel_buffer.height()
            )
        else:
            layer_selector_tabs = None

        self.widget = AppLayout(
            center=self.image,
            right_sidebar=layer_selector_tabs,
            left_sidebar=None,
            # footer=VBox([self.wheel_info, self.mouse_info]),
            align_items="top",
            justify_items="left",
        )

    def button_toggle(self, button):
        button.style.button_color = (
            "transparent"
            if (button.style.button_color == button.default_color)
            else button.default_color
        )

        layer_iter = self.layout_view.begin_layers()

        while not layer_iter.at_end():
            props = layer_iter.current()
            if props.name == button.layer_props.name:
                props.visible = not props.visible
                self.layout_view.set_layer_properties(layer_iter, props)
                self.layout_view.reload_layout(self.layout_view.current_layer_list)
                break
            layer_iter.next()
        self.refresh()

    def build_layer_toggle(self, prop_iter: lay.LayerPropertiesIterator):
        from gdsfactory.utils.color_utils import ensure_six_digit_hex_color

        props = prop_iter.current()
        layer_color = ensure_six_digit_hex_color(props.eff_fill_color())

        # Would be nice to use LayoutView.icon_for_layer() rather than simple colored box
        button_layout = Layout(
            width="5px",
            height="20px",
            border=f"solid 2px {layer_color}",
            display="block",
        )

        layer_checkbox = Button(
            style={"button_color": layer_color if props.visible else "transparent"},
            layout=button_layout,
        )
        layer_checkbox.default_color = layer_color
        layer_checkbox.layer_props = props

        if props.has_children():
            prop_iter = prop_iter.down_first_child()
            n_children = prop_iter.num_siblings()
            # print(f"{props.name} has {n_children} children!")
            children = []
            for _i in range(n_children):
                prop_iter = prop_iter.next()
                children.append(self.build_layer_toggle(prop_iter))
            layer_label = Accordion([VBox(children)], titles=(props.name,))
        else:
            layer_label = Label(props.name)
        layer_checkbox.label = layer_label

        layer_checkbox.on_click(self.button_toggle)
        return HBox([layer_checkbox, layer_label])

    def build_layer_selector(self, max_height: float):
        """Builds a widget for toggling layer displays.

        Args:
            max_height: Maximum height to set for the widget (likely the height of the pixel buffer).
        """

        all_boxes = []

        prop_iter = self.layout_view.begin_layers()
        while not prop_iter.at_end():
            layer_toggle = self.build_layer_toggle(prop_iter)
            all_boxes.append(layer_toggle)
            prop_iter.next()

        layers_layout = Layout(
            max_height=f"{max_height}px", overflow_y="auto", display="block"
        )
        layer_selector = VBox(all_boxes, layout=layers_layout)

        # For when tabs are implemented
        layer_selector_tabs = Tab([layer_selector])
        layer_selector_tabs.titles = ("Layers",)
        return layer_selector_tabs

    def load_layout(self, filepath: str, layer_properties: Optional[str]):
        """Loads a GDS layout.

        Args:
            filepath: path for the GDS layout.
            layer_properties: Optional path for the layer_properties klayout file (lyp).
        """
        self.layout_view.load_layout(filepath)
        self.layout_view.max_hier()
        if layer_properties:
            self.layout_view.load_layer_props(layer_properties)

    def refresh(self):
        pixel_buffer = self.layout_view.get_pixels_with_options(800, 600)
        png_data = pixel_buffer.to_png_data()
        self.image.value = png_data

    def _get_modifier_buttons(self, event):
        shift = event["shiftKey"]
        alt = event["altKey"]
        ctrl = event["ctrlKey"]
        # meta = event["metaKey"]

        mouse_buttons = event["buttons"]

        buttons = 0
        if shift:
            buttons |= lay.ButtonState.ShiftKey
        if alt:
            buttons |= lay.ButtonState.AltKey
        if ctrl:
            buttons |= lay.ButtonState.ControlKey

        if mouse_buttons == 1:
            buttons |= lay.ButtonState.LeftButton
        elif mouse_buttons == 2:
            buttons |= lay.ButtonState.RightButton
        elif mouse_buttons == 4:
            buttons |= lay.ButtonState.MidButton

        return buttons

    def on_scroll(self, event):
        delta = event["deltaY"]
        # x = event["offsetX"]
        # y = event["offsetY"]
        self.wheel_info.value = f"scroll event: {event}"
        # buttons = self._get_modifier_buttons(event)
        # TODO: this is what I *want* to respond with, but it doesn't work, so I am using zoom_in/zoom_out instead
        # self.layout_view.send_wheel_event(delta, False, db.Point(x, y), buttons)
        if delta < 0:
            self.layout_view.zoom_in()
        else:
            self.layout_view.zoom_out()
        self.refresh()

    def on_mouse_down(self, event):
        x = event["offsetX"]
        y = event["offsetY"]
        moved_x = event["movementX"]
        moved_y = event["movementY"]
        buttons = self._get_modifier_buttons(event)
        # TODO: this part is also not working. why?
        if event == "mousedown":
            self.layout_view.send_mouse_press_event(db.Point(x, y), buttons)
        elif event == "mouseup":
            self.layout_view.send_mouse_release_event(db.Point(x, y), buttons)
        elif event == "mousemove":
            self.layout_view.send_mouse_move_event(db.Point(moved_x, moved_y), buttons)
        self.refresh()
        self.mouse_info.value = f"mouse event: {event}"
