try:
    from ipywidgets import HTML, Image
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
    def __init__(self, filepath: str, layer_properties: Optional[str]):
        filepath = str(filepath)
        layer_properties = str(layer_properties)
        self.filepath = filepath
        self.layer_properties = layer_properties
        self.layout_view = lay.LayoutView()
        self.load_layout(filepath, layer_properties)
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
