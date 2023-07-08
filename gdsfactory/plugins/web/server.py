#!/usr/bin/env python3
# type: ignore

import asyncio
import json
from typing import Optional

import klayout.db as db
import klayout.lay as lay
from fastapi import WebSocket
from loguru import logger
from starlette.endpoints import WebSocketEndpoint

import gdsfactory as gf
from gdsfactory.component import GDSDIR_TEMP

host = "localhost"
port = 8765


class LayoutViewServerEndpoint(WebSocketEndpoint):
    encoding = "text"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        logger.info("Initialized websocket")
        _params = self.scope["query_string"].decode("utf-8")
        _params_splitted = _params.split("&")
        params = {}
        for _param in _params_splitted:
            key, value = _param.split("=")
            params[key] = value

        # print("args:", args)
        # print("kwargs:", kwargs)
        # self.url = params["gds_file"].replace('/', '\\')
        # self.layer_props = params.get("layer_props", None)
        lyp_path = GDSDIR_TEMP / "layer_props.lyp"
        gf.get_active_pdk().layer_views.to_lyp(lyp_path)
        self.layer_props = lyp_path
        # path_params = args[0]['path_params']
        # cell_name = path_params["cell_name"]
        cell_name = params["variant"]
        # c = gf.get_component(cell_name)
        gds_path = GDSDIR_TEMP / f"{cell_name}.gds"
        # c.write_gds(gds_path)
        self.url = self.gds_path = str(gds_path)

    async def on_connect(self, websocket) -> None:
        await websocket.accept()
        await self.connection(websocket)

    async def on_receive(self, websocket, data) -> None:
        await self.reader(websocket, data)

    async def on_disconnect(self, websocket, close_code) -> None:
        pass

    async def send_image(self, websocket, data) -> None:
        await websocket.send_text(data)

    def image_updated(self, websocket) -> None:
        pixel_buffer = self.layout_view.get_screenshot_pixels()
        asyncio.create_task(self.send_image(websocket, pixel_buffer.to_png_data()))

    def mode_dump(self):
        return self.layout_view.mode_names()

    def annotation_dump(self):
        return [d[1] for d in self.layout_view.annotation_templates()]

    def layer_dump(self):
        js = []
        for layer in self.layout_view.each_layer():
            js.append(
                {
                    "dp": layer.eff_dither_pattern(),
                    "ls": layer.eff_line_style(),
                    "c": layer.eff_fill_color(),
                    "fc": layer.eff_frame_color(),
                    "m": layer.marked,
                    "s": layer.source,
                    "t": layer.transparent,
                    "va": layer.valid,
                    "v": layer.visible,
                    "w": layer.width,
                    "x": layer.xfill,
                    "name": layer.name,
                    "id": layer.id(),
                }
            )
        return js

    async def connection(
        self, websocket: WebSocket, path: Optional[str] = None
    ) -> None:
        self.layout_view = lay.LayoutView()
        url = self.url
        self.layout_view.load_layout(url)
        if self.layer_props is not None:
            self.layout_view.load_layer_props(str(self.layer_props))
        self.layout_view.max_hier()

        await websocket.send_text(
            json.dumps(
                {
                    "msg": "loaded",
                    "modes": self.mode_dump(),
                    "annotations": self.annotation_dump(),
                    "layers": self.layer_dump(),
                }
            )
        )

        asyncio.create_task(self.timer(websocket))

    async def timer(self, websocket) -> None:
        self.layout_view.on_image_updated_event = lambda: self.image_updated(websocket)
        while True:
            self.layout_view.timer()
            await asyncio.sleep(0.01)

    def buttons_from_js(self, js):
        buttons = 0
        k = js["k"]
        b = js["b"]
        if (k & 1) != 0:
            buttons |= lay.ButtonState.ShiftKey
        if (k & 2) != 0:
            buttons |= lay.ButtonState.ControlKey
        if (k & 4) != 0:
            buttons |= lay.ButtonState.AltKey
        if (b & 1) != 0:
            buttons |= lay.ButtonState.LeftButton
        if (b & 2) != 0:
            buttons |= lay.ButtonState.RightButton
        if (b & 4) != 0:
            buttons |= lay.ButtonState.MidButton
        return buttons

    def wheel_event(self, function, js) -> None:
        delta = 0
        dx = js["dx"]
        dy = js["dy"]
        if dx != 0:
            delta = -dx
            horizontal = True
        elif dy != 0:
            delta = -dy
            horizontal = False
        if delta != 0:
            function(
                delta, horizontal, db.Point(js["x"], js["y"]), self.buttons_from_js(js)
            )

    def mouse_event(self, function, js) -> None:
        function(db.Point(js["x"], js["y"]), self.buttons_from_js(js))

    async def reader(self, websocket, data: str) -> None:
        js = json.loads(data)
        msg = js["msg"]
        if msg == "clear-annotations":
            self.layout_view.clear_annotations()
        elif msg == "initialize":
            self.layout_view.resize(js["width"], js["height"])
            await websocket.send_text(json.dumps({"msg": "initialized"}))
        elif msg == "layer-v":
            layer_id = js["id"]
            vis = js["value"]
            for layer in self.layout_view.each_layer():
                if layer.id() == layer_id:
                    layer.visible = vis
        elif msg == "layer-v-all":
            vis = js["value"]
            for layer in self.layout_view.each_layer():
                layer.visible = vis
        elif msg == "mode_select":
            self.layout_view.switch_mode(js["mode"])
        elif msg == "mouse_dblclick":
            self.mouse_event(self.layout_view.send_mouse_double_clicked_event, js)
        elif msg == "mouse_enter":
            self.layout_view.send_enter_event()
        elif msg == "mouse_leave":
            self.layout_view.send_leave_event()
        elif msg == "mouse_move":
            self.mouse_event(self.layout_view.send_mouse_move_event, js)
        elif msg == "mouse_pressed":
            self.mouse_event(self.layout_view.send_mouse_press_event, js)
        elif msg == "mouse_released":
            self.mouse_event(self.layout_view.send_mouse_release_event, js)
        elif msg == "quit":
            return
        elif msg == "resize":
            self.layout_view.resize(js["width"], js["height"])
        elif msg == "select-mode":
            mode = js["value"]
            self.layout_view.switch_mode(mode)
        elif msg == "select-ruler":
            ruler = js["value"]
            self.layout_view.set_config("current-ruler-template", str(ruler))
        elif msg == "wheel":
            self.wheel_event(self.layout_view.send_wheel_event, js)


def get_layer_properties() -> str:
    lyp_path = GDSDIR_TEMP / "layers.lyp"
    lyp_path = gf.get_active_pdk().layer_views.to_lyp(lyp_path)
    return str(lyp_path)


def get_layout_view(component: gf.Component) -> lay.LayoutView:
    """Returns klayout layout view for a gdsfactory Component."""
    gds_path = GDSDIR_TEMP / f"{component.name}.gds"
    component.write_gds(gdspath=str(gds_path))
    layout_view = lay.LayoutView()
    layout_view.load_layout(str(gds_path))
    lyp_path = get_layer_properties()
    layout_view.load_layer_props(str(lyp_path))
    layout_view.max_hier()
    return layout_view
