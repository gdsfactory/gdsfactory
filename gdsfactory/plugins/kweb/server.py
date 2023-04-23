#!/usr/bin/env python3

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

# NOTE: import db to enable stream format readers
import klayout.db as db
import klayout.lay as lay
from fastapi import WebSocket
from starlette.endpoints import WebSocketEndpoint

host = "localhost"
port = 8765

layout_url = (
    "https://github.com/KLayout/klayout/blob/master/testdata/gds/t10.gds?raw=true"
)


# TODO: Not sure `WebSocket` is the right type here


class LayoutViewServerEndpoint(WebSocketEndpoint):
    encoding = "text"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        _params = self.scope["query_string"].decode("utf-8")
        _params_splitted = _params.split("&")
        params = {}
        for _param in _params_splitted:
            key, value = _param.split("=")
            params[key] = value

        self.url = params["gds_file"]
        self.layer_props = params.get("layer_props", None)

    async def on_connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        await self.connection(websocket)

    async def on_receive(self, websocket: WebSocket, data: str) -> None:
        await self.reader(websocket, data)

    async def on_disconnect(self, websocket: WebSocket, close_code: int) -> None:
        pass

    async def send_image(self, websocket: WebSocket, data: bytes) -> None:
        await websocket.send_bytes(data)

    def image_updated(self, websocket: WebSocket) -> None:
        pixel_buffer = self.layout_view.get_screenshot_pixels()
        asyncio.create_task(self.send_image(websocket, pixel_buffer.to_png_data()))

    def mode_dump(self) -> list[str]:
        return self.layout_view.mode_names()

    def annotation_dump(self) -> list[str]:
        return [d[1] for d in self.layout_view.annotation_templates()]

    def layer_dump(self) -> list[dict[str, object]]:
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

    async def connection(self, websocket: WebSocket, path: str | None = None) -> None:
        self.layout_view = lay.LayoutView()
        self.layout_view.load_layout(self.url)
        if Path(self.layer_props).is_file():
            self.layout_view.load_layer_props(self.layer_props)
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

    async def timer(self, websocket: WebSocket) -> None:
        self.layout_view.on_image_updated_event = (  # type: ignore[attr-defined]
            lambda: self.image_updated(websocket)
        )
        while True:
            self.layout_view.timer()  # type: ignore[attr-defined]
            await asyncio.sleep(0.01)

    def buttons_from_js(self, js: dict[str, int]) -> int:
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

    def wheel_event(
        self, function: Callable[[int, bool, db.Point, int], None], js: dict[str, int]
    ) -> None:
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

    def mouse_event(
        self, function: Callable[[db.Point, int], None], js: dict[str, int]
    ) -> None:
        function(db.Point(js["x"], js["y"]), self.buttons_from_js(js))

    async def reader(self, websocket: WebSocket, data: str) -> None:
        js = json.loads(data)
        msg = js["msg"]
        if msg == "quit":
            return
        elif msg == "resize":
            self.layout_view.resize(js["width"], js["height"])
        elif msg == "clear-annotations":
            self.layout_view.clear_annotations()
        elif msg == "select-ruler":
            ruler = js["value"]
            self.layout_view.set_config("current-ruler-template", str(ruler))
        elif msg == "select-mode":
            mode = js["value"]
            self.layout_view.switch_mode(mode)
        elif msg == "layer-v-all":
            vis = js["value"]
            for layer in self.layout_view.each_layer():
                layer.visible = vis
        elif msg == "layer-v":
            id = js["id"]
            vis = js["value"]
            for layer in self.layout_view.each_layer():
                if layer.id() == id:
                    layer.visible = vis
        elif msg == "initialize":
            self.layout_view.resize(js["width"], js["height"])
            await websocket.send_text(json.dumps({"msg": "initialized"}))
        elif msg == "mode_select":
            self.layout_view.switch_mode(js["mode"])
        elif msg == "mouse_move":
            self.mouse_event(
                self.layout_view.send_mouse_move_event, js  # type: ignore[arg-type]
            )
        elif msg == "mouse_pressed":
            self.mouse_event(
                self.layout_view.send_mouse_press_event, js  # type: ignore[arg-type]
            )
        elif msg == "mouse_released":
            self.mouse_event(
                self.layout_view.send_mouse_release_event, js  # type: ignore[arg-type]
            )
        elif msg == "mouse_enter":
            self.layout_view.send_enter_event()
        elif msg == "mouse_leave":
            self.layout_view.send_leave_event()
        elif msg == "mouse_dblclick":
            self.mouse_event(
                self.layout_view.send_mouse_double_clicked_event,
                js,
            )
        elif msg == "wheel":
            self.wheel_event(
                self.layout_view.send_wheel_event, js  # type: ignore[arg-type]
            )


# server = LayoutViewServerEndpoint(layout_url)
# server.run()
