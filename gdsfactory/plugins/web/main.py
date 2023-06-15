import base64
import importlib
import os
from pathlib import Path
from typing import Optional

import orjson
from fastapi import FastAPI, Form, Request, status
from gdsfactory.plugins.web.middleware import ProxiedHeadersMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from starlette.routing import WebSocketRoute

import gdsfactory as gf
from gdsfactory.config import PATH
from gdsfactory.plugins.web.server import LayoutViewServerEndpoint, get_layout_view

module_path = Path(__file__).parent.absolute()

app = FastAPI(
    routes=[WebSocketRoute("/view/{cell_name}/ws", endpoint=LayoutViewServerEndpoint)]
)
app.add_middleware(ProxiedHeadersMiddleware)
# app = FastAPI()
app.mount("/static", StaticFiles(directory=PATH.web / "static"), name="static")

# gdsfiles = StaticFiles(directory=home_path)
# app.mount("/gds_files", gdsfiles, name="gds_files")
templates = Jinja2Templates(directory=PATH.web / "templates")


def load_pdk() -> gf.Pdk:
    pdk = os.environ.get("PDK", "generic")

    if pdk == "generic":
        active_pdk = gf.get_active_pdk()
    else:
        active_module = importlib.import_module(pdk)
        active_pdk = active_module.PDK
        active_pdk.activate()
    return active_pdk


def get_url(request: Request) -> str:
    port_mod = ""
    if request.url.port is not None and len(str(request.url).split(".")) < 3:
        port_mod = ":" + str(request.url.port)

    hostname = request.url.hostname

    if "github" in hostname:
        port_mod = ""

    url = str(
        request.url.scheme
        + "://"
        + (hostname or "localhost")
        + port_mod
        + request.url.path
    )

    return url


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if "preview.app.github" in str(request.url):
        return RedirectResponse(str(request.url).replace(".preview", ""))
    active_pdk = load_pdk()
    pdk_name = active_pdk.name
    components = list(active_pdk.cells.keys())
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Main",
            "pdk_name": pdk_name,
            "components": sorted(components),
        },
    )


LOADED_COMPONENTS = {}


@app.get("/view/{cell_name}", response_class=HTMLResponse)
async def view_cell(request: Request, cell_name: str, variant: Optional[str] = None):
    if "preview.app.github" in str(request.url):
        return RedirectResponse(str(request.url).replace(".preview", ""))

    if variant in LOADED_COMPONENTS:
        component = LOADED_COMPONENTS[variant]
    else:
        component = gf.get_component(cell_name)
    layout_view = get_layout_view(component)
    pixel_data = layout_view.get_pixels_with_options(800, 400).to_png_data()
    # pixel_data = layout_view.get_screenshot_pixels().to_png_data()
    b64_data = base64.b64encode(pixel_data).decode("utf-8")
    return templates.TemplateResponse(
        "viewer.html",
        {
            "request": request,
            "cell_name": str(cell_name),
            "variant": variant,
            "title": "Viewer",
            "initial_view": b64_data,
            "component": component,
            "url": get_url(request),
        },
    )


def _parse_value(value: str):
    if value.startswith("{") or value.startswith("["):
        try:
            return orjson.loads(value.replace("'", '"'))
        except orjson.JSONDecodeError as e:
            raise ValueError(f"Unable to decode parameter value, {value}: {e.msg}")
    else:
        return value


@app.post("/update/{cell_name}")
async def update_cell(request: Request, cell_name: str):
    data = await request.form()
    changed_settings = {k: _parse_value(v) for k, v in data.items() if v != ""}
    new_component = gf.get_component(
        {"component": cell_name, "settings": changed_settings}
    )
    LOADED_COMPONENTS[new_component.name] = new_component
    logger.info(data)
    return RedirectResponse(
        f"/view/{cell_name}?variant={new_component.name}",
        status_code=status.HTTP_302_FOUND,
    )


@app.post("/search", response_class=RedirectResponse)
async def search(name: str = Form(...)):
    logger.info(f"Searching for {name}...")
    try:
        gf.get_component(name)
    except ValueError:
        return RedirectResponse("/", status_code=status.HTTP_404_NOT_FOUND)
    logger.info(f"Successfully found {name}! Redirecting...")
    return RedirectResponse(f"/view/{name}", status_code=status.HTTP_302_FOUND)


if __name__ == "__main__":
    pdk = load_pdk()
