import base64
import importlib
import os
from pathlib import Path
import pathlib
from typing import Optional

from glob import glob
import orjson

from fastapi import Form, HTTPException
from fastapi import FastAPI, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from loguru import logger
from starlette.routing import WebSocketRoute

import gdsfactory as gf
from gdsfactory.plugins.web.middleware import ProxiedHeadersMiddleware
from gdsfactory.config import PATH, GDSDIR_TEMP, CONF
from gdsfactory.plugins.web.server import LayoutViewServerEndpoint, get_layout_view

from gdsfactory.watch import FileWatcher
from gdsfactory.cell import Settings

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
        port_mod = f":{str(request.url.port)}"

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
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/gds_list", response_class=HTMLResponse)
async def gds_list(request: Request):
    """List all saved GDS files."""
    files_root = GDSDIR_TEMP
    paths_list = glob(str(files_root / "*.gds"))
    files_list = sorted(Path(gdsfile).stem for gdsfile in paths_list)
    files_metadata = [
        {"name": file_name, "url": f"view/{file_name}"} for file_name in files_list
    ]
    return templates.TemplateResponse(
        "file_browser.html",
        {
            "request": request,
            "message": f"GDS files in {str(files_root)!r}",
            "files_root": files_root,
            "files_metadata": files_metadata,
        },
    )


@app.get("/gds_current", response_class=HTMLResponse)
async def gds_current(request: Request) -> RedirectResponse:
    """List all saved GDS files."""
    if CONF.last_saved_files:
        return RedirectResponse(f"/view/{CONF.last_saved_files[-1].stem}")
    else:
        return RedirectResponse(
            "/",
            status_code=status.HTTP_302_FOUND,
        )


@app.get("/pdk", response_class=HTMLResponse)
async def pdk(request: Request):
    if "preview.app.github" in str(request.url):
        return RedirectResponse(str(request.url).replace(".preview", ""))
    active_pdk = load_pdk()
    pdk_name = active_pdk.name
    components = list(active_pdk.cells.keys())
    return templates.TemplateResponse(
        "pdk.html",
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
    gds_files = GDSDIR_TEMP.glob("*.gds")
    gds_names = [gdspath.stem for gdspath in gds_files]

    if "preview.app.github" in str(request.url):
        return RedirectResponse(str(request.url).replace(".preview", ""))

    if variant in LOADED_COMPONENTS:
        component = LOADED_COMPONENTS[variant]
    else:
        try:
            component = gf.get_component(cell_name)
        except Exception as e:
            if cell_name not in gds_names:
                raise HTTPException(
                    status_code=400, detail=f"Component not found. {e}"
                ) from e

            gdspath = GDSDIR_TEMP / cell_name
            component = gf.import_gds(gdspath=gdspath.with_suffix(".gds"))
            component.settings = Settings(name=component.name)
    layout_view = get_layout_view(component)
    pixel_data = layout_view.get_pixels_with_options(800, 400).to_png_data()
    b64_data = base64.b64encode(pixel_data).decode("utf-8")
    return templates.TemplateResponse(
        "viewer.html",
        {
            "request": request,
            "cell_name": cell_name,
            "variant": variant,
            "title": "Viewer",
            "initial_view": b64_data,
            "component": component,
            "url": get_url(request),
        },
    )


def _parse_value(value: str):
    if not value.startswith("{") and not value.startswith("["):
        return value
    try:
        return orjson.loads(value.replace("'", '"'))
    except orjson.JSONDecodeError as e:
        raise ValueError(f"Unable to decode parameter value, {value}: {e.msg}") from e


@app.post("/update/{cell_name}")
async def update_cell(request: Request, cell_name: str):
    """Cell name is the name of the PCell function."""
    data = await request.form()
    settings = {k: _parse_value(v) for k, v in data.items() if v != ""}
    if not settings:
        return RedirectResponse(
            f"/view/{cell_name}",
            status_code=status.HTTP_302_FOUND,
        )
    component = gf.get_component({"component": cell_name, "settings": settings})
    variant = component.name

    LOADED_COMPONENTS[component.name] = component
    return RedirectResponse(
        f"/view/{cell_name}?variant={variant}",
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


#########################
# filewatcher
#######################

watched_folder = None
watcher = None
output = ""
component = None


@app.get("/filewatcher", response_class=HTMLResponse)
async def filewatcher(request: Request):
    global component

    if CONF.last_saved_files:
        component = gf.import_gds(gf.CONF.last_saved_files[-1])
        component.settings = Settings(name=component.name)
    else:
        component = gf.components.straight()

    layout_view = get_layout_view(component)
    pixel_data = layout_view.get_pixels_with_options(800, 400).to_png_data()
    b64_data = base64.b64encode(pixel_data).decode("utf-8")

    return templates.TemplateResponse(
        "filewatcher.html",
        {
            "request": request,
            "output": output,
            "cell_name": str(component.name),
            "variant": None,
            "title": "Viewer",
            "initial_view": b64_data,
            "component": component,
            "url": get_url(request),
        },
    )


@app.post("/filewatcher_start")
async def watch_folder(request: Request, folder_path: str = Form(...)):
    global component
    global output
    global watched_folder
    global watcher

    if folder_path is None or not folder_path.strip():
        raise HTTPException(status_code=400, detail="Folder path is required.")
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail="Folder does not exist.")

    watched_folder = folder_path
    watched_folder = pathlib.Path(folder_path)
    watcher = FileWatcher(path=folder_path)
    watcher.start()
    output += f"watching {watched_folder}\n"
    return RedirectResponse(
        "/filewatcher",
        status_code=status.HTTP_302_FOUND,
    )


@app.get("/filewatcher_stop")
def stop_watcher(request: Request) -> str:
    """Stops filewacher."""
    global watcher
    global watched_folder
    global output

    if watcher:
        watcher.stop()

    message = f"stopped watching {watched_folder}\n"
    output += message
    return message
