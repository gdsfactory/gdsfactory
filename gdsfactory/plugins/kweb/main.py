from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.routing import WebSocketRoute
from starlette.templating import _TemplateResponse

from kweb import __version__ as version
from kweb.server import LayoutViewServerEndpoint

module_path = Path(__file__).parent.absolute()
home_path = Path.home() / ".gdsfactory" / "extra"
home_path.mkdir(exist_ok=True, parents=True)

app = FastAPI(routes=[WebSocketRoute("/gds/ws", endpoint=LayoutViewServerEndpoint)])
app.mount("/static", StaticFiles(directory=module_path / "static"), name="static")

# gdsfiles = StaticFiles(directory=home_path)
# app.mount("/gds_files", gdsfiles, name="gds_files")
templates = Jinja2Templates(directory=module_path / "templates")


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Welcome to kweb visualizer"}


@app.get("/gds", response_class=HTMLResponse)
async def gds_view(
    request: Request, gds_file: str, layer_props: str = str(home_path)
) -> _TemplateResponse:
    url = str(
        request.url.scheme
        + "://"
        + (request.url.hostname or "localhost")
        + ":"
        + str(request.url.port)
        + "/gds"
    )
    return templates.TemplateResponse(
        "client.html",
        {
            "request": request,
            "url": url,
            "gds_file": gds_file,
            "layer_props": layer_props,
        },
    )


@app.get("/gds/{gds_name}", response_class=HTMLResponse)
async def gds_view_static(
    request: Request, gds_name: str, layer_props: str = str(home_path)
) -> _TemplateResponse:
    gds_file = (Path(__file__).parent / f"gds_files/{gds_name}").with_suffix(".gds")

    url = str(
        request.url.scheme
        + "://"
        + (request.url.hostname or "localhost")
        + ":"
        + str(request.url.port)
        + "/gds"
    )

    return templates.TemplateResponse(
        "client.html",
        {
            "request": request,
            "url": url,
            "gds_file": gds_file,
            "layer_props": layer_props,
        },
    )


@app.get("/status")
async def status() -> dict[str, Any]:
    return {"server": "kweb", "version": version}
