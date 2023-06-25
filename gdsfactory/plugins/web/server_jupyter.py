import asyncio

import uvicorn

from gdsfactory.plugins.web.main import app

global jupyter_server
jupyter_server = None


def _run(port: int = 8000) -> None:
    global jupyter_server

    config = uvicorn.Config(app, port=port)
    jupyter_server = uvicorn.Server(config)
    loop = asyncio.get_event_loop()
    loop.create_task(jupyter_server.serve())


def _server_is_running() -> bool:
    global jupyter_server
    return False if jupyter_server is None else jupyter_server.started


def start(port: int = 8000) -> None:
    """Start a jupyter_server if it's not already started."""
    if not _server_is_running():
        _run(port=port)
