import asyncio
import os
import socket

# import requests
import uvicorn

from kweb.main import app

global jupyter_server
jupyter_server = None
global host
host = None
global port
port = None


# TODO: Don't start a server if there is one running and the version
# is similar or higher


def is_port_in_use(
    port: int,
    host: str = "localhost",
) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _run() -> None:
    global jupyter_server
    global host
    global port

    _port = os.getenv("KWEB_PORT")
    port = 8081 if _port is None else int(_port)
    host = os.getenv("KWEB_HOST")
    if host is None:
        host = "localhost"
    # if is_port_in_use(host=host, port=port):
    #     req = requests.request("get", f"http://{host}:{port}/status")

    config = uvicorn.Config(app)
    config.port = port
    config.host = host

    jupyter_server = uvicorn.Server(config)
    loop = asyncio.get_event_loop()
    loop.create_task(jupyter_server.serve())


def _server_is_running() -> bool:
    global jupyter_server
    return False if jupyter_server is None else jupyter_server.started


def start() -> None:
    """Start a jupyter_server if it's nor already started."""
    if not _server_is_running():
        _run()
