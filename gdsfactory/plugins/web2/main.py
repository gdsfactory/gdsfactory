from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, WebSocket

from gdsfactory.watch import watch


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/graph", response_class=HTMLResponse)
async def graph_view(request: Request):
    return templates.TemplateResponse("graph.html", {"request": request})


@app.get("/editor", response_class=HTMLResponse)
async def editor_view(request: Request):
    return templates.TemplateResponse("editor.html", {"request": request})


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    while True:
        dirpath = await websocket.receive_text()
        # Do something with the data
        # For instance, start a file watcher on the received path
        # Send back a message to the client
        await websocket.send_text(f"Monitoring folder: {dirpath}")
        watch(str(dirpath))


@app.get("/filewatcher", response_class=HTMLResponse)
async def filewatcher(request: Request):
    return templates.TemplateResponse("filewatcher.html", {"request": request})
