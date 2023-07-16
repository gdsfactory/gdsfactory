
//  TODO: shouldn't be explicit here ..
// let url = 'ws://localhost:8765/ws';

let ws_url = current_url.replace("http://", "ws://");
ws_url = ws_url.replace("https://", "wss://");
ws_url += '/ws?' + "variant=" + cell_variant;
let url = ws_url;
console.log(url);


var canvas = document.getElementById("layout_canvas");
var context = canvas.getContext("2d");

var message = document.getElementById("message");

//  HTML5 does not have a resize event, so we need to poll
//  to generate events for the canvas resize

var lastCanvasWidth = 0;
var lastCanvasHeight = 0;

setInterval(function () {

  var view = document.getElementById('layout-view');
  var w = view.clientWidth;
  var h = view.clientHeight;

  if (lastCanvasWidth !== w || lastCanvasHeight !== h) {

    //  this avoids flicker:

    var tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    var tempContext = tempCanvas.getContext("2d");
    tempContext.drawImage(context.canvas, 0, 0);

    lastCanvasWidth = w;
    lastCanvasHeight = h;
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    context.drawImage(tempContext.canvas, 0, 0);
    if (socket.connected) {
      socket.send(JSON.stringify({ msg: "resize", width: canvas.width, height: canvas.height }));
    }
    else {
      console.log("Socket is not connected. Loading static image data...")
      const img = new Image();
      img.src = initial_image_data;
      img.onload = () => {
        context.drawImage(img, 0, 0, img.width, img.height,     // source rectangle
          0, 0, canvas.width, canvas.height); // destination rectangle
      }

      //  resizes the layer list:

      let layers = document.getElementById("layers");

      var padding = 10; //  padding in pixels
      layers.style.height = (h - 2 * padding) + "px";

    };
  }

}, 10)

let socket = new WebSocket(url);
socket.binaryType = "blob";
var initialized = false;

//  Installs a handler called when the connection is established
socket.onopen = function (evt) {

  var ev = { msg: "initialize", width: canvas.width, height: canvas.height };
  socket.send(JSON.stringify(ev));

  //  Prevents the context menu to show up over the canvas area
  canvas.addEventListener('contextmenu', function (evt) {
    evt.preventDefault();
  });

  canvas.addEventListener('mousemove', function (evt) {
    sendMouseEvent(canvas, "mouse_move", evt);
    evt.preventDefault();
  }, false);

  canvas.addEventListener('click', function (evt) {
    sendMouseEvent(canvas, "mouse_click", evt);
    evt.preventDefault();
  }, false);

  canvas.addEventListener('dblclick', function (evt) {
    sendMouseEvent(canvas, "mouse_dblclick", evt);
    evt.preventDefault();
  }, false);

  canvas.addEventListener('mousedown', function (evt) {
    sendMouseEvent(canvas, "mouse_pressed", evt);
    evt.preventDefault();
  }, false);

  canvas.addEventListener('mouseup', function (evt) {
    sendMouseEvent(canvas, "mouse_released", evt);
    evt.preventDefault();
  }, false);

  canvas.addEventListener('mouseenter', function (evt) {
    sendMouseEvent(canvas, "mouse_enter", evt);
    evt.preventDefault();
  }, false);

  canvas.addEventListener('mouseout', function (evt) {
    sendMouseEvent(canvas, "mouse_leave", evt);
    evt.preventDefault();
  }, false);

  canvas.addEventListener('wheel', function (evt) {
    sendWheelEvent(canvas, "wheel", evt);
    evt.preventDefault();
  }, false);
}

//  Installs a handler for the messages delivered by the web socket
socket.onmessage = function (evt) {

  let data = evt.data;
  if (typeof (data) === "string") {

    //  For debugging:
    //  message.textContent = data;

    //  incoming messages are JSON objects
    js = JSON.parse(data);
    if (js.msg == "initialized") {
      initialized = true;
    } else if (js.msg == "loaded") {
      showLayers(js.layers);
      showMenu(js.modes, js.annotations);
    }

  } else if (initialized) {

    //  incoming blob messages are paint events
    createImageBitmap(data).then(function (image) {
      context.drawImage(image, 0, 0)
    });

  }

};

socket.onclose = evt => console.log(`Closed ${evt.code}`);

function mouseEventToJSON(canvas, type, evt) {

  var rect = canvas.getBoundingClientRect();
  var x = evt.clientX - rect.left;
  var y = evt.clientY - rect.top;
  var keys = 0;
  if (evt.shiftKey) {
    keys += 1;
  }
  if (evt.ctrlKey) {
    keys += 2;
  }
  if (evt.altKey) {
    keys += 4;
  }
  return { msg: type, x: x, y: y, b: evt.buttons, k: keys };

}

function sendMouseEvent(canvas, type, evt) {

  if (socket.readyState == 1 /*OPEN*/) {
    var ev = mouseEventToJSON(canvas, type, evt);
    socket.send(JSON.stringify(ev));
  }

}

function sendWheelEvent(canvas, type, evt) {

  if (socket.readyState == 1 /*OPEN*/) {
    var ev = mouseEventToJSON(canvas, type, evt);
    ev.dx = evt.deltaX;
    ev.dy = evt.deltaY;
    ev.dm = evt.deltaMode;
    socket.send(JSON.stringify(ev));
  }

}



//  Updates the layer list
function showMenu(modes, annotations) {

  var modeElement = document.getElementById("modes");
  modeElement.childNodes = new Array();

  var modeTable = document.createElement("table");
  modeTable.className = "modes-table";
  modeElement.appendChild(modeTable)

  var modeRow = document.createElement("tr");
  modeRow.className = "mode-row-header";
  modeRow.id = "mode-row";
  modeTable.appendChild(modeRow)

  var cell;
  var inner;

  modes.forEach(function (m) {

    cell = document.createElement("td");
    cell.className = "mode-cell";

    var inner = document.createElement("input");
    inner.value = m;
    inner.type = "button";
    inner.className = "unchecked";
    inner.onclick = function () {
      var modeRow = document.getElementById("mode-row");
      modeRow.childNodes.forEach(function (e) {
        e.firstChild.className = "unchecked";
      });
      inner.className = "checked";
      socket.send(JSON.stringify({ msg: "select-mode", value: m }));
    };

    cell.appendChild(inner);
    modeRow.appendChild(cell);

  });

  var menuElement = document.getElementById("menu");

  var menuTable = document.createElement("table");
  menuTable.className = "menu-table";
  menuElement.appendChild(menuTable)

  var menuRow = document.createElement("tr");
  menuRow.className = "menu-row-header";
  menuTable.appendChild(menuRow)

  cell = document.createElement("td");
  cell.className = "menu-cell";
  menuRow.appendChild(cell);

  var rulersSelect = document.createElement("select");
  rulersSelect.onchange = function () {
    socket.send(JSON.stringify({ msg: "select-ruler", value: rulersSelect.selectedIndex }));
  };
  cell.appendChild(rulersSelect);

  cell = document.createElement("td");
  cell.className = "menu-cell";
  menuRow.appendChild(cell);

  var clearRulers = document.createElement("input");
  clearRulers.value = "Clear Rulers";
  clearRulers.type = "button";
  clearRulers.onclick = function () {
    socket.send(JSON.stringify({ msg: "clear-annotations" }));
  };
  cell.appendChild(clearRulers);

  var index = 0;

  annotations.forEach(function (a) {

    var option = document.createElement("option");
    option.value = index;
    option.text = a;

    rulersSelect.appendChild(option);

    index += 1;

  });
}

//  Updates the layer list
function showLayers(layers) {

  var layerElement = document.getElementById("layers");
  layerElement.childNodes = new Array();

  var layerTable = document.createElement("table");
  layerTable.className = "layer-table";
  layerElement.appendChild(layerTable)

  var cell;
  var inner;
  var s;
  var visibilityCheckboxes = [];

  var layerRow = document.createElement("tr");
  layerRow.className = "layer-row-header";

  //  create a top level entry for resetting/setting all visible flags

  cell = document.createElement("td");
  cell.className = "layer-visible-cell";

  inner = document.createElement("input");
  inner.type = "checkbox";
  inner.checked = true;
  inner.onclick = function () {
    var checked = this.checked;
    visibilityCheckboxes.forEach(function (cb) {
      cb.checked = checked;
    });
    socket.send(JSON.stringify({ msg: "layer-v-all", value: checked }));
  };
  cell.appendChild(inner);

  layerRow.appendChild(cell);
  layerTable.appendChild(layerRow);

  //  create table rows for each layer

  layers.forEach(function (l) {

    var layerRow = document.createElement("tr");
    layerRow.className = "layer-row";

    cell = document.createElement("td");
    cell.className = "layer-visible-cell";

    inner = document.createElement("input");
    visibilityCheckboxes.push(inner);
    inner.type = "checkbox";
    inner.checked = l.v;
    inner.onclick = function () {
      socket.send(JSON.stringify({ msg: "layer-v", id: l.id, value: this.checked }));
    };
    cell.appendChild(inner);

    layerRow.appendChild(cell);

    cell = document.createElement("td");
    cell.className = "layer-color-cell";
    s = "border-style: solid; border-width: " + (l.w < 0 ? 1 : l.w) + "px; border-color: #" + (l.fc & 0xffffff).toString(16) + ";";
    cell.style = s;
    layerRow.appendChild(cell);

    inner = document.createElement("div");
    s = "width: 2rem; height: 1em;";
    s += "margin: 1px;";
    s += "background: #" + (l.c & 0xffffff).toString(16) + ";";
    inner.style = s;
    cell.appendChild(inner);

    cell = document.createElement("td");
    cell.className = "layer-name-cell";
    cell.textContent = (l.name != 0 ? l.name : l.s);
    layerRow.appendChild(cell);

    layerTable.appendChild(layerRow);

  });

}
