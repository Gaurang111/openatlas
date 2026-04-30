"""
OpenAtlas — Gesture-Controlled Knowledge Graph
-----------------------------------------------
Usage:
  openatlas                            run with built-in AI/ML sample data
  openatlas --csv my_graph.csv         load your own graph
  openatlas --camera 1                 use a different webcam
  openatlas --camera http://YOUR_PHONE_IP:4747/video  use DroidCam / IP camera
  openatlas --port 8765                custom WebSocket port
  openatlas --http-port 8080           custom HTTP port
  openatlas --no-window                hide the OpenCV debug window
  openatlas --stream-every 2           frame skip (lower = smoother, more CPU)
  openatlas --zoom-sensitivity 5.0     tune zoom speed
  openatlas --pan-sensitivity 1800     tune pan speed
  openatlas --pinch-threshold 0.07     tune pinch detection

Gestures:
  ☝  One hand, index extended      → Move cursor (tracks index tip)
  ☝→✊ Point then fist             → Select hovered node
  ✊  One hand fist + move          → Pan graph
  ✊✊ Both hands pinch + expand    → Zoom in / out
  R key                             → Fit / reset view

Install:  pip install git+https://github.com/YOURUSER/openatlas.git
Run:      openatlas
"""

import argparse
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import asyncio
import http.server
import functools
import webbrowser
import websockets
import json
import threading
import math
import urllib.request
import os
import base64

# ── Paths ──────────────────────────────────────────────────────────────────────
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(os.path.expanduser("~"), ".openatlas")
MODEL_PATH  = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print(f"[MODEL] {MODEL_PATH} found.")
        return
    print("[MODEL] Downloading hand_landmarker.task (~5 MB, first run only)…")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[MODEL] Download complete.")

# ── WebSocket ──────────────────────────────────────────────────────────────────
clients    = set()
loop       = asyncio.new_event_loop()
loaded_csv = None   # CSV string pushed to every new browser client on connect

async def ws_handler(websocket):
    clients.add(websocket)
    print(f"[WS] Client connected (total: {len(clients)})")
    if loaded_csv:
        await websocket.send(json.dumps({"command": "load_graph", "value": loaded_csv}))
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)

async def ws_main(port):
    async with websockets.serve(ws_handler, "localhost", port):
        print(f"[WS] Listening on ws://localhost:{port}")
        await asyncio.Future()

def start_ws(port):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ws_main(port))

def broadcast(command, value=None):
    if not clients:
        return
    msg = json.dumps({"command": command, "value": value})
    async def _go():
        await asyncio.gather(*[c.send(msg) for c in list(clients)],
                             return_exceptions=True)
    asyncio.run_coroutine_threadsafe(_go(), loop)

def broadcast_frame(frame, quality=100):
    if not clients:
        return
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    msg = json.dumps({"type": "frame", "data": b64})
    async def _go():
        await asyncio.gather(*[c.send(msg) for c in list(clients)],
                             return_exceptions=True)
    asyncio.run_coroutine_threadsafe(_go(), loop)

# ── HTTP server ────────────────────────────────────────────────────────────────
def make_http_handler(ws_port):
    """Returns an HTTP handler that serves index.html with the correct WS port."""
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path in ('/', '/index.html'):
                path = os.path.join(PACKAGE_DIR, 'index.html')
                with open(path, 'rb') as f:
                    content = f.read().decode('utf-8')
                content = content.replace('ws://localhost:8765',
                                          f'ws://localhost:{ws_port}')
                data = content.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                # fall back to static file serving for any other assets
                self.directory = PACKAGE_DIR
                super().do_GET()

        def log_message(self, *args):
            pass  # silence HTTP logs

    return Handler

def start_http(http_port, ws_port):
    Handler = make_http_handler(ws_port)
    httpd = http.server.HTTPServer(('localhost', http_port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"[HTTP] Serving on http://localhost:{http_port}")

# ── Shared landmark state ──────────────────────────────────────────────────────
latest_hands = []
lm_lock      = threading.Lock()

def on_result(result, _image, _ts):
    global latest_hands
    with lm_lock:
        latest_hands = result.hand_landmarks

# ── Geometry ───────────────────────────────────────────────────────────────────
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def palm_center(lm):
    return (lm[0].x + lm[9].x) / 2, (lm[0].y + lm[9].y) / 2

def pinch_point(lm):
    return (lm[4].x + lm[8].x) / 2, (lm[4].y + lm[8].y) / 2

def is_pinching(lm, threshold=0.07):
    return dist(lm[4], lm[8]) < threshold

def is_fist(lm):
    wrist = lm[0]
    pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]
    curled = sum(1 for tip, mcp in pairs
                 if dist(lm[tip], wrist) < dist(lm[mcp], wrist))
    return curled >= 3

def is_pointing(lm):
    wrist = lm[0]
    index_ext  = dist(lm[8], wrist) > dist(lm[5], wrist) * 1.6
    middle_not = dist(lm[12], wrist) < dist(lm[9], wrist) * 1.5
    return index_ext and middle_not

def two_hand_dist(lm0, lm1):
    c0 = palm_center(lm0)
    c1 = palm_center(lm1)
    return math.hypot(c1[0] - c0[0], c1[1] - c0[1])

# ── EMA smoother ───────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, alpha=0.30):
        self.alpha = alpha
        self.val   = None

    def update(self, x):
        self.val = x if self.val is None else self.alpha * x + (1 - self.alpha) * self.val
        return self.val

    def reset(self):
        self.val = None

# ── Hand skeleton ───────────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

_PALETTE = {
    "zoom":   ((0,   220, 100), (0,   180,  80), (0,   255, 160)),
    "pan":    ((0,   150, 255), (0,   120, 200), (0,   200, 255)),
    "cursor": ((255, 180,   0), (220, 140,   0), (255, 230,  60)),
    "idle":   ((70,   70,  70), (50,   50,  50), (110, 110, 110)),
}

def draw_hand(frame, lm, h, w, mode):
    pts = [(int(l.x * w), int(l.y * h)) for l in lm]
    lc, jc, tc = _PALETTE.get(mode, _PALETTE["idle"])

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], lc, 1)

    tip_indices = {"cursor": (8,), "zoom": (4, 8), "pan": (), "idle": ()}.get(mode, ())

    for i, (x, y) in enumerate(pts):
        r = 6 if i in tip_indices else 3
        c = tc if i in tip_indices else jc
        cv2.circle(frame, (x, y), r, c, -1)
        cv2.circle(frame, (x, y), r, (0, 0, 0), 1)

def draw_zoom_line(frame, pp0, pp1, h, w):
    p0 = (int(pp0[0] * w), int(pp0[1] * h))
    p1 = (int(pp1[0] * w), int(pp1[1] * h))
    cv2.line(frame, p0, p1, (0, 255, 160), 1)
    mx, my = (p0[0]+p1[0])//2, (p0[1]+p1[1])//2
    d = math.hypot(pp1[0]-pp0[0], pp1[1]-pp0[1])
    cv2.putText(frame, f"{d:.2f}", (mx-20, my-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 160), 1)

def draw_cursor_ray(frame, lm, h, w):
    tip = (int(lm[8].x * w), int(lm[8].y * h))
    mcp = (int(lm[5].x * w), int(lm[5].y * h))
    cv2.line(frame, mcp, tip, (255, 200, 0), 1)
    cv2.circle(frame, tip, 7, (255, 230, 60), -1)
    cv2.circle(frame, tip, 7, (0, 0, 0), 1)

# ── Gesture Controller ─────────────────────────────────────────────────────────
class GestureCtrl:
    SMOOTH_ALPHA     = 0.28
    PINCH_THRESHOLD  = 0.07
    ZOOM_SENSITIVITY = 5.0
    ZOOM_DEADZONE    = 0.006
    PAN_SENSITIVITY  = 1800
    PAN_DEADZONE     = 0.004

    def __init__(self):
        self.dist_ema          = EMA(self.SMOOTH_ALPHA)
        self.prev_dist         = None
        self.zoom_pp           = None
        self.pan_x             = EMA(self.SMOOTH_ALPHA)
        self.pan_y             = EMA(self.SMOOTH_ALPHA)
        self.prev_px           = None
        self.prev_py           = None
        self.cur_x             = EMA(self.SMOOTH_ALPHA)
        self.cur_y             = EMA(self.SMOOTH_ALPHA)
        self.prev_was_pointing = False
        self.msg               = "Waiting for hands…"
        self.mode              = "–"

    def _reset_zoom(self):
        self.dist_ema.reset()
        self.prev_dist = None
        self.zoom_pp   = None

    def _reset_pan(self):
        self.pan_x.reset(); self.pan_y.reset()
        self.prev_px = None; self.prev_py = None

    def _reset_cursor(self):
        self.cur_x.reset(); self.cur_y.reset()

    def tick(self, hands):
        if len(hands) == 2:
            self._reset_pan()
            self._reset_cursor()
            self.prev_was_pointing = False

            p0 = is_pinching(hands[0], self.PINCH_THRESHOLD)
            p1 = is_pinching(hands[1], self.PINCH_THRESHOLD)

            if p0 and p1:
                pp0 = pinch_point(hands[0])
                pp1 = pinch_point(hands[1])
                self.zoom_pp = (pp0, pp1)

                raw_d = math.hypot(pp1[0]-pp0[0], pp1[1]-pp0[1])
                sd    = self.dist_ema.update(raw_d)

                if self.prev_dist is not None:
                    dd = sd - self.prev_dist
                    if abs(dd) > self.ZOOM_DEADZONE:
                        val = round(abs(dd) * self.ZOOM_SENSITIVITY, 4)
                        if dd > 0:
                            broadcast("zoom_in",  val)
                            self.msg  = f"ZOOM IN   +{val:.3f}"
                            self.mode = "ZOOM IN"
                        else:
                            broadcast("zoom_out", val)
                            self.msg  = f"ZOOM OUT  -{val:.3f}"
                            self.mode = "ZOOM OUT"
                    else:
                        self.msg  = "Both pinched — expand / shrink"
                        self.mode = "PINCHED"

                self.prev_dist = sd
                return ["zoom", "zoom"]

            else:
                self._reset_zoom()
                self.msg  = "Pinch both hands to zoom"
                self.mode = "idle"
                return ["idle", "idle"]

        elif len(hands) == 1:
            self._reset_zoom()
            lm = hands[0]

            pointing = is_pointing(lm)
            fst      = is_fist(lm)

            cx, cy_n = palm_center(lm)
            sx = self.pan_x.update(cx)
            sy = self.pan_y.update(cy_n)

            if pointing:
                self._reset_pan()
                tx = self.cur_x.update(lm[8].x)
                ty = self.cur_y.update(lm[8].y)
                broadcast("cursor", {"x": round(tx, 4), "y": round(ty, 4), "active": True})
                self.prev_was_pointing = True
                self.msg  = f"CURSOR  ({tx:.2f}, {ty:.2f})"
                self.mode = "CURSOR"
                return ["cursor"]

            elif fst and self.prev_was_pointing:
                self._reset_cursor()
                self.prev_was_pointing = False
                broadcast("select", None)
                self.msg  = "SELECT"
                self.mode = "SELECT"
                self.prev_px = sx; self.prev_py = sy
                return ["pan"]

            elif fst:
                self._reset_cursor()
                self.prev_was_pointing = False
                if self.prev_px is not None:
                    dx = (sx - self.prev_px) * self.PAN_SENSITIVITY
                    dy = (sy - self.prev_py) * self.PAN_SENSITIVITY
                    if (abs(sx - self.prev_px) > self.PAN_DEADZONE or
                            abs(sy - self.prev_py) > self.PAN_DEADZONE):
                        broadcast("pan", {"x": round(dx, 1), "y": round(dy, 1)})
                        self.msg  = f"PAN  dx={dx:+.0f}  dy={dy:+.0f}"
                        self.mode = "PAN"
                    else:
                        self.msg  = "Fist — move to pan"
                        self.mode = "FIST"
                else:
                    self.msg  = "Fist — move to pan"
                    self.mode = "FIST"
                self.prev_px = sx; self.prev_py = sy
                return ["pan"]

            else:
                self._reset_cursor()
                self.prev_was_pointing = False
                broadcast("cursor", {"active": False})
                self.msg  = "Point=cursor  Fist=pan"
                self.mode = "open"
                self.prev_px = sx; self.prev_py = sy
                return ["idle"]

        else:
            self._reset_zoom()
            self._reset_pan()
            self._reset_cursor()
            self.prev_was_pointing = False
            broadcast("cursor", {"active": False})
            self.msg  = "No hands detected"
            self.mode = "–"
            return []

# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        prog="openatlas",
        description="OpenAtlas — Gesture-Controlled Knowledge Graph Visualizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",
                   default=None,
                   help="Path to CSV file with columns: source, target, relationship. "
                        "Omit to use the built-in AI/ML sample graph.")
    p.add_argument("--camera",
                   default="0",
                   help="Webcam index (0, 1, …) or stream URL (e.g. http://IP:4747/video)")
    p.add_argument("--port",
                   type=int, default=8765,
                   help="WebSocket port")
    p.add_argument("--http-port",
                   type=int, default=8080,
                   help="HTTP port used to serve the browser UI")
    p.add_argument("--no-window",
                   action="store_true",
                   help="Hide the OpenCV debug window")
    p.add_argument("--stream-every",
                   type=int, default=2,
                   help="Send every Nth camera frame to the browser (lower = smoother but more CPU)")
    p.add_argument("--zoom-sensitivity",
                   type=float, default=5.0,
                   help="Zoom gesture sensitivity")
    p.add_argument("--pan-sensitivity",
                   type=float, default=1800.0,
                   help="Pan gesture sensitivity")
    p.add_argument("--pinch-threshold",
                   type=float, default=0.07,
                   help="Pinch detection threshold (0.0–1.0, smaller = tighter pinch needed)")
    return p.parse_args()

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    global loaded_csv
    args = parse_args()

    ensure_model()

    # Load CSV — custom or bundled sample
    csv_path = args.csv or os.path.join(PACKAGE_DIR, "sample_data.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return
    with open(csv_path, encoding="utf-8") as f:
        loaded_csv = f.read()
    label = "sample data" if args.csv is None else args.csv
    print(f"[DATA] Loaded {label}  ({loaded_csv.count(chr(10))} rows)")

    # Camera source — integer index or URL string
    camera_src = args.camera
    try:
        camera_src = int(camera_src)
    except ValueError:
        pass  # keep as URL string

    # Start servers
    threading.Thread(target=start_ws, args=(args.port,), daemon=True).start()
    start_http(args.http_port, args.port)

    url = f"http://localhost:{args.http_port}/index.html"
    print(f"[BROWSER] Opening {url}")
    webbrowser.open(url)

    # MediaPipe
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        result_callback=on_result,
    )
    lander = mp_vision.HandLandmarker.create_from_options(opts)
    ctrl   = GestureCtrl()
    ctrl.ZOOM_SENSITIVITY = args.zoom_sensitivity
    ctrl.PAN_SENSITIVITY  = args.pan_sensitivity
    ctrl.PINCH_THRESHOLD  = args.pinch_threshold

    cap = cv2.VideoCapture(camera_src)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera: {camera_src}")
        return

    ts = 0
    print("[CAM] Q = quit   R = fit/reset graph")
    print("[GESTURES]  ☝=cursor  ☝→✊=select  ✊=pan  ✊✊pinch both=zoom")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts += 1
        lander.detect_async(mpi, ts)

        with lm_lock:
            hands = list(latest_hands)

        draw_modes = ctrl.tick(hands)

        for lm, mode in zip(hands, draw_modes):
            draw_hand(frame, lm, h, w, mode)
            if mode == "cursor":
                draw_cursor_ray(frame, lm, h, w)

        if ctrl.zoom_pp:
            draw_zoom_line(frame, ctrl.zoom_pp[0], ctrl.zoom_pp[1], h, w)

        cv2.rectangle(frame, (0, 0), (w, 100), (10, 10, 22), -1)
        cv2.putText(frame, ctrl.msg, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (60, 210, 255), 2)

        mode_colors = {
            "ZOOM IN":  (100, 255, 100),
            "ZOOM OUT": (100, 180, 255),
            "PAN":      (0,   165, 255),
            "FIST":     (200, 200,   0),
            "PINCHED":  (180, 255, 180),
            "CURSOR":   (255, 200,  50),
            "SELECT":   (255, 100, 100),
            "open":     (70,   70,  70),
            "idle":     (70,   70,  70),
            "–":        (70,   70,  70),
        }
        mc = mode_colors.get(ctrl.mode, (80, 80, 80))
        cv2.putText(frame, f"MODE: {ctrl.mode}", (10, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, mc, 2)
        cv2.putText(frame, f"hands: {len(hands)}   ws: {len(clients)}",
                    (w - 230, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 255, 120), 1)

        cv2.rectangle(frame, (0, h - 34), (w, h), (10, 10, 22), -1)
        cv2.putText(frame,
                    "☝=cursor  ☝>✊=select  ✊=pan  pinch both hands=zoom  R=reset",
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 120), 1)

        if ts % args.stream_every == 0:
            broadcast_frame(frame)

        if not args.no_window:
            cv2.imshow("OpenAtlas  |  Q=quit  R=reset", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            broadcast("fit")
            ctrl.msg = "FIT ALL"

    lander.close()
    cap.release()
    cv2.destroyAllWindows()
