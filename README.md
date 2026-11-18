# OpenAtlas

**Explore a knowledge graph with your bare hands.** OpenAtlas tracks your hands through a webcam and lets you point, select, pan and zoom a graph in the browser, with no mouse or touchscreen.

[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

Hand tracking uses [MediaPipe](https://developers.google.com/mediapipe). The graph is drawn with [Cytoscape.js](https://js.cytoscape.org/).

<p align="center">
  <a href="https://www.youtube.com/watch?v=VCilobkDwnU">
    <img src="https://img.youtube.com/vi/VCilobkDwnU/maxresdefault.jpg" alt="OpenAtlas demo video" width="500"/>
  </a>
  <br/>
  <a href="https://www.youtube.com/watch?v=VCilobkDwnU">Watch the demo on YouTube</a>
</p>

---

## Quick start

```bash
pip install git+https://github.com/Gaurang111/openatlas.git
openatlas
```

A browser window opens with a sample graph. Raise your hand to start.

> The first run downloads a hand-tracking model of about 5 MB. This happens once.

## Gestures

| Gesture | Action |
|---|---|
| ☝ Point with your index finger | Move the cursor |
| ☝ → ✊ Point, then close into a fist | Select the node under the cursor |
| ✊ Fist and move your arm | Pan the graph |
| ✊✊ Pinch with both hands, then pull apart or push together | Zoom in or out |
| `R` | Fit the whole graph back into view |
| `Q` | Quit |

## How it works

```
Webcam ──> OpenCV ──> MediaPipe hand landmarks ──> gesture recognition
                                                          │
                                                   WebSocket (:8765)
                                                          │
                              Browser UI (:8080) <────────┘
                              Cytoscape.js graph, camera feed, cursor
```

A Python process reads frames, detects hand landmarks, and classifies gestures such as pointing, fist and two-hand pinch. It streams the resulting events over a WebSocket to a browser page, which applies them to the graph.

## Use your own data

```bash
openatlas --csv my_data.csv
```

The CSV needs three columns:

```csv
source,target,relationship
Python,NumPy,uses
NumPy,Linear Algebra,implements
PyTorch,NumPy,built_on
```

## Use your phone as the camera

Install a webcam app such as DroidCam, then pass its stream URL:

```bash
openatlas --camera http://YOUR_PHONE_IP:4747/video
```

## Command-line options

Run `openatlas --help` to see them all.

| Flag | Default | Description |
|---|---|---|
| `--csv PATH` | built-in sample | Graph data to load. |
| `--camera N` | `0` | Webcam index (`0` built-in, `1` USB), or a stream URL. |
| `--port N` | `8765` | WebSocket port. |
| `--http-port N` | `8080` | Port for the browser UI. |
| `--no-window` | off | Hide the OpenCV debug window. |
| `--stream-every N` | `2` | Send every Nth camera frame to the browser. Lower is smoother, higher uses less CPU. |
| `--zoom-sensitivity N` | `5.0` | How strongly zoom gestures respond. |
| `--pan-sensitivity N` | `1800` | How far a small fist movement pans the graph. |
| `--pinch-threshold N` | `0.07` | How tight a pinch must be to register. |

## Requirements

- Python 3.8 or newer
- A webcam (or a phone used as one)
- Good lighting. Two hands are needed for zoom.

Dependencies installed automatically: `opencv-python`, `mediapipe`, `websockets`.

## Project structure

```
openatlas/
├── __init__.py       # CLI entry point (`openatlas`)
├── server.py         # Camera loop, hand tracking, gesture logic, WebSocket and HTTP servers
├── index.html        # Browser UI (Cytoscape.js)
└── sample_data.csv   # Default graph
```

## Troubleshooting

- **Wrong or black camera:** try `--camera 1`, or `--camera 2` if you have several.
- **Jittery cursor:** improve the lighting, or raise `--stream-every`.
- **Zoom too sensitive or too sluggish:** adjust `--zoom-sensitivity`.

## License

Released under the [MIT License](LICENSE).
