# OpenAtlas ✋

[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Wave your hands at your laptop to explore a knowledge graph. That's the whole thing.

Uses MediaPipe to track your hands and Cytoscape.js to draw the graph. No PhD required.

<a href="https://www.youtube.com/watch?v=VCilobkDwnU">
  <img src="https://img.youtube.com/vi/VCilobkDwnU/maxresdefault.jpg" width="400"/>
</a>

---

## Just run it

```bash
pip install git+https://github.com/Gaurang111/openatlas.git
openatlas
```

Browser opens. Wave hand. Graph appears.

> First run downloads a ~5 MB hand tracking model. It only does this once.

---

## Hand gestures

| Do this | Get that |
|---|---|
| ☝ Point with index finger | Move cursor around |
| ☝ → ✊ Point then make a fist | Select the node you're hovering |
| ✊ Fist + move your arm | Pan the graph |
| ✊✊ Pinch with both hands and pull apart / push together | Zoom |
| `R` key | Fit everything back in view |
| `Q` key | Quit (or just close the window like a normal person) |

---

## Options

```bash
openatlas --help
```

| Flag | Default              | What it does |
|---|----------------------|---|
| `--csv PATH` | built-in sample data | Load your own data |
| `--camera N` | `0`                  | Webcam index (`0` = built-in, `1` = external/USB), or a stream URL if you're using your phone as a webcam |
| `--port N` | `8765`               | WebSocket port |
| `--http-port N` | `8080`               | The port the browser UI runs on |
| `--no-window` | off                  | Hide the OpenCV debug window |
| `--stream-every N` | `2`                  | Lower = smoother camera feed, higher = less CPU |
| `--zoom-sensitivity N` | `5.0`                | How dramatic your zoom gestures are |
| `--pan-sensitivity N` | `1800`               | How far a small fist-move pans |
| `--pinch-threshold N` | `0.07`               | How tight a pinch needs to be |

### Using your phone as a webcam (DroidCam)

```bash
openatlas --camera http://YOUR_PHONE_IP:4747/video
```

Find your phone's IP in the DroidCam app and swap it in.

### Load your own graph

```bash
openatlas --csv my_data.csv
```

CSV needs three columns:

```csv
source,target,relationship
Python,NumPy,uses
NumPy,Linear Algebra,implements
PyTorch,NumPy,built_on
```

---

## Requirements

- Python 3.8+
- A webcam
- Hands (two recommended)
