"""HTTP server for the QueenNet dashboard and user-image inference."""
import base64
import io
import http.server
import json
import os
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from queennet.data.patches import HSIPatchDataset
from queennet.data.preprocessing import normalize_spectral
from queennet.models.queennet import QueenNet
from queennet.data.loaders import detect_mat_key
import scipy.io as sio

os.chdir(r"C:\Users\makke\.gemini\antigravity\scratch\queennet-hsi")
ROOT = Path.cwd()
CHECKPOINT = ROOT / "outputs" / "indian_pines" / "checkpoints" / "best_model.pt"
MODEL_CONFIG = {"bands": 200, "classes": 16, "qubits": 12, "depth": 2, "patch": 32}
_model = None


def load_uploaded_image(filename, content):
    """Load a user cube from MAT, NPY, or NPZ and return H x W x bands."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".mat":
        mat = sio.loadmat(io.BytesIO(content))
        key = detect_mat_key(mat, ["indian_pines", "Indian_pines", "data", "image"])
        image = np.asarray(mat[key])
    elif suffix == ".npy":
        image = np.load(io.BytesIO(content), allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(io.BytesIO(content), allow_pickle=False)
        if not archive.files:
            raise ValueError("The NPZ file does not contain an array.")
        image = archive["image"] if "image" in archive.files else archive[archive.files[0]]
    else:
        raise ValueError("Use a .mat, .npy, or .npz hyperspectral cube.")

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D cube (height, width, bands), received {image.ndim}D data.")
    if image.shape[-1] != MODEL_CONFIG["bands"]:
        raise ValueError(
            f"This checkpoint expects {MODEL_CONFIG['bands']} spectral bands; received {image.shape[-1]}."
        )
    if not np.isfinite(image).all():
        raise ValueError("The uploaded cube contains NaN or infinite values.")
    return image


def get_model():
    global _model
    if _model is None:
        _model = QueenNet(
            in_bands=MODEL_CONFIG["bands"],
            num_classes=MODEL_CONFIG["classes"],
            num_qubits=MODEL_CONFIG["qubits"],
            depth_L=MODEL_CONFIG["depth"],
            patch_size=MODEL_CONFIG["patch"],
        )
        if not CHECKPOINT.exists():
            raise FileNotFoundError(f"Trained checkpoint not found: {CHECKPOINT}")
        checkpoint = torch.load(str(CHECKPOINT), map_location="cpu")
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.eval()
    return _model


def classify_image(image):
    normalized = normalize_spectral(image, method="minmax")
    height, width = image.shape[:2]
    coords = [(row, col) for row in range(height) for col in range(width)]
    dataset = HSIPatchDataset(normalized, coords, [0] * len(coords), patch_size=MODEL_CONFIG["patch"])
    predictions = []
    model = get_model()
    with torch.no_grad():
        for patches, _ in DataLoader(dataset, batch_size=32, shuffle=False):
            predictions.extend(model(patches).argmax(dim=-1).cpu().tolist())

    class_map = np.asarray(predictions, dtype=np.int32).reshape(height, width) + 1
    counts = np.bincount(class_map.ravel(), minlength=MODEL_CONFIG["classes"] + 1)[1:]
    figure = io.BytesIO()
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.imshow(class_map, cmap="tab20", interpolation="nearest")
    axis.set_title("QueenNet classification map")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(figure, format="png", dpi=160)
    plt.close(fig)
    return {
        "height": height,
        "width": width,
        "bands": image.shape[-1],
        "classes": [{"class": index + 1, "pixels": int(count)} for index, count in enumerate(counts)],
        "map": "data:image/png;base64," + base64.b64encode(figure.getvalue()).decode("ascii"),
    }

class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if path == "/" or path == "":
            return os.path.join(os.getcwd(), "dashboard", "index.html")
        return super().translate_path(path)

    def do_POST(self):
        if self.path != "/api/classify":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            message = BytesParser(policy=default).parsebytes(
                (f"Content-Type: {self.headers.get('Content-Type')}\r\n\r\n").encode() + raw
            )
            upload = next((part for part in message.iter_attachments() if part.get_filename()), None)
            if upload is None:
                raise ValueError("Choose a hyperspectral cube before classifying.")
            result = classify_image(load_uploaded_image(upload.get_filename(), upload.get_payload(decode=True)))
            body = json.dumps(result).encode("utf-8")
            self.send_response(200)
        except Exception as error:
            body = json.dumps({"error": str(error)}).encode("utf-8")
            self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

print("=" * 50)
print("QueenNet Research Dashboard")
print("Open: http://localhost:8888")
print("=" * 50)

server = http.server.HTTPServer(("0.0.0.0", 8888), Handler)
server.serve_forever()
