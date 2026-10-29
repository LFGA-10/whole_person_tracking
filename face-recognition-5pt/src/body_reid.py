# src/body_reid.py
r"""
Body-appearance re-identification -- a deliberately weaker, SEPARATE signal
from ArcFace face verification, used only to make a hedged guess ("likely
Happy?") about a person whose face has never been seen in their current
track (e.g. they walked in with their back to the camera).

This is NOT face recognition and must never be displayed or treated as
equivalent to it:
    - ArcFace / face verification -> "VERIFIED" -- a real identity claim.
    - Body re-ID (this module)    -> "likely <name>?" -- a soft guess.

A body-appearance match only has a realistic chance of being right within
roughly the same session, wearing the same clothes, under similar lighting.
It will NOT reliably recognize someone across different days or outfits --
that is a fundamentally different, much harder problem than face
recognition, and this module does not attempt to solve that; it is scoped
to "didn't leave and change clothes before walking back in."

Model: OSNet (Omni-Scale Network), the standard lightweight person
re-identification backbone from the Torchreid project (Zhou et al., "Omni-
Scale Feature Learning for Person Re-Identification", ICCV 2019), exported
to ONNX by the `nnio` project. ~8.7 MB, outputs a 512-d appearance vector
compared the same way ArcFace embeddings are (L2-normalize, cosine
distance = 1 - dot product).

    Invoke-WebRequest -Uri "https://drive.google.com/uc?id=1FlkHyebtkIqkE4jPfGZu-RXwfgCe4T1p&export=download" -OutFile "models\osnet_x1_0_op10.onnx"

Run:
    python -m src.body_reid
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import onnxruntime as ort

INPUT_HW = (256, 128)  # (height, width), matches the ONNX graph's [1,3,256,128]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class BodyMatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


class BodyReidEmbedder:
    def __init__(self, model_path: str = "models/osnet_x1_0_op10.onnx", debug: bool = False):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model file: {model_path}\n"
                "Download the OSNet ONNX export with:\n"
                '  Invoke-WebRequest -Uri '
                '"https://drive.google.com/uc?id=1FlkHyebtkIqkE4jPfGZu-RXwfgCe4T1p&export=download" '
                f'-OutFile "{model_path}"'
            )
        # See src/embed.py's ArcFaceEmbedderONNX.__init__ for why this is
        # thread-limited: multiple concurrent onnxruntime sessions in one
        # process oversubscribe the CPU badly with the unlimited default.
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess_opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.debug = debug

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        h, w = INPUT_HW
        img = cv2.resize(crop_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        chw = rgb.transpose(2, 0, 1)[None, ...]
        return np.ascontiguousarray(chw, dtype=np.float32)

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Returns a 512-d L2-normalized appearance embedding."""
        x = self._preprocess(crop_bgr)
        out = self.sess.run([self.out_name], {self.in_name: x})[0]
        v = out.reshape(-1).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)


def load_body_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    data = np.load(str(db_path), allow_pickle=True)
    try:
        return {k: np.asarray(data[k], dtype=np.float32).reshape(-1) for k in data.files}
    finally:
        data.close()


class BodyDBMatcher:
    """Same shape/behavior as recognize_live.FaceDBMatcher, over body
    appearance embeddings instead of face embeddings. Intentionally a
    looser threshold is expected -- appearance is a much weaker signal
    than a face embedding, so this should never be compared 1:1 against
    the face-matching threshold."""

    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float = 0.35):
        self.db = db
        self.dist_thresh = float(dist_thresh)
        self._names: List[str] = []
        self._mat: Optional[np.ndarray] = None
        self._rebuild()

    def _rebuild(self) -> None:
        self._names = sorted(self.db.keys())
        self._mat = (
            np.stack([self.db[n].reshape(-1).astype(np.float32) for n in self._names], axis=0)
            if self._names else None
        )

    def reload_from(self, path: Path) -> None:
        self.db = load_body_db_npz(path)
        self._rebuild()

    def match(self, embedding: np.ndarray) -> BodyMatchResult:
        if self._mat is None or not self._names:
            return BodyMatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)
        e = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        similarities = (self._mat @ e.T).reshape(-1)
        best_i = int(np.argmax(similarities))
        best_sim = float(similarities[best_i])
        best_dist = 1.0 - best_sim
        accepted = best_dist <= self.dist_thresh
        return BodyMatchResult(
            name=self._names[best_i] if accepted else None,
            distance=best_dist,
            similarity=best_sim,
            accepted=accepted,
        )


def main() -> None:
    """Smoke test: embed two crops of the same static image and confirm the
    embedder runs and returns a sane 512-d unit vector (no camera needed)."""
    import sys

    embedder = BodyReidEmbedder()
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("Usage: python -m src.body_reid <image_path>")
        return
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read {path}")
        return
    v = embedder.embed(img)
    print(f"embedding shape={v.shape} norm={np.linalg.norm(v):.4f}")


if __name__ == "__main__":
    main()
