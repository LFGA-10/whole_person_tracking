# src/yolox_person.py
"""
YOLOX person detection, used ONLY to keep a tentative body track alive
while a previously *verified* face is temporarily occluded.

YOLOX does not establish identity. It never replaces ArcFace. It only
tells face_tracking.LockedFaceTracker "something shaped like a person
is still roughly where the locked face used to be," so the tracker can
enter BODY_HOLD instead of immediately dropping to SEARCHING. Identity
is only ever re-confirmed by matching a face again through the same
ArcFace pipeline as before.

Model: official YOLOX ONNX export (Megvii-BaseDetection/YOLOX releases,
e.g. yolox_nano.onnx / yolox_tiny.onnx). These exports do NOT include
the final grid decode -- decode_outputs() below reproduces the exact
math from YOLOX's own demo/ONNXRuntime/onnx_inference.py so the boxes
land in the right place. Preprocessing matches that same demo: BGR,
no mean/std normalization, letterboxed (aspect-preserving resize +
gray padding) to a square input, CHW float32.

Run:
    python -m src.yolox_person
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .cam_config import get_cam_index

COCO_PERSON_CLASS_ID = 0  # COCO class 0 == "person"

# YOLOX-Nano / YOLOX-Tiny both export at 416x416 with these FPN strides.
INPUT_SIZE = (416, 416)
STRIDES = (8, 16, 32)


@dataclass
class PersonBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


def _letterbox(img: np.ndarray, input_size: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    """Aspect-preserving resize onto a gray canvas, padded bottom/right only."""
    ih, iw = input_size
    h, w = img.shape[:2]
    r = min(ih / h, iw / w)

    resized = cv2.resize(
        img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR
    )

    canvas = np.full((ih, iw, 3), 114, dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return canvas, r


def _preprocess(img: np.ndarray, input_size: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    padded, ratio = _letterbox(img, input_size)
    chw = padded.transpose(2, 0, 1).astype(np.float32)  # BGR, no normalization
    return np.ascontiguousarray(chw)[None, ...], ratio


def _make_grids(input_size: Tuple[int, int], strides: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    grids = []
    expanded_strides = []
    for stride in strides:
        hsize = input_size[0] // stride
        wsize = input_size[1] // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((1, grid.shape[1], 1), stride))
    return np.concatenate(grids, axis=1), np.concatenate(expanded_strides, axis=1)


def _decode_outputs(outputs: np.ndarray, grids: np.ndarray, expanded_strides: np.ndarray) -> np.ndarray:
    outputs = outputs.copy()
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        order = order[1:][iou <= iou_thresh]

    return keep


class YoloxPersonDetector:
    """Lightweight CPU person detector, run at a configurable interval.

    Call maybe_detect() every frame; it only runs the network every
    `interval` calls and returns the cached boxes otherwise, which is
    what the project brief asks for: don't assume both detection models
    need to run on every frame.
    """

    def __init__(
        self,
        model_path: str = "models/yolox_nano.onnx",
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        interval: int = 3,
        debug: bool = False,
    ) -> None:
        self.debug = bool(debug)
        self.interval = max(1, int(interval))
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model file: {model_path}\n"
                "Download a YOLOX ONNX export, e.g.:\n"
                "  Invoke-WebRequest -Uri "
                '"https://github.com/Megvii-BaseDetection/YOLOX/releases/download/'
                f'0.1.1rc0/{model_path.name}" -OutFile "{model_path}"'
            )

        # See src/embed.py's ArcFaceEmbedderONNX.__init__ for why this is
        # thread-limited: multiple concurrent onnxruntime sessions in one
        # process oversubscribe the CPU badly with the unlimited default.
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess_opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

        self._grids, self._expanded_strides = _make_grids(INPUT_SIZE, STRIDES)

        self._frame_index = 0
        self._last_boxes: List[PersonBox] = []
        self.last_latency_ms: Optional[float] = None

    def _infer(self, frame_bgr: np.ndarray) -> List[PersonBox]:
        t0 = time.perf_counter()

        x, ratio = _preprocess(frame_bgr, INPUT_SIZE)
        raw = self.sess.run(None, {self.in_name: x})[0]  # (1, N, 85)
        raw = _decode_outputs(raw, self._grids, self._expanded_strides)[0]

        boxes_cxcywh = raw[:, :4]
        obj_conf = raw[:, 4]
        cls_conf = raw[:, 5:]

        person_scores = obj_conf * cls_conf[:, COCO_PERSON_CLASS_ID]
        mask = person_scores >= self.conf_threshold

        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        if not np.any(mask):
            return []

        cxcywh = boxes_cxcywh[mask]
        scores = person_scores[mask]

        x1 = cxcywh[:, 0] - cxcywh[:, 2] / 2.0
        y1 = cxcywh[:, 1] - cxcywh[:, 3] / 2.0
        x2 = cxcywh[:, 0] + cxcywh[:, 2] / 2.0
        y2 = cxcywh[:, 1] + cxcywh[:, 3] / 2.0
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1) / ratio  # undo letterbox scale

        keep = _nms(boxes_xyxy, scores, self.nms_threshold)

        h, w = frame_bgr.shape[:2]
        out: List[PersonBox] = []
        for i in keep:
            bx1, by1, bx2, by2 = boxes_xyxy[i]
            out.append(
                PersonBox(
                    x1=int(max(0, min(w - 1, bx1))),
                    y1=int(max(0, min(h - 1, by1))),
                    x2=int(max(0, min(w - 1, bx2))),
                    y2=int(max(0, min(h - 1, by2))),
                    score=float(scores[i]),
                )
            )
        return out

    def maybe_detect(self, frame_bgr: np.ndarray, force: bool = False) -> List[PersonBox]:
        """Run YOLOX every `interval` frames; reuse the cached result otherwise."""
        self._frame_index += 1
        if force or self._frame_index % self.interval == 0 or not self._last_boxes:
            self._last_boxes = self._infer(frame_bgr)
        return self._last_boxes


def main() -> None:
    cap = cv2.VideoCapture(get_cam_index())
    det = YoloxPersonDetector(model_path="models/yolox_nano.onnx", interval=1, debug=True)

    print("YOLOX person detector test. Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        boxes = det.maybe_detect(frame)
        vis = frame.copy()

        for b in boxes:
            cv2.rectangle(vis, (b.x1, b.y1), (b.x2, b.y2), (255, 170, 0), 2)
            cv2.putText(
                vis, f"person {b.score:.2f}", (b.x1, max(0, b.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 170, 0), 2,
            )

        if det.last_latency_ms is not None:
            cv2.putText(
                vis, f"yolox {det.last_latency_ms:.1f}ms", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

        cv2.imshow("yolox_person", vis)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
