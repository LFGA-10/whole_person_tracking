# src/yunet_5pt.py
"""
YuNet face detection + native 5-point landmarks (OpenCV FaceDetectorYN).

Replaces the Haar cascade + MediaPipe FaceLandmarker combination in
haar_5pt.py / recognize_live.py's HaarFaceLandmarker5pt with a single
CNN pass that gives both the face box and the five landmarks YuNet was
trained to output directly:

    right eye, left eye, nose tip, right mouth corner, left mouth corner

("right"/"left" are anatomical -- the subject's own right/left -- which
for a face looking at the camera means the "right eye" point lands on
the left side of the image.) That already matches the point order
align_face_5pt() expects (image-left eye, image-right eye, nose,
image-left mouth corner, image-right mouth corner), but this module
still applies the same defensive x-sort used elsewhere in this codebase
(haar_5pt.py, recognize_live.py) so a sideways head pose can never
silently swap the eyes or mouth corners.

Model:
    OpenCV Zoo "face_detection_yunet" ONNX export, e.g.
    face_detection_yunet_2023mar.onnx (~230 KB). Download with:

        Invoke-WebRequest -Uri ^
          "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" ^
          -OutFile "models\face_detection_yunet_2023mar.onnx"

    (raw.githubusercontent.com returns a git-lfs pointer file for this
    model, not the binary -- use media.githubusercontent.com instead.)

Run:
    python -m src.yunet_5pt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .cam_config import get_cam_index

MODEL_PATH = Path("models/face_detection_yunet_2023mar.onnx")


@dataclass
class FaceKpsBox:
    """Detected face: bounding box + five landmarks, full-frame coordinates.

    Field names match haar_5pt.FaceKpsBox and recognize_live.FaceDet so
    this class is a drop-in replacement for either -- every call site
    only ever reads .x1/.y1/.x2/.y2/.score/.kps.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5, 2) float32


def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width - 1, round(x2))))
    y2 = int(max(0, min(height - 1, round(y2))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _reorder_5pt(kps: np.ndarray) -> np.ndarray:
    """Guarantee [left-eye, right-eye, nose, left-mouth, right-mouth] by image x."""
    kps = kps.copy()
    if kps[0, 0] > kps[1, 0]:
        kps[[0, 1]] = kps[[1, 0]]
    if kps[3, 0] > kps[4, 0]:
        kps[[3, 4]] = kps[[4, 3]]
    return kps


class YuNetDetector5pt:
    """Multi-face detector backed by cv2.FaceDetectorYN.

    Interface-compatible with haar_5pt.Haar5ptDetector and
    recognize_live.HaarFaceLandmarker5pt: .detect(frame, max_faces) and
    .close(). Use it anywhere either of those is used today.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_size: Tuple[int, int] = (60, 60),
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 50,
        debug: bool = False,
    ) -> None:
        self.debug = bool(debug)
        self.min_size = tuple(map(int, min_size))

        model_path = Path(model_path) if model_path is not None else MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model file: {model_path}\n"
                "Download it with:\n"
                "  Invoke-WebRequest -Uri "
                '"https://media.githubusercontent.com/media/opencv/opencv_zoo/main/'
                'models/face_detection_yunet/face_detection_yunet_2023mar.onnx" '
                f'-OutFile "{model_path}"'
            )

        self._detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),  # placeholder, reset on first real frame
            score_threshold=float(score_threshold),
            nms_threshold=float(nms_threshold),
            top_k=int(top_k),
        )
        self._input_size: Optional[Tuple[int, int]] = None

    def _ensure_input_size(self, frame_bgr: np.ndarray) -> None:
        """YuNet needs setInputSize() called whenever the frame size changes."""
        h, w = frame_bgr.shape[:2]
        if self._input_size != (w, h):
            self._detector.setInputSize((w, h))
            self._input_size = (w, h)
            if self.debug:
                print(f"[yunet_5pt] input size set to {(w, h)}")

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 5) -> List[FaceKpsBox]:
        height, width = frame_bgr.shape[:2]
        self._ensure_input_size(frame_bgr)

        _, faces = self._detector.detect(frame_bgr)

        if faces is None or len(faces) == 0:
            return []

        detections: List[FaceKpsBox] = []
        for row in faces:
            x, y, w, h = row[0:4]
            score = float(row[14])

            if w < self.min_size[0] or h < self.min_size[1]:
                continue

            x1, y1, x2, y2 = _clip_xyxy(x, y, x + w, y + h, width, height)

            # row[4:14] = right eye, left eye, nose, right mouth, left mouth (x, y pairs)
            kps = row[4:14].reshape(5, 2).astype(np.float32)
            kps = _reorder_5pt(kps)

            detections.append(
                FaceKpsBox(x1=x1, y1=y1, x2=x2, y2=y2, score=score, kps=kps)
            )

        # Largest face first, matching the other detectors' convention.
        detections.sort(key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1), reverse=True)
        return detections[:max_faces]

    def close(self) -> None:
        """No persistent resources to release, kept for interface parity."""
        return None


def main() -> None:
    cap = cv2.VideoCapture(get_cam_index())
    det = YuNetDetector5pt(min_size=(60, 60), debug=True)

    print("YuNet + native 5pt test. Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = det.detect(frame, max_faces=5)
        vis = frame.copy()

        for f in faces:
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
            for (x, y) in f.kps.astype(int):
                cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(
                vis, f"{f.score:.2f}", (f.x1, max(0, f.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

        if not faces:
            cv2.putText(vis, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("yunet_5pt", vis)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
