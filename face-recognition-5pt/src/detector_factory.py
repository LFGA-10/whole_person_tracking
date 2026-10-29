# src/detector_factory.py
"""
Single place that decides which face detector a script runs with.

YuNet is now the default across the project (better tolerance of head
angle and partial occlusion than Haar, plus it gives 5-point landmarks
directly). Haar is kept as an explicit fallback / comparison option --
pass detector="haar" or set FALCON_FACE_DETECTOR=haar.

Every detector returned here exposes the same interface:
    .detect(frame_bgr, max_faces) -> list of objects with
        .x1 .y1 .x2 .y2 .score .kps   (kps: (5,2) float32,
        [left-eye, right-eye, nose, left-mouth, right-mouth] by image x)
    .close()

Switching detectors changes the alignment crop slightly, which shifts
the genuine/impostor cosine-distance distributions. Re-run
`python -m src.evaluate --detector yunet` (or whichever detector you
pick) before trusting an old threshold with a new detector.
"""

from __future__ import annotations

import os
from typing import Optional

DEFAULT_DETECTOR = "yunet"


def get_default_detector_name() -> str:
    return os.environ.get("FALCON_FACE_DETECTOR", DEFAULT_DETECTOR).strip().lower()


def build_face_detector(
    name: Optional[str] = None,
    min_size=(60, 60),
    debug: bool = False,
):
    """Construct a multi-face, 5-point detector by name ("yunet" or "haar")."""

    name = (name or get_default_detector_name()).strip().lower()

    if name == "yunet":
        from .yunet_5pt import YuNetDetector5pt

        return YuNetDetector5pt(min_size=min_size, debug=debug)

    if name == "haar":
        from .recognize_live import HaarFaceLandmarker5pt

        return HaarFaceLandmarker5pt(min_size=min_size, debug=debug)

    raise ValueError(f"Unknown detector '{name}'. Use 'yunet' or 'haar'.")
