# src/face_tracking.py
"""
Target-locked face tracking pipeline with real-time signal analysis (EAR, blinks, expressions)
and occlusion-tolerant body-assisted tracking.

Usage:
    python -m src.face_tracking --target <name>
    python -m src.face_tracking --target <name> --detector yunet --yolox tiny

States
------
    SEARCHING    No lock. Scanning every detected face for --target.
    LOCKED       Target verified and being followed.
    LOST         The face dropped out and no YOLOX person box is available to
                 hold the position with -- still retried every frame via
                 geometric continuation, same as before this feature existed.
    BODY_HOLD    The verified face disappeared, but YOLOX still finds a person
                 box roughly where it was last seen. TENTATIVE ONLY -- this is
                 never treated as proof of identity, only as "keep watching
                 this spot a little longer." See `is_action_authorized`.
    REVERIFYING  (display-only, see `display_state`) A face reappeared while
                 LOST or BODY_HOLD and ArcFace is being asked to confirm it is
                 still --target before the lock is trusted again.

YOLOX only ever answers "is a person-shaped box still roughly where the face
used to be" -- it never assigns or confirms identity. That stays exclusively
ArcFace's job, through the same accept/reject threshold as before. A
BODY_HOLD lock must never be treated as equivalent to a verified identity by
anything downstream (e.g. a future servo integration must gate movement on
`is_action_authorized`, not merely `state != LOST`).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .align import align_face_5pt
from .detector_factory import build_face_detector
from .embed import ArcFaceEmbedderONNX
from .face_signals import FaceSignalExtractor
from .cam_config import get_cam_index
from .camera import open_camera
from .recognize_live import (
    DEFAULT_THRESHOLD,
    FaceDBMatcher,
    HaarFaceLandmarker5pt,
    load_db_npz,
)

Box = Tuple[int, int, int, int]


class LockState(Enum):
    SEARCHING = auto()
    LOCKED = auto()
    LOST = auto()
    BODY_HOLD = auto()
    REVERIFYING = auto()  # display-only; see LockedFaceTracker.display_state


def iou(a: Box, b: Box) -> float:
    """Calculate Intersection over Union (IoU) between two XYXY boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def contains_fraction(inner: Box, outer: Box) -> float:
    """Fraction of `inner`'s area that falls inside `outer`.

    Plain IoU is near-zero for a small face box inside a much larger person
    box even when the face clearly belongs to that person -- this is what
    associates the two instead.
    """
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    ax1, ay1 = max(ix1, ox1), max(iy1, oy1)
    ax2, ay2 = min(ix2, ox2), min(iy2, oy2)
    inter = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    inner_area = max(1, (ix2 - ix1) * (iy2 - iy1))
    return inter / float(inner_area)


def center(box: Box) -> np.ndarray:
    """Return center (x, y) coordinates of an XYXY box."""
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def box_diag(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(1.0, float(np.linalg.norm([x2 - x1, y2 - y1])))


MIN_BODY_CONTINUITY = -0.30  # same scale/threshold as the original associate()


def best_overlap(last_box: Optional[Box], candidates: List[Box]) -> Optional[int]:
    """Pick the candidate box that best continues `last_box` frame-to-frame.

    Same displacement-aware overlap scoring as the original face associate(),
    generalized so BODY_HOLD can reuse it for person-box continuity.
    """
    if last_box is None or not candidates:
        return None
    last_center = center(last_box)
    last_diag = box_diag(last_box)
    scored = []
    for i, box in enumerate(candidates):
        overlap = iou(last_box, box)
        displacement = float(np.linalg.norm(center(box) - last_center)) / last_diag
        scored.append((overlap - 0.35 * displacement, i))
    best_score, best_i = max(scored, key=lambda item: item[0])
    return best_i if best_score > MIN_BODY_CONTINUITY else None


@dataclass
class TrackingSignal:
    """Nose-tip position relative to the frame center."""

    error_x: float  # normalized, -1 (left edge) .. +1 (right edge)
    error_y: float  # normalized, -1 (top edge) .. +1 (bottom edge)
    horizontal: str
    vertical: str
    offset_x_px: float  # nose tip minus frame center, pixels (+ = right)
    offset_y_px: float  # nose tip minus frame center, pixels (+ = down)
    distance_px: float
    nose_xy: Tuple[float, float]

    @property
    def direction(self) -> str:
        parts = [p for p in (self.vertical, self.horizontal) if p != "CENTER"]
        return "-".join(parts) if parts else "CENTER"


# Tuning for the body-assisted hold, kept separate from the original
# lost_timeout so a short face-only dropout and a longer occlusion-with-body
# can be tuned independently.
BODY_HOLD_TIMEOUT = 60             # frames a body-only hold is allowed to run
MIN_FACE_BODY_OVERLAP = 0.55       # fraction of the face box inside a person box


class LockedFaceTracker:
    def __init__(
        self,
        target_name: str,
        detector,
        embedder,
        matcher,
        body_detector=None,
        verify_every: int = 10,
        lost_timeout: int = 24,
        body_hold_timeout: int = BODY_HOLD_TIMEOUT,
        ema_alpha: float = 0.30,
        dead_zone: float = 0.07,
    ):
        self.target_name = target_name
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher
        self.body_detector = body_detector
        self.verify_every = verify_every
        self.lost_timeout = lost_timeout
        self.body_hold_timeout = body_hold_timeout
        self.ema_alpha = ema_alpha
        self.dead_zone = dead_zone

        self.state = LockState.SEARCHING
        self.display_state = LockState.SEARCHING  # what the UI should label this tick
        self.last_box: Optional[Box] = None
        self.last_body_box: Optional[Box] = None
        self.smooth_center: Optional[np.ndarray] = None  # smoothed nose tip
        self.lost_frames = 0
        self.frame_index = 0
        self.other_faces: list = []  # detected faces that are not the locked target
        self.last_person_boxes: List[Box] = []
        self.occlusion_start: Optional[float] = None

        self._next_track_id = 1
        self.track_id: Optional[int] = None

        self.detector_latency_ms = 0.0
        self.body_latency_ms: Optional[float] = None

    # -------------------------------------------------------------
    # Safety gate -- anything that moves hardware or otherwise acts
    # on "this is a verified identity" must check this, not just
    # "state != LOST". BODY_HOLD is explicitly tentative.
    # -------------------------------------------------------------
    @property
    def is_action_authorized(self) -> bool:
        return self.state == LockState.LOCKED

    @property
    def occlusion_seconds(self) -> float:
        if self.occlusion_start is None:
            return 0.0
        return time.time() - self.occlusion_start

    @staticmethod
    def box(face) -> Box:
        return (face.x1, face.y1, face.x2, face.y2)

    def identity(self, frame, face):
        aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
        return self.matcher.match(self.embedder.embed(aligned))

    def target_is_verified(self, frame, face) -> bool:
        match = self.identity(frame, face)
        return bool(match.accepted and match.name == self.target_name)

    def acquire(self, frame, faces):
        best = None
        best_similarity = -1.0
        for face in faces:
            match = self.identity(frame, face)
            if (
                match.accepted
                and match.name == self.target_name
                and match.similarity > best_similarity
            ):
                best, best_similarity = face, match.similarity
        return best

    def associate(self, faces):
        if self.last_box is None or not faces:
            return None
        boxes = [self.box(f) for f in faces]
        i = best_overlap(self.last_box, boxes)
        return faces[i] if i is not None else None

    def find_face_in_region(self, faces, region: Optional[Box]):
        """BODY_HOLD: look for any face inside the body-hold region."""
        if region is None or not faces:
            return None
        best, best_frac = None, 0.0
        for face in faces:
            frac = contains_fraction(self.box(face), region)
            if frac > best_frac:
                best, best_frac = face, frac
        return best if best_frac >= 0.3 else None

    def _match_person_box(self, face_box: Optional[Box], person_boxes: List[Box]) -> Optional[Box]:
        if face_box is None:
            return None
        best, best_frac = None, 0.0
        for pbox in person_boxes:
            frac = contains_fraction(face_box, pbox)
            if frac > best_frac:
                best, best_frac = pbox, frac
        return best if best_frac >= MIN_FACE_BODY_OVERLAP else None

    def _continue_body(self, person_boxes: List[Box]) -> Optional[Box]:
        i = best_overlap(self.last_body_box, person_boxes)
        return person_boxes[i] if i is not None else None

    def _clear_lock(self) -> None:
        self.last_box = None
        self.last_body_box = None
        self.smooth_center = None
        self.lost_frames = 0
        self.occlusion_start = None
        self.track_id = None

    def update(self, frame):
        self.frame_index += 1

        t0 = time.perf_counter()
        faces = self.detector.detect(frame, max_faces=8)
        self.detector_latency_ms = (time.perf_counter() - t0) * 1000.0

        person_boxes: List[Box] = []
        if self.body_detector is not None:
            person_boxes = [(b.x1, b.y1, b.x2, b.y2) for b in self.body_detector.maybe_detect(frame)]
            self.body_latency_ms = self.body_detector.last_latency_ms
        self.last_person_boxes = person_boxes

        if self.state == LockState.SEARCHING:
            candidate = self.acquire(frame, faces)
        elif self.state == LockState.BODY_HOLD:
            # No tracked face right now -- only look inside the hold region.
            candidate = self.find_face_in_region(faces, self.last_body_box)
        else:  # LOCKED or LOST
            candidate = self.associate(faces)

        came_from_hold = self.state == LockState.BODY_HOLD
        must_reverify = self.state in (LockState.LOST, LockState.BODY_HOLD)
        verifying_now = candidate is not None and (
            must_reverify or self.frame_index % self.verify_every == 0
        )

        self.display_state = self.state
        if verifying_now and must_reverify:
            # One-tick "checking..." label for the UI before the outcome lands.
            self.display_state = LockState.REVERIFYING

        rejected_as_wrong_person = False
        if verifying_now and not self.target_is_verified(frame, candidate):
            rejected_as_wrong_person = must_reverify
            candidate = None

        self.other_faces = [f for f in faces if f is not candidate]

        if candidate is None:
            if rejected_as_wrong_person:
                # A different person (enrolled or not) is in the hold region.
                # Don't guess -- clear fully rather than keep watching them.
                # display_state stays REVERIFYING for this one tick so the UI
                # shows the check happened, even though it failed; `state`
                # (used for next tick's branch logic) moves on immediately.
                self._clear_lock()
                self.state = LockState.SEARCHING
                return None, None

            # No face this tick -- can a YOLOX person box keep the position
            # alive instead of dropping straight to LOST?
            body_box = None
            if self.body_detector is not None and self.state != LockState.SEARCHING:
                body_box = self._match_person_box(self.last_box, person_boxes)
                if body_box is None and self.last_body_box is not None:
                    body_box = self._continue_body(person_boxes)

            if body_box is not None:
                if self.state != LockState.BODY_HOLD:
                    self.occlusion_start = time.time()
                self.state = LockState.BODY_HOLD
                self.display_state = LockState.BODY_HOLD
                self.last_body_box = body_box
                self.lost_frames += 1
                if self.lost_frames > self.body_hold_timeout:
                    self._clear_lock()
                    self.state = LockState.SEARCHING
                    self.display_state = LockState.SEARCHING
                return None, None

            self.lost_frames += 1
            if self.last_box is not None:
                self.state = LockState.LOST
                self.display_state = LockState.LOST
            if self.lost_frames > self.lost_timeout:
                self._clear_lock()
                self.state = LockState.SEARCHING
                self.display_state = LockState.SEARCHING
            return None, None

        # --- candidate accepted this tick -----------------------------
        self.state = LockState.LOCKED
        self.display_state = (
            LockState.REVERIFYING if self.display_state == LockState.REVERIFYING else LockState.LOCKED
        )
        self.lost_frames = 0
        self.occlusion_start = None
        if self.track_id is None:
            self.track_id = self._next_track_id
            self._next_track_id += 1

        self.last_box = self.box(candidate)
        self.last_body_box = self._match_person_box(self.last_box, person_boxes) or self.last_body_box
        raw_center = candidate.kps[2].astype(np.float32)  # nose tip

        if self.smooth_center is None or came_from_hold:
            # Don't EMA-blend across an occlusion gap of unknown duration --
            # snap cleanly to the reacquired position instead.
            self.smooth_center = raw_center
        else:
            a = self.ema_alpha
            self.smooth_center = a * raw_center + (1.0 - a) * self.smooth_center

        return candidate, self.position_signal(frame.shape)

    def position_signal(self, shape) -> TrackingSignal:
        height, width = shape[:2]
        dx = float(self.smooth_center[0] - width / 2.0)
        dy = float(self.smooth_center[1] - height / 2.0)
        ex = dx / (width / 2.0)
        ey = dy / (height / 2.0)
        horizontal = "CENTER"
        vertical = "CENTER"

        if ex < -self.dead_zone:
            horizontal = "LEFT"
        elif ex > self.dead_zone:
            horizontal = "RIGHT"

        if ey < -self.dead_zone:
            vertical = "UP"
        elif ey > self.dead_zone:
            vertical = "DOWN"

        return TrackingSignal(
            error_x=ex,
            error_y=ey,
            horizontal=horizontal,
            vertical=vertical,
            offset_x_px=dx,
            offset_y_px=dy,
            distance_px=float(np.hypot(dx, dy)),
            nose_xy=(float(self.smooth_center[0]), float(self.smooth_center[1])),
        )


def draw_label(frame, text, xy, color, scale=0.62):
    cv2.putText(
        frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA
    )


def draw_missing_banner(view, target: str, others: int, lost: bool) -> None:
    """Flashing red/orange 'face missing' warning across the frame."""
    h, w = view.shape[:2]
    flash = int(time.time() * 2) % 2 == 0
    color = (0, 0, 255) if flash else (0, 140, 255)  # red <-> orange (BGR)

    cv2.rectangle(view, (0, 0), (w - 1, h - 1), color, 8)

    band_h = max(70, h // 8)
    y0 = h // 2 - band_h // 2
    overlay = view.copy()
    cv2.rectangle(overlay, (0, y0), (w, y0 + band_h), color, -1)
    cv2.addWeighted(overlay, 0.75, view, 0.25, 0, view)

    title = "FACE MISSING" if lost else "SEARCHING FOR FACE"
    if others:
        sub = f"{target} not found - {others} other face(s) ignored"
    else:
        sub = f"{target} not in frame"
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
    cv2.putText(view, title, ((w - tw) // 2, y0 + int(band_h * 0.55)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4, cv2.LINE_AA)
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(view, sub, ((w - sw) // 2, y0 + int(band_h * 0.88)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_body_hold_banner(view, target: str, state: LockState, occlusion_s: float, timeout_s: float) -> None:
    """Calm (non-flashing) banner for BODY_HOLD / REVERIFYING.

    Deliberately styled differently from draw_missing_banner: the system has
    NOT lost the person -- it is just not claiming to recognize them from
    their body alone, which is the thing this banner needs to communicate.
    """
    h, w = view.shape[:2]
    color = (210, 140, 40)  # muted blue-orange, distinct from the red/orange "missing" flash

    band_h = max(60, h // 10)
    y0 = h - band_h - 10
    overlay = view.copy()
    cv2.rectangle(overlay, (0, y0), (w, y0 + band_h), color, -1)
    cv2.addWeighted(overlay, 0.55, view, 0.45, 0, view)

    if state == LockState.REVERIFYING:
        title = "IDENTITY REVERIFICATION IN PROGRESS"
    else:
        title = "FACE TEMPORARILY OCCLUDED -- BODY TRACK MAINTAINED"
    sub = f"{target} not re-identified from body alone  |  occluded {occlusion_s:0.1f}s / {timeout_s:0.0f}s max"

    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(view, title, (max(10, (w - tw) // 2), y0 + int(band_h * 0.45)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.putText(view, sub, (max(10, (w - sw) // 2), y0 + int(band_h * 0.82)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


def draw_corner_brackets(frame, box: Box, color, length_frac: float = 0.22, thickness: int = 2):
    """Subtle corner-bracket style for a person box co-detected alongside a LOCKED face."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    lx, ly = max(6, int(w * length_frac)), max(6, int(h * length_frac))
    for (cx, cy), (sx, sy) in [((x1, y1), (1, 1)), ((x2, y1), (-1, 1)), ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))]:
        cv2.line(frame, (cx, cy), (cx + sx * lx, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + sy * ly), color, thickness, cv2.LINE_AA)


def draw_dashed_rect(frame, box: Box, color, thickness: int = 2, dash_len: int = 10, phase: float = 0.0):
    """Dashed, gently pulsing outline -- used only for the tentative BODY_HOLD box."""
    x1, y1, x2, y2 = box
    pulse = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(phase))
    c = tuple(int(ch * pulse) for ch in color)

    def dashed_line(p1, p2):
        p1, p2 = np.array(p1, dtype=np.float32), np.array(p2, dtype=np.float32)
        length = float(np.linalg.norm(p2 - p1))
        if length < 1:
            return
        direction = (p2 - p1) / length
        for i in range(max(1, int(length // (dash_len * 2))) + 1):
            start = p1 + direction * (i * dash_len * 2)
            end = start + direction * dash_len
            cv2.line(frame, tuple(start.astype(int)), tuple(np.minimum(end, p2).astype(int)), c, thickness, cv2.LINE_AA)

    dashed_line((x1, y1), (x2, y1))
    dashed_line((x2, y1), (x2, y2))
    dashed_line((x2, y2), (x1, y2))
    dashed_line((x1, y2), (x1, y1))


PERSON_BOX_COLOR = (255, 170, 0)


def draw_status_panel(frame, lines, origin):
    x0, y0 = origin
    pad = 8
    line_h = 20
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0] for t in lines) + pad * 2
    h = line_h * len(lines) + pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - pad, y0 - pad), (x0 - pad + w, y0 - pad + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x0, y0 + i * line_h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="enrolled identity to lock")
    parser.add_argument("--camera", type=int, default=get_cam_index())
    parser.add_argument("--width", type=int, default=1280, help="requested capture width")
    parser.add_argument("--height", type=int, default=720, help="requested capture height")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--detector", choices=("haar", "yunet"), default="haar",
        help="Default stays 'haar' (matches DEFAULT_THRESHOLD's calibration); "
             "'yunet' is the project-wide default elsewhere -- re-tune --threshold if you switch.",
    )
    parser.add_argument(
        "--yolox", choices=("nano", "tiny", "off"), default="nano",
        help="YOLOX variant for body-assisted occlusion tracking, or 'off' to disable it.",
    )
    parser.add_argument(
        "--yolox-interval", type=int, default=3,
        help="Run YOLOX every Nth frame and reuse the cached result in between (default: 3).",
    )
    args = parser.parse_args()

    detector = (
        build_face_detector("yunet", min_size=(70, 70), debug=False)
        if args.detector == "yunet"
        else HaarFaceLandmarker5pt(min_size=(70, 70), debug=False)
    )
    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
    )
    matcher = FaceDBMatcher(
        load_db_npz(Path("data/db/face_db.npz")),
        dist_thresh=args.threshold,
    )

    body_detector = None
    if args.yolox != "off":
        from .yolox_person import YoloxPersonDetector

        model_path = f"models/yolox_{args.yolox}.onnx"
        if Path(model_path).exists():
            body_detector = YoloxPersonDetector(model_path=model_path, interval=args.yolox_interval, debug=False)
        else:
            print(f"[face_tracking] {model_path} not found -- running without body-assisted tracking.")

    tracker = LockedFaceTracker(args.target, detector, embedder, matcher, body_detector=body_detector)
    signals = FaceSignalExtractor()

    cap, cam_used, (cam_w, cam_h) = open_camera(args.camera, args.width, args.height)
    cam_text = f"cam {cam_used} {cam_w}x{cam_h}"

    detector_label = args.detector.upper()
    yolox_label = f"YOLOX-{args.yolox}" if body_detector is not None else "OFF"
    show_panel = False
    fps, fps_t0, fps_frames = 0.0, time.time(), 0

    blink_total = 0
    blink_flash = 0
    frame_counter = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_counter += 1

            locked_face, position = tracker.update(frame)
            view = frame.copy()
            h, w = view.shape[:2]

            fps_frames += 1
            if time.time() - fps_t0 >= 1.0:
                fps = fps_frames / (time.time() - fps_t0)
                fps_frames, fps_t0 = 0, time.time()

            # Other faces are deliberately ignored - show that instead of hiding them.
            for other in tracker.other_faces:
                cv2.rectangle(view, (other.x1, other.y1), (other.x2, other.y2), (150, 150, 150), 1)
                draw_label(view, "other (ignored)", (other.x1, max(14, other.y1 - 6)), (170, 170, 170), 0.5)

            state_text = f"{tracker.display_state.name}: {args.target}"
            state_color = (0, 180, 0) if locked_face is not None else (0, 140, 255)
            draw_label(view, state_text, (12, 28), state_color, 0.72)
            draw_label(view, cam_text, (w - 230, 28), (200, 200, 200), 0.55)

            # --- person box (co-detected while LOCKED, or the tentative hold) ---
            person_box = tracker.last_body_box
            if person_box is not None:
                if tracker.state == LockState.BODY_HOLD:
                    draw_dashed_rect(view, person_box, PERSON_BOX_COLOR, phase=frame_counter * 0.25)
                elif locked_face is not None:
                    draw_corner_brackets(view, person_box, PERSON_BOX_COLOR)

            if locked_face is not None:
                box = tracker.box(locked_face)
                x1, y1, x2, y2 = box
                cv2.rectangle(view, (x1, y1), (x2, y2), (255, 170, 0), 3)

                face_state = signals.analyze(frame, box)
                if face_state is not None:
                    if face_state.blink:
                        blink_total += 1
                        blink_flash = 6
                    blink_flash = max(0, blink_flash - 1)

                    eye_text = "EYES CLOSED" if face_state.eyes_closed else "EYES OPEN"
                    if blink_flash:
                        eye_text = "BLINK!"

                    expr_color = (0, 255, 255) if face_state.expression != "NEUTRAL" else (200, 200, 200)
                    draw_label(view, face_state.expression, (x1, max(55, y1 - 50)), expr_color)
                    draw_label(
                        view,
                        f"{eye_text} blinks={blink_total}",
                        (x1, max(78, y1 - 25)),
                        (255, 255, 0),
                    )
                    draw_label(
                        view,
                        f"EAR={face_state.ear:.3f} smile={face_state.smile_score * 100:.0f}% "
                        f"frown={face_state.frown_score * 100:.0f}% "
                        f"sad={face_state.sad_score * 100:.0f}% "
                        f"grimace={face_state.grimace_score * 100:.0f}%",
                        (12, h - 18),
                        (255, 255, 255),
                        0.52,
                    )

                if position is not None:
                    nx, ny = int(position.nose_xy[0]), int(position.nose_xy[1])
                    cv2.line(view, (w // 2, h // 2), (nx, ny), (255, 170, 0), 2, cv2.LINE_AA)
                    cv2.circle(view, (nx, ny), 6, (0, 255, 255), -1, cv2.LINE_AA)
                    draw_label(
                        view,
                        f"NOSE {position.direction}  "
                        f"dx={position.offset_x_px:+.0f}px dy={position.offset_y_px:+.0f}px "
                        f"dist={position.distance_px:.0f}px",
                        (12, 56),
                        (255, 170, 0),
                        0.60,
                    )
            elif tracker.state in (LockState.BODY_HOLD, LockState.REVERIFYING) or tracker.display_state == LockState.REVERIFYING:
                signals.reset()
                blink_flash = 0
                draw_body_hold_banner(
                    view, args.target, tracker.display_state,
                    tracker.occlusion_seconds, tracker.body_hold_timeout / 30.0,
                )
            else:
                signals.reset()
                blink_flash = 0
                draw_missing_banner(
                    view,
                    args.target,
                    len(tracker.other_faces),
                    lost=tracker.state == LockState.LOST,
                )

            # Frame-center crosshair.
            cv2.drawMarker(view, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 24, 1)
            dz = tracker.dead_zone
            cv2.rectangle(
                view,
                (int(w * (0.5 - dz / 2)), int(h * (0.5 - dz / 2))),
                (int(w * (0.5 + dz / 2)), int(h * (0.5 + dz / 2))),
                (120, 120, 120),
                1,
            )

            if show_panel:
                recognition = (
                    "Verified" if locked_face is not None
                    else "Pending" if tracker.display_state == LockState.REVERIFYING
                    else "Unknown"
                )
                panel = [
                    f"Face detector : {detector_label}",
                    f"Person detector: {yolox_label}",
                    f"Recognition   : {recognition}",
                    f"Tracking      : {tracker.display_state.name}",
                    f"Face visible  : {'Yes' if locked_face is not None else 'No'}",
                    f"Track ID      : {tracker.track_id if tracker.track_id is not None else '-'}",
                    f"FPS           : {fps:0.1f}",
                    f"Face latency  : {tracker.detector_latency_ms:0.1f} ms",
                ]
                if tracker.body_latency_ms is not None:
                    panel.append(f"Body latency  : {tracker.body_latency_ms:0.1f} ms")
                draw_status_panel(view, panel, origin=(w - 230, 60))
            else:
                draw_label(view, "d: details", (w - 150, 56), (180, 180, 180), 0.5)

            cv2.imshow("Locked Face Tracking", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_panel = not show_panel
    finally:
        cap.release()
        signals.close()
        if hasattr(detector, "close"):
            detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
