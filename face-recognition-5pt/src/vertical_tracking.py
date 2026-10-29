# src/vertical_tracking.py
"""
Vertical face tracking: the needle (tachometer) tilts with the locked person's nose as they
move up and down. A gauge drawn on the camera image shows the same angle the needle is sent.

Usage:
    py -m src.vertical_tracking --target Happy            # on-screen gauge only
    py -m src.vertical_tracking --target Happy --mqtt     # also drive the ESP8266 needle

Angle mapping (absolute mode): nose at frame center -> 90 deg, nose higher -> towards 180,
nose lower -> towards 0. --gain scales the movement (a head rarely reaches the frame edge),
--invert flips it if the physical needle moves the wrong way.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from .cam_config import get_cam_index
from .camera import open_camera
from .embed import ArcFaceEmbedderONNX
from .face_tracking import LockState, LockedFaceTracker, draw_label, draw_missing_banner
from .recognize_live import (
    DEFAULT_THRESHOLD,
    FaceDBMatcher,
    HaarFaceLandmarker5pt,
    load_db_npz,
)


def tilt_from_error_y(error_y: float, gain: float, invert: bool) -> float:
    """Nose vertical error (-1 top .. +1 bottom) -> tilt (-1 down .. +1 up), clipped."""
    tilt = -error_y * gain
    if invert:
        tilt = -tilt
    return float(np.clip(tilt, -1.0, 1.0))


def draw_vertical_gauge(view, angle: float, active: bool) -> None:
    """
    Tachometer on the right edge of the image, opening to the left:
    180 deg points up, 90 deg points straight left (level), 0 deg points down.
    """
    h, w = view.shape[:2]
    radius = int(min(h * 0.28, 170))
    cx, cy = w - 25, h // 2

    overlay = view.copy()
    cv2.ellipse(overlay, (cx, cy), (radius + 18, radius + 18), 0, 90, 270, (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, view, 0.45, 0, view)
    cv2.ellipse(view, (cx, cy), (radius, radius), 0, 90, 270, (220, 220, 220), 2, cv2.LINE_AA)

    def point(a_deg: float, r: float):
        t = math.radians(a_deg - 90.0)  # -90 (down) .. +90 (up)
        return int(cx - r * math.cos(t)), int(cy - r * math.sin(t))

    for a in range(0, 181, 15):
        major = a % 45 == 0
        p1 = point(a, radius)
        p2 = point(a, radius - (16 if major else 8))
        cv2.line(view, p1, p2, (220, 220, 220), 2 if major else 1, cv2.LINE_AA)
        if major:
            tx, ty = point(a, radius - 38)
            cv2.putText(view, str(a), (tx - 14, ty + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (220, 220, 220), 1, cv2.LINE_AA)

    color = (0, 220, 255) if active else (130, 130, 130)
    tip = point(angle, radius - 6)
    cv2.line(view, (cx, cy), tip, color, 4, cv2.LINE_AA)
    cv2.circle(view, (cx, cy), 9, color, -1, cv2.LINE_AA)
    draw_label(view, f"TILT {angle:.0f} deg", (cx - radius - 10, cy + radius + 40), color, 0.6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="enrolled identity to lock")
    parser.add_argument("--camera", type=int, default=get_cam_index())
    parser.add_argument("--width", type=int, default=1280, help="requested capture width")
    parser.add_argument("--height", type=int, default=720, help="requested capture height")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--gain", type=float, default=1.5,
                        help="scale vertical movement before mapping to the needle (1 = frame edge is full tilt)")
    parser.add_argument("--mqtt", action="store_true", help="publish needle angle to the MQTT broker")
    parser.add_argument("--mqtt-host", default=None, help="broker host (default: FACE_MQTT_HOST or localhost)")
    parser.add_argument("--invert", action="store_true", help="flip needle direction")
    parser.add_argument(
        "--mode",
        choices=("center", "absolute"),
        default="absolute",
        help="absolute: needle angle follows the face's vertical position (fixed camera); "
        "center: tilt toward the face until it is centred (only if the camera rides on the needle)",
    )
    parser.add_argument("--no-mirror", action="store_true", help="show the raw (un-mirrored) camera image")
    args = parser.parse_args()

    needle = None
    if args.mqtt:
        from .mqtt_out import BROKER_HOST, NeedlePublisher

        # Inversion is applied in tilt_from_error_y, so the publisher gets it un-inverted.
        needle = NeedlePublisher(host=args.mqtt_host or BROKER_HOST, mode=args.mode, dead_zone=0.07)

    detector = HaarFaceLandmarker5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
    )
    matcher = FaceDBMatcher(
        load_db_npz(Path("data/db/face_db.npz")),
        dist_thresh=args.threshold,
    )
    tracker = LockedFaceTracker(args.target, detector, embedder, matcher)

    cap, cam_used, (cam_w, cam_h) = open_camera(args.camera, args.width, args.height)
    cam_text = f"cam {cam_used} {cam_w}x{cam_h}"

    gauge_angle = 90.0  # what the on-screen gauge shows; holds its value while the face is lost
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            locked_face, position = tracker.update(frame)
            view = frame.copy()
            h, w = view.shape[:2]

            if position is not None:
                tilt = tilt_from_error_y(position.error_y, args.gain, args.invert)
                if args.mode == "absolute":
                    gauge_angle = 90.0 + tilt * 90.0
                if needle is not None:
                    # The publisher works on any normalised error; here it is the vertical tilt.
                    needle.send_error_x(tilt)
                    if args.mode == "center":
                        gauge_angle = needle.angle
            elif needle is not None:
                needle.send_stop()

            for other in tracker.other_faces:
                cv2.rectangle(view, (other.x1, other.y1), (other.x2, other.y2), (150, 150, 150), 1)
                draw_label(view, "other (ignored)", (other.x1, max(14, other.y1 - 6)), (170, 170, 170), 0.5)

            state_color = (0, 180, 0) if locked_face is not None else (0, 140, 255)
            draw_label(view, f"{tracker.state.name}: {args.target}", (12, 28), state_color, 0.72)
            draw_label(view, cam_text, (w - 230, 28), (200, 200, 200), 0.55)

            # Frame-center level line and the dead zone band around it.
            dz_px = int(tracker.dead_zone * h / 2)
            cv2.line(view, (0, h // 2), (w, h // 2), (200, 200, 200), 1)
            cv2.rectangle(view, (0, h // 2 - dz_px), (w, h // 2 + dz_px), (110, 110, 110), 1)

            if locked_face is not None and position is not None:
                x1, y1, x2, y2 = tracker.box(locked_face)
                cv2.rectangle(view, (x1, y1), (x2, y2), (255, 170, 0), 3)

                nx, ny = int(position.nose_xy[0]), int(position.nose_xy[1])
                cv2.line(view, (0, ny), (w, ny), (0, 255, 255), 1, cv2.LINE_AA)
                cv2.arrowedLine(view, (nx, h // 2), (nx, ny), (255, 170, 0), 3, cv2.LINE_AA, tipLength=0.2)
                cv2.circle(view, (nx, ny), 6, (0, 255, 255), -1, cv2.LINE_AA)
                draw_label(
                    view,
                    f"NOSE {position.vertical}  dy={position.offset_y_px:+.0f}px  "
                    f"needle={gauge_angle:.0f} deg",
                    (12, 56),
                    (255, 170, 0),
                    0.62,
                )
            else:
                draw_missing_banner(
                    view,
                    args.target,
                    len(tracker.other_faces),
                    lost=tracker.state == LockState.LOST,
                )

            draw_vertical_gauge(view, gauge_angle, active=locked_face is not None)

            cv2.imshow("Vertical Face Tracking", view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        if needle is not None:
            needle.send_stop()
            needle.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
