# src/person_tracking.py
"""
Anonymous multi-person tracking with opportunistic ArcFace verification.

Unlike face_tracking.py (which locks onto one named --target and ignores
everyone else), this tracks EVERY person YOLOX finds, from the moment they
enter the frame, whether or not a face has ever been visible -- front, side
or back. Each person gets an anonymous track id immediately. A face is only
ever used to try to attach a verified name to an existing track; it never
creates identity on its own, and YOLOX's person detection is never treated
as identity.

Labels shown on screen (exactly the states below):
    PERSON 01 - UNVERIFIED       body tracked, no face matched yet
    PERSON 01 - VERIFYING        a face just matched this track, checking it
    PERSON 01 - Alice - VERIFIED       ArcFace confirmed who it is
    PERSON 01 - Alice - FACE OCCLUDED  was verified, face not visible right now
    PERSON 01 - TRACK LOST       about to be dropped (shown briefly, then removed)

Identity never silently moves between tracks. Once a track is verified as
someone, a periodic recheck must keep matching that SAME name or the track
is dropped (not relabeled) -- see MultiPersonTracker._recheck(). A reappearing
person after their track is lost gets a brand-new track id and starts over
at UNVERIFIED; there is no body-appearance re-identification here (that is a
materially harder problem -- see the project README).

This script is passive monitoring only: it never opens an MQTT connection or
drives a servo. If a future turret integration wants to act on a track, it
must gate that on `Track.is_action_authorized`, exactly like
face_tracking.LockedFaceTracker.is_action_authorized -- a track's VERIFIED
state is the only state that counts as proof of identity; UNVERIFIED,
VERIFYING and FACE_OCCLUDED never do, FACE_OCCLUDED included (it is a lapse
in a verified track's face visibility, not fresh proof).

Run:
    python -m src.person_tracking
    python -m src.person_tracking --detector yunet --yolox tiny
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .align import align_face_5pt
from .detector_factory import build_face_detector
from .embed import ArcFaceEmbedderONNX
from .cam_config import get_cam_index
from .camera import open_camera
from .recognize_live import DEFAULT_THRESHOLD, FaceDBMatcher, HaarFaceLandmarker5pt, load_db_npz

Box = Tuple[int, int, int, int]


class TrackState(Enum):
    UNVERIFIED = auto()
    VERIFYING = auto()
    VERIFIED = auto()
    FACE_OCCLUDED = auto()
    LOST = auto()  # shown for one frame, then the track is dropped


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def contains_fraction(inner: Box, outer: Box) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    ax1, ay1 = max(ix1, ox1), max(iy1, oy1)
    ax2, ay2 = min(ix2, ox2), min(iy2, oy2)
    inter = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    inner_area = max(1, (ix2 - ix1) * (iy2 - iy1))
    return inter / float(inner_area)


def center(box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def box_diag(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(1.0, float(np.linalg.norm([x2 - x1, y2 - y1])))


def continuity_score(last_box: Box, candidate: Box) -> float:
    """Same displacement-aware overlap scoring used in face_tracking.py."""
    overlap = iou(last_box, candidate)
    displacement = float(np.linalg.norm(center(candidate) - center(last_box))) / box_diag(last_box)
    return overlap - 0.35 * displacement


MIN_CONTINUITY = -0.30
MIN_FACE_BODY_OVERLAP = 0.55


@dataclass
class Track:
    track_id: int
    body_box: Box
    state: TrackState = TrackState.UNVERIFIED
    verified_name: Optional[str] = None
    face_box: Optional[Box] = None
    face_kps: Optional[np.ndarray] = None
    lost_frames: int = 0
    frames_seen: int = 0
    last_seen_at: float = field(default_factory=time.time)

    # What the UI should show THIS tick. Equals `state` on every ordinary
    # frame; forced to VERIFYING for exactly the one tick a fresh ArcFace
    # check is actually running on this track (see MultiPersonTracker.
    # update()), regardless of whether that check then succeeds or fails --
    # `state` itself always resolves straight to the real next value
    # (VERIFIED or UNVERIFIED) so logic never rests in VERIFYING.
    display_state: Optional["TrackState"] = None  # set to `state` in __post_init__

    # Only meaningful when MultiPersonTracker was given a --target. None =
    # never checked against the target yet (e.g. face never seen -- could
    # still turn out to be the target). True/False = a face WAS checked
    # against it. Never invented without an actual ArcFace check having run.
    is_target: Optional[bool] = None

    # Body-appearance re-ID (src/body_reid.py) -- a soft, hedged GUESS about
    # identity from clothing/build, used only while verified_name is still
    # None (no face seen yet on this track). Never promotes to VERIFIED and
    # never overrides a real ArcFace result. See Track.label / target_tag.
    body_guess_name: Optional[str] = None
    body_guess_score: Optional[float] = None
    body_reid_attempts: int = 0
    next_body_reid_frame: int = 0

    def __post_init__(self):
        if self.display_state is None:
            self.display_state = self.state

    @property
    def label(self) -> str:
        tag = f"PERSON {self.track_id:02d}"
        ds = self.display_state
        if ds == TrackState.VERIFIED:
            return f"{tag} - {self.verified_name} - VERIFIED"
        if ds == TrackState.FACE_OCCLUDED:
            return f"{tag} - {self.verified_name} - FACE OCCLUDED"
        if ds == TrackState.VERIFYING:
            return f"{tag} - VERIFYING"
        if ds == TrackState.LOST:
            return f"{tag} - TRACK LOST"
        if self.body_guess_name is not None:
            # Deliberately worded differently from VERIFIED ("- X -") so it
            # can never be misread as a real identity claim.
            return f"{tag} - UNVERIFIED (likely {self.body_guess_name}?)"
        return f"{tag} - UNVERIFIED"

    def target_tag(self, target: Optional[str]) -> str:
        """" | TARGET" / " | NOT <target>", or "" if unchecked/no target set.

        Only ever reflects an actual completed ArcFace check (is_target is
        None until one has run on this track) -- never guesses.
        """
        if target is None or self.is_target is None:
            return ""
        return " | TARGET" if self.is_target else f" | NOT {target.upper()}"

    @property
    def is_action_authorized(self) -> bool:
        """Only a currently-VERIFIED track counts as proof of identity.

        FACE_OCCLUDED deliberately does NOT authorize anything -- it is a
        lapse in an already-verified track's face visibility, not fresh
        proof the body in frame right now is still that person.
        """
        return self.state == TrackState.VERIFIED


# Face verification is re-checked periodically even after VERIFIED, so a
# track can never keep an identity on stale evidence forever.
RECHECK_EVERY = 15
# A LOST track is dropped (not kept trying) after this many frames with no
# matching person box at all.
DROP_AFTER = 20


class MultiPersonTracker:
    """Greedy IoU+displacement multi-object tracker over YOLOX person boxes,
    with opportunistic per-track ArcFace verification from any face that
    falls inside a track's body box.

    This is intentionally a simple greedy assignment, not a Kalman-filter
    tracker (SORT/DeepSORT/ByteTrack) -- CPU budget here is already split
    between a face detector, YOLOX and ArcFace, and a project of this size
    does not need motion prediction to keep a handful of people's track ids
    stable. If fast-moving crowds become a real use case, that is the
    documented upgrade path.
    """

    def __init__(
        self, detector, embedder, matcher, body_detector,
        recheck_every: int = RECHECK_EVERY, target: Optional[str] = None,
        body_reid_embedder=None, body_matcher=None,
        body_reid_interval: int = 20, body_reid_max_attempts: int = 6,
    ):
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher
        self.body_detector = body_detector
        self.recheck_every = recheck_every
        # Optional: when set, every track that has had a face checked gets
        # flagged is_target True/False (see Track.is_target). This is a
        # display-only overlay on top of the same real identity logic --
        # it never changes who a track is verified as, only whether the UI
        # also says "this one is/isn't <target>".
        self.target = target

        # Optional body-appearance re-ID (src/body_reid.py). OSNet measured
        # ~126 ms/crop on this machine -- far too slow to run every frame,
        # let alone once per unverified track. At most ONE check happens
        # per update() call, on whichever eligible track is most overdue
        # (see _maybe_body_reid), and a track stops being tried at all
        # after body_reid_max_attempts misses.
        self.body_reid_embedder = body_reid_embedder
        self.body_matcher = body_matcher
        self.body_reid_interval = body_reid_interval
        self.body_reid_max_attempts = body_reid_max_attempts
        self.body_reid_latency_ms: Optional[float] = None

        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self.frame_index = 0

        self.detector_latency_ms = 0.0
        self.body_latency_ms: Optional[float] = None

    def _new_track(self, box: Box) -> Track:
        t = Track(track_id=self._next_id, body_box=box)
        self._next_id += 1
        self.tracks[t.track_id] = t
        return t

    def _verify_face(self, frame, face) -> Tuple[Optional[str], bool]:
        """Returns (name_or_None, accepted)."""
        aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
        match = self.matcher.match(self.embedder.embed(aligned))
        return (match.name if match.accepted else None), bool(match.accepted)

    def _maybe_body_reid(self, frame) -> None:
        """At most one body-appearance check per frame (see __init__ for why).

        Only considers tracks that have never had a face verified
        (verified_name is None) -- the moment a face is actually seen,
        ArcFace is strictly better evidence and this stops being useful for
        that track. Never writes verified_name or changes state; only ever
        sets the soft, hedged body_guess_* fields.
        """
        if self.body_reid_embedder is None or self.body_matcher is None:
            return

        candidates = [
            t for t in self.tracks.values()
            if t.verified_name is None
            and t.body_reid_attempts < self.body_reid_max_attempts
            and t.next_body_reid_frame <= self.frame_index
            and (t.body_box[2] - t.body_box[0]) >= 40
            and (t.body_box[3] - t.body_box[1]) >= 80
        ]
        if not candidates:
            return

        track = min(candidates, key=lambda t: t.next_body_reid_frame)
        x1, y1, x2, y2 = track.body_box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        t0 = time.perf_counter()
        embedding = self.body_reid_embedder.embed(crop)
        match = self.body_matcher.match(embedding)
        self.body_reid_latency_ms = (time.perf_counter() - t0) * 1000.0

        track.body_reid_attempts += 1
        track.next_body_reid_frame = self.frame_index + self.body_reid_interval
        if match.accepted:
            track.body_guess_name = match.name
            track.body_guess_score = match.similarity
        # A miss does not clear a previous guess by itself -- it just means
        # this particular attempt didn't clear the threshold; attempts are
        # capped above so a genuine stranger stops costing cycles.

    def update(self, frame) -> List[Track]:
        self.frame_index += 1

        t0 = time.perf_counter()
        faces = self.detector.detect(frame, max_faces=10)
        self.detector_latency_ms = (time.perf_counter() - t0) * 1000.0

        person_boxes: List[Box] = []
        if self.body_detector is not None:
            person_boxes = [(b.x1, b.y1, b.x2, b.y2) for b in self.body_detector.maybe_detect(frame)]
            self.body_latency_ms = self.body_detector.last_latency_ms

        # --- 1. associate existing tracks to this frame's person boxes ---
        unmatched_boxes = list(range(len(person_boxes)))
        assignments: Dict[int, int] = {}  # track_id -> person_boxes index

        pairs = []
        for tid, track in self.tracks.items():
            for bi in unmatched_boxes:
                score = continuity_score(track.body_box, person_boxes[bi])
                if score > MIN_CONTINUITY:
                    pairs.append((score, tid, bi))
        pairs.sort(key=lambda p: p[0], reverse=True)

        claimed_tracks, claimed_boxes = set(), set()
        for score, tid, bi in pairs:
            if tid in claimed_tracks or bi in claimed_boxes:
                continue
            assignments[tid] = bi
            claimed_tracks.add(tid)
            claimed_boxes.add(bi)
        unmatched_boxes = [bi for bi in unmatched_boxes if bi not in claimed_boxes]

        # --- 2. update matched tracks, age out the rest ---
        for tid, track in list(self.tracks.items()):
            if tid in assignments:
                track.body_box = person_boxes[assignments[tid]]
                track.lost_frames = 0
                track.frames_seen += 1
                track.last_seen_at = time.time()
                if track.state == TrackState.LOST:
                    # Shouldn't normally happen (LOST tracks are dropped
                    # below), but never resurrect a lost track's old
                    # identity silently -- start it over clean.
                    track.state = TrackState.UNVERIFIED
                    track.verified_name = None
            else:
                track.lost_frames += 1
                if track.lost_frames > DROP_AFTER:
                    track.state = TrackState.LOST

        for tid in [tid for tid, tr in self.tracks.items() if tr.state == TrackState.LOST]:
            del self.tracks[tid]

        # --- 3. spawn new anonymous tracks for unclaimed person boxes ---
        for bi in unmatched_boxes:
            self._new_track(person_boxes[bi])

        # --- 4. associate faces to tracks by containment, then verify ---
        # A track's face_box is cleared every frame and only re-set below if
        # a face is actually found inside it THIS frame -- FACE_OCCLUDED has
        # to track "no face associated this frame", not "no body box this
        # frame" (a person can keep walking in frame, fully tracked by
        # YOLOX, while turning away from the camera and hiding their face).
        for track in self.tracks.values():
            track.face_box = None
            track.face_kps = None
            # Default this tick's display to the real state; overridden
            # below to VERIFYING for the one tick a fresh check runs.
        face_claimed = set()
        matched_face_this_frame = set()
        verifying_flash_ids = set()
        for track in self.tracks.values():
            best_face, best_frac, best_i = None, 0.0, -1
            for i, face in enumerate(faces):
                if i in face_claimed:
                    continue
                frac = contains_fraction((face.x1, face.y1, face.x2, face.y2), track.body_box)
                if frac > best_frac:
                    best_face, best_frac = face, frac
                    best_i = i
            if best_face is None or best_frac < MIN_FACE_BODY_OVERLAP:
                continue
            face_claimed.add(best_i)
            matched_face_this_frame.add(track.track_id)
            track.face_box = (best_face.x1, best_face.y1, best_face.x2, best_face.y2)
            track.face_kps = best_face.kps

            needs_check = (
                track.state in (TrackState.UNVERIFIED, TrackState.FACE_OCCLUDED)
                or (track.state == TrackState.VERIFIED and self.frame_index % self.recheck_every == 0)
            )
            if not needs_check:
                continue

            if track.verified_name is None:
                # Fresh first-time check on this track -- flash VERIFYING
                # for this one tick regardless of the outcome below. Not
                # done for an already-verified track's periodic recheck or
                # FACE_OCCLUDED reappearance; those have their own states.
                track.display_state = TrackState.VERIFYING
                verifying_flash_ids.add(track.track_id)

            name, accepted = self._verify_face(frame, best_face)

            if self.target is not None:
                # A real ArcFace check just ran on this track's face -- now
                # we actually know whether it's the target or not, so record
                # it. Never set this without a check having run.
                track.is_target = bool(accepted and name == self.target)

            if track.verified_name is not None:
                # This track has claimed an identity before -- whether it is
                # currently VERIFIED (periodic recheck) or FACE_OCCLUDED
                # (the face just came back), re-verification must keep
                # agreeing with THAT name, or the track is dropped rather
                # than relabeled. This is what stops identity from silently
                # transferring onto a different person who ends up inside
                # the same track after a crossing or a mis-association.
                if accepted and name == track.verified_name:
                    track.state = TrackState.VERIFIED
                else:
                    track.state = TrackState.LOST
                continue

            if accepted:
                track.verified_name = name
                track.state = TrackState.VERIFIED
            else:
                # Stays UNVERIFIED, not VERIFYING: needs_check above only
                # ever re-triggers from UNVERIFIED/FACE_OCCLUDED (or a
                # periodic VERIFIED recheck) -- landing in VERIFYING here
                # and staying there would permanently stop this track from
                # ever being checked again (confirmed live: a stranger's
                # track got stuck showing "VERIFYING" for 24+ straight
                # frames). A rejected match is a normal, repeatable outcome
                # for an unenrolled person, not a stuck state.
                track.state = TrackState.UNVERIFIED

        # A VERIFIED track with no face associated THIS frame is occluded,
        # not re-identified from its body alone -- demote, but keep the name
        # on display (see Track.label) and don't touch anything else.
        for track in self.tracks.values():
            if track.state == TrackState.VERIFIED and track.track_id not in matched_face_this_frame:
                track.state = TrackState.FACE_OCCLUDED

        for tid in [tid for tid, tr in self.tracks.items() if tr.state == TrackState.LOST]:
            del self.tracks[tid]

        # --- 5. body-appearance re-ID, for tracks with no face seen yet ---
        self._maybe_body_reid(frame)

        # Sync display_state to the now-final real state for everything
        # except the track(s) flashing VERIFYING this tick (set above,
        # deliberately left alone so the flash survives every mutation
        # after it -- including the FACE_OCCLUDED demotion, which runs
        # later in this same method).
        for track in self.tracks.values():
            if track.track_id not in verifying_flash_ids:
                track.display_state = track.state

        return list(self.tracks.values())


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

STATE_COLOR = {
    TrackState.UNVERIFIED: (150, 150, 150),
    TrackState.VERIFYING: (0, 215, 255),
    TrackState.VERIFIED: (60, 220, 90),
    TrackState.FACE_OCCLUDED: (210, 140, 40),
    TrackState.LOST: (60, 60, 220),
}


def draw_label(frame, text, xy, color, scale=0.58):
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_track(frame, track: Track, target: Optional[str] = None):
    color = STATE_COLOR[track.display_state]
    x1, y1, x2, y2 = track.body_box
    thickness = 3 if track.display_state == TrackState.VERIFIED else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    draw_label(frame, track.label, (x1, max(16, y1 - 8)), color)

    tag = track.target_tag(target)
    if tag:
        tag_color = (0, 215, 255) if track.is_target else (0, 0, 255)  # gold vs red
        draw_label(frame, tag.strip(" |"), (x1, min(frame.shape[0] - 6, y2 + 20)), tag_color, 0.6)

    if track.face_box is not None:
        fx1, fy1, fx2, fy2 = track.face_box
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 1)
        if track.face_kps is not None:
            for (x, y) in track.face_kps.astype(int):
                cv2.circle(frame, (int(x), int(y)), 2, color, -1)


def draw_status_panel(frame, lines, origin):
    x0, y0 = origin
    pad, line_h = 8, 20
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0] for t in lines) + pad * 2
    h = line_h * len(lines) + pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - pad, y0 - pad), (x0 - pad + w, y0 - pad + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x0, y0 + i * line_h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=get_cam_index())
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--detector", choices=("haar", "yunet"), default="haar")
    parser.add_argument("--yolox", choices=("nano", "tiny", "off"), default="nano")
    parser.add_argument("--yolox-interval", type=int, default=3)
    parser.add_argument(
        "--target", default=None,
        help=(
            "Optional enrolled name. Every track still gets tracked and "
            "labeled normally (this does not turn into face_tracking.py's "
            "single-target lock) -- once a track's face has actually been "
            "checked, it additionally shows '| TARGET' if it matched this "
            "name, or '| NOT <TARGET>' if it matched someone else or no "
            "one. A track that has never shown a face yet (e.g. still "
            "back-facing) shows neither -- that is genuinely unknown, not "
            "guessed."
        ),
    )
    parser.add_argument(
        "--body-reid", choices=("auto", "on", "off"), default="auto",
        help=(
            "Hedged body-appearance guess ('likely <name>?') for a track "
            "that has never shown a face -- e.g. walked in with their back "
            "turned. 'auto' (default) enables it if models/osnet_x1_0_op10.onnx "
            "and data/db/body_db.npz both exist, else runs without it. This "
            "only has a realistic chance of working within the SAME session, "
            "same clothes -- it is not face-level recognition. Needs "
            "`python -m src.enroll_body` first."
        ),
    )
    parser.add_argument("--body-reid-threshold", type=float, default=0.35)
    args = parser.parse_args()

    if args.yolox == "off":
        raise SystemExit(
            "person_tracking needs YOLOX to detect anonymous bodies -- "
            "pass --yolox nano or --yolox tiny, not 'off'."
        )

    detector = (
        build_face_detector("yunet", min_size=(60, 60), debug=False)
        if args.detector == "yunet"
        else HaarFaceLandmarker5pt(min_size=(70, 70), debug=False)
    )
    embedder = ArcFaceEmbedderONNX(model_path="models/embedder_arcface.onnx", input_size=(112, 112), debug=False)
    matcher = FaceDBMatcher(load_db_npz(Path("data/db/face_db.npz")), dist_thresh=args.threshold)

    from .yolox_person import YoloxPersonDetector

    model_path = f"models/yolox_{args.yolox}.onnx"
    if not Path(model_path).exists():
        raise SystemExit(f"{model_path} not found -- see README setup steps.")
    body_detector = YoloxPersonDetector(model_path=model_path, interval=args.yolox_interval, debug=False)

    body_reid_embedder, body_matcher = None, None
    if args.body_reid != "off":
        osnet_path = Path("models/osnet_x1_0_op10.onnx")
        body_db_path = Path("data/db/body_db.npz")
        if osnet_path.exists() and body_db_path.exists():
            from .body_reid import BodyReidEmbedder, BodyDBMatcher, load_body_db_npz

            body_reid_embedder = BodyReidEmbedder(model_path=str(osnet_path))
            body_matcher = BodyDBMatcher(load_body_db_npz(body_db_path), dist_thresh=args.body_reid_threshold)
            print(f"[person_tracking] body re-ID enabled: {sorted(body_matcher.db.keys())}")
        elif args.body_reid == "on":
            raise SystemExit(
                f"--body-reid on requires both {osnet_path} and {body_db_path} "
                "-- download the OSNet model and run `python -m src.enroll_body` first."
            )
        else:
            print("[person_tracking] body re-ID unavailable (model or body_db.npz missing) -- running without it.")

    tracker = MultiPersonTracker(
        detector, embedder, matcher, body_detector, target=args.target,
        body_reid_embedder=body_reid_embedder, body_matcher=body_matcher,
    )

    cap, cam_used, (cam_w, cam_h) = open_camera(args.camera, args.width, args.height)
    cam_text = f"cam {cam_used} {cam_w}x{cam_h}"

    show_panel = False
    fps, fps_t0, fps_frames = 0.0, time.time(), 0

    print("Anonymous person tracking (passive monitoring only -- no MQTT/servo).")
    print("q=quit  d=details")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            tracks = tracker.update(frame)
            view = frame.copy()
            h, w = view.shape[:2]

            fps_frames += 1
            if time.time() - fps_t0 >= 1.0:
                fps = fps_frames / (time.time() - fps_t0)
                fps_frames, fps_t0 = 0, time.time()

            for track in tracks:
                draw_track(view, track, target=args.target)

            verified_count = sum(1 for t in tracks if t.state == TrackState.VERIFIED)
            header = f"people={len(tracks)} verified={verified_count}"
            if args.target:
                header += f"  target={args.target}"
            draw_label(view, header, (12, 28), (0, 255, 0), 0.72)
            draw_label(view, cam_text, (w - 230, 28), (200, 200, 200), 0.55)

            if show_panel:
                panel = [
                    "Mode          : anonymous person tracking",
                    f"Target        : {args.target or '(none)'}",
                    f"Face detector : {args.detector.upper()}",
                    f"Person detector: YOLOX-{args.yolox}",
                    f"Tracks        : {len(tracks)}",
                    f"Verified      : {verified_count}",
                    f"FPS           : {fps:0.1f}",
                    f"Face latency  : {tracker.detector_latency_ms:0.1f} ms",
                ]
                if tracker.body_latency_ms is not None:
                    panel.append(f"Body latency  : {tracker.body_latency_ms:0.1f} ms")
                if tracker.body_reid_embedder is not None:
                    reid_ms = tracker.body_reid_latency_ms
                    panel.append(f"Body reID     : {'on' if reid_ms is None else f'{reid_ms:0.1f} ms'}")
                draw_status_panel(view, panel, origin=(w - 230, 60))
            else:
                draw_label(view, "d: details", (w - 150, 56), (180, 180, 180), 0.5)

            cv2.imshow("Anonymous Person Tracking", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_panel = not show_panel
    finally:
        cap.release()
        if hasattr(detector, "close"):
            detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
