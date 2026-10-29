# src/enroll_body.py
"""
Body-appearance enrollment: camera -> YOLOX person box -> OSNet embedding.
Stores a template per identity (mean embedding, L2-normalized) in
data/db/body_db.npz, the input person_tracking.py's --body-reid reads.

This is a SEPARATE gallery from the face database (data/db/face_db.npz) --
body appearance is a much weaker, clothing-dependent signal, never meant to
be compared 1:1 against ArcFace. Re-enroll whenever you change clothes if
you want --body-reid to keep working; it is only realistically useful
within roughly the same session/outfit, not across days.

Capture a person turning in place -- front, both sides, back -- so the
gallery isn't just one pose. 15-20 samples across that turn is reasonable.

Run:
    python -m src.enroll_body
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .body_reid import BodyReidEmbedder
from .cam_config import get_cam_index
from .yolox_person import YoloxPersonDetector


@dataclass
class EnrollBodyConfig:
    out_db_npz: Path = Path("data/db/body_db.npz")
    out_db_json: Path = Path("data/db/body_db.json")
    save_crops: bool = True
    crops_dir: Path = Path("data/enroll_body")
    samples_needed: int = 15
    auto_capture_every_s: float = 0.4
    max_existing_crops: int = 200
    window_main: str = "enroll_body"


def ensure_dirs(cfg: EnrollBodyConfig) -> None:
    cfg.out_db_npz.parent.mkdir(parents=True, exist_ok=True)
    if cfg.save_crops:
        cfg.crops_dir.mkdir(parents=True, exist_ok=True)


def load_db(cfg: EnrollBodyConfig) -> Dict[str, np.ndarray]:
    if cfg.out_db_npz.exists():
        data = np.load(cfg.out_db_npz, allow_pickle=True)
        return {k: data[k].astype(np.float32) for k in data.files}
    return {}


def save_db(cfg: EnrollBodyConfig, db: Dict[str, np.ndarray], meta: dict) -> None:
    ensure_dirs(cfg)
    np.savez(cfg.out_db_npz, **{k: v.astype(np.float32) for k, v in db.items()})
    cfg.out_db_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def mean_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    E = np.stack([e.reshape(-1) for e in embeddings], axis=0).astype(np.float32)
    m = E.mean(axis=0)
    return (m / (np.linalg.norm(m) + 1e-12)).astype(np.float32)


def _list_existing_crops(person_dir: Path, max_count: int) -> List[Path]:
    if not person_dir.exists():
        return []
    files = sorted([p for p in person_dir.glob("*.jpg") if p.is_file()])
    return files[-max_count:] if len(files) > max_count else files


def load_existing_samples_from_crops(cfg, emb: BodyReidEmbedder, person_dir: Path) -> List[np.ndarray]:
    if not cfg.save_crops:
        return []
    base: List[np.ndarray] = []
    for p in _list_existing_crops(person_dir, cfg.max_existing_crops):
        img = cv2.imread(str(p))
        if img is None:
            continue
        try:
            base.append(emb.embed(img))
        except Exception:
            continue
    return base


def draw_status(frame, name, base_count, new_count, needed, auto, msg=""):
    total = base_count + new_count
    lines = [
        f"ENROLL BODY: {name}",
        f"Existing: {base_count} | New: {new_count} | Total: {total} / {needed}",
        "Turn in place: front -> side -> back -> side -> front",
        f"Auto: {'ON' if auto else 'OFF'} (toggle: a)",
        "SPACE=capture | s=save | r=reset NEW | q=quit",
    ]
    if msg:
        lines.insert(0, msg)
    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 24


def main():
    cfg = EnrollBodyConfig()
    ensure_dirs(cfg)

    name = input("Enter person name to enroll (body appearance, e.g. Happy): ").strip()
    if not name:
        print("No name provided. Exiting.")
        return
    if name.startswith(("py ", "python ")) or " -m " in name:
        print(f"'{name}' looks like a command, not a person's name. Exiting without enrolling.")
        return

    det = YoloxPersonDetector(model_path="models/yolox_nano.onnx", interval=1, debug=False)
    emb = BodyReidEmbedder(model_path="models/osnet_x1_0_op10.onnx", debug=False)

    db = load_db(cfg)
    person_dir = cfg.crops_dir / name
    if cfg.save_crops:
        person_dir.mkdir(parents=True, exist_ok=True)

    base_samples = load_existing_samples_from_crops(cfg, emb, person_dir)
    new_samples: List[np.ndarray] = []
    status_msg = f"Loaded {len(base_samples)} existing samples from disk." if base_samples else ""

    auto = False
    last_auto = 0.0

    cap = cv2.VideoCapture(get_cam_index())
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera.")

    print("\nBody enrollment started. Stand so your whole body is in frame.")
    print("Turn slowly in a circle while capturing so the gallery covers front/side/back.")
    print("Controls: SPACE=capture, a=auto, s=save, r=reset NEW, q=quit\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            vis = frame.copy()
            boxes = det.maybe_detect(frame, force=True)
            crop: Optional[np.ndarray] = None
            box = None

            if boxes:
                box = max(boxes, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))
                cv2.rectangle(vis, (box.x1, box.y1), (box.x2, box.y2), (255, 170, 0), 2)
                raw = frame[box.y1:box.y2, box.x1:box.x2]
                if raw.size > 0:
                    crop = raw

            now = time.time()
            if auto and crop is not None and (now - last_auto) >= cfg.auto_capture_every_s:
                new_samples.append(emb.embed(crop))
                last_auto = now
                status_msg = f"Auto captured NEW ({len(new_samples)})"
                if cfg.save_crops:
                    cv2.imwrite(str(person_dir / f"{int(now * 1000)}.jpg"), crop)

            draw_status(vis, name, len(base_samples), len(new_samples), cfg.samples_needed, auto, status_msg)
            cv2.imshow(cfg.window_main, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                auto = not auto
                status_msg = f"Auto mode {'ON' if auto else 'OFF'}"
            if key == ord("r"):
                new_samples.clear()
                status_msg = "NEW samples reset (existing kept)."
            if key == ord(" "):
                if crop is None:
                    status_msg = "No person detected. Not captured."
                else:
                    new_samples.append(emb.embed(crop))
                    status_msg = f"Captured NEW ({len(new_samples)})"
                    if cfg.save_crops:
                        cv2.imwrite(str(person_dir / f"{int(time.time() * 1000)}.jpg"), crop)
            if key == ord("s"):
                total = len(base_samples) + len(new_samples)
                if total < max(3, cfg.samples_needed // 2):
                    status_msg = f"Not enough total samples to save (have {total})."
                    continue
                template = mean_embedding(base_samples + new_samples)
                db[name] = template
                meta = {
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "embedding_dim": int(template.size),
                    "names": sorted(db.keys()),
                    "samples_total_used": int(total),
                    "note": (
                        "Body-appearance (OSNet) embeddings, L2-normalized. "
                        "Clothing-dependent -- re-enroll after changing outfits."
                    ),
                }
                save_db(cfg, db, meta)
                status_msg = f"Saved '{name}' to body_db. Total identities: {len(db)}"
                print(status_msg)
                base_samples = load_existing_samples_from_crops(cfg, emb, person_dir)
                new_samples.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
