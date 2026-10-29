# Eagle Eye — Face Recognition, Face Locking and Expression Tracking

A CPU-only face recognition system built for the Intelligent Robotics course. A webcam
feeds a vision pipeline that finds faces, works out who each one is, locks onto one chosen
person, and reports where their nose is relative to the center of the frame and what
expression they are making. An optional motorized camera mount (ESP8266/ESP32 servo,
driven over MQTT) can physically sweep and hold on a recognized face.

The project has two ways to run, sharing the same vision pipeline:

| Mode | Command | What it is for |
|---|---|---|
| **Software-only** (no hardware) | `recognize_live`, `face_tracking`, `person_tracking` | Recognition on screen, target locking, nose tracking, expressions, missing-face warning, anonymous multi-person tracking |
| **Turret** (needs the servo board) | `recognize` | Sweeps a servo through preset angles until it sees a known face, then holds on it |

## What it does

- **Recognizes people** against an enrolled database using ArcFace embeddings.
- **Labels every visible face** (`recognize_live`): enrolled people in green with their
  name, everyone else as **Unknown** in red, including small, far-away faces.
- **Locks onto one person** (`face_tracking`): finds the chosen name among all visible
  faces, follows them, ignores everyone else, and keeps checking it is still the right
  person.
- **Measures the nose tip** relative to the center of the frame: direction
  (`UP-LEFT`, `CENTER`, ...) and the offset in pixels.
- **Reads expressions:** smile, frown, sad, grimace, blinking and eyes closed.
- **Warns when the face is gone:** a flashing red/orange **FACE MISSING** banner.

## How it works

```
Camera frame (default 1280x720)
   -> MediaPipe FaceLandmarker on the full frame: up to 10 faces, 5 key landmarks each
   -> Far-face pass: Haar cascade proposes small candidates, the landmark model
      must confirm each one (so objects are not boxed as faces)
   -> Similarity alignment / warp of each face to a canonical 112x112 pose
   -> ArcFace ONNX embedding (512-d identity vector, L2-normalized)
   -> Cosine distance to every enrolled identity
   -> Name if distance <= threshold, otherwise "Unknown"
```

Aligning every face to the same pose before embedding is what makes the matching
threshold meaningful; without it, head pose alone would move the embedding more than
identity does.

**Face locking** (`face_tracking.py`) is a small state machine:

```
SEARCHING    -> target recognized among the visible faces  ->  LOCKED
LOCKED       -> target no longer found, no YOLOX person box ->  LOST
LOCKED       -> target no longer found, a YOLOX person box
                still overlaps where the face was           ->  BODY_HOLD
BODY_HOLD    -> a face reappears in that region              ->  (re-verify, see below)
LOST         -> found again (identity re-verified)            ->  LOCKED
LOST / BODY_HOLD -> not found for the hold's timeout           ->  SEARCHING
```

While locked, the same face is followed frame to frame by box overlap plus movement, and
its identity is re-verified every 10 frames so the lock cannot silently jump to someone
else. `BODY_HOLD` is new: when `--yolox` is enabled, a YOLOX person-box detector (never
used for identity, only for "is a person still roughly here") can keep the tracker
holding position through a brief occlusion -- a hand over the face, turning around,
briefly stepping behind something. The moment a face is visible again, ArcFace is asked
to re-verify it before the lock is trusted again (shown on screen as a one-frame
"re-verification" flash); if it's the wrong person, or nobody returns before the hold's
timeout, the lock clears and the tracker goes back to `SEARCHING` -- a body box alone
never grants or restores an identity.

Detector/body-tracker options for both `recognize_live` and `face_tracking`:

| Flag | Values | Notes |
|---|---|---|
| `--detector` | `haar` (default), `yunet` | `haar` is what `DEFAULT_THRESHOLD` was tuned against here (incl. the far-face pass in `recognize_live.py`). `yunet` (OpenCV's `FaceDetectorYN`) is the project-wide default for *new* work and handles head angle / partial occlusion better, but re-run `src.evaluate` and pass `--threshold` if you switch, since the alignment crop shifts slightly. |
| `--yolox` (`face_tracking` only) | `nano` (default), `tiny`, `off` | Person detector backing `BODY_HOLD`. Nano measured ~13ms/frame on this machine (CPU-only), Tiny ~25ms -- Nano is the better default unless you see it losing the body box too easily. |
| `--yolox-interval` | integer, default `3` | Run YOLOX every Nth frame and reuse the cached boxes in between, since it doesn't need to run as often as face detection. |

Models: `models/face_detection_yunet_2023mar.onnx` (OpenCV Zoo) and `models/yolox_nano.onnx`
/ `models/yolox_tiny.onnx` (Megvii-BaseDetection/YOLOX releases) -- see Setup below.

**Nose tracking.** The nose tip landmark is smoothed (exponential average) and compared
with the frame center. Output: direction, `dx` / `dy` in pixels (+ is right/down) and the
straight-line distance. A small dead zone in the middle counts as `CENTER`.

**Expressions.** Blink and eyes-closed come from the eye aspect ratio (EAR) of the face
mesh. Smile comes from mouth width and mouth-corner lift. Frown, sad and grimace come
from MediaPipe blendshape scores:

| Expression | Driven by |
|---|---|
| Frown | brows pulled down + mouth corners down |
| Sad | inner brows raised + mouth corners down |
| Grimace | mouth stretched sideways + eyes squinted + nose wrinkled |

Each score is smoothed and uses on/off hysteresis so labels do not flicker. The live
percentage for each is printed at the bottom of the window.

## Quick start

Run everything from the `face-recognition-5pt` folder.

```powershell
cd face-recognition-5pt

# Pick the camera (0 = built-in, 1 = usually the external USB camera)
$env:FALCON_CAM_INDEX = "1"

py -m src.camera                       # confirm the right camera opens
py -m src.enroll                       # enroll people (needed once)
py -m src.recognize_live               # label every face: name or Unknown
py -m src.face_tracking --target Happy # lock onto one enrolled person
```

Both demos also accept `--camera 1` directly. To change the default for every script,
edit `DEFAULT_CAM_INDEX` in `src/cam_config.py`.

### `recognize_live` — everyone in view

Boxes every face it can find (up to 10). Enrolled people are green with their name;
everyone else is red **Unknown**. Faces too small to identify reliably (eyes under 20 px
apart) are still boxed and shown as Unknown ("too far to identify"), because a blurry
crop can match an enrolled person by chance.

Keys: `q` quit, `r` reload database, `+` / `-` adjust the threshold, `d` debug overlay.
Options: `--camera`, `--width`, `--height`, `--threshold`.

### `face_tracking` — one locked person

Locks onto `--target`. Other people are drawn as gray "other (ignored)" boxes and never
affect the lock. While locked it shows the nose offset, the expression label, the
blink count and the per-expression scores. When the target is not locked it shows the
flashing **FACE MISSING** banner.

Options: `--target <name>` (required), `--camera`, `--width`, `--height`,
`--threshold`. Key: `q` quit.

### `person_tracking` — everyone, anonymous until verified

```powershell
py -m src.person_tracking
py -m src.person_tracking --detector yunet --yolox tiny
```

Different job from `face_tracking`: instead of locking onto one named person and
ignoring everyone else, this tracks **every** person YOLOX finds -- front, side or back,
whether or not a face has ever been visible -- and only tries to attach a verified name
once a face is seen. Each person gets an anonymous id the moment they enter the frame:

| Label | Meaning |
|---|---|
| `PERSON 01 - UNVERIFIED` | Body tracked, no face matched to it yet |
| `PERSON 01 - VERIFYING` | A face just matched; checking it against the database |
| `PERSON 01 - Alice - VERIFIED` | ArcFace confirmed the identity |
| `PERSON 01 - Alice - FACE OCCLUDED` | Was verified; no face currently associated with this body (turned away, hand up, etc.) -- name stays shown, but this is *not* being claimed as proof it's still them |
| `PERSON 01 - TRACK LOST` | Shown for one frame before the track is dropped |

A verified identity never silently moves to a different track: a periodic recheck must
keep matching the *same* name, whether the track is currently `VERIFIED` or
`FACE_OCCLUDED` -- a mismatch drops the track rather than relabeling it. A lost track's
id is never reused and never keeps its name; a person who walks back into frame gets a
new id and starts at `UNVERIFIED` again -- *unless* `--body-reid` is enabled and a body
match is found (see below), which is a hedged guess, never a real identity claim.

### Optional: body-appearance re-ID (recognize someone whose face was never seen)

By default, a track that never shows a face stays `UNVERIFIED` forever, even if it's
someone enrolled -- there is no way to know who it is from a body shape alone. 
`--body-reid on` adds a second, much weaker signal: an OSNet appearance embedding
(clothing/build, not a face) compared against a separate gallery. It never promotes a
track to `VERIFIED` -- it only adds a hedge to the label:

```
PERSON 01 - UNVERIFIED (likely Happy?)
```

**This only has a realistic chance of working within the same session, same outfit.**
It is not face-level recognition and was never claimed to be -- clothing-based matching
cannot reliably survive an outfit change, different lighting, or two people in similar
clothes. `Track.is_action_authorized` stays gated on `VERIFIED` only; a body guess never
authorizes anything or relabels a track. A track that's already `FACE_OCCLUDED` keeps
its real verified name from ArcFace -- a body guess only ever applies to a track whose
face has never been seen at all.

Setup:
```powershell
# OSNet (Torchreid's model, exported to ONNX by the nnio project), ~8.7 MB:
Invoke-WebRequest -Uri "https://drive.google.com/uc?id=1FlkHyebtkIqkE4jPfGZu-RXwfgCe4T1p&export=download" -OutFile "models\osnet_x1_0_op10.onnx"

# Build the body-appearance gallery (separate from the face database) --
# turn in place (front/side/back) while capturing so it isn't one pose:
py -m src.enroll_body

py -m src.person_tracking --body-reid on --target Happy
```

Measured ~126 ms per body crop on this machine (standalone) -- far slower than the face
pipeline, so `person_tracking.py` runs at most **one** body-reid check per frame total
(round-robin across unverified tracks), and gives up on a track after 6 misses so a
genuine stranger doesn't keep eating that budget. `--body-reid auto` (the default) only
turns it on if both the model and `data/db/body_db.npz` exist; it never errors out if
you haven't set it up.

Association across frames is a simple greedy IoU + displacement match (the same scoring
`face_tracking.py` uses), generalized to many simultaneous boxes -- not a Kalman-filter
tracker like SORT/ByteTrack. That is a deliberate scope call for a CPU budget already
split between a face detector, YOLOX and ArcFace; the periodic identity recheck is what
bounds how long a mis-association could show a wrong label (at most `recheck_every`
frames, default 15) rather than motion prediction preventing it in the first place.

This mode is **passive monitoring only** -- it never opens MQTT or drives a servo.
`Track.is_action_authorized` exists for exactly one reason: if a future turret
integration wants to act on a track, it must check that (true only while `VERIFIED`),
never merely `state != LOST`, and `FACE_OCCLUDED` must never authorize movement.

**Optional `--target <name>`** adds an explicit flag to every track's label once its
face has actually been checked: `| TARGET` for the one that matches, `| NOT <NAME>`
for everyone else (whether they're a different enrolled person or a stranger). This
does **not** turn into `face_tracking.py`'s single-target lock -- everyone is still
tracked and labeled with their own real identity; `--target` only adds this extra tag
on top. A track that has never shown a face yet (e.g. still back-facing) gets no tag
either way -- that is genuinely unknown, never guessed.

```powershell
py -m src.person_tracking --target Happy
```

Options: `--detector`, `--yolox` (`nano`/`tiny`, no `off` -- this mode needs YOLOX),
`--yolox-interval`, `--target`, `--camera`, `--width`, `--height`, `--threshold`.
Key: `q` quit, `d` toggle the status panel.

### Turret mode

```powershell
py -m src.recognize     # needs the servo board online and reachable over MQTT
```

## Demo checklist

| Requirement | How to show it |
|---|---|
| External HD camera | Run with `--camera 1`; the window and terminal show the camera index and resolution in use |
| Recognizes you without confusing others | `recognize_live` with a second person in frame: you in green, them Unknown |
| Other faces in the background | `recognize_live` labels them Unknown; `face_tracking` ignores them |
| Nose tip distance and direction from center | `face_tracking`: move your head, watch `NOSE <direction> dx dy dist` and the line from the center |
| Smile, frown, sad, blink, grimace | `face_tracking`: make each expression, watch the label and the percentages |
| Red/orange warning when the face is missing | `face_tracking`: cover your face or step out of frame |

## Recognition threshold and accuracy

A match is accepted when cosine distance is at or below the threshold. The default is
**0.73**, chosen with `py -m src.evaluate` on the enrolled data:

| Threshold | False accepts (FAR) | Genuine faces rejected (FRR) |
|---|---|---|
| 0.65 | 0.04% | 21% |
| 0.73 | 0.8% | 20% |
| 0.81 (old default) | roughly 10-20% (estimated between the 0.76 and 0.87 sweep points) | ~16% |

Lower is stricter. Re-tune it after re-enrolling, and enroll with the same camera and
lighting you will demo with, since that improves accuracy more than the threshold does.
The figures come from the enrollment crops themselves, so real-world error is somewhat
higher.

## Known limits

- **Distance:** at 1280x720, faces are found down to a box about 60 px wide, roughly
  3 m from a typical webcam. Smaller faces are not detected.
- **Head angle:** with `--detector haar` (default), faces turned well sideways or partly
  hidden are not detected, so they get no box until they face the camera again. YuNet
  (`--detector yunet`) tolerates more angle, at a higher per-frame cost (~27ms vs Haar's
  far-face pass) -- not yet re-tuned for this project's enrolled identities.
- **`BODY_HOLD` only covers brief occlusion of an already-locked face.** It is not
  identification from the body: it cannot tell two people wearing similar clothing
  apart, and it was only verified here against scripted scenarios (correct person
  returning, wrong person returning, and timeout), not a live multi-person session --
  see the Benchmarks section.
- **`person_tracking` has no body re-identification either.** A lost track (person
  walked out of frame, or association failed) is gone for good -- a reappearing person
  gets a brand-new id and starts over at `UNVERIFIED`, even if it's the same enrolled
  person from a minute ago. There is no clothing/gait model here, deliberately: that is
  a materially different, much harder problem than detecting *that* a body is there.
- **`person_tracking`'s multi-object association is greedy IoU, not a motion-predicting
  tracker.** Two people crossing paths can, in principle, confuse which track owns which
  box for a frame or two. The periodic identity recheck is what bounds the damage (a
  wrong label can last at most `recheck_every` frames, default 15, before the track is
  dropped rather than left mislabeled) -- it was verified with a scripted test that
  forces exactly this disagreement, not with two real people crossing on camera.
- **Small faces are labeled Unknown, not identified.** This is deliberate.
- **Expression thresholds** are starting values. Tune them in `expression_scores()` in
  `src/face_signals.py` by watching the on-screen percentages.
- **Speed:** roughly 15-20 fps in `recognize_live` on a laptop CPU with Haar; see
  Benchmarks for YuNet/YOLOX numbers.

## Benchmarks (this machine, CPU-only, single real camera frame)

| Component | Latency | Notes |
|---|---|---|
| Haar + MediaPipe full-frame (no face in view) | ~7-10 ms | Early-exits when nothing is found |
| Haar + MediaPipe full-frame (`detect_far=True`) | ~7 ms (no face) | Far-face Haar-proposal pass only runs every other frame |
| YuNet (`face_detection_yunet_2023mar.onnx`) | ~27-29 ms | Roughly constant whether or not a face is present -- it's a single CNN pass, it doesn't early-exit like Haar |
| YOLOX-Nano (`yolox_nano.onnx`, 416x416) | ~13 ms | Warmed-up; first call is ~2-3x slower (ORT graph optimization) |
| YOLOX-Tiny (`yolox_tiny.onnx`, 416x416) | ~25 ms | ~2x Nano's cost; only worth it if Nano's box recall proves too coarse |
| OSNet body re-ID (`osnet_x1_0_op10.onnx`, 256x128) | ~100-130 ms | Slow -- `person_tracking.py` runs at most one check per frame total, round-robin, capped at 6 attempts per track |

**Running multiple ONNX Runtime sessions in one process oversubscribes the CPU badly
with its unlimited-by-default thread pool.** Measured on this 20-core machine: ArcFace +
YOLOX + OSNet running together went from ~1.2s down to ~0.24s per combined pass just by
capping each session to `intra_op_num_threads=4`. This is applied in `embed.py`,
`yolox_person.py` and `body_reid.py` -- without it, `person_tracking.py --body-reid on`
was briefly measured taking **2.4-3.5 *seconds*** per body-reid check instead of ~100ms.

**What's verified vs. not:**
- YuNet: confirmed finding real faces (0.92-0.94 confidence) on enrolled crops.
- YOLOX decode/NMS math: checked against hand-computed expected boxes.
- `person_tracking.py` (multi-person, `--target`, `--body-reid`): run live against real
  people walking in frame, including `VERIFIED -> FACE_OCCLUDED -> VERIFIED` cycling as
  a person turned their face away and back, multiple simultaneous anonymous tracks, and
  a track being dropped and replaced with a new id when someone left. The `VERIFYING`
  one-tick flash, the `FACE_OCCLUDED` transition, and a stuck-state bug in `VERIFYING`
  were all caught and fixed by this live testing, not just the scripted tests.
- `face_tracking.py`'s `BODY_HOLD` specifically (one locked target, occluded, body held
  by YOLOX, re-verified on reappearance) is still only checked with scripted fake
  detections, not a live occlusion -- nobody deliberately covered their face on camera
  during this session to exercise that exact path. Try it yourself before trusting it.
- Body re-ID's actual accuracy (does "likely Happy?" really fire correctly, and not on
  the wrong person) is **unverified** -- it needs `python -m src.enroll_body` run first,
  which nothing in this session could do without a live person choosing to enroll.

## Setup

1. **Virtual environment**
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
2. **Dependencies**
   ```powershell
   pip install opencv-python numpy onnxruntime scipy tqdm mediapipe paho-mqtt
   ```
3. **ArcFace ONNX model**
   ```bash
   curl -L -o buffalo_l.zip "https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download"
   unzip -o buffalo_l.zip
   cp w600k_r50.onnx models/embedder_arcface.onnx
   rm -f buffalo_l.zip w600k_r50.onnx 1k3d68.onnx 2d106det.onnx det_10g.onnx genderage.onnx
   ```
4. **MediaPipe landmark model** — place `face_landmarker.task` (from
   [MediaPipe's model zoo](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker))
   at `models/face_landmarker.task`. Blendshapes (frown/sad/grimace) need this model.
5. **YuNet (only if you pass `--detector yunet`)**
   ```powershell
   # raw.githubusercontent.com serves a git-lfs pointer for this file, not the
   # binary -- use media.githubusercontent.com instead:
   Invoke-WebRequest -Uri "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -OutFile "models\face_detection_yunet_2023mar.onnx"
   ```
6. **YOLOX (only for `face_tracking`'s `BODY_HOLD` or `person_tracking`, both default to it)**
   ```powershell
   Invoke-WebRequest -Uri "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx" -OutFile "models\yolox_nano.onnx"
   # optional, for comparison:
   Invoke-WebRequest -Uri "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx" -OutFile "models\yolox_tiny.onnx"
   ```
7. **Turret only:** set real WiFi / MQTT values in a local, gitignored copy of
   `src/main.py` or `sketch_sep11a.ino` before flashing the board.

Each pipeline stage can be checked on its own:

```powershell
py -m src.camera      # webcam
py -m src.detect      # face detection
py -m src.landmarks   # 5-point landmarks
py -m src.align       # alignment / warp
py -m src.embed       # embedding
py -m src.evaluate    # threshold sweep (needs 2+ enrolled identities)
```

## Turret: how the two sides communicate

The Python side never drives GPIO pins, and the microcontroller never runs recognition.
They coordinate only through an MQTT broker, so either side can be restarted or replaced
independently.

| Topic | Direction | Meaning |
|---|---|---|
| `falcon/eye/servo/cmd` | desktop -> board | `ANGLE:<0-180>`, `STOP`, `HOME` |
| `falcon/eye/servo/status` | board -> desktop | current angle / motion state |
| `falcon/eye/recognition` | desktop -> broker | recognition events (who / stranger) |
| `falcon/eye/status/<client_id>` | board -> broker | online/offline presence (MQTT last-will) |

If nobody recognized is in frame, the turret sweeps through preset angles. When it
recognizes a face it stops and holds that position, and resumes sweeping if the face is
lost. Two firmware options implement the same command set:

- `src/main.py` — MicroPython, ESP8266
- `sketch_sep11a/sketch_sep11a.ino` — Arduino C++, ESP32

## Repository layout

```
├── README.md
├── face-recognition-5pt/
│   ├── data/
│   │   ├── enroll/<name>/         # enrollment crops per identity (real faces, do not publish)
│   │   ├── debug_aligned/         # alignment debug snapshots
│   │   └── db/                    # face_db.npz / face_db.json (enrolled embeddings)
│   ├── models/
│   │   ├── embedder_arcface.onnx  # ~166 MB, downloaded separately
│   │   ├── face_landmarker.task   # MediaPipe landmark + blendshape model
│   │   ├── face_detection_yunet_2023mar.onnx  # OpenCV Zoo, ~230 KB, downloaded separately
│   │   └── yolox_nano.onnx / yolox_tiny.onnx  # Megvii-BaseDetection releases, downloaded separately
│   ├── src/
│   │   ├── recognize_live.py      # multi-face recognition, far-face pass, Unknown labels
│   │   ├── face_tracking.py       # face locking, nose tracking, missing-face banner, BODY_HOLD
│   │   ├── person_tracking.py     # anonymous multi-person tracking + opportunistic verification
│   │   ├── yunet_5pt.py           # YuNet face detector (--detector yunet everywhere)
│   │   ├── yolox_person.py        # YOLOX person detector (body-assisted tracking)
│   │   ├── detector_factory.py    # picks yunet/haar by name, shared by every script
│   │   ├── face_signals.py        # blink, eyes closed, smile, frown, sad, grimace
│   │   ├── recognize.py           # turret loop: recognition + MQTT + servo logic
│   │   ├── enroll.py              # enrollment tool
│   │   ├── evaluate.py            # threshold tuning (FAR/FRR sweep)
│   │   ├── camera.py              # camera opening helper + webcam test
│   │   ├── cam_config.py          # default camera index / FALCON_CAM_INDEX
│   │   ├── detect.py, landmarks.py, align.py, embed.py, haar_5pt.py
│   │   │                          # pipeline stages and their sanity checks
│   │   └── main.py                # ESP8266 firmware (flash to the board, not the PC)
│   ├── init_project.py
│   ├── run.bat, run.ps1           # launchers for the turret scanner
└── sketch_sep11a/sketch_sep11a.ino  # ESP32 firmware alternative
```

## Privacy and secrets

- `data/enroll/`, `data/debug_aligned/` and `data/db/` hold real people's face crops and
  embeddings. Do not publish them.
- WiFi credentials in `src/main.py` and `sketch_sep11a.ino` are placeholders; put real
  values only in a local copy.
- `models/embedder_arcface.onnx` is about 166 MB, over GitHub's 100 MB file limit, so
  keep it out of any repository and download it per the setup steps.

## References

- Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). *ArcFace: Additive Angular Margin
  Loss for Deep Face Recognition.* CVPR 2019.
- InsightFace Project — 2D & 3D Face Analysis.
- ONNX / ONNX Runtime documentation.
- Lugaresi, C., Tang, J., Nash, H., et al. (2019). *MediaPipe: A Framework for Building
  Perception Pipelines.*
