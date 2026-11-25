# clipbro — Architecture & Contracts (Phase 1: Motion data layer)

clipbro is a local, original 9:16 clipper. Phase 1 builds only the **motion data layer**:
analyse a time range of a video and output saliency, timestamped events and a raw focus path.
No rendering, no UI.

Content-agnostic: works for talk / IRL / gaming streams. Nothing is game-specific; behaviour
differences live in config **profiles**, not in code.

## 1. Design rules (non-negotiable)

1. **Ports & adapters.** `clipbro/core` defines data types and `Protocol` interfaces (ports).
   Everything that touches FFmpeg, OpenCV model files, disk or HTTP is an **adapter** implementing one port.
2. **Dependency direction**
   - `clipbro.core` imports only stdlib + numpy.
   - `clipbro.motion` imports only `clipbro.core` (+ numpy, cv2 for pure maths on arrays). Never `clipbro.adapters`, `clipbro.api`, `clipbro.cli`, `subprocess`.
   - `clipbro.adapters` imports `clipbro.core` only.
   - `clipbro.bootstrap` is the **only** module that imports concrete adapters and wires them.
   - `clipbro.api` / `clipbro.cli` import `clipbro.bootstrap` + `clipbro.core`.
   A test (`tests/test_architecture.py`) enforces this by parsing imports with `ast`.
3. **Plug-and-play.** Channels, detectors and smoothers self-register with `@register(kind, name)`
   and are auto-discovered from their package with `pkgutil`. Adding one = adding one file
   (+ optional config entry). No other file changes.
4. **Config, not constants.** Every threshold is read through `ctx.config.threshold(plugin, key, default)`.
   Defaults live next to the code that uses them, documented in a module-level `DEFAULTS` dict.
5. **Style.** Readable Python 3.10+: type hints, docstrings, functions <= ~40 lines,
   no semicolons, no bare `except Exception` (catch specific errors), no one-letter names except loop indices / maths.
6. **Deps.** numpy, opencv-python-headless, FFmpeg binary. FastAPI optional (`[api]` extra). Nothing else.

## 2. Package layout

```
clipbro/
  __init__.py                 # __version__ = "0.1.0"
  core/
    __init__.py               # re-exports public names
    types.py                  # TimeRange, FrameBatch, AudioTrack, Box, Word, Event, FocusPoint, Keyframe, SaliencyMap
    context.py                # AnalysisContext
    ports.py                  # Protocols
    registry.py               # Registry, REGISTRY, register, discover
    config.py                 # MotionConfig, PROFILES
    errors.py                 # ClipbroError, InvalidRange, MediaError
  adapters/
    ffmpeg_video.py           # FfmpegFrameSource            (FrameSource)
    ffmpeg_audio.py           # FfmpegAudioSource            (AudioSource)
    ffmpeg_cuts.py            # FfmpegCutDetector            (CutDetector)
    yunet_faces.py            # YuNetFaceDetector            (FaceDetector)
    json_cache.py             # JsonFileCache                (Cache)
  hype/
    lexicon.py                # LexiconWordScorer            (WordScorer)
  motion/
    grid.py                   # pure helpers: cell_reduce, cell_centre, argmax_cell
    audio_features.py         # pure helpers on AudioTrack: rms_track, band_ratio, loud_jumps
    channels/                 # one file per SignalChannel (auto-discovered)
      hud.py motion.py flash.py smoke.py face.py ui_change.py centre.py
    detectors/                # one file per EventDetector (auto-discovered)
      boom.py smoke.py scream.py hype_word.py hard_cut.py dead_air.py
    saliency.py               # run_channels(ctx) -> SaliencyMap
    events.py                 # run_detectors(ctx) -> list[Event]; dedupe()
    focus.py                  # raw_focus(); IdentitySmoother; focus_path()
    keyframes.py              # save(), load() JSON v1 (Keyframe type lives in core.types)
    pipeline.py               # MotionAnalyzer, MotionResult, VERSION
  bootstrap.py                # build_analyzer(...)
  api/motion_routes.py        # make_router(...) -> fastapi.APIRouter
  cli/debug.py                # python -m clipbro.cli.debug
models/                       # face_detection_yunet_2023mar.onnx (downloaded, git-ignored)
scripts/fetch_models.py       # downloads YuNet from the OpenCV model zoo
tests/
```

## 3. Core types (`clipbro/core/types.py`)

All coordinates are **normalised 0..1** of the frame (x right, y down). Times are **absolute seconds in the source media**.

```python
@dataclass(frozen=True)
class TimeRange:
    start: float
    end: float
    # __post_init__: finite, start >= 0, end > start, else raise InvalidRange
    @property
    def duration(self) -> float: ...

@dataclass(frozen=True)
class FrameBatch:
    frames: np.ndarray            # uint8 [N, H, W, 3] RGB
    times: np.ndarray             # float64 [N], absolute seconds
    source_size: tuple[int, int]  # (width, height) of the original video
    fps: float
    # properties: n -> int, size -> (W, H) of sampled frames

@dataclass(frozen=True)
class AudioTrack:
    samples: np.ndarray           # float32 mono, -1..1
    sr: int
    start: float                  # absolute time of samples[0]
    # property duration

@dataclass(frozen=True)
class Box:        x: float; y: float; w: float; h: float; score: float      # normalised
@dataclass(frozen=True)
class Word:       text: str; start: float; end: float

@dataclass
class Event:
    t: float; kind: str; strength: float; x: float; y: float; reason: str = ""
    # __post_init__: clamp strength/x/y into 0..1, t finite
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "Event": ...

@dataclass(frozen=True)
class FocusPoint: t: float; cx: float; cy: float; weight: float = 1.0

@dataclass(frozen=True)
class Keyframe:
    t: float; cx: float; cy: float
    zoom: float = 1.0; shake: float = 0.0; speed: float = 1.0; reason: str = ""
    # __post_init__ raises ValueError unless: t finite >= 0, 0<=cx<=1, 0<=cy<=1,
    #   1.0<=zoom<=2.0, 0.25<=speed<=2.0, shake>=0
    # to_dict / from_dict

@dataclass
class SaliencyMap:
    times: np.ndarray                     # [N]
    grid: tuple[int, int]                 # (rows, cols)
    channels: dict[str, np.ndarray]       # name -> float32 [N, rows, cols], 0..1
    combined: np.ndarray                  # float32 [N, rows, cols]
    hud: np.ndarray                       # bool [rows, cols]
    def channel(self, name: str) -> np.ndarray   # zeros if absent
    def nearest(self, t: float) -> int           # index of nearest frame
```

## 4. Context (`clipbro/core/context.py`)

```python
@dataclass
class AnalysisContext:
    frames: FrameBatch
    rng: TimeRange
    config: "MotionConfig"
    audio: AudioTrack | None = None
    cuts: list[float] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    faces: FaceDetector | None = None
    scorer: WordScorer | None = None
    channels: dict[str, np.ndarray] = field(default_factory=dict)  # filled by saliency runner
    saliency: SaliencyMap | None = None                             # set before detectors run
    _memo: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def grid(self) -> tuple[int, int]: return self.config.grid
    def memo(self, key: str, compute: Callable[[], T]) -> T: ...   # compute once, share (e.g. "gray", "flow")
    def threshold(self, plugin: str, key: str, default: float) -> float: ...  # delegates to config
```

## 5. Ports (`clipbro/core/ports.py`) — all `@runtime_checkable Protocol`

```python
class FrameSource(Protocol):
    def read(self, media: Path, rng: TimeRange, fps: float, width: int) -> FrameBatch: ...
class AudioSource(Protocol):
    def read(self, media: Path, rng: TimeRange, sr: int) -> AudioTrack | None: ...   # None = no audio stream
class CutDetector(Protocol):
    def cuts(self, media: Path, rng: TimeRange) -> list[float]: ...                   # absolute times in rng
class FaceDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[Box]: ...                             # RGB uint8 in, normalised boxes out
class WordScorer(Protocol):
    def score(self, word: str) -> float: ...                                          # 0..1
class Cache(Protocol):
    def get(self, key: str) -> dict | None: ...
    def put(self, key: str, value: dict) -> None: ...
    def put_array(self, key: str, name: str, array: np.ndarray) -> None: ...
    def get_array(self, key: str, name: str) -> np.ndarray | None: ...
class SignalChannel(Protocol):
    name: str
    requires: tuple[str, ...]
    def compute(self, ctx: AnalysisContext) -> np.ndarray: ...   # float32 [N, rows, cols] in 0..1
class EventDetector(Protocol):
    kind: str
    def detect(self, ctx: AnalysisContext) -> list[Event]: ...
class FocusSmoother(Protocol):
    def smooth(self, path: list[FocusPoint]) -> list[FocusPoint]: ...
```

## 6. Registry (`clipbro/core/registry.py`)

```python
class Registry:
    def register(self, kind: str, name: str) -> Callable[[type], type]   # decorator; sets cls.name/kind if missing; duplicate name -> ValueError
    def unregister(self, kind: str, name: str) -> None                   # for tests
    def names(self, kind: str) -> list[str]
    def get(self, kind: str, name: str) -> type                          # KeyError with helpful message
    def create(self, kind: str, name: str, **kwargs) -> Any
    def ordered(self, kind: str, names: Iterable[str]) -> list[str]      # topological by class attr `requires`;
                                                                         # missing dep / cycle -> ValueError
def discover(package: str) -> None     # import every submodule of package (pkgutil.iter_modules), idempotent
REGISTRY = Registry()
register = REGISTRY.register
```
Kinds used: `"channel"`, `"detector"`, `"smoother"`.

## 7. Config (`clipbro/core/config.py`)

```python
@dataclass
class MotionConfig:
    fps: float = 6.0
    width: int = 320
    grid: tuple[int, int] = (8, 8)
    audio_sr: int = 16000
    face_every: int = 2
    top_k: int = 3
    dedupe_window: float = 0.5
    channels: list[str]   = ["hud", "motion", "flash", "smoke", "face", "ui_change", "centre"]
    detectors: list[str]  = ["boom", "smoke", "scream", "hype_word", "hard_cut", "dead_air"]
    # keys are channel names; "a|b" means element-wise max of channels a and b
    weights: dict[str, float] = {"motion": .35, "flash|smoke": .25, "face": .25, "ui_change": .10, "centre": .05}
    thresholds: dict[str, dict[str, float]] = {}   # plugin name -> key -> value (overrides plugin DEFAULTS)

    def threshold(self, plugin: str, key: str, default: float) -> float
    def fingerprint(self) -> str                        # stable sha256 of all fields (used in cache key)
    def with_overrides(self, **changes) -> "MotionConfig"   # deep-merges dicts
    @classmethod
    def from_dict(cls, data: dict) -> "MotionConfig"    # unknown keys -> ValueError
    @classmethod
    def from_toml(cls, path: Path) -> "MotionConfig"    # tomllib (py>=3.11) or ValueError message on 3.10
    @classmethod
    def profile(cls, name: str) -> "MotionConfig"       # from PROFILES; unknown -> ValueError

PROFILES = {
  "default": {},
  "talk":   {"weights": {"face": .45, "motion": .20, "flash|smoke": .10, "ui_change": .10, "centre": .15}},
  "gaming": {"weights": {"motion": .40, "flash|smoke": .30, "face": .15, "ui_change": .10, "centre": .05}},
}
```

## 8. Grid helpers (`clipbro/motion/grid.py`)

```python
def cell_slices(shape_hw: tuple[int,int], grid: tuple[int,int]) -> list[list[tuple[slice, slice]]]
def cell_reduce(img: np.ndarray, grid, fn=np.mean) -> np.ndarray        # [H,W] -> [rows, cols]
def cell_centre(row: int, col: int, grid) -> tuple[float, float]        # normalised (x, y)
def argmax_cell(scores: np.ndarray) -> tuple[int, int]                  # [rows, cols] -> (row, col)
def to_gray(frames: np.ndarray) -> np.ndarray                           # [N,H,W,3] -> uint8 [N,H,W]
```

## 9. Channels (`clipbro/motion/channels/*.py`)

Each: a class decorated `@register("channel", "<name>")`, attributes `name`, `requires`, module-level `DEFAULTS`.
Output float32 `[N, rows, cols]` in 0..1. Shared intermediates via `ctx.memo("gray", ...)`.

| name | requires | algorithm (defaults in brackets) |
|---|---|---|
| `hud` | () | Per pixel on the sampled gray (`ctx.memo("gray")`): static = temporal std < `std_max` [6.0]; overlay pixel = static **and** within 1 px (3x3 dilation) of a Canny(80,160) edge present on >= `edge_persist` [0.6] of the frames. Cell is HUD if the fraction of its pixels that are overlay >= `cover` [0.04] (so a small logo over moving footage marks its own cell). Needs N >= `min_frames` [6] else none. Output = mask broadcast to N (1.0 / 0.0). Also `ctx.memo("hud_mask", ...)` bool [rows, cols]. |
| `motion` | ("hud",) | Farneback flow on consecutive gray frames downscaled by `flow_scale` [0.5] (pyr .5, levels 3, win 15, iters 3, poly_n 5, sigma 1.2); magnitudes are divided by `flow_scale` so units stay sampled-frame px. Per cell 90th percentile magnitude / `scale_px` [2.0 * width/160 px per frame], clip 0..1. Frame i is zeroed if mean absdiff(gray[i], gray[i-1]) > `cut_diff` [38] or a cut in `ctx.cuts` lies in (t[i-1], t[i]]. Frame 0 copies frame 1. HUD cells -> 0. |
| `flash` | () | Per-cell mean luminance L and orange fraction O (R>200, 90<G<190, B<90). Burst = L rises by >= `lum_jump` [35] in one step and within `return_frames` [3] falls back by >= half the rise. score = clip(rise / (2*lum_jump)) on burst frames; plus fire = clip(dO / `orange_jump` [0.12]) for rises in O. Max of both. |
| `smoke` | ("flash",) | Per cell: saturation mean S and Laplacian variance V; baseline = mean of previous 1 s (round(fps) frames). sat_drop=(Sb-S)/max(Sb,1), sharp_drop=(Vb-V)/max(Vb,1). score = clip(min(sat_drop/`sat_drop` [0.25], sharp_drop/`sharp_drop` [0.4])) only where 60 <= L <= 190. **Gate:** frame counts only if some frame in the previous `gate_s` [2.0] s had flash max >= `trigger` [0.5] or an audio loud jump (audio_features.loud_jumps). **Cuts:** a cut frame has full-frame mean absdiff of stats_gray to the previous frame > `cut_diff` [30] or a `ctx.cuts` time in (t[i-1], t[i]]. The baseline never reaches back past the last cut; smoke = 0 on frames c .. c + round(fps) + round(`cut_reset_s` [1.0] * fps) for each cut c; cut frames never count as flashes for the gate. |
| `face` | () | If `ctx.faces` is None -> zeros. Run detector every `config.face_every` frames, hold result for skipped frames. Rasterise each Box: per cell, fraction of the cell covered by the box x score, summed, clip 1. |
| `ui_change` | ("hud",) | Only in HUD cells: per-cell mean absdiff of consecutive gray / `scale` [30], clip. Else 0. (Overlay pops: alerts, kill feeds, donation boxes.) |
| `centre` | () | exp(-d^2 / (2*`sigma`^2)) [0.35], d = distance of cell centre from (0.5, 0.5) in normalised units; constant over time. |

## 10. Audio features (`clipbro/motion/audio_features.py`) — pure numpy

```python
def rms_track(track: AudioTrack, hop: float = 0.04) -> tuple[np.ndarray, np.ndarray]          # (abs times, rms)
def band_ratio(track: AudioTrack, lo: float, hi: float, hop: float = 0.04) -> tuple[np.ndarray, np.ndarray]
def loud_jumps(track: AudioTrack | None, ratio: float = 3.0, floor: float = 0.02,
               rise: float = 1.8, lookback: float = 0.12) -> list[tuple[float, float]]
    # ONSETS only: (time, rms/median) where rms > max(ratio*median, floor) AND
    # rms > rise * mean(rms of the previous `lookback` s). Hits within 0.25 s merge to the EARLIEST
    # (onset time), carrying the max ratio seen in that group. A sustained loud tone yields one hit at its start.
```

## 11. Detectors (`clipbro/motion/detectors/*.py`)

Each: `@register("detector", "<kind>")`, attribute `kind`, module `DEFAULTS`, `detect(ctx) -> list[Event]`.
Run **after** saliency, so `ctx.saliency` is set. x/y = normalised source position; default
locator = `clipbro.motion.events.locate(ctx, t)` (centre of the argmax `combined` cell at nearest frame).

| kind | rule (defaults) |
|---|---|
| `boom` | Frames where flash max >= `flash_min` [0.5] **and** an audio loud jump within +-`window` [0.3] s. strength = mean(flash_max, min(1, ratio/6)). x/y = argmax flash cell. No audio -> no booms. |
| `smoke` | Runs of frames with smoke max >= `min` [0.4] lasting >= `min_s` [0.5] s. t = run start, strength = run mean of max, x/y = argmax smoke cell at peak. |
| `scream` | z = (ratio - median) / (1.4826 MAD + 1e-3) of the 2-4 kHz band ratio over the window. A hop qualifies if z >= `k` [6] and rms >= `loud_ratio` [2.0] x median rms. Runs of >= `min_hops` [5] consecutive hops (0.2 s) give one event at the peak-z hop, if face channel max >= `face_min` [0.2] at that time. strength = min(1, mean(clip((z - k)/k, 0, 1), clip(rms / median / 4, 0, 1)) + 0.25). x/y = argmax face cell. `MIN_STRENGTH` 0.35. |
| `hype_word` | For each word with `ctx.scorer.score(text)` >= `min` [0.5]: t = word.start, strength = score, x/y = locate(ctx, t). No scorer/words -> []. |
| `hard_cut` | Each t in `ctx.cuts`: strength 1.0, x/y = locate(ctx, t). |
| `dead_air` | Spans where rms < max(`rms_ratio` [0.25] x median, `rms_floor` [0.01]) **and** motion max < `motion_max` [0.15], lasting >= `min_s` [1.5] s. t = span start, strength = min(1, dur/5), reason = f"silent+static {dur:.1f}s". |

Generic floor: `run_detectors` drops events with strength < `min_strength` (threshold of the detector's kind; default = its class attribute `MIN_STRENGTH`, 0.0 if absent). scream and smoke set `MIN_STRENGTH = 0.35`.

`clipbro/motion/events.py`: `run_detectors(ctx, registry=REGISTRY) -> list[Event]` (config.detectors order, sorted by t)
and `dedupe(events, window) -> list[Event]` (same kind within `window` s -> keep strongest).

## 12. Focus & keyframes

- `focus.raw_focus(sal: SaliencyMap, top_k=3) -> list[FocusPoint]`: per frame, weighted centroid of the top_k `combined` cells (weights = scores; all-zero frame -> previous point or (0.5, 0.5)).
- `focus.IdentitySmoother` registered as `("smoother", "identity")`.
- `focus.focus_path(sal, smoother: FocusSmoother | None = None, top_k=3)` -> smoothed (Phase 2 plugs a real smoother in here).
- `keyframes.save(path, payload: dict)` writes `{"version": 1, **payload}`; `keyframes.load(path)` checks version == 1, returns dict; Keyframe objects serialise via to_dict/from_dict.

## 13. Pipeline (`clipbro/motion/pipeline.py`)

```python
VERSION = 1

@dataclass
class MotionResult:
    payload: dict                         # JSON-safe, see below
    saliency: SaliencyMap | None = None   # only on fresh (non-cached) runs

class MotionAnalyzer:
    def __init__(self, frames: FrameSource, config: MotionConfig, *, audio: AudioSource | None = None,
                 cuts: CutDetector | None = None, faces: FaceDetector | None = None,
                 scorer: WordScorer | None = None, cache: Cache | None = None,
                 smoother: FocusSmoother | None = None, registry: Registry = REGISTRY): ...
    def analyze(self, media: Path, rng: TimeRange, words: list[Word] | None = None,
                use_cache: bool = True) -> MotionResult: ...
    def cache_key(self, media: Path, rng: TimeRange) -> str
        # sha256 of f"{abs path}|{st_mtime_ns}|{start:.3f}|{end:.3f}|{VERSION}|{config.fingerprint()}"
```
Payload:
```json
{"version": 1, "source_size": [w, h], "range": [start, end], "grid": [rows, cols],
 "events": [Event.to_dict()...], "focus": [[t, cx, cy], ...], "hud": [[0/1...]...],
 "timing": {"sample_s": .., "saliency_s": .., "events_s": .., "total_s": ..}}
```
Cache: payload via `cache.put(key, payload)`; `combined` downsampled to 2 fps as float16 via `put_array(key, "combined", ...)`.
A cache hit returns before any adapter call.

## 14. Bootstrap, API, CLI

- `bootstrap.build_analyzer(config: MotionConfig | None = None, data_dir: Path = Path("data"), faces: bool = True) -> MotionAnalyzer` — wires Ffmpeg adapters, YuNet if model file exists (else None, logs a warning), LexiconWordScorer, JsonFileCache(data_dir / "motion"); calls `discover()` for channels, detectors, smoothers.
- `api.motion_routes.make_router(analyzer: MotionAnalyzer, resolve_media: Callable[[str], Path | None], duration_of: Callable[[Path], float]) -> APIRouter` — `GET /motion/{source_id}?start=&end=`; 400 if id not alphanumeric, media missing, non-finite, start < 0, end <= start, end - start > 600, end > duration + 0.1.
- `cli.debug`: `python -m clipbro.cli.debug MEDIA START END [--out sheet.jpg] [--profile NAME] [--cols 5]` — fresh analysis (use_cache=False), contact sheet of 1 frame/s: heatmap of `combined` blended 35 %, raw focus circle, HUD cells hatched, event labels (`boom 0.82`) on the frame nearest each event; prints an aligned event table to stdout.

## 15. Testing

`python -m unittest discover -s tests -v`. No network, no real videos.
- `tests/fixtures.py`: FFmpeg lavfi clip builders into a temp dir + in-memory `FakeFrameSource`, `FakeAudioSource`, `FakeFaceDetector`.
- Face/YuNet tests `skipUnless` model file exists.
- `tests/test_architecture.py`: import-direction rules from §1.2.
- `tests/test_plugins.py`: registering a fake channel/detector changes output without touching pipeline code.
