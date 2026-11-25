# clipbro

clipbro is a local, original 9:16 clipper for long streams: talk, IRL or gaming. It is
content-agnostic. Behaviour differences live in config profiles, not in code.

**Phase 1 (current) is the motion data layer only.** For a time range of a video it returns
per-frame saliency, timestamped events (booms, smoke, dead air, hype words, hard cuts, ...)
and a raw focus path. There is no rendering and no UI yet. The design contract is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Install

Requirements: Python 3.10+, `ffmpeg` and `ffprobe` on `PATH`.

```bash
pip install -e .[api]          # [api] adds FastAPI and httpx; omit it if you only need Python
python scripts/fetch_models.py # downloads the YuNet face model into models/ (checksum verified)
```

Without the YuNet model the face channel is empty and a warning is logged.

## Quick start (Python)

```python
from pathlib import Path

from clipbro.bootstrap import build_analyzer
from clipbro.core import TimeRange

analyzer = build_analyzer(profile="talk")  # or config=MotionConfig(...); cache in data/motion
result = analyzer.analyze(Path("stream.mp4"), TimeRange(120.0, 150.0))

for event in result.payload["events"]:
    print(event["t"], event["kind"], event["strength"], event["reason"])
print(result.payload["focus"][:3])  # rows of [t, cx, cy], normalised 0..1
```

The payload is JSON-safe. It has `version`, `source_size`, `range`, `grid`, `events`,
`focus`, `hud` and `timing`. A repeated call with the same media, range and config is served
from the cache.

## Debug contact sheet (CLI)

```bash
python -m clipbro.cli.debug MEDIA START END --out sheet.jpg [--profile talk|gaming]
```

The command runs a fresh analysis (no cache). It writes a contact sheet at 1 frame per
second, with a saliency heatmap, the focus point, hatched HUD cells and event labels such as
`boom 0.82`. It also prints an aligned event table to stdout.

## API

```bash
uvicorn clipbro.api.motion_routes:create_app --factory --port 8000
# media files are read from ./data/media/<source_id>.{mp4,mkv,mov,webm}
curl "http://localhost:8000/motion/stream01?start=120&end=150"
```

`GET /motion/{source_id}?start=&end=` returns the payload. It answers 400 for a bad id, a
missing file, a non-finite or negative start, `end <= start`, a range over 600 s, or an end
beyond the media length. It answers 422 when media decoding or analysis fails.

## Architecture

Ports and adapters. `clipbro.core` defines the types and the `Protocol` ports. Everything that
touches FFmpeg, model files, disk or HTTP is an adapter. `clipbro.bootstrap` is the only module
that wires concrete adapters.

```
clipbro/
  core/            types, ports (Protocols), registry, config, errors   (stdlib + numpy only)
  adapters/        ffmpeg_video, ffmpeg_audio, ffmpeg_cuts, yunet_faces, json_cache
  hype/            lexicon.py               (WordScorer)
  motion/          pure analysis; channels/ and detectors/ auto-register
    channels/      hud, motion, flash, smoke, face, ui_change, centre
    detectors/     boom, smoke, scream, hype_word, hard_cut, dead_air
    pipeline.py    MotionAnalyzer
  bootstrap.py     build_analyzer(): the composition root
  api/             motion_routes.py (FastAPI, optional)
  cli/             debug.py
```

Import rules (enforced by `tests/test_architecture.py`): `core` imports only `core`, `motion`
imports only `core` and `motion`, `adapters` imports only `core` and `adapters`, and only
`bootstrap`, `api` and `cli` import `bootstrap`.

### Add a channel

A channel is one file in `clipbro/motion/channels/`. It must also be listed in
`config.channels` to run. The example below adds a per-cell brightness channel:

```python
# clipbro/motion/channels/bright.py
import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.motion.grid import cell_mean, stats_gray

DEFAULTS = {"scale": 255.0}


@register("channel", "bright")
class BrightChannel:
    name = "bright"
    requires: tuple[str, ...] = ()

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        """Per-cell mean luma as float32 [N, rows, cols] in 0..1."""
        scale = ctx.threshold("bright", "scale", DEFAULTS["scale"])
        means = cell_mean(stats_gray(ctx).astype(np.float32), ctx.grid)
        return np.clip(means / scale, 0, 1).astype(np.float32)
```

Add `"bright"` to `channels` and, if it should score, to `weights`. Nothing else changes.

### Add a detector

A detector is one file in `clipbro/motion/detectors/`. It must also be listed in
`config.detectors`. It runs after saliency, so `ctx.saliency` is available:

```python
# clipbro/motion/detectors/bright_peak.py
import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.core.types import Event
from clipbro.motion.events import locate

DEFAULTS = {"min": 0.9}


@register("detector", "bright_peak")
class BrightPeakDetector:
    kind = "bright_peak"

    def detect(self, ctx: AnalysisContext) -> list[Event]:
        """One event per sampled frame whose brightest cell clears the threshold."""
        limit = ctx.threshold("bright_peak", "min", DEFAULTS["min"])
        peak = ctx.saliency.channel("bright").max(axis=(1, 2))
        events: list[Event] = []
        for index in np.flatnonzero(peak >= limit):
            t = float(ctx.frames.times[index])
            x, y = locate(ctx, t)
            events.append(Event(t=t, kind=self.kind, strength=float(peak[index]), x=x, y=y, reason="bright"))
        return events
```

Add `"bright_peak"` to `detectors`. Events of the same kind within `dedupe_window` seconds
are merged, keeping the strongest.

### Add an adapter

An adapter implements one port from `clipbro/core/ports.py`. It lives in `clipbro/adapters/`
and imports only `core`. Wire it in `bootstrap.build_analyzer`, the only place allowed to
construct it:

```python
# clipbro/adapters/fixed_faces.py
from clipbro.core.types import Box


class FixedFaceDetector:
    """FaceDetector that always returns the same boxes (useful for tests and dry runs)."""

    def __init__(self, boxes: list[Box]) -> None:
        self.boxes = boxes

    def detect(self, image) -> list[Box]:
        return list(self.boxes)
```

Then pass it as `faces=` into `MotionAnalyzer` in `bootstrap.py`. Ports are `@runtime_checkable`
Protocols, so no base class is needed.

## Config, profiles and tuning

`MotionConfig` holds the grid, sampling, channel and detector lists, the weights and the
per-plugin thresholds. `PROFILES` names presets: `default`, `talk` (favours faces and the
centre) and `gaming` (favours motion and flashes). Select one with `build_analyzer(profile=...)`
or `--profile` on the CLI.

To tune, write a TOML file with top-level keys and load it with `MotionConfig.from_toml`
(Python 3.11+). A table replaces the default table of the same name, so list every weight you
want to keep:

```toml
# clipbro.toml
[weights]
motion = 0.50
"flash|smoke" = 0.20
face = 0.15
ui_change = 0.10
centre = 0.05

[thresholds.boom]
flash_min = 0.6      # default 0.5: fewer, stronger booms

[thresholds.hype_word]
min = 0.7            # default 0.5
```

```python
from pathlib import Path

from clipbro.bootstrap import build_analyzer
from clipbro.core import MotionConfig

analyzer = build_analyzer(config=MotionConfig.from_toml(Path("clipbro.toml")))
```

Every threshold has its default in the module-level `DEFAULTS` of the plugin that reads it.
Changing the config changes the cache fingerprint, so old cache entries are not reused.

## Testing

```bash
python -m unittest discover -s tests -v
```

The tests need no network and no real videos. FFmpeg clips are rendered on the fly and the
pipeline tests skip when `ffmpeg` is missing. The API tests need the `[api]` extra and skip
without it. Face tests skip when the model file is absent.
