"""Debug contact sheet for one motion analysis.

    python -m clipbro.cli.debug MEDIA START END [--out PATH] [--profile NAME] [--cols N]
                                [--data-dir DIR] [--no-faces]

Runs a fresh analysis of the range, renders one tile per second (frame, saliency heatmap,
raw focus point, HUD hatching, event labels) into a JPEG grid and prints an event table.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from clipbro.bootstrap import build_analyzer
from clipbro.core import ClipbroError, Event, FrameSource, InvalidRange, MediaError, SaliencyMap, TimeRange

DEFAULT_COLS = 5
TILE_WIDTH = 480
TILE_FPS = 1.0
HEAT_ALPHA = 0.35
EVENT_HALF_WINDOW = 0.5
JPEG_QUALITY = 90
HEADER_HEIGHT = 56
GAP = 4
HATCH_SPACING = 10
LABEL_PAD = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
GRID_COLOUR = (110, 110, 110)
HATCH_COLOUR = (200, 200, 200)
SHEET_BACKGROUND = 24
TABLE_COLUMNS = ("t", "kind", "strength", "x", "y", "reason")


class Analysis(Protocol):
    """The parts of a MotionResult the debug sheet reads."""

    payload: dict[str, Any]
    saliency: SaliencyMap | None


class DebugAnalyzer(Protocol):
    """The parts of a MotionAnalyzer the debug sheet uses: its frame source and analyze()."""

    frames: FrameSource

    def analyze(self, media: Path, rng: TimeRange, use_cache: bool = True) -> Analysis:
        """Analyse rng of media and return the payload with saliency."""


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line; argparse exits with status 2 on bad arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m clipbro.cli.debug",
        description="Render a contact sheet of the motion analysis of MEDIA between START and END seconds.",
    )
    parser.add_argument("media", type=Path, help="path to the source video")
    parser.add_argument("start", type=float, help="range start in seconds")
    parser.add_argument("end", type=float, help="range end in seconds")
    parser.add_argument(
        "--out", type=Path, default=None, help="JPEG output path (default: <stem>_<start>_<end>_sheet.jpg)"
    )
    parser.add_argument("--profile", default=None, help="motion config profile name")
    parser.add_argument("--cols", type=_positive_int, default=DEFAULT_COLS, help="tiles per row (default 5)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="cache directory (default ./data)")
    parser.add_argument("--no-faces", action="store_true", help="skip the face detector")
    return parser.parse_args(argv)


def _positive_int(text: str) -> int:
    """argparse type: an integer of at least 1."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _cell_bounds(length: int, parts: int) -> list[int]:
    """parts + 1 integer pixel edges splitting 0..length into near-equal spans (same rule as the grid)."""
    return np.linspace(0, length, parts + 1).round().astype(int).tolist()


def _to_pixel(tile: np.ndarray, x: float, y: float) -> tuple[int, int]:
    """Normalised (x, y) to integer pixel coordinates of a tile."""
    height, width = tile.shape[:2]
    return int(round(x * (width - 1))), int(round(y * (height - 1)))


def draw_heatmap(tile: np.ndarray, saliency: SaliencyMap, index: int) -> np.ndarray:
    """Blend a JET heatmap of the combined saliency at frame index over the tile (HEAT_ALPHA)."""
    height, width = tile.shape[:2]
    grid = np.clip(saliency.combined[index] * 255.0, 0.0, 255.0).astype(np.float32)
    upsampled = cv2.resize(grid, (width, height), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(np.clip(upsampled, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(tile, 1.0 - HEAT_ALPHA, heat, HEAT_ALPHA, 0.0)


def draw_grid(tile: np.ndarray, grid: tuple[int, int]) -> None:
    """Draw faint lines along the cell edges of the saliency grid."""
    rows, cols = grid
    height, width = tile.shape[:2]
    for x in _cell_bounds(width, cols)[1:-1]:
        cv2.line(tile, (x, 0), (x, height - 1), GRID_COLOUR, 1, cv2.LINE_AA)
    for y in _cell_bounds(height, rows)[1:-1]:
        cv2.line(tile, (0, y), (width - 1, y), GRID_COLOUR, 1, cv2.LINE_AA)


def draw_hud_hatch(tile: np.ndarray, hud: np.ndarray) -> None:
    """Hatch every HUD cell with diagonal lines, drawn cell by cell so they stay inside the cell."""
    rows, cols = hud.shape
    y_edges = _cell_bounds(tile.shape[0], rows)
    x_edges = _cell_bounds(tile.shape[1], cols)
    for row, col in zip(*np.nonzero(hud)):
        y0, y1 = y_edges[int(row)], y_edges[int(row) + 1]
        x0, x1 = x_edges[int(col)], x_edges[int(col) + 1]
        cell = tile[y0:y1, x0:x1].copy()
        cell_h, cell_w = cell.shape[:2]
        for offset in range(-cell_h, cell_w, HATCH_SPACING):
            cv2.line(cell, (offset, cell_h - 1), (offset + cell_h - 1, 0), HATCH_COLOUR, 1, cv2.LINE_AA)
        tile[y0:y1, x0:x1] = cell


def draw_focus(tile: np.ndarray, x: float, y: float) -> None:
    """A white dot with a black outline at the focus point."""
    centre = _to_pixel(tile, x, y)
    cv2.circle(tile, centre, 9, BLACK, 2, cv2.LINE_AA)
    cv2.circle(tile, centre, 7, WHITE, -1, cv2.LINE_AA)


def _text_metrics(text: str) -> tuple[int, int, int]:
    """(width, height, baseline) of text at the label font."""
    (width, height), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    return width, height, baseline


def label_height(text: str) -> int:
    """Pixel height of the label box that put_label draws for text."""
    _, height, baseline = _text_metrics(text)
    return height + baseline + 2 * LABEL_PAD


def put_label(tile: np.ndarray, text: str, x: int, y: int) -> None:
    """Draw anti-aliased text on a dark box whose top-left corner is (x, y)."""
    width, height, _ = _text_metrics(text)
    cv2.rectangle(tile, (x, y), (x + width + 2 * LABEL_PAD, y + label_height(text)), BLACK, -1)
    cv2.putText(
        tile, text, (x + LABEL_PAD, y + LABEL_PAD + height),
        FONT, FONT_SCALE, WHITE, FONT_THICKNESS, cv2.LINE_AA,
    )


def draw_events(tile: np.ndarray, events: list[Event]) -> None:
    """Mark each event position with a dot, then stack its label box bottom-left with the strength."""
    for event in events:
        centre = _to_pixel(tile, event.x, event.y)
        cv2.circle(tile, centre, 5, BLACK, 2, cv2.LINE_AA)
        cv2.circle(tile, centre, 3, YELLOW, -1, cv2.LINE_AA)
    labels = [f"{event.kind} {event.strength:.2f}" for event in events]
    y = tile.shape[0] - GAP - sum(label_height(text) + GAP for text in labels)
    for text in labels:
        put_label(tile, text, GAP, y)
        y += label_height(text) + GAP


def nearest_focus(focus: list[list[float]], t: float) -> tuple[float, float] | None:
    """(cx, cy) of the payload focus entry nearest t, or None when the path is empty."""
    if not focus:
        return None
    _, cx, cy = min(focus, key=lambda entry: abs(entry[0] - t))
    return cx, cy


def events_at(events: list[Event], t: float) -> list[Event]:
    """Events whose time falls in [t - 0.5, t + 0.5) (one tile per second)."""
    return [event for event in events if t - EVENT_HALF_WINDOW <= event.t < t + EVENT_HALF_WINDOW]


def render_tile(
    frame_rgb: np.ndarray,
    t: float,
    saliency: SaliencyMap,
    focus: list[list[float]],
    events: list[Event],
) -> np.ndarray:
    """Render one BGR tile for absolute time t: heatmap, grid, HUD hatching, focus, events, timestamp."""
    tile = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    tile = draw_heatmap(tile, saliency, saliency.nearest(t))
    draw_grid(tile, saliency.grid)
    draw_hud_hatch(tile, saliency.hud)
    spot = nearest_focus(focus, t)
    if spot is not None:
        draw_focus(tile, *spot)
    draw_events(tile, events_at(events, t))
    put_label(tile, f"{t:.1f}s", GAP, GAP)
    return tile


def format_table(events: list[Event]) -> str:
    """Aligned table of events sorted by time: t | kind | strength | x | y | reason."""
    rows = [
        [f"{event.t:.2f}", event.kind, f"{event.strength:.2f}", f"{event.x:.3f}", f"{event.y:.3f}", event.reason]
        for event in sorted(events, key=lambda event: event.t)
    ]
    widths = [max([len(TABLE_COLUMNS[i])] + [len(row[i]) for row in rows]) for i in range(len(TABLE_COLUMNS) - 1)]
    lines = [_table_line(list(TABLE_COLUMNS), widths)]
    lines.append("-+-".join("-" * width for width in widths) + "-+-" + "-" * len(TABLE_COLUMNS[-1]))
    lines.extend(_table_line(row, widths) for row in rows)
    if not rows:
        lines.append("(no events)")
    return "\n".join(lines)


def _table_line(cells: list[str], widths: list[int]) -> str:
    """Join one table row, padding every column except the last (free-text) one."""
    padded = [cell.ljust(width) for cell, width in zip(cells[:-1], widths)]
    return " | ".join(padded + [cells[-1]])


def format_timing(timing: dict[str, float]) -> str:
    """One-line timing summary of an analysis, from the payload timing block."""
    return (
        f"timing: sample {timing.get('sample_s', 0.0):.2f} s, saliency {timing.get('saliency_s', 0.0):.2f} s, "
        f"events {timing.get('events_s', 0.0):.2f} s, total {timing.get('total_s', 0.0):.2f} s"
    )


def format_header(media: Path, rng: TimeRange, profile: str | None, events: list[Event]) -> str:
    """Single header line for stdout: media, range, profile and the event count."""
    return (
        f"clipbro debug: {media.name} {rng.start:g}-{rng.end:g} s, profile {profile or 'default'}, "
        f"{len(events)} events"
    )


def _header_lines(
    media: Path, rng: TimeRange, profile: str | None, timing: dict[str, float], events: list[Event]
) -> list[str]:
    """Two text lines for the sheet header strip: identity and timing, then event counts per kind."""
    counts = Counter(event.kind for event in events)
    kinds = ", ".join(f"{kind} {count}" for kind, count in sorted(counts.items())) or "none"
    return [
        f"{media.name}  range {rng.start:g}-{rng.end:g} s  profile {profile or 'default'}",
        f"{format_timing(timing)}  |  events: {kinds}",
    ]


def compose_sheet(tiles: list[np.ndarray], header: list[str], cols: int) -> np.ndarray:
    """Lay the tiles out in a grid of cols columns under a header strip; returns BGR uint8."""
    tile_h, tile_w = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    width = cols * tile_w + (cols + 1) * GAP
    height = HEADER_HEIGHT + rows * tile_h + (rows + 1) * GAP
    sheet = np.full((height, width, 3), SHEET_BACKGROUND, dtype=np.uint8)
    for number, tile in enumerate(tiles):
        row, col = divmod(number, cols)
        y = HEADER_HEIGHT + GAP + row * (tile_h + GAP)
        x = GAP + col * (tile_w + GAP)
        sheet[y : y + tile_h, x : x + tile_w] = tile
    for line_number, text in enumerate(header):
        cv2.putText(sheet, text, (GAP + 4, 20 + line_number * 22), FONT, FONT_SCALE, WHITE, FONT_THICKNESS, cv2.LINE_AA)
    return sheet


def write_sheet(path: Path, sheet: np.ndarray) -> None:
    """Save the sheet as JPEG (quality 90), creating parent directories first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        raise OSError(f"could not write {path}")


def default_out_path(media: Path, rng: TimeRange) -> Path:
    """Default sheet location: <media stem>_<start>_<end>_sheet.jpg in the working directory."""
    return Path(f"{media.stem}_{rng.start:g}_{rng.end:g}_sheet.jpg")


def build_sheet(
    analyzer: DebugAnalyzer, args: argparse.Namespace, rng: TimeRange
) -> tuple[np.ndarray, Analysis, list[Event]]:
    """Run a fresh analysis, sample one frame per second and render the tiles into a sheet."""
    analysis = analyzer.analyze(args.media, rng, use_cache=False)
    if analysis.saliency is None:
        raise ClipbroError("analysis returned no saliency map")
    batch = analyzer.frames.read(args.media, rng, TILE_FPS, TILE_WIDTH)
    events = [Event.from_dict(item) for item in analysis.payload["events"]]
    focus = analysis.payload["focus"]
    tiles = [
        render_tile(batch.frames[index], float(batch.times[index]), analysis.saliency, focus, events)
        for index in range(batch.n)
    ]
    header = _header_lines(args.media, rng, args.profile, analysis.payload["timing"], events)
    return compose_sheet(tiles, header, args.cols), analysis, events


def run(args: argparse.Namespace, rng: TimeRange) -> int:
    """Build the analyzer, render the sheet, write it and print the event table and timing."""
    analyzer = build_analyzer(profile=args.profile, data_dir=args.data_dir, faces=not args.no_faces)
    sheet, analysis, events = build_sheet(analyzer, args, rng)
    out = args.out if args.out is not None else default_out_path(args.media, rng)
    write_sheet(out, sheet)
    print(format_header(args.media, rng, args.profile, events))
    print(format_table(events))
    print(format_timing(analysis.payload["timing"]))
    print(f"sheet: {out}")
    return 0


def _exit_code(request: SystemExit) -> int:
    """Map a SystemExit from argparse to an integer status (2 for usage errors)."""
    return request.code if isinstance(request.code, int) else 2


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: returns 0 on success, 2 on bad arguments or missing or unreadable media, 1 on write errors."""
    try:
        args = parse_args(argv)
    except SystemExit as request:
        return _exit_code(request)
    if not args.media.is_file():
        print(f"error: media not found: {args.media}", file=sys.stderr)
        return 2
    try:
        rng = TimeRange(args.start, args.end)
        return run(args, rng)
    except InvalidRange as error:
        print(f"error: invalid range: {error}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except MediaError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
