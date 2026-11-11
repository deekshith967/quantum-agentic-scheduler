"""Render synthetic task-board images (demo input and benchmark cases).

The renderer draws a whiteboard with three columns and pastel sticky notes. It is
used to create the demo photo and the benchmark boards; it is not part of the
runtime perception path.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

COLUMNS = ["To Do", "In Progress", "Done"]
NOTE_COLOURS = [(255, 229, 138), (255, 200, 221), (189, 224, 254), (199, 240, 189), (255, 214, 165), (220, 205, 255)]


def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "Arial Bold.ttf", "Arial.ttf"]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_board(notes: list[tuple[str, str]], path: Path, width: int = 1500, height: int = 1000,
                 blur: float = 0.0, seed: int = 0) -> Path:
    """notes: list of (column, text). Returns the written path."""
    img = Image.new("RGB", (width, height), (244, 244, 241))
    d = ImageDraw.Draw(img)
    colw = width // 3
    hf = _font(44, bold=False)
    for i, c in enumerate(COLUMNS):
        tw = d.textlength(c, font=hf)
        d.text((i * colw + (colw - tw) / 2, 30), c, fill=(40, 40, 40), font=hf)
        if i:
            d.line([(i * colw, 20), (i * colw, height - 20)], fill=(150, 150, 145), width=4)
    nf = _font(34)
    per_col: dict[str, int] = {c: 0 for c in COLUMNS}
    for k, (col, text) in enumerate(notes):
        ci = COLUMNS.index(col)
        row = per_col[col]
        per_col[col] += 1
        x0 = ci * colw + 60 + (row % 2) * 40
        y0 = 130 + row * 210
        w, h = colw - 140, 170
        d.rectangle([x0, y0, x0 + w, y0 + h], fill=NOTE_COLOURS[(k + seed) % len(NOTE_COLOURS)], outline=(160, 160, 160), width=2)
        lines = textwrap.wrap(text, 16)
        ty = y0 + (h - 44 * len(lines)) / 2
        for ln in lines:
            lw = d.textlength(ln, font=nf)
            d.text((x0 + (w - lw) / 2, ty), ln, fill=(25, 25, 25), font=nf)
            ty += 44
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path
