"""Layer 2, vision: read sticky notes from a task-board photo.

Pipeline (real, runs offline):
1. Find coloured note regions by saturation thresholding and connected components.
2. OCR each note crop with Tesseract and keep the mean word confidence.
3. OCR the top band to locate the column headers; assign each note to the nearest
   header centre (Calculation 5.1 in the report). Notes in "Done" are excluded later.

If no coloured regions are found (for example a plain photo), the whole image is
OCR'd and Tesseract's paragraph blocks are used as notes instead.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..schemas import Extracted

HEADERS = {"to do": "To Do", "in progress": "In Progress", "done": "Done"}


def _tess():
    import pytesseract  # imported lazily so the package works without OCR installed
    return pytesseract


def _header_centres(img: Image.Image) -> dict[str, float]:
    pt = _tess()
    w, h = img.size
    band = img.crop((0, 0, w, int(h * 0.12)))
    data = pt.image_to_data(ImageOps.grayscale(band), output_type=pt.Output.DICT, config="--psm 11")
    words = [(data["text"][i].strip().lower(), data["left"][i] + data["width"][i] / 2)
             for i in range(len(data["text"])) if data["text"][i].strip()]
    centres: dict[str, float] = {}
    for i, (t, x) in enumerate(words):
        nxt = words[i + 1][0] if i + 1 < len(words) else ""
        if t == "to" and nxt.startswith("do"):
            centres["To Do"] = (x + words[i + 1][1]) / 2
        elif t == "in" and nxt.startswith("progress"):
            centres["In Progress"] = (x + words[i + 1][1]) / 2
        elif t.startswith("done"):
            centres["Done"] = x
    if len(centres) < 3:  # fall back to equal-width columns for anything not read
        for k, name in enumerate(["To Do", "In Progress", "Done"]):
            centres.setdefault(name, w * (k + 0.5) / 3)
    return centres


def _note_boxes(img: Image.Image) -> list[tuple[int, int, int, int]]:
    from scipy import ndimage
    hsv = np.asarray(img.convert("HSV")).astype(float)
    mask = hsv[..., 1] > 40  # board and ink are low-saturation; notes are pastel
    mask = ndimage.binary_closing(mask, iterations=3)
    lab, n = ndimage.label(mask)
    boxes = []
    min_area = img.size[0] * img.size[1] * 0.004
    for sl in ndimage.find_objects(lab):
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        if (x1 - x0) * (y1 - y0) >= min_area:
            boxes.append((x0, y0, x1, y1))
    return sorted(boxes, key=lambda b: (b[0] // 200, b[1]))


def _ocr_crop(crop: Image.Image) -> tuple[str, float]:
    pt = _tess()
    g = ImageOps.grayscale(crop)
    data = pt.image_to_data(g, output_type=pt.Output.DICT, config="--psm 6")
    words, confs = [], []
    for t, c in zip(data["text"], data["conf"]):
        t = t.strip()
        if t and float(c) >= 0:
            words.append(t)
            confs.append(float(c))
    text = re.sub(r"\s+", " ", " ".join(words)).strip()
    text = re.sub(r"[^A-Za-z0-9 .,'()/-]", "", text)
    return text, (float(np.mean(confs)) / 100.0 if confs else 0.0)


def read_board(path: str | Path) -> list[Extracted]:
    img = Image.open(path).convert("RGB")
    centres = _header_centres(img)
    boxes = _note_boxes(img)
    items: list[Extracted] = []
    if boxes:
        for k, (x0, y0, x1, y1) in enumerate(boxes, 1):
            text, conf = _ocr_crop(img.crop((x0, y0, x1, y1)))
            if not text:
                continue
            cx = (x0 + x1) / 2
            col = min(centres, key=lambda c: abs(centres[c] - cx))
            items.append(Extracted(kind="task", value={"title": text, "column": col, "box": [x0, y0, x1, y1], "x_centre": cx},
                                   src=f"ocr#{k}", conf=round(conf, 3), text=text))
    else:  # fallback: paragraph blocks of a plain photo
        pt = _tess()
        data = pt.image_to_data(ImageOps.grayscale(img), output_type=pt.Output.DICT)
        blocks: dict[tuple[int, int], list[int]] = {}
        for i, t in enumerate(data["text"]):
            if t.strip() and data["top"][i] > img.size[1] * 0.12:
                blocks.setdefault((data["block_num"][i], data["par_num"][i]), []).append(i)
        for k, idx in enumerate(blocks.values(), 1):
            text = " ".join(data["text"][i] for i in idx)
            conf = float(np.mean([float(data["conf"][i]) for i in idx])) / 100
            cx = float(np.mean([data["left"][i] + data["width"][i] / 2 for i in idx]))
            col = min(centres, key=lambda c: abs(centres[c] - cx))
            items.append(Extracted(kind="task", value={"title": text, "column": col, "x_centre": cx}, src=f"ocr#{k}", conf=round(conf, 3), text=text))
    return items
