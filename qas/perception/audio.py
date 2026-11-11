"""Layer 2, audio: speech-to-text.

Two paths:
* A transcript file (.txt) is read directly with confidence 1.0. This is the default
  in the demo because the cloud build environment cannot download Whisper weights.
* An audio file (.wav, .mp3, .m4a, ...) is transcribed with faster-whisper if it is
  installed (`pip install faster-whisper`). Segment confidence is exp(avg_logprob).
"""
from __future__ import annotations

import math
import os
from pathlib import Path


class STTUnavailable(RuntimeError):
    pass


def transcribe(path: str | Path) -> tuple[str, float, str]:
    """Returns (text, confidence, method)."""
    path = Path(path)
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8").strip(), 1.0, "transcript-file"
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise STTUnavailable("faster-whisper is not installed; upload a .txt transcript or run `pip install faster-whisper`") from exc
    model = WhisperModel(os.environ.get("QAS_WHISPER_MODEL", "small"), device="cpu", compute_type="int8")  # pragma: no cover
    segs, _info = model.transcribe(str(path), beam_size=5)  # pragma: no cover
    segs = list(segs)  # pragma: no cover
    text = " ".join(s.text.strip() for s in segs)  # pragma: no cover
    conf = sum(math.exp(s.avg_logprob) for s in segs) / max(len(segs), 1)  # pragma: no cover
    return text, round(conf, 3), "faster-whisper"  # pragma: no cover
