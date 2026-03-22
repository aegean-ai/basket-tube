"""Text timeline service — normalise transcript segments for the action recognition dataset builder."""

import math
import re
from datetime import datetime, timezone

from api.src.core.artifacts import config_key as _config_key


def normalize_text(raw: str) -> str:
    """Normalise a raw transcript segment string.

    Steps:
    1. Lowercase
    2. Strip trailing punctuation (! ? . , : ;) while preserving apostrophes
    3. Apply basketball domain lexicon substitutions
    """
    text = raw.lower()

    # Strip trailing punctuation (not apostrophes)
    text = text.rstrip("!?.,;:")

    # Basketball domain lexicon substitutions
    # three-pointer synonyms: bare digit "3", "three pointer", "trey"
    text = re.sub(r"\bthree pointer\b", "three", text)
    text = re.sub(r"\btrey\b", "three", text)
    text = re.sub(r"\b3\b", "three", text)

    # dunk synonyms
    text = re.sub(r"\bslam\b", "dunk", text)
    text = re.sub(r"\bjam\b", "dunk", text)

    # layup synonyms
    text = re.sub(r"\blay-up\b", "layup", text)
    text = re.sub(r"\blay up\b", "layup", text)

    # and-one synonyms
    text = re.sub(r"\band-one\b", "and one", text)
    text = re.sub(r"\band 1\b", "and one", text)

    return text


def build_timeline(
    transcript: dict,
    *,
    source: str,
    lexicon_version: str,
    stt_model_dir: str = "whisper",
) -> dict:
    """Build a normalised text timeline from a Whisper-format transcript dict.

    Parameters
    ----------
    transcript:
        Whisper-format dict with a ``segments`` list.
    source:
        Either ``"stt"`` or ``"caption"``.
    lexicon_version:
        Version tag for the basketball domain lexicon applied.
    stt_model_dir:
        Model directory name used for STT (used in config_key params).

    Returns
    -------
    dict
        Contains ``_meta``, ``source``, ``lexicon_version``, and ``segments``.
    """
    params = {"stt_model_dir": stt_model_dir, "lexicon_version": lexicon_version}
    cfg_key = _config_key(params)

    segments_out = []
    for seg in transcript.get("segments", []):
        raw_text = seg.get("text", "")
        if not raw_text or not raw_text.strip():
            continue

        if source == "stt":
            avg_logprob = seg.get("avg_logprob")
            if avg_logprob is not None:
                asr_confidence = round(math.exp(avg_logprob), 4)
            else:
                asr_confidence = None
        else:
            asr_confidence = None

        segments_out.append(
            {
                "segment_id": seg.get("id"),
                "t_start": seg.get("start"),
                "t_end": seg.get("end"),
                "raw_text": raw_text,
                "normalized_text": normalize_text(raw_text),
                "source": source,
                "asr_confidence": asr_confidence,
            }
        )

    return {
        "_meta": {
            "stage": "text_timeline",
            "config_key": cfg_key,
            "upstream": {"stt_model_dir": stt_model_dir},
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        },
        "source": source,
        "lexicon_version": lexicon_version,
        "segments": segments_out,
    }
