"""Extractive meeting summary and simple action-item heuristics."""

from __future__ import annotations

import re
from typing import Any

_ACTION_PATTERNS = [
    re.compile(r"\b(will|shall|should|need to|have to|must|todo|action item|follow up|請|會|應該|需要)\b", re.I),
    re.compile(r"\b(assign|deadline|by friday|next week|負責|截止)\b", re.I),
]


def _segment_sentences(segments: list[dict[str, Any]]) -> list[str]:
    sentences: list[str] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        prefix = f"[{speaker}] " if speaker else ""
        # Prefer keeping ASR segments as summary units (already sentence-ish).
        sentences.append(prefix + text)
    return sentences


def extractive_summary(
    segments: list[dict[str, Any]],
    *,
    max_bullets: int = 5,
) -> str:
    """
    Build a short extractive summary from the longest informative segments.

    No external LLM required — useful as a local baseline.
    """
    candidates = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if len(text) < 12:
            continue
        score = len(text) + (8 if seg.get("speaker") else 0)
        candidates.append((score, seg))
    candidates.sort(key=lambda item: item[0], reverse=True)

    picked = []
    seen = set()
    for _, seg in candidates:
        text = str(seg.get("text", "")).strip()
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        speaker = seg.get("speaker")
        line = f"- [{speaker}] {text}" if speaker else f"- {text}"
        picked.append((float(seg.get("start", 0.0)), line))
        if len(picked) >= max_bullets:
            break

    picked.sort(key=lambda item: item[0])
    if not picked:
        return "No summary available (transcript too short)."
    return "Summary\n" + "\n".join(line for _, line in picked)


def extract_action_items(segments: list[dict[str, Any]], *, max_items: int = 8) -> str:
    """Return bullet lines that look like commitments / follow-ups."""
    items: list[str] = []
    for sentence in _segment_sentences(segments):
        if any(pattern.search(sentence) for pattern in _ACTION_PATTERNS):
            items.append(f"- {sentence}")
        if len(items) >= max_items:
            break
    if not items:
        return "No clear action items detected."
    return "Action items\n" + "\n".join(items)


def build_meeting_notes(segments: list[dict[str, Any]]) -> dict[str, str]:
    """Return summary + action-item blocks for UI / export."""
    return {
        "summary": extractive_summary(segments),
        "action_items": extract_action_items(segments),
    }
