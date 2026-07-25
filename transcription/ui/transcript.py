"""Pure helpers for editable transcripts (no Streamlit dependency)."""

from __future__ import annotations

from typing import Any


def non_empty_lines(text: str) -> list[str]:
    """Split transcript text into non-empty trimmed lines."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def redistribute_segments(
    reference_segments: list[dict[str, Any]],
    lines: list[str],
) -> list[dict[str, Any]]:
    """
    Build timed segments for edited lines.

    When line count matches the reference, keep original timestamps.
    Otherwise redistribute evenly across the original time span.
    """
    if not lines:
        return []

    if reference_segments and len(lines) == len(reference_segments):
        return [
            {
                "start": float(ref.get("start", 0.0)),
                "end": float(ref.get("end", 0.0)),
                "text": line,
            }
            for ref, line in zip(reference_segments, lines)
        ]

    if reference_segments:
        t0 = float(reference_segments[0].get("start", 0.0))
        t1 = float(reference_segments[-1].get("end", t0 + 1.0))
    else:
        t0 = 0.0
        t1 = float(max(len(lines), 1))

    span = max(t1 - t0, 0.1 * len(lines))
    step = span / len(lines)
    return [
        {
            "start": round(t0 + index * step, 2),
            "end": round(t0 + (index + 1) * step, 2),
            "text": line,
        }
        for index, line in enumerate(lines)
    ]


def apply_edited_transcript(
    segments: list[dict[str, Any]],
    edited_text: str,
) -> tuple[list[dict[str, Any]], str]:
    """
    Apply user edits to a transcript.

    Returns:
        (updated_segments, normalized_full_text)
    """
    lines = non_empty_lines(edited_text)
    updated = redistribute_segments(segments, lines)
    full_text = "\n".join(line["text"] for line in updated)
    return updated, full_text
