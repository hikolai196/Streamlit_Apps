"""Pure helpers for batch result filtering and retry selection."""

from __future__ import annotations

from typing import Any

STATUS_ALL = "All"
STATUS_SUCCESS = "Success"
STATUS_EMPTY = "Empty"
STATUS_FAILED = "Failed"
STATUS_CANCELLED = "Cancelled"

STATUS_FILTER_OPTIONS = [
    STATUS_ALL,
    STATUS_SUCCESS,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_CANCELLED,
]


def normalize_status(status: str | None) -> str:
    return (status or "").strip().lower()


def filter_batch_results(
    results: list[dict[str, Any]],
    status_filter: str = STATUS_ALL,
) -> list[dict[str, Any]]:
    """Filter batch results by status label (All / Success / Empty / Failed / Cancelled)."""
    if status_filter == STATUS_ALL:
        return list(results)
    wanted = status_filter.strip().lower()
    return [item for item in results if normalize_status(item.get("status")) == wanted]


def retryable_batch_items(
    results: list[dict[str, Any]],
    upload_items_by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build upload items that can be retried (failed / empty / cancelled with a saved path).
    """
    retryable: list[dict[str, Any]] = []
    for result in results:
        status = normalize_status(result.get("status"))
        if status not in {"failed", "empty", "cancelled"}:
            continue
        filename = result.get("filename")
        if not filename:
            continue
        source = upload_items_by_name.get(filename)
        if not source or source.get("path") is None or source.get("error"):
            continue
        retryable.append(source)
    return retryable


def merge_batch_results(
    existing: list[dict[str, Any]],
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace existing results with updates matched by filename; keep others."""
    by_name = {item["filename"]: item for item in existing if item.get("filename")}
    for item in updates:
        name = item.get("filename")
        if name:
            by_name[name] = item
    return sorted(by_name.values(), key=lambda row: row.get("filename") or "")


def cancelled_result(filename: str, message: str = "Cancelled by user") -> dict[str, Any]:
    """Build a batch result row for a cancelled file."""
    return {
        "filename": filename,
        "status": "cancelled",
        "segments": [],
        "full_text": "",
        "segment_count": 0,
        "error": message,
        "export_paths": {},
    }
