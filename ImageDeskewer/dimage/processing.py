"""Image loading, validation, corner geometry, and perspective deskewing."""

from __future__ import annotations

import io
import math
from collections.abc import Mapping, Sequence
from typing import Any, BinaryIO, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageEnhance, ImageOps

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_IMAGE_DIM = 10000

Point = Sequence[float]
Points = Sequence[Point]
ImageArray = NDArray[np.uint8]
CanvasDrawing = dict[str, Any]


def validate_upload(file: BinaryIO | Any, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
    """
    Ensure an uploaded file is within the allowed size limit.

    Raises:
        ValueError: If the file exceeds ``max_bytes``.
    """
    size = getattr(file, "size", None)
    if size is None:
        pos = file.tell()
        file.seek(0, io.SEEK_END)
        size = file.tell()
        file.seek(pos)
    if size > max_bytes:
        mb = size / (1024 * 1024)
        limit_mb = max_bytes / (1024 * 1024)
        raise ValueError(
            f"File is too large ({mb:.1f} MB). Maximum allowed is {limit_mb:.0f} MB."
        )


def validate_image_dimensions(img_np: ImageArray, max_dim: int = MAX_IMAGE_DIM) -> None:
    """
    Ensure image width and height are within limits.

    Raises:
        ValueError: If either dimension exceeds ``max_dim``.
    """
    h, w = img_np.shape[:2]
    if h > max_dim or w > max_dim:
        raise ValueError(
            f"Image dimensions {w}x{h} exceed the maximum of {max_dim}px on either side."
        )


def load_image(
    file: BinaryIO | Any,
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_dim: int = MAX_IMAGE_DIM,
) -> tuple[Image.Image, ImageArray]:
    """
    Load an image file, apply EXIF orientation, and convert to RGB + NumPy.

    Raises:
        ValueError: If size/dimension limits are exceeded or the image is invalid.
    """
    validate_upload(file, max_bytes=max_bytes)
    try:
        opened = Image.open(file)
        transposed = ImageOps.exif_transpose(opened)
        image = (transposed or opened).convert("RGB")
    except Exception as exc:
        raise ValueError(
            "Could not open image. Please upload a valid PNG or JPG."
        ) from exc
    img_np = cast(ImageArray, np.asarray(image, dtype=np.uint8))
    validate_image_dimensions(img_np, max_dim=max_dim)
    return image, img_np


def resize_image(img_np: ImageArray, max_dim: int = 600) -> tuple[ImageArray, float]:
    """
    Resize an image to fit within max_dim, preserving aspect ratio (never upscales).

    Returns:
        tuple: (resized_img, scale) where scale maps original -> display size.
    """
    h, w = img_np.shape[:2]
    if max(h, w) > max_dim:
        scale = min(max_dim / w, max_dim / h)
    else:
        scale = 1.0
    display_w, display_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized_img = cast(
        ImageArray,
        cv2.resize(img_np, (display_w, display_h), interpolation=cv2.INTER_AREA),
    )
    return resized_img, scale


def order_corners(points: Points) -> list[list[float]]:
    """
    Order four points as top-left, top-right, bottom-right, bottom-left.

    Uses the classic sum/diff heuristic so click order does not matter.
    """
    if len(points) != 4:
        raise ValueError("Exactly 4 points are required to order corners.")
    pts = np.asarray(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = pts[:, 0] - pts[:, 1]
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]  # top-left
    ordered[2] = pts[np.argmax(sums)]  # bottom-right
    ordered[1] = pts[np.argmax(diffs)]  # top-right
    ordered[3] = pts[np.argmin(diffs)]  # bottom-left
    # Guard against a point being selected twice (degenerate / near-square edge cases)
    if len({tuple(p) for p in ordered}) != 4:
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        pts_sorted = pts[np.argsort(angles)]
        start = int(np.argmin(pts_sorted.sum(axis=1)))
        pts_sorted = np.roll(pts_sorted, -start, axis=0)
        ordered = pts_sorted.astype(np.float32)
    return cast(list[list[float]], ordered.tolist())


def are_points_valid(points: Points, min_distance: float = 5) -> tuple[bool, str]:
    """Validate that four points are suitable for a perspective transform."""
    if len(points) != 4:
        return False, "Exactly 4 points are required."
    pts = np.asarray(points, dtype=np.float32)
    if len({tuple(map(int, pt)) for pt in pts}) != 4:
        return False, "Points must be unique."
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(pts[i] - pts[j]) < min_distance:
                return False, "Points are too close together."

    def area_quad(quad: NDArray[np.float32]) -> float:
        return float(
            0.5
            * abs(
                quad[0][0] * quad[1][1]
                + quad[1][0] * quad[2][1]
                + quad[2][0] * quad[3][1]
                + quad[3][0] * quad[0][1]
                - (
                    quad[1][0] * quad[0][1]
                    + quad[2][0] * quad[1][1]
                    + quad[3][0] * quad[2][1]
                    + quad[0][0] * quad[3][1]
                )
            )
        )

    if area_quad(pts) < 1.0:
        return (
            False,
            "Selected points are nearly collinear or form a degenerate quadrilateral.",
        )
    return True, ""


def deskew_image(img_np: ImageArray, points: Points) -> Image.Image:
    """
    Apply a perspective transform to deskew an image based on four corner points.

    Points are auto-ordered; click order does not matter.
    """
    ordered = order_corners(points)
    valid, msg = are_points_valid(ordered)
    if not valid:
        raise ValueError(f"Invalid points for deskewing: {msg}")
    pts_src = np.asarray(ordered, dtype=np.float32)
    w_a = float(np.linalg.norm(pts_src[0] - pts_src[1]))
    w_b = float(np.linalg.norm(pts_src[2] - pts_src[3]))
    max_w = max(1, math.ceil(max(w_a, w_b)))
    h_a = float(np.linalg.norm(pts_src[0] - pts_src[3]))
    h_b = float(np.linalg.norm(pts_src[1] - pts_src[2]))
    max_h = max(1, math.ceil(max(h_a, h_b)))
    pts_dst = np.asarray(
        [
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(img_np, matrix, (max_w, max_h))
    return Image.fromarray(warped)


def label_corners(points: Points, scale: float) -> list[list[int]]:
    """Convert display-coordinate points to original image coordinates."""
    if scale <= 0:
        raise ValueError("Scale must be positive.")
    return [[int(x / scale), int(y / scale)] for x, y in points]


def annotate_corners(img_np: ImageArray, points: Points) -> ImageArray:
    """
    Draw numbered corner markers and a connecting quadrilateral on a copy of the image.

    When four points are provided they are auto-ordered before drawing.
    """
    annotated = cast(ImageArray, img_np.copy())
    if not points:
        return annotated

    pts: list[list[float]] = [list(map(float, p)) for p in points]
    draw_pts: list[list[float]] = order_corners(pts) if len(pts) == 4 else pts

    for i, (x, y) in enumerate(draw_pts):
        center = (int(round(x)), int(round(y)))
        cv2.circle(annotated, center, 8, (255, 0, 0), 2)
        cv2.putText(
            annotated,
            str(i + 1),
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )

    if len(draw_pts) >= 2:
        poly = np.asarray(
            [[int(round(x)), int(round(y))] for x, y in draw_pts],
            dtype=np.int32,
        )
        closed = len(draw_pts) == 4
        cv2.polylines(
            annotated, [poly], isClosed=closed, color=(0, 255, 255), thickness=2
        )

    return annotated


def detect_document_corners(img_np: ImageArray | None) -> list[list[float]] | None:
    """
    Attempt to find a document-like quadrilateral in the image.

    Returns:
        Four [x, y] points (unordered) or None if detection fails.
    """
    if img_np is None or img_np.size == 0:
        return None

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), dtype=np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = int(img_np.shape[0] * img_np.shape[1])

    for contour in contours[:15]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > 0.05 * img_area:
            points = cast(list[list[float]], approx.reshape(4, 2).astype(float).tolist())
            valid, _ = are_points_valid(order_corners(points), min_distance=2)
            if valid:
                return points
    return None


def canvas_circles_from_points(points: Points, radius: float = 7) -> CanvasDrawing:
    """Build a Fabric.js-style initial_drawing dict of circle markers for st_canvas."""
    objects: list[dict[str, Any]] = []
    for x, y in points:
        objects.append(
            {
                "type": "circle",
                "left": float(x) - radius,
                "top": float(y) - radius,
                "radius": radius,
                "fill": "rgba(255, 0, 0, 0.3)",
                "stroke": "#000",
                "strokeWidth": 1,
                "originX": "left",
                "originY": "top",
            }
        )
    return {"version": "4.4.0", "objects": objects}


def extract_canvas_points(canvas_json: Mapping[str, Any] | None) -> list[list[float]]:
    """Extract circle center points from st_canvas JSON data."""
    points: list[list[float]] = []
    if not canvas_json or "objects" not in canvas_json:
        return points
    for obj in canvas_json["objects"]:
        if obj.get("type") == "circle":
            x = float(obj["left"]) + float(obj["radius"])
            y = float(obj["top"]) + float(obj["radius"])
            points.append([x, y])
    return points


def image_to_bytes(img: Image.Image, fmt: str, quality: int = 92) -> bytes:
    """Convert a PIL Image to bytes in the specified format."""
    buf = io.BytesIO()
    ext = "PNG" if fmt.upper() == "PNG" else "JPEG"
    save_kwargs: dict[str, Any] = {}
    to_save = img
    if ext == "JPEG":
        save_kwargs["quality"] = int(quality)
        if to_save.mode != "RGB":
            to_save = to_save.convert("RGB")
    to_save.save(buf, format=ext, **save_kwargs)
    return buf.getvalue()


def rotate_image(img: Image.Image, degrees: int) -> Image.Image:
    """
    Rotate an image clockwise by 0/90/180/270 degrees.

    Args:
        img: Source PIL image.
        degrees: Clockwise rotation; values outside {0,90,180,270} are normalized.
    """
    normalized = int(degrees) % 360
    if normalized not in {0, 90, 180, 270}:
        # Snap to nearest right angle for post-warp UI simplicity
        normalized = min({0, 90, 180, 270}, key=lambda d: abs(d - normalized))
    if normalized == 0:
        return img
    # PIL rotate is counter-clockwise; negate for clockwise UI semantics
    return img.rotate(-normalized, expand=True, fillcolor=(255, 255, 255))


def enhance_image(
    img: Image.Image,
    contrast: float = 1.0,
    sharpness: float = 1.0,
) -> Image.Image:
    """
    Adjust contrast and sharpness for scanned-document readability.

    Factors of 1.0 leave the image unchanged. Typical useful range: 0.5–2.0.
    """
    if contrast <= 0 or sharpness <= 0:
        raise ValueError("Contrast and sharpness must be positive.")
    out = img
    if contrast != 1.0:
        out = ImageEnhance.Contrast(out).enhance(contrast)
    if sharpness != 1.0:
        out = ImageEnhance.Sharpness(out).enhance(sharpness)
    return out


def crop_margins(
    img: Image.Image,
    left_pct: float = 0.0,
    top_pct: float = 0.0,
    right_pct: float = 0.0,
    bottom_pct: float = 0.0,
) -> Image.Image:
    """
    Crop by percentage margins from each edge (0–49).

    Raises:
        ValueError: If margins are invalid or leave an empty region.
    """
    for name, value in (
        ("left", left_pct),
        ("top", top_pct),
        ("right", right_pct),
        ("bottom", bottom_pct),
    ):
        if value < 0 or value >= 50:
            raise ValueError(f"{name} margin must be in [0, 50).")
    if left_pct + right_pct >= 100 or top_pct + bottom_pct >= 100:
        raise ValueError("Crop margins leave an empty image.")

    width, height = img.size
    left = int(width * left_pct / 100.0)
    top = int(height * top_pct / 100.0)
    right = int(width * (100.0 - right_pct) / 100.0)
    bottom = int(height * (100.0 - bottom_pct) / 100.0)
    if right <= left or bottom <= top:
        raise ValueError("Crop margins leave an empty image.")
    return img.crop((left, top, right, bottom))


def apply_post_process(
    img: Image.Image,
    *,
    rotate_degrees: int = 0,
    contrast: float = 1.0,
    sharpness: float = 1.0,
    left_pct: float = 0.0,
    top_pct: float = 0.0,
    right_pct: float = 0.0,
    bottom_pct: float = 0.0,
) -> Image.Image:
    """Apply rotate → crop → enhance in a stable order for the deskew UI."""
    out = rotate_image(img, rotate_degrees)
    out = crop_margins(out, left_pct, top_pct, right_pct, bottom_pct)
    return enhance_image(out, contrast=contrast, sharpness=sharpness)
