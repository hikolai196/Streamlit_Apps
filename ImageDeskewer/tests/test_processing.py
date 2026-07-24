import io

import cv2
import numpy as np
import pytest
from dimage.processing import (
    annotate_corners,
    apply_post_process,
    are_points_valid,
    canvas_circles_from_points,
    crop_margins,
    deskew_image,
    detect_document_corners,
    enhance_image,
    extract_canvas_points,
    image_to_bytes,
    label_corners,
    load_image,
    order_corners,
    resize_image,
    rotate_image,
    validate_image_dimensions,
    validate_upload,
)
from PIL import Image


def _make_png_bytes(size=(100, 80), color=(10, 20, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_order_corners_ignores_click_order():
    tl, tr, br, bl = [10, 10], [90, 12], [88, 80], [12, 78]
    shuffled = [br, tl, bl, tr]
    ordered = order_corners(shuffled)
    assert ordered[0] == pytest.approx(tl, abs=1e-3)
    assert ordered[1] == pytest.approx(tr, abs=1e-3)
    assert ordered[2] == pytest.approx(br, abs=1e-3)
    assert ordered[3] == pytest.approx(bl, abs=1e-3)


def test_order_corners_requires_four_points():
    with pytest.raises(ValueError, match="Exactly 4"):
        order_corners([[0, 0], [1, 1]])


def test_are_points_valid_rejects_degenerate():
    ok, _ = are_points_valid([[0, 0], [100, 0], [100, 80], [0, 80]])
    assert ok
    # Spread-out collinear points so the failure is area, not min-distance
    bad, msg = are_points_valid([[0, 0], [40, 0], [80, 0], [120, 0]])
    assert not bad
    assert "degenerate" in msg.lower() or "collinear" in msg.lower()


def test_are_points_valid_rejects_close_points():
    ok, msg = are_points_valid([[0, 0], [1, 0], [100, 80], [0, 80]], min_distance=5)
    assert not ok
    assert "close" in msg.lower()


def test_resize_image_scale_math():
    img = np.zeros((1200, 800, 3), dtype=np.uint8)
    resized, scale = resize_image(img, max_dim=600)
    assert scale == pytest.approx(0.5)
    assert resized.shape[0] == 600
    assert resized.shape[1] == 400

    small = np.zeros((100, 80, 3), dtype=np.uint8)
    resized_small, scale_small = resize_image(small, max_dim=600)
    assert scale_small == 1.0
    assert resized_small.shape[:2] == (100, 80)


def test_label_corners_maps_to_original():
    points = [[10.0, 20.0], [30.0, 40.0]]
    mapped = label_corners(points, scale=0.5)
    assert mapped == [[20, 40], [60, 80]]


def test_label_corners_rejects_nonpositive_scale():
    with pytest.raises(ValueError):
        label_corners([[1, 2]], scale=0)


def test_deskew_image_on_synthetic_quad():
    # White canvas with a solid colored trapezoid-ish rectangle region
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[:] = (255, 255, 255)
    # Fill a rectangle that we will "deskew" using its corners
    img[40:160, 50:250] = (200, 40, 40)
    points = [[50, 40], [250, 40], [250, 160], [50, 160]]
    # Pass in reverse order to ensure auto-order works
    result = deskew_image(img, list(reversed(points)))
    assert result.size[0] > 0 and result.size[1] > 0
    arr = np.array(result)
    # Center of output should be mostly the red fill
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    assert arr[cy, cx, 0] > 150


def test_annotate_corners_draws_without_error():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    points = [[10, 10], [90, 10], [90, 90], [10, 90]]
    annotated = annotate_corners(img, points)
    assert annotated.shape == img.shape
    # Some non-black pixels from drawing
    assert annotated.sum() > 0


def test_validate_upload_size_limit():
    big = io.BytesIO(b"x" * 100)
    big.seek(0)
    validate_upload(big, max_bytes=200)

    too_big = io.BytesIO(b"x" * 250)
    too_big.seek(0)
    with pytest.raises(ValueError, match="too large"):
        validate_upload(too_big, max_bytes=200)


def test_validate_image_dimensions():
    ok = np.zeros((100, 100, 3), dtype=np.uint8)
    validate_image_dimensions(ok, max_dim=500)
    bad = np.zeros((600, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="dimensions"):
        validate_image_dimensions(bad, max_dim=500)


def test_load_image_png_and_limits():
    buf = _make_png_bytes((64, 48))
    image, img_np = load_image(buf, max_bytes=1024 * 1024, max_dim=1000)
    assert image.mode == "RGB"
    assert img_np.shape == (48, 64, 3)

    huge = _make_png_bytes((120, 80))
    with pytest.raises(ValueError, match="dimensions"):
        load_image(huge, max_bytes=1024 * 1024, max_dim=100)


def test_load_image_rejects_oversized_file():
    # Craft a file object with a large reported size
    buf = _make_png_bytes()
    buf.size = 30 * 1024 * 1024  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="too large"):
        load_image(buf, max_bytes=20 * 1024 * 1024)


def test_image_to_bytes_png_and_jpeg():
    img = Image.new("RGB", (20, 10), (1, 2, 3))
    png = image_to_bytes(img, "PNG")
    jpg = image_to_bytes(img, "JPG", quality=80)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert jpg[:2] == b"\xff\xd8"


def test_canvas_circles_and_extract_roundtrip():
    points = [[15.0, 25.0], [40.0, 50.0]]
    drawing = canvas_circles_from_points(points, radius=7)
    extracted = extract_canvas_points(drawing)
    assert len(extracted) == 2
    assert extracted[0] == pytest.approx(points[0])
    assert extracted[1] == pytest.approx(points[1])


def test_detect_document_corners_finds_clear_rectangle():
    img = np.full((400, 300, 3), 255, dtype=np.uint8)
    # Dark document frame on white background with light interior
    cv2.rectangle(img, (40, 50), (260, 350), (30, 30, 30), thickness=-1)
    cv2.rectangle(img, (50, 60), (250, 340), (220, 220, 220), thickness=-1)

    detected = detect_document_corners(img)
    assert detected is not None
    assert len(detected) == 4
    ordered = order_corners(detected)
    # Roughly near the outer document corners
    assert ordered[0][0] < 80 and ordered[0][1] < 90
    assert ordered[2][0] > 220 and ordered[2][1] > 300


def test_detect_document_corners_returns_none_on_blank():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert detect_document_corners(img) is None


def test_package_exports():
    import dimage

    assert hasattr(dimage, "deskew_image")
    assert hasattr(dimage, "apply_post_process")
    assert dimage.__version__


def test_rotate_image_clockwise_swaps_dimensions():
    img = Image.new("RGB", (40, 20), (10, 20, 30))
    rotated = rotate_image(img, 90)
    assert rotated.size == (20, 40)
    assert rotate_image(img, 0).size == (40, 20)
    assert rotate_image(img, 180).size == (40, 20)


def test_enhance_image_changes_pixels_and_rejects_nonpositive():
    img = Image.new("RGB", (8, 8), (40, 40, 40))
    img.putpixel((3, 3), (200, 200, 200))
    boosted = enhance_image(img, contrast=1.8, sharpness=1.0)
    assert boosted.size == img.size
    assert boosted.getpixel((3, 3)) != img.getpixel((3, 3)) or boosted.getpixel(
        (0, 0)
    ) != img.getpixel((0, 0))
    with pytest.raises(ValueError):
        enhance_image(img, contrast=0)


def test_crop_margins_and_post_process():
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    # Paint a white center so crop keeps white
    for x in range(20, 80):
        for y in range(20, 80):
            img.putpixel((x, y), (255, 255, 255))
    cropped = crop_margins(img, left_pct=20, top_pct=20, right_pct=20, bottom_pct=20)
    assert cropped.size == (60, 60)
    assert cropped.getpixel((0, 0)) == (255, 255, 255)

    with pytest.raises(ValueError):
        crop_margins(img, left_pct=60)

    processed = apply_post_process(
        img,
        rotate_degrees=90,
        contrast=1.1,
        sharpness=1.0,
        left_pct=10,
        top_pct=10,
        right_pct=10,
        bottom_pct=10,
    )
    assert processed.size[0] > 0 and processed.size[1] > 0
