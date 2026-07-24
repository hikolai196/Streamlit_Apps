"""Streamlit UI for interactive image deskewing."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from dimage.processing import (
    ImageArray,
    annotate_corners,
    apply_post_process,
    canvas_circles_from_points,
    deskew_image,
    detect_document_corners,
    extract_canvas_points,
    image_to_bytes,
    label_corners,
    load_image,
    order_corners,
    resize_image,
)

_MAX_HISTORY = 12

_TIPS_MARKDOWN = """
**Tips:**
- Click four corners of the region to deskew — order does not matter.
- Prefer a stylus or larger taps on mobile; pinch-zoom the page if needed.
- Use **Edit corners by coordinates** for keyboard-precise placement.
- Use **Clear points** to reset, or **Auto-detect corners** as a starting guess.
- After deskewing, rotate / crop / enhance, then **Save to history**.
"""

_PRIVACY_NOTE = (
    "Privacy: images stay in your browser session for this app run. "
    "Nothing is uploaded to an external storage service from this tool, "
    "and Streamlit usage stats collection is disabled in config."
)

_THEME_COLORS = {
    "dark": {
        "bg": "#0e1117",
        "secondary": "#262730",
        "text": "#fafafa",
        "primary": "#3d9b7a",
        "muted": "#a3a8b4",
    },
    "light": {
        "bg": "#ffffff",
        "secondary": "#f0f2f6",
        "text": "#31333f",
        "primary": "#1a5f4a",
        "muted": "#5c6370",
    },
}


def _apply_theme(mode: str) -> None:
    """Override Streamlit chrome colors for the selected light/dark mode."""
    colors = _THEME_COLORS.get(mode, _THEME_COLORS["dark"])
    st.markdown(
        f"""
        <style>
        .stApp,
        .stAppViewContainer,
        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {{
            background-color: {colors["bg"]};
            color: {colors["text"]};
        }}
        section[data-testid="stSidebar"] > div {{
            background-color: {colors["secondary"]};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {colors["secondary"]};
            color: {colors["text"]};
        }}
        .stMarkdown, .stCaption, .stText, label, p, h1, h2, h3, h4 {{
            color: {colors["text"]} !important;
        }}
        [data-testid="stCaption"] {{
            color: {colors["muted"]} !important;
        }}
        div[data-testid="stAlert"] {{
            color: {colors["text"]};
        }}
        .stButton > button {{
            border-color: {colors["primary"]};
        }}
        .stDownloadButton > button {{
            border-color: {colors["primary"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session() -> None:
    if "canvas_version" not in st.session_state:
        st.session_state.canvas_version = 0
    if "last_upload_id" not in st.session_state:
        st.session_state.last_upload_id = None
    if "initial_drawing" not in st.session_state:
        st.session_state.initial_drawing = None
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True
    if "deskew_history" not in st.session_state:
        st.session_state.deskew_history = []
    if "history_counter" not in st.session_state:
        st.session_state.history_counter = 0


def _reset_canvas(clear_drawing: bool = True) -> None:
    st.session_state.canvas_version += 1
    if clear_drawing:
        st.session_state.initial_drawing = None


def _clear_image_cache() -> None:
    st.session_state.pop("image_cache", None)


def _load_or_cache_image(img_file: Any, upload_id: str) -> tuple[ImageArray, ImageArray, float]:
    """Decode and resize once per upload; reuse across Streamlit reruns."""
    cached = st.session_state.get("image_cache")
    if (
        isinstance(cached, dict)
        and cached.get("upload_id") == upload_id
        and isinstance(cached.get("img_np"), np.ndarray)
        and isinstance(cached.get("resized_img"), np.ndarray)
        and isinstance(cached.get("scale"), (int, float))
    ):
        return cached["img_np"], cached["resized_img"], float(cached["scale"])

    _image, img_np = load_image(img_file)
    resized_img, scale = resize_image(img_np)
    st.session_state.image_cache = {
        "upload_id": upload_id,
        "img_np": img_np,
        "resized_img": resized_img,
        "scale": scale,
    }
    return img_np, resized_img, scale


def _save_to_history(image: Image.Image, source_name: str) -> None:
    st.session_state.history_counter += 1
    entry = {
        "id": st.session_state.history_counter,
        "name": source_name,
        "created_at": datetime.now().strftime("%H:%M:%S"),
        "image": image.copy(),
        "size": image.size,
    }
    history: list[dict[str, Any]] = list(st.session_state.deskew_history)
    history.insert(0, entry)
    st.session_state.deskew_history = history[:_MAX_HISTORY]


def _render_history(output_format: str, jpeg_quality: int) -> None:
    history: list[dict[str, Any]] = st.session_state.deskew_history
    st.subheader("History")
    if not history:
        st.caption("Saved deskews from this session will appear here.")
        return

    if st.button("Clear history", key="clear_history"):
        st.session_state.deskew_history = []
        st.rerun()

    for entry in history:
        with st.expander(
            f"#{entry['id']} · {entry['name']} · {entry['created_at']} · "
            f"{entry['size'][0]}×{entry['size'][1]}"
        ):
            st.image(entry["image"], use_container_width=True)
            ext = "png" if output_format == "PNG" else "jpg"
            st.download_button(
                label=f"Download #{entry['id']} as {output_format}",
                data=image_to_bytes(entry["image"], output_format, quality=jpeg_quality),
                file_name=f"deskewed_{entry['id']}.{ext}",
                mime=f"image/{ext}",
                key=f"dl_hist_{entry['id']}",
            )


def _manual_corner_editor(
    display_w: int,
    display_h: int,
    current_points: list[list[float]],
) -> None:
    with st.expander("Edit corners by coordinates (keyboard-friendly)", expanded=False):
        st.caption(
            "Enter display-pixel coordinates for four corners. "
            "Tab between fields, then apply to update the canvas."
        )
        defaults = current_points if len(current_points) == 4 else [
            [0.0, 0.0],
            [float(display_w - 1), 0.0],
            [float(display_w - 1), float(display_h - 1)],
            [0.0, float(display_h - 1)],
        ]
        labels = ("Top-left", "Top-right", "Bottom-right", "Bottom-left")
        edited: list[list[float]] = []
        for i, label in enumerate(labels):
            c1, c2 = st.columns(2)
            with c1:
                x = st.number_input(
                    f"{label} X",
                    min_value=0.0,
                    max_value=float(max(display_w - 1, 0)),
                    value=float(defaults[i][0]),
                    step=1.0,
                    key=f"manual_x_{i}_{st.session_state.canvas_version}",
                )
            with c2:
                y = st.number_input(
                    f"{label} Y",
                    min_value=0.0,
                    max_value=float(max(display_h - 1, 0)),
                    value=float(defaults[i][1]),
                    step=1.0,
                    key=f"manual_y_{i}_{st.session_state.canvas_version}",
                )
            edited.append([x, y])

        if st.button("Apply coordinates to canvas", use_container_width=True):
            ordered = order_corners(edited)
            st.session_state.initial_drawing = canvas_circles_from_points(ordered)
            _reset_canvas(clear_drawing=False)
            st.rerun()


def run_app() -> None:
    st.set_page_config(page_title="Image Deskewer", layout="wide")
    _init_session()

    st.sidebar.header("Appearance")
    dark_mode = st.sidebar.toggle(
        "Dark mode",
        key="dark_mode",
        help="Switch between dark and light theme",
    )
    _apply_theme("dark" if dark_mode else "light")

    st.title("🖼️ Image Deskewer")
    st.write(
        "Select four corner points to deskew a document or region. "
        "Click order does not matter."
    )
    st.caption(_PRIVACY_NOTE)

    st.sidebar.header("1. Upload Image")
    img_file = st.sidebar.file_uploader(
        "Choose a PNG or JPG image", type=["png", "jpg", "jpeg"]
    )
    st.sidebar.caption("Max 20 MB · max 10000px on either side")

    st.sidebar.header("2. Tools")
    clear_clicked = st.sidebar.button("Clear points", use_container_width=True)
    detect_clicked = st.sidebar.button("Auto-detect corners", use_container_width=True)

    st.sidebar.header("3. Output")
    output_format = st.sidebar.radio("Format", ["PNG", "JPG"])
    jpeg_quality = 92
    if output_format == "JPG":
        jpeg_quality = int(
            st.sidebar.slider("JPEG quality", min_value=50, max_value=100, value=92)
        )

    st.sidebar.header("4. Post-process")
    rotate_degrees = int(
        st.sidebar.selectbox("Rotate (clockwise)", options=[0, 90, 180, 270], index=0)
    )
    contrast = float(
        st.sidebar.slider("Contrast", min_value=0.5, max_value=2.0, value=1.0, step=0.05)
    )
    sharpness = float(
        st.sidebar.slider("Sharpen", min_value=0.5, max_value=2.0, value=1.0, step=0.05)
    )
    st.sidebar.caption("Crop margins (% from each edge)")
    crop_left = float(st.sidebar.slider("Crop left %", 0.0, 40.0, 0.0, 0.5))
    crop_top = float(st.sidebar.slider("Crop top %", 0.0, 40.0, 0.0, 0.5))
    crop_right = float(st.sidebar.slider("Crop right %", 0.0, 40.0, 0.0, 0.5))
    crop_bottom = float(st.sidebar.slider("Crop bottom %", 0.0, 40.0, 0.0, 0.5))

    if not img_file:
        _clear_image_cache()
        st.info("Upload a PNG or JPG image to get started.")
        st.markdown(
            """
            **Accessibility / mobile**
            - On phones/tablets, tap carefully or use a stylus.
            - Prefer landscape orientation; avoid rapid multi-touch on the canvas.
            - Use **Edit corners by coordinates** if tapping four points is hard.
            """
        )
        st.markdown(_TIPS_MARKDOWN)
        _render_history(output_format, jpeg_quality)
        st.caption(_PRIVACY_NOTE)
        return

    upload_id = f"{img_file.name}:{img_file.size}"
    if upload_id != st.session_state.last_upload_id:
        st.session_state.last_upload_id = upload_id
        _reset_canvas(clear_drawing=True)
        _clear_image_cache()

    if clear_clicked:
        _reset_canvas(clear_drawing=True)
        st.rerun()

    try:
        img_np, resized_img, scale = _load_or_cache_image(img_file, upload_id)

        if detect_clicked:
            detected = detect_document_corners(resized_img)
            if detected is None:
                st.sidebar.warning(
                    "Could not auto-detect a document. Select corners manually."
                )
            else:
                ordered = order_corners(detected)
                st.session_state.initial_drawing = canvas_circles_from_points(ordered)
                _reset_canvas(clear_drawing=False)
                st.rerun()

        st.write(
            "Click on the image to add points, or enter coordinates below. "
            "Drag canvas points to adjust."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=3,
                background_image=Image.fromarray(resized_img),
                update_streamlit=True,
                height=resized_img.shape[0],
                width=resized_img.shape[1],
                drawing_mode="point",
                point_display_radius=7,
                initial_drawing=st.session_state.initial_drawing,
                key=f"canvas_{st.session_state.canvas_version}",
            )

            points = extract_canvas_points(
                canvas_result.json_data if canvas_result else None
            )

            _manual_corner_editor(
                resized_img.shape[1],
                resized_img.shape[0],
                points,
            )

            if points:
                annotated = annotate_corners(resized_img, points)
                st.image(
                    annotated,
                    caption="Corner preview (numbered after auto-order when 4 points)",
                    use_container_width=True,
                )

            if len(points) > 4:
                st.error(
                    "You selected more than 4 points. Please clear and select exactly 4."
                )
            elif len(points) < 4:
                st.info(f"Select exactly 4 points to deskew ({len(points)}/4).")

        with col2:
            st.subheader("Deskewed")
            if len(points) == 4:
                try:
                    orig_points = label_corners(points, scale)
                    result_img = deskew_image(img_np, orig_points)
                    result_img = apply_post_process(
                        result_img,
                        rotate_degrees=rotate_degrees,
                        contrast=contrast,
                        sharpness=sharpness,
                        left_pct=crop_left,
                        top_pct=crop_top,
                        right_pct=crop_right,
                        bottom_pct=crop_bottom,
                    )
                except ValueError as e:
                    st.warning(str(e))
                else:
                    st.success("Deskewed image ready.")
                    st.image(
                        result_img,
                        caption="Deskewed output (with post-process)",
                        use_container_width=True,
                    )
                    img_bytes = image_to_bytes(
                        result_img, output_format, quality=jpeg_quality
                    )
                    ext = "png" if output_format == "PNG" else "jpg"
                    st.download_button(
                        label=f"Download as {output_format}",
                        data=img_bytes,
                        file_name=f"deskewed.{ext}",
                        mime=f"image/{ext}",
                    )
                    if st.button("Save to history", use_container_width=True):
                        _save_to_history(result_img, img_file.name)
                        st.toast(f"Saved to history (max {_MAX_HISTORY} items).")
                        st.rerun()
            else:
                st.info(
                    "The corrected image will appear here once four corners are set."
                )

        _render_history(output_format, jpeg_quality)

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unexpected error: {e}")

    st.markdown(_TIPS_MARKDOWN)
    st.caption(_PRIVACY_NOTE)
