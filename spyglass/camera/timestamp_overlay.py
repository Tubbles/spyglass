"""Draw the local time into the top left corner of the main stream frames.

The text is rendered into a small greyscale patch only when it changes (once
a second). Each frame only gets that cached patch copied into its pixel buffer.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# numpy and Pillow come with picamera2. They are not spyglass requirements, so
# the type check environment does not have them.
import numpy as np  # pyright: ignore[reportMissingImports]
from PIL import Image, ImageDraw, ImageFont  # pyright: ignore[reportMissingImports]

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
YUV420_FORMATS = ("YUV420", "YVU420")
RGB_FORMATS = ("BGR888", "RGB888", "XBGR8888", "XRGB8888")
TEXT_HEIGHT_DIVISOR = 30
MINIMUM_TEXT_HEIGHT = 16
TEXT_LUMA = 255
BOX_LUMA = 0
NEUTRAL_CHROMA = 128


@dataclass
class TimestampOverlay:
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    padding: int
    margin: int
    text: str = ""
    patch: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.uint8))


def text_height_for_frame(frame_height: int) -> int:
    return max(MINIMUM_TEXT_HEIGHT, frame_height // TEXT_HEIGHT_DIVISOR)


def round_up_to_even(value: int) -> int:
    return value + value % 2


def create_overlay(frame_height: int) -> TimestampOverlay:
    text_height = text_height_for_frame(frame_height)
    padding = text_height // 4
    # Twice the padding is even, so the box starts on a chroma sample.
    return TimestampOverlay(
        font=ImageFont.load_default(size=text_height),
        padding=padding,
        margin=2 * padding,
    )


def render_patch(
    text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, padding: int
) -> np.ndarray:
    """Render white text on a black box as a 2D luma array of even size."""
    left, top, right, bottom = font.getbbox(text)
    width = round_up_to_even(math.ceil(right - left) + 2 * padding)
    height = round_up_to_even(math.ceil(bottom - top) + 2 * padding)
    image = Image.new("L", (width, height), BOX_LUMA)
    position = (padding - left, padding - top)
    ImageDraw.Draw(image).text(position, text, fill=TEXT_LUMA, font=font)
    return np.asarray(image)


def update_patch(overlay: TimestampOverlay, text: str) -> np.ndarray:
    if text != overlay.text:
        overlay.patch = render_patch(text, overlay.font, overlay.padding)
        overlay.text = text
    return overlay.patch


def yuv420_planes(
    frame: np.ndarray, width: int, height: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return views of the Y and both chroma planes without the stride padding.

    Each chroma plane has height // 2 rows of stride // 2 bytes and follows
    the Y plane in the flat buffer.
    """
    flat = frame.reshape(-1)
    luma_size = height * stride
    chroma_shape = (height // 2, stride // 2)
    chroma_size = chroma_shape[0] * chroma_shape[1]
    first_chroma_end = luma_size + chroma_size
    luma = flat[:luma_size].reshape(height, stride)[:, :width]
    first_chroma = flat[luma_size:first_chroma_end].reshape(chroma_shape)
    second_chroma = flat[first_chroma_end : first_chroma_end + chroma_size]
    second_chroma = second_chroma.reshape(chroma_shape)
    chroma_width = width // 2
    return luma, first_chroma[:, :chroma_width], second_chroma[:, :chroma_width]


def draw_patch_yuv420(
    frame: np.ndarray,
    patch: np.ndarray,
    width: int,
    height: int,
    stride: int,
    margin: int,
) -> None:
    luma, first_chroma, second_chroma = yuv420_planes(frame, width, height, stride)
    patch_height, patch_width = patch.shape
    # Slicing clips the region at the frame edges.
    region = luma[margin : margin + patch_height, margin : margin + patch_width]
    rows, columns = region.shape
    region[:, :] = patch[:rows, :columns]
    chroma_rows = slice(margin // 2, (margin + rows) // 2)
    chroma_columns = slice(margin // 2, (margin + columns) // 2)
    first_chroma[chroma_rows, chroma_columns] = NEUTRAL_CHROMA
    second_chroma[chroma_rows, chroma_columns] = NEUTRAL_CHROMA


def draw_patch_rgb(frame: np.ndarray, patch: np.ndarray, margin: int) -> None:
    """Write the patch into the three colour channels, leaving the X byte."""
    patch_height, patch_width = patch.shape
    # Slicing clips the region at the frame edges.
    region = frame[margin : margin + patch_height, margin : margin + patch_width, :3]
    rows, columns = region.shape[:2]
    region[:, :, :] = patch[:rows, :columns, np.newaxis]


def create_frame_drawer(
    stream_config: dict[str, Any],
) -> Callable[[np.ndarray], None] | None:
    """Return a function drawing the current local time into a frame of the
    configured stream, or ``None`` if the pixel format is not supported."""
    pixel_format = stream_config["format"]
    if pixel_format not in YUV420_FORMATS + RGB_FORMATS:
        return None
    width, height = stream_config["size"]
    stride = stream_config["stride"]
    overlay = create_overlay(height)

    def draw_yuv420(frame: np.ndarray) -> None:
        patch = update_patch(overlay, time.strftime(TIMESTAMP_FORMAT))
        draw_patch_yuv420(frame, patch, width, height, stride, overlay.margin)

    def draw_rgb(frame: np.ndarray) -> None:
        patch = update_patch(overlay, time.strftime(TIMESTAMP_FORMAT))
        draw_patch_rgb(frame, patch, overlay.margin)

    return draw_yuv420 if pixel_format in YUV420_FORMATS else draw_rgb
