import sys
from unittest.mock import MagicMock, patch

import numpy as np

mock_libcamera = MagicMock()
mock_picamera2 = MagicMock()
mock_picamera2.encoders._hw_encoder_available = False
mock_picamera2.outputs.Output = MagicMock
# setdefault keeps mocks installed by other test modules, whose tests rely on
# the libcamera constants set on them.
for module_name, module in {
    "libcamera": mock_libcamera,
    "picamera2": mock_picamera2,
    "picamera2.encoders": mock_picamera2.encoders,
    "picamera2.outputs": mock_picamera2.outputs,
}.items():
    sys.modules.setdefault(module_name, module)

WIDTH = 64
HEIGHT = 48
STRIDE = 80
MARGIN = 4
LUMA_FILL = 16
FIRST_CHROMA_FILL = 90
SECOND_CHROMA_FILL = 160
PIXEL_FILL = 50


def _make_patch(height, width):
    return (np.arange(height * width) % 251 + 1).astype(np.uint8).reshape(height, width)


def _make_yuv420_frame(width, height, stride):
    """Build a frame shaped like picamera2's YUV420 MappedArray: the Y plane
    followed by two chroma planes of height // 2 rows of stride // 2 bytes."""
    luma = np.full((height, stride), LUMA_FILL, np.uint8)
    first_chroma = np.full((height // 2, stride // 2), FIRST_CHROMA_FILL, np.uint8)
    second_chroma = np.full((height // 2, stride // 2), SECOND_CHROMA_FILL, np.uint8)
    flat = np.concatenate(
        [luma.reshape(-1), first_chroma.reshape(-1), second_chroma.reshape(-1)]
    )
    return flat.reshape(height * 3 // 2, stride)


def _split_yuv420_frame(frame, height, stride):
    flat = frame.reshape(-1)
    luma_size = height * stride
    chroma_size = (height // 2) * (stride // 2)
    luma = flat[:luma_size].reshape(height, stride)
    first_chroma = flat[luma_size : luma_size + chroma_size]
    second_chroma = flat[luma_size + chroma_size :]
    return (
        luma,
        first_chroma.reshape(height // 2, stride // 2),
        second_chroma.reshape(height // 2, stride // 2),
    )


def test_yuv420_patch_goes_into_luma_and_box_chroma_is_neutral():
    from spyglass.camera.timestamp_overlay import NEUTRAL_CHROMA, draw_patch_yuv420

    frame = _make_yuv420_frame(WIDTH, HEIGHT, STRIDE)
    patch = _make_patch(10, 20)
    draw_patch_yuv420(frame, patch, WIDTH, HEIGHT, STRIDE, MARGIN)

    luma, first_chroma, second_chroma = _split_yuv420_frame(frame, HEIGHT, STRIDE)
    expected_luma = np.full((HEIGHT, STRIDE), LUMA_FILL, np.uint8)
    expected_luma[4:14, 4:24] = patch
    np.testing.assert_array_equal(luma, expected_luma)
    for chroma, fill in (
        (first_chroma, FIRST_CHROMA_FILL),
        (second_chroma, SECOND_CHROMA_FILL),
    ):
        expected_chroma = np.full((HEIGHT // 2, STRIDE // 2), fill, np.uint8)
        expected_chroma[2:7, 2:12] = NEUTRAL_CHROMA
        np.testing.assert_array_equal(chroma, expected_chroma)


def test_yuv420_patch_wider_than_frame_leaves_stride_padding_untouched():
    from spyglass.camera.timestamp_overlay import NEUTRAL_CHROMA, draw_patch_yuv420

    frame = _make_yuv420_frame(WIDTH, HEIGHT, STRIDE)
    patch = _make_patch(10, 70)
    draw_patch_yuv420(frame, patch, WIDTH, HEIGHT, STRIDE, MARGIN)

    luma, first_chroma, second_chroma = _split_yuv420_frame(frame, HEIGHT, STRIDE)
    np.testing.assert_array_equal(luma[4:14, 4:WIDTH], patch[:, : WIDTH - 4])
    assert (luma[:, WIDTH:] == LUMA_FILL).all()
    assert (first_chroma[2:7, 2 : WIDTH // 2] == NEUTRAL_CHROMA).all()
    assert (second_chroma[2:7, 2 : WIDTH // 2] == NEUTRAL_CHROMA).all()
    assert (first_chroma[:, WIDTH // 2 :] == FIRST_CHROMA_FILL).all()
    assert (second_chroma[:, WIDTH // 2 :] == SECOND_CHROMA_FILL).all()


def test_rgb_patch_goes_into_all_three_channels():
    from spyglass.camera.timestamp_overlay import draw_patch_rgb

    frame = np.full((HEIGHT, WIDTH, 3), PIXEL_FILL, np.uint8)
    patch = _make_patch(10, 20)
    draw_patch_rgb(frame, patch, MARGIN)

    expected = np.full((HEIGHT, WIDTH, 3), PIXEL_FILL, np.uint8)
    expected[4:14, 4:24, :] = patch[:, :, np.newaxis]
    np.testing.assert_array_equal(frame, expected)


def test_xrgb_patch_leaves_fourth_byte_alone():
    from spyglass.camera.timestamp_overlay import draw_patch_rgb

    frame = np.full((HEIGHT, WIDTH, 4), PIXEL_FILL, np.uint8)
    patch = _make_patch(10, 20)
    draw_patch_rgb(frame, patch, MARGIN)

    expected = np.full((HEIGHT, WIDTH, 4), PIXEL_FILL, np.uint8)
    expected[4:14, 4:24, :3] = patch[:, :, np.newaxis]
    np.testing.assert_array_equal(frame, expected)


def test_patch_is_clipped_on_tiny_frames():
    from spyglass.camera.timestamp_overlay import draw_patch_rgb, draw_patch_yuv420

    patch = _make_patch(10, 20)
    yuv420_frame = _make_yuv420_frame(8, 6, 8)
    draw_patch_yuv420(yuv420_frame, patch, 8, 6, 8, 2)
    luma, _, _ = _split_yuv420_frame(yuv420_frame, 6, 8)
    np.testing.assert_array_equal(luma[2:6, 2:8], patch[:4, :6])

    rgb_frame = np.full((6, 8, 3), PIXEL_FILL, np.uint8)
    draw_patch_rgb(rgb_frame, patch, 2)
    np.testing.assert_array_equal(rgb_frame[2:6, 2:8, 0], patch[:4, :6])

    frame_smaller_than_margin = np.full((2, 2, 3), PIXEL_FILL, np.uint8)
    draw_patch_rgb(frame_smaller_than_margin, patch, MARGIN)
    assert (frame_smaller_than_margin == PIXEL_FILL).all()


def test_patch_is_rendered_again_only_when_text_changes():
    from spyglass.camera import timestamp_overlay

    overlay = timestamp_overlay.create_overlay(480)
    with patch.object(
        timestamp_overlay, "render_patch", wraps=timestamp_overlay.render_patch
    ) as render_patch:
        first = timestamp_overlay.update_patch(overlay, "2026-10-27 12:00:00")
        second = timestamp_overlay.update_patch(overlay, "2026-10-27 12:00:00")
        assert render_patch.call_count == 1
        assert second is first
        third = timestamp_overlay.update_patch(overlay, "2026-10-27 12:00:01")
        assert render_patch.call_count == 2
        assert third is not first


def test_rendered_patch_is_even_sized_white_on_black():
    from spyglass.camera.timestamp_overlay import create_overlay, render_patch

    overlay = create_overlay(960)
    rendered = render_patch("2026-10-27 12:00:00", overlay.font, overlay.padding)

    assert rendered.dtype == np.uint8
    assert rendered.shape[0] % 2 == 0
    assert rendered.shape[1] % 2 == 0
    assert rendered.max() == 255
    assert (rendered[0, :] == 0).all()
    assert (rendered[:, 0] == 0).all()


def test_text_height_scales_with_frame_height_down_to_minimum():
    from spyglass.camera.timestamp_overlay import text_height_for_frame

    assert text_height_for_frame(960) == 32
    assert text_height_for_frame(1080) == 36
    assert text_height_for_frame(480) == 16
    assert text_height_for_frame(240) == 16


def test_frame_drawer_is_none_for_unsupported_format():
    from spyglass.camera.timestamp_overlay import create_frame_drawer

    stream_config = {"format": "NV12", "size": (WIDTH, HEIGHT), "stride": WIDTH}
    assert create_frame_drawer(stream_config) is None


def test_frame_drawer_draws_into_yuv420_frame():
    from spyglass.camera.timestamp_overlay import create_frame_drawer

    stream_config = {"format": "YUV420", "size": (640, 480), "stride": 640}
    draw_timestamp = create_frame_drawer(stream_config)
    assert draw_timestamp is not None
    frame = _make_yuv420_frame(640, 480, 640)
    draw_timestamp(frame)

    luma, _, _ = _split_yuv420_frame(frame, 480, 640)
    assert luma.max() == 255
    assert (luma[:, 400:] == LUMA_FILL).all()


def test_csi_installs_pre_callback_drawing_into_main_stream():
    from spyglass.camera.csi import CSI

    picam2 = MagicMock()
    picam2.pre_callback = None
    picam2.camera_config = {
        "main": {"format": "XBGR8888", "size": (640, 480), "stride": 2560}
    }
    frame = np.full((480, 640, 4), PIXEL_FILL, np.uint8)
    request = MagicMock()
    with patch("spyglass.camera.csi.MappedArray") as mapped_array:
        mapped_array.return_value.__enter__.return_value.array = frame
        CSI(picam2)._install_timestamp_overlay()
        picam2.pre_callback(request)

    mapped_array.assert_called_once_with(request, "main")
    assert frame[:, :, :3].max() == 255
    assert (frame[:, :, 3] == PIXEL_FILL).all()


def test_csi_installs_no_pre_callback_for_unsupported_format():
    from spyglass.camera.csi import CSI

    picam2 = MagicMock()
    picam2.pre_callback = None
    picam2.camera_config = {"main": {"format": "NV12", "size": (640, 480)}}
    CSI(picam2)._install_timestamp_overlay()

    assert picam2.pre_callback is None
