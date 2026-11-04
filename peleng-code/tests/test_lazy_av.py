"""PyAV грузится только для телефона: окно и записи живут с одной копией FFmpeg (из OpenCV)."""

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import phone_usb


def _h264_stream(frames: int = 5, width: int = 64, height: int = 48) -> list[bytes]:
    """Пакеты Annex-B: так сервер камеры телефона шлёт кадры."""
    av = phone_usb._pyav()
    for name in ("libx264", "h264"):
        try:
            encoder = av.CodecContext.create(name, "w")
            break
        except Exception:
            continue
    else:
        raise unittest.SkipTest("в сборке PyAV нет кодировщика H.264")
    encoder.width = width
    encoder.height = height
    encoder.pix_fmt = "yuv420p"
    encoder.options = {"tune": "zerolatency", "preset": "ultrafast"}
    packets = []
    for index in range(frames):
        image = np.full((height, width, 3), 40 * index, np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        packets += [bytes(packet) for packet in encoder.encode(frame)]
    packets += [bytes(packet) for packet in encoder.encode(None)]
    return packets


class LazyPyAvTest(unittest.TestCase):
    def test_window_import_does_not_load_pyav(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys, app.main; print('av' in sys.modules)"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")
        self.assertNotIn("is implemented in both", result.stderr)

    def test_phone_decoder_still_decodes(self):
        decoder = phone_usb._H264Decoder()
        images = [decoder.feed(packet, config=False) for packet in _h264_stream()]
        decoded = [image for image in images if image is not None]
        self.assertTrue(decoded)
        self.assertEqual(decoded[-1].shape, (48, 64, 3))

    def test_broken_packet_is_skipped(self):
        decoder = phone_usb._H264Decoder()
        self.assertIsNone(decoder.feed(b"\x00\x00\x00\x01\x65\xff\xff\xff", config=False))


if __name__ == "__main__":
    unittest.main()
