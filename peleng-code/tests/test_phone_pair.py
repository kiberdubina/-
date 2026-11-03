import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import phone_usb
from app.phone_usb import LOCAL_PORT, believable_latency_ms, phone_slot, slower_profile


class _Result:
    def __init__(self, stdout: str):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


class _FakePreview:
    def __init__(self, serial: str, camera_id: str = "phone-1", port: int | None = None):
        self.serial = serial
        self.camera_id = camera_id
        self.port = LOCAL_PORT if port is None else port
        self.error = ""
        self.index = 0
        self.stale = False
        self.closed = False

    def is_open(self) -> bool:
        return not self.closed

    def read(self):
        return object()

    def close(self) -> None:
        self.closed = True


class PhonePairTest(unittest.TestCase):
    def test_slots_follow_sorted_serials(self):
        serials = ["bbb", "aaa"]
        self.assertEqual(phone_slot("aaa", serials), (LOCAL_PORT, "phone-1"))
        self.assertEqual(phone_slot("bbb", serials), (LOCAL_PORT + 1, "phone-2"))
        self.assertEqual(LOCAL_PORT + 1, 27184)

    def test_ready_serials_lists_every_authorized_device(self):
        text = "List of devices attached\nzzz\tdevice\naaa\tdevice\nqq\tunauthorized\n"
        with mock.patch.object(phone_usb, "_run", return_value=_Result(text)), mock.patch.object(
            phone_usb.time, "sleep"
        ) as sleep:
            found = phone_usb._ready_serials()
        self.assertEqual(found, ["aaa", "zzz"])
        sleep.assert_not_called()

    def test_ready_serials_keeps_the_short_russian_errors(self):
        cases = (
            ("List of devices attached\n", "Жду телефон по кабелю."),
            (
                "List of devices attached\nabc\tunauthorized\n",
                "На экране телефона один раз нажмите «Разрешить» "
                "и оставьте «Всегда с этого компьютера».",
            ),
            (
                "List of devices attached\nabc\toffline\n",
                "Телефон на кабеле, связь на секунду пропала. Подключаю снова.",
            ),
        )
        for text, expected in cases:
            with mock.patch.object(phone_usb, "_run", return_value=_Result(text)), mock.patch.object(
                phone_usb.time, "sleep"
            ):
                found = phone_usb._ready_serials()
            self.assertEqual(found, [])
            self.assertEqual(phone_usb.last_error, expected)
            lowered = phone_usb.last_error.lower()
            self.assertNotIn("разработ", lowered)
            self.assertNotIn("miui", lowered)

    def test_second_serial_does_not_restart_the_open_phone(self):
        live = _FakePreview("bbb", "phone-1", LOCAL_PORT)
        created: list[str] = []

        class Recording(_FakePreview):
            def __init__(self, serial: str, camera_id: str = "phone-1", port: int | None = None):
                created.append(serial)
                super().__init__(serial, camera_id, port)

        with mock.patch.object(phone_usb, "_adb", return_value="adb"), mock.patch.object(
            phone_usb, "_server_jar", return_value="jar"
        ), mock.patch.object(
            phone_usb, "_ready_serials", return_value=["aaa", "bbb"]
        ), mock.patch.object(phone_usb, "PhonePreviewSource", Recording):
            opened = phone_usb.open_phones(already=[live])
        self.assertEqual(created, ["aaa"])
        self.assertFalse(live.closed)
        self.assertEqual(live.port, LOCAL_PORT)
        self.assertEqual(live.camera_id, "phone-2")
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0].serial, "aaa")
        self.assertEqual(opened[0].camera_id, "phone-1")
        self.assertEqual(opened[0].port, LOCAL_PORT + 1)
        self.assertNotEqual(opened[0].port, live.port)

    def test_open_phones_gives_each_serial_its_own_port(self):
        created: list[tuple[str, str, int]] = []

        class Recording(_FakePreview):
            def __init__(self, serial: str, camera_id: str = "phone-1", port: int | None = None):
                super().__init__(serial, camera_id, port)
                created.append((serial, camera_id, self.port))

        with mock.patch.object(phone_usb, "_adb", return_value="adb"), mock.patch.object(
            phone_usb, "_server_jar", return_value="jar"
        ), mock.patch.object(
            phone_usb, "_ready_serials", return_value=["aaa", "bbb"]
        ), mock.patch.object(phone_usb, "PhonePreviewSource", Recording):
            opened = phone_usb.open_phones()
        self.assertEqual(
            created,
            [("aaa", "phone-1", LOCAL_PORT), ("bbb", "phone-2", LOCAL_PORT + 1)],
        )
        self.assertEqual(len(opened), 2)

    def test_open_phone_usb_does_not_open_the_second_serial(self):
        created: list[str] = []

        class Recording(_FakePreview):
            def __init__(self, serial: str, camera_id: str = "phone-1", port: int | None = None):
                created.append(serial)
                super().__init__(serial, camera_id, port)

        with mock.patch.object(phone_usb, "_adb", return_value="adb"), mock.patch.object(
            phone_usb, "_server_jar", return_value="jar"
        ), mock.patch.object(
            phone_usb, "_ready_serials", return_value=["aaa", "bbb"]
        ), mock.patch.object(phone_usb, "PhonePreviewSource", Recording):
            source = phone_usb.open_phone_usb()
        self.assertEqual(created, ["aaa"])
        self.assertIsNotNone(source)
        self.assertEqual(source.camera_id, "phone-1")
        self.assertEqual(source.port, LOCAL_PORT)

    def test_android_10_is_not_opened_again(self):
        phone_usb.skipped.clear()

        def fake_run(serial, *args, timeout=40):
            del timeout
            if args[:3] == ("shell", "getprop", "ro.build.version.sdk"):
                return _Result("29\n" if serial == "old" else "31\n")
            if args[:3] == ("shell", "getprop", "ro.build.version.release"):
                return _Result("10\n")
            return _Result("")

        created: list[str] = []

        class Recording(_FakePreview):
            def __init__(self, serial: str, camera_id: str = "phone-1", port: int | None = None):
                created.append(serial)
                super().__init__(serial, camera_id, port)

        with mock.patch.object(phone_usb, "_adb", return_value="adb"), mock.patch.object(
            phone_usb, "_server_jar", return_value="jar"
        ), mock.patch.object(
            phone_usb, "_ready_serials", return_value=["new", "old"]
        ), mock.patch.object(phone_usb, "_run", side_effect=fake_run), mock.patch.object(
            phone_usb, "PhonePreviewSource", Recording
        ):
            opened = phone_usb.open_phones()
            again = phone_usb.open_phones(already=opened)
        self.assertEqual(created, ["new"])
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0].camera_id, "phone-1")
        self.assertIn("Android 10", phone_usb.skipped["old"])
        self.assertEqual(again, [])
        phone_usb.skipped.clear()

    def test_latency_sample_1194_is_rejected(self):
        self.assertFalse(believable_latency_ms(1194))
        self.assertFalse(believable_latency_ms(401))
        self.assertFalse(believable_latency_ms(0))
        self.assertTrue(believable_latency_ms(400))
        self.assertTrue(believable_latency_ms(50))
        self.assertIsNone(slower_profile(0, 1194))


if __name__ == "__main__":
    unittest.main()
