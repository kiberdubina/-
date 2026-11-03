"""Телефон как камера только по кабелю к этому ноутбуку.

На телефоне из временного файла поднимается сервер камеры: он снимает сенсор,
не экран, без звука и без управления. Картинка приходит на localhost через кабель.
Чужой компьютер и сеть её не видят. Кабель вынули или окно закрыли — съёмка гаснет.
"""

import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import av
import cv2
import numpy as np

from app.source import Frame, VideoSource

PACKAGE = "peleng.cam"
SERVICE = "peleng.cam/.CamService"
LOCAL_PORT = 27183
ROOT = Path(__file__).resolve().parent.parent
APK = ROOT / "phone" / "PelengCam.apk"
SERVER_JAR = ROOT / "phone" / "scrcpy-server"
DEVICE_JAR = "/data/local/tmp/scrcpy-server.jar"
SOCKET_NAME = "scrcpy"
SERVER_VERSION = "5.0.1"
SERVER_CLASS = "com.genymobile.scrcpy.Server"
H264_CODEC = 0x68323634
_SESSION = 1 << 63
_CONFIG = 1 << 62
_KEY = 1 << 61
_PTS_MASK = _KEY - 1
# CREATE_NO_WINDOW: консоль adb не всплывает. Флаг есть только в Windows,
# в macOS и Linux subprocess отвергает любой ненулевой creationflags.
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_TARGET_MS = 50
# Выше этого — часы телефона против ноутбука (сон, uptime), не стекло-стекло.
MAX_LATENCY_MS = 400
_WARMUP_FRAMES = 15
_SAMPLE_FRAMES = 20
_RECV_BYTES = 64 * 1024
# Сначала около 640 px по широкой стороне. Не уложились в 50 мс — кадр меньше и реже.
_PROFILES = (
    ("640x360", "30", "800000"),
    ("320x240", "24", "400000"),
)
_LOW_OPTIONS = (
    "latency=0,priority=0,i-frame-interval=1,"
    "vendor.qti-ext-enc-low-latency.enable=1"
)
_SAFE_OPTIONS = "latency=0,priority=0,i-frame-interval=1"
MAX_STREAM_RESTARTS = 3
RESTART_PAUSE = 1.2
# Сокет жив, а нового кадра нет: кодировщик замер. Это не экран телефона.
STALL_S = 1.0
# Последний кадр можно подержать только на время короткого перезапуска.
KEEP_LAST_S = 1.5
AWAKE_REFRESH_S = 30.0
VIDEO_SOURCE = "camera"

last_error = ""
# Телефон, с которого камера по кабелю не пойдёт. Повторно его не открываем.
skipped: dict[str, str] = {}
# После неудачного переключения не дёргаем шторку на каждом опросе.
_file_transfer_retry = 0.0
# «Без передачи данных» не отдаёт ни управление, ни картинку. Нужен пункт шторки.
FILE_TRANSFER_REQUIRED = (
    "В шторке USB выберите «Передача файлов», не «Без передачи данных». "
    "Иначе ноутбук не получает ни управление, ни картинку."
)
_USB_NAMES = {
    "mtp",
    "ptp",
    "adb",
    "rndis",
    "midi",
    "ncm",
    "acm",
    "mass_storage",
    "accessory",
    "audio_source",
    "uvc",
    "none",
    "charging",
}
_CHARGING_ONLY = {"adb", "none", "charging"}


class _CodecRejected(Exception):
    pass


class _DeviceGone(Exception):
    """adb уже не видит телефон. Сервер камеры надо гасить, а не держать кадр."""


def capture_pts_us(header: int) -> int | None:
    """Метка сенсора, микросекунды. У конфигурации и смены размера её нет."""
    if header & (_SESSION | _CONFIG):
        return None
    return header & _PTS_MASK


def suspend_seconds(text: str) -> float | None:
    """Секунды сна телефона: метка кадра их не считает, часы с загрузки считают."""
    parts = text.split()
    if not parts:
        return None
    try:
        uptime = float(parts[0])
    except ValueError:
        return None
    match = re.search(r"\s(\d+\.\d{3})\s+\d+\s+\d+\s+I pelengclk:", text)
    if match is None:
        return None
    return uptime - float(match.group(1))


def median_latency_ms(samples: list[float]) -> int:
    ordered = sorted(samples)
    return int(round(ordered[len(ordered) // 2]))


def believable_latency_ms(sample: float) -> bool:
    """Кадр успел дойти до окна. 1194 мс — сдвиг часов, такой замер отбрасывается."""
    return 0 < sample <= MAX_LATENCY_MS


def slower_profile(index: int, latency_ms: int) -> int | None:
    """Более мелкий и редкий кадр, если живой замер не уложился в 50 мс.

    Сдвиг часов сюда не входит: из-за него профиль не уменьшаем.
    """
    if not believable_latency_ms(latency_ms):
        return None
    if latency_ms <= _TARGET_MS or index + 1 >= len(_PROFILES):
        return None
    return index + 1


def usb_link_mode(text: str) -> str:
    """Режим кабеля по ответу телефона.

    files — в конфигурации есть mtp, это «Передача файлов».
    none — только зарядка или одна отладка, это «Без передачи данных».
    other — иной обмен, например фото (PTP).
    unknown — телефон ничего не ответил, режим не гадаем.
    Последняя строка с известными функциями важнее предыдущих.
    """
    mode = "unknown"
    for line in text.splitlines():
        lower = line.strip().lower()
        # Это ответ adb на ноутбуке, не список функций телефона.
        if not lower or lower.startswith("adb.exe") or "no devices" in lower:
            continue
        tokens = {
            part.strip("[]\"'")
            for part in re.split(r"[,;\s]+", lower)
        }
        tokens &= _USB_NAMES
        if not tokens:
            continue
        if "mtp" in tokens:
            mode = "files"
        elif tokens <= _CHARGING_ONLY:
            mode = "none"
        else:
            mode = "other"
    return mode


def device_listed(text: str, serial: str) -> bool:
    """В выводе adb devices есть этот телефон и он уже разрешён."""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == serial and parts[1] == "device":
            return True
    return False


def _adb() -> Path | None:
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "peleng-build" / "platform-tools" / "adb.exe"
    if local.is_file():
        return local
    from shutil import which

    found = which("adb")
    return Path(found) if found else None


def _stop_remote(serial: str) -> None:
    """Гасит только процесс сервера scrcpy."""
    try:
        _run(serial, "shell", "pkill", "-f", SERVER_CLASS, timeout=4)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _run(serial: str | None, *args: str, timeout: float = 40) -> subprocess.CompletedProcess[str]:
    command = [str(_adb())]
    if serial:
        command += ["-s", serial]
    command += list(args)
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=_NO_WINDOW,
    )


def _phone_usb_config(serial: str, full: bool = True) -> str:
    """sys.usb.config, затем текущие функции. Пустая строка — ответа не было."""
    queries = [("shell", "getprop", "sys.usb.config")]
    if full:
        queries.append(("shell", "svc", "usb", "getFunctions"))
    parts: list[str] = []
    for args in queries:
        try:
            result = _run(serial, *args, timeout=4)
        except (OSError, subprocess.TimeoutExpired):
            continue
        lines = []
        for line in f"{result.stdout or ''}\n{result.stderr or ''}".splitlines():
            if line.strip() and "adb.exe" not in line.lower():
                lines.append(line.strip())
        if lines:
            parts.append("\n".join(lines))
    return "\n".join(parts)


def _wait_for_file_transfer(serial: str, seconds: float) -> bool:
    """Ждём, пока кабель снова в adb и среди функций есть mtp."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            listed = _run(None, "devices", timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            time.sleep(0.3)
            continue
        if device_listed(listed.stdout or "", serial):
            if usb_link_mode(_phone_usb_config(serial)) == "files":
                return True
        time.sleep(0.35)
    return False


def _switch_to_file_transfer(serial: str) -> bool:
    """Ставит «Передача файлов» (mtp), отладку не снимает.

    На части прошивок вторая форма команды без true. Если шторка не
    переключилась, следующую попытку откладываем, чтобы не рвать кабель.
    """
    global _file_transfer_retry
    if time.monotonic() < _file_transfer_retry:
        return False
    commands = (
        ("shell", "svc", "usb", "setFunctions", "mtp", "true"),
        ("shell", "svc", "usb", "setFunctions", "mtp"),
    )
    for args in commands:
        try:
            result = _run(serial, *args, timeout=6)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if _wait_for_file_transfer(serial, 3.5):
            _file_transfer_retry = 0.0
            return True
        combined = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
        if result.returncode == 0 and "error" not in combined and "expected" not in combined:
            break
    _file_transfer_retry = time.monotonic() + 8
    return False


def _wake_usb(serial: str) -> None:
    """Экран не гаснет, пока кабель этого телефона в adb. Чужой серийник не трогает."""
    if not serial or _adb() is None:
        return
    try:
        _run(serial, "shell", "svc", "power", "stayon", "usb", timeout=4)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _cable_accepts_camera(serial: str) -> bool:
    """Камера идёт по уже разрешённой отладке. Шторку не трогаем.

    Смена «Передача файлов» на секунду рвёт кабель, и ноутбук теряет телефон.
    Пока adb видит аппарат, картинку можно включать сразу.
    """
    try:
        listed = _run(None, "devices", timeout=4)
    except (OSError, subprocess.TimeoutExpired):
        return True
    if device_listed(listed.stdout or "", serial):
        return True
    mode = usb_link_mode(_phone_usb_config(serial))
    if mode in {"files", "unknown"}:
        return True
    return _switch_to_file_transfer(serial)


def _getprop(serial: str, name: str) -> str:
    try:
        result = _run(serial, "shell", "getprop", name, timeout=4)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (result.stdout or "").strip()


def cable_camera_block(serial: str) -> str:
    """Пустая строка — камеру можно включать. Иначе причина, без новой попытки."""
    known = skipped.get(serial)
    if known:
        return known
    try:
        sdk = int(_getprop(serial, "ro.build.version.sdk"))
    except ValueError:
        return ""
    if sdk >= 31:
        return ""
    release = _getprop(serial, "ro.build.version.release") or str(sdk)
    message = (
        f"Android {release} не отдаёт камеру по этому кабелю. "
        "Поток камеры в программе начинается с Android 12."
    )
    skipped[serial] = message
    return message


def phone_slot(serial: str, serials: list[str]) -> tuple[int, str]:
    """Порт ноутбука и имя камеры. Порядок серийных номеров не прыгает."""
    ordered = sorted(set(serials))
    index = ordered.index(serial)
    return LOCAL_PORT + index, f"phone-{index + 1}"


def _ready_serials() -> list[str]:
    global last_error
    unauthorized = False
    offline = False
    for attempt in range(4):
        result = _run(None, "devices")
        ready: list[str] = []
        unauthorized = False
        offline = False
        for line in result.stdout.splitlines()[1:]:
            parts = line.split()
            if len(parts) < 2:
                continue
            if parts[1] == "device":
                ready.append(parts[0])
            elif parts[1] == "unauthorized":
                unauthorized = True
            elif parts[1] == "offline":
                offline = True
        if ready:
            return sorted(set(ready))
        if attempt < 3:
            time.sleep(0.4)
    if unauthorized:
        last_error = (
            "На экране телефона один раз нажмите «Разрешить» "
            "и оставьте «Всегда с этого компьютера»."
        )
    elif offline:
        last_error = "Телефон на кабеле, связь на секунду пропала. Подключаю снова."
    else:
        last_error = "Жду телефон по кабелю."
    return []


def _serial() -> str | None:
    found = _ready_serials()
    return found[0] if found else None


class PhoneUsbSource(VideoSource):
    def __init__(self, serial: str, port: int = LOCAL_PORT, camera_id: str = "phone-1"):
        self.serial = serial
        self.port = port
        self._stopped = False
        super().__init__(f"http://127.0.0.1:{port}/video", camera_id)

    def close(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        super().close()
        if _adb() is None:
            return
        _run(self.serial, "forward", "--remove", f"tcp:{self.port}")
        _run(self.serial, "shell", "am", "stopservice", "-n", SERVICE)


def _server_jar() -> Path | None:
    cached = Path(os.environ.get("LOCALAPPDATA", "")) / "peleng-build" / "scrcpy" / "scrcpy-server"
    for path in (SERVER_JAR, cached):
        if path.is_file() and path.stat().st_size > 100_000:
            return path
    return None


def _explain(text: str) -> str:
    lower = text.lower()
    if "does not match the client" in lower or "server version" in lower:
        return "Сервер камеры на телефоне не совпал с программой на ноутбуке."
    if "not supported before android 12" in lower:
        return "Эта версия Android не отдаёт камеру по кабелю."
    if "security" in lower or "permission" in lower or "disconnected" in lower:
        return "Телефон не пустил к камере. Разблокируйте экран и закройте приложение «Камера»."
    return "Кабель на месте, камера телефона не открылась. Разблокируйте экран."


def _avcc_extradata(payload: bytes) -> tuple[bytes, int]:
    length_size = (payload[4] & 3) + 1
    out = bytearray()
    index = 5
    sps_count = payload[index] & 0x1F
    index += 1
    for _ in range(sps_count):
        if index + 2 > len(payload):
            break
        size = int.from_bytes(payload[index : index + 2], "big")
        index += 2
        out += b"\x00\x00\x00\x01"
        out += payload[index : index + size]
        index += size
    if index < len(payload):
        pps_count = payload[index]
        index += 1
        for _ in range(pps_count):
            if index + 2 > len(payload):
                break
            size = int.from_bytes(payload[index : index + 2], "big")
            index += 2
            out += b"\x00\x00\x00\x01"
            out += payload[index : index + size]
            index += size
    return bytes(out), length_size


def _length_prefixed(payload: bytes, length_size: int) -> bytes:
    out = bytearray()
    index = 0
    limit = len(payload)
    while index + length_size <= limit:
        size = int.from_bytes(payload[index : index + length_size], "big")
        index += length_size
        if size <= 0 or index + size > limit:
            break
        out += b"\x00\x00\x00\x01"
        out += payload[index : index + size]
        index += size
    return bytes(out)


class _H264Decoder:
    """Один декодер. Наружу отдаётся только что разобранный кадр, без очереди."""

    def __init__(self) -> None:
        self._context = av.CodecContext.create("h264", "r")
        self._context.thread_count = 1
        self._context.thread_type = av.codec.context.ThreadType.SLICE
        self._context.flags |= av.codec.context.Flags.low_delay
        self._context.flags2 |= av.codec.context.Flags2.fast
        self._length_size: int | None = None

    def feed(self, payload: bytes, config: bool) -> np.ndarray | None:
        annex = self._annex(payload, config)
        if not annex:
            return None
        latest = None
        try:
            for packet in self._context.parse(annex):
                for frame in self._context.decode(packet):
                    latest = frame.to_ndarray(format="bgr24")
        except av.FFmpegError:
            return None
        return latest

    def _annex(self, payload: bytes, config: bool) -> bytes:
        if config and payload[:1] == b"\x01" and len(payload) >= 7:
            annex, self._length_size = _avcc_extradata(payload)
            return annex
        if payload.startswith((b"\x00\x00\x00\x01", b"\x00\x00\x01")):
            return payload
        if self._length_size:
            return _length_prefixed(payload, self._length_size)
        return payload


class PhonePreviewSource:
    """Кадры камеры телефона. Идут только по кабелю к этому ноутбуку.

    Снимается сенсор, не экран. Пока кабель на месте, экран телефона не гаснет,
    а оборванный или замерший поток поднимается снова. Последний кадр держится
    не дольше полутора секунд: дальше окно ждёт новую картинку.
    """

    def __init__(self, serial: str, camera_id: str = "phone-1", port: int | None = None):
        self.serial = serial
        self.camera_id = camera_id
        self.port = LOCAL_PORT if port is None else port
        self.kind = "phone"
        self.rewound = False
        self.index = 0
        self.fps = 30
        self.latency_ms: int | None = None
        self.error = ""
        self.stale = False
        self._presented_capture: float | None = None
        self._shown: list[float] = []
        self._seq = 0
        self._served = 0
        self._suspend: float | None = None
        self._capture_mono: float | None = None
        self._present_count = 0
        self._closed = False
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._started = time.monotonic()
        self._proc: subprocess.Popen[bytes] | None = None
        self._sock: socket.socket | None = None
        self._log = bytearray()
        self._thread: threading.Thread | None = None
        self._restarts = 0
        self._keep_last = False
        self._keep_since: float | None = None
        self._fresh_at: float | None = None
        self._last_decoded = time.monotonic()
        self._had_frame = False
        self._released = False
        self._life = threading.Lock()
        self._awake: threading.Thread | None = None
        try:
            self._start()
        except Exception as exc:
            self.error = str(exc) or _explain("")
            self.close()

    def is_open(self) -> bool:
        return not self._closed

    def _start(self) -> None:
        self._power_stayon("usb")
        if not _cable_accepts_camera(self.serial):
            raise OSError(FILE_TRANSFER_REQUIRED)
        jar = _server_jar()
        if jar is None:
            raise OSError("На ноутбуке нет сервера камеры для кабеля.")
        _stop_remote(self.serial)
        _run(self.serial, "shell", "am", "force-stop", "com.android.camera", timeout=5)
        time.sleep(0.25)
        pushed = _run(self.serial, "push", str(jar), DEVICE_JAR)
        if pushed.returncode != 0:
            raise OSError("Не удалось передать сервер камеры на телефон.")
        _run(self.serial, "forward", "--remove", f"tcp:{self.port}", timeout=5)
        forwarded = _run(self.serial, "forward", f"tcp:{self.port}", f"localabstract:{SOCKET_NAME}")
        if forwarded.returncode != 0:
            raise OSError("Кабель не принял канал картинки.")
        self._power_stayon("usb")
        self._awake = threading.Thread(target=self._awake_loop, daemon=True)
        self._awake.start()
        self._thread = threading.Thread(target=self._pull, daemon=True)
        self._thread.start()

    def _drain(self, proc: subprocess.Popen[bytes], log: bytearray) -> None:
        if proc.stdout is None:
            return
        while True:
            line = proc.stdout.readline()
            if not line:
                return
            with self._lock:
                log += line
                if len(log) > 8000:
                    del log[:-4000]

    def _connect(self) -> socket.socket:
        """Кабель принимает соединение и до сервера. Пустой обрыв — ещё не слушаем, пробуем снова."""
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if self._stop.is_set():
                raise EOFError
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                raise OSError(_explain(self._server_text()))
            try:
                sock = socket.create_connection(("127.0.0.1", self.port), timeout=1)
            except OSError:
                time.sleep(0.05)
                continue
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _RECV_BYTES)
            sock.settimeout(0.4)
            try:
                peeked = sock.recv(4, socket.MSG_PEEK)
            except socket.timeout:
                peeked = b"\x00"
            except OSError:
                peeked = b""
            if peeked:
                sock.settimeout(0.5)
                with self._life:
                    if self._stop.is_set() or self._released:
                        reject = True
                    else:
                        self._sock = sock
                        reject = False
                if reject:
                    try:
                        sock.close()
                    except OSError:
                        pass
                    raise EOFError
                return sock
            try:
                sock.close()
            except OSError:
                pass
            time.sleep(0.05)
        raise OSError(_explain(self._server_text()))

    def _server_text(self) -> str:
        with self._lock:
            local = bytes(self._log).decode("utf-8", "replace")
        try:
            remote = _run(self.serial, "logcat", "-d", "-t", "60", "-s", "scrcpy:W", timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return local
        return local + "\n" + remote.stdout

    def _recv_timeout(self) -> float:
        """До первого кадра ждём спокойно. Дальше просыпаемся ровно к порогу зависания."""
        if not self._had_frame:
            return 0.5
        left = STALL_S - (time.monotonic() - self._last_decoded)
        if left <= 0.05:
            return 0.05
        return min(0.5, left)

    def _read_exact(self, size: int) -> bytes:
        sock = self._sock
        if sock is None:
            raise EOFError
        chunks = bytearray()
        while len(chunks) < size:
            if self._stop.is_set():
                raise EOFError
            try:
                sock.settimeout(self._recv_timeout())
                part = sock.recv(size - len(chunks))
            except socket.timeout:
                self._on_idle()
                continue
            if not part:
                raise EOFError
            chunks += part
        return bytes(chunks)

    def _device_online(self) -> bool:
        """True, только если adb прямо сейчас перечисляет этот телефон как device.

        Пустой или оборванный ответ adb — не «телефона нет» и не повод включать отладку.
        """
        if _adb() is None:
            return False
        try:
            result = _run(None, "devices", timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            return True
        text = result.stdout or ""
        if "List of devices attached" not in text:
            return True
        return device_listed(text, self.serial)

    def _awake_loop(self) -> None:
        """Пока окно открыто, этот телефон не гаснет. Пропуск в adb поток не убивает."""
        next_power = 0.0
        while not self._stop.wait(5.0):
            if self._closed or self._stop.is_set():
                return
            if not self._device_online():
                continue
            now = time.monotonic()
            if now >= next_power:
                next_power = now + AWAKE_REFRESH_S
                self._power_stayon("usb")

    def _raise_if_stalled(self) -> None:
        """Сокет открыт, но нового кадра нет около секунды — кодировщик замер.

        Это перезапуск потока, не закрытие телефона. Последний живой кадр остаётся.
        """
        if self._stop.is_set() or self._closed:
            raise EOFError
        if not self._had_frame or time.monotonic() - self._last_decoded < STALL_S:
            return
        with self._lock:
            if self._frame is not None and (
                self._keep_since is None or time.monotonic() - self._keep_since <= KEEP_LAST_S
            ):
                if self._keep_since is None:
                    self._keep_since = time.monotonic()
                self._keep_last = True
        raise EOFError

    def _on_idle(self) -> None:
        self._raise_if_stalled()

    def _power_stayon(self, mode: str) -> None:
        """Экран не гаснет, только пока кабель воткнут. На close() режим снова false."""
        arming = mode != "false"
        if arming:
            with self._life:
                if self._stop.is_set() or self._released:
                    return
        if _adb() is None:
            return
        try:
            _run(self.serial, "shell", "svc", "power", "stayon", mode, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if not arming:
            return
        with self._life:
            cancel = self._stop.is_set() or self._released
        if cancel:
            self._power_stayon("false")

    def _launch(self, size: str, fps: str, bitrate: str, options: str) -> None:
        if VIDEO_SOURCE != "camera":
            raise OSError("В окно идёт только камера, не экран телефона.")
        if self._stop.is_set():
            raise EOFError
        self._power_stayon("usb")
        if self._stop.is_set():
            raise EOFError
        adb = _adb()
        assert adb is not None
        self.fps = int(fps)
        self._shown.clear()
        self._present_count = 0
        self._presented_capture = None
        self.latency_ms = None
        log = bytearray()
        with self._lock:
            self._log = log
        wide = size.split("x", 1)[0]
        with self._life:
            if self._stop.is_set():
                raise EOFError
            proc = subprocess.Popen(
                [
                    str(adb),
                    "-s",
                    self.serial,
                    "shell",
                    f"CLASSPATH={DEVICE_JAR}",
                    "app_process",
                    "/",
                    SERVER_CLASS,
                    SERVER_VERSION,
                    "log_level=info",
                    "video=true",
                    "audio=false",
                    "control=false",
                    "video_codec=h264",
                    f"video_source={VIDEO_SOURCE}",
                    "camera_facing=back",
                    f"camera_size={size}",
                    f"camera_fps={fps}",
                    f"max_fps={fps}",
                    f"max_size={wide}",
                    f"video_bit_rate={bitrate}",
                    f"video_codec_options={options}",
                    "capture_orientation=@0",
                    "tunnel_forward=true",
                    "send_device_meta=false",
                    "send_frame_meta=true",
                    "send_dummy_byte=false",
                    "send_stream_meta=true",
                    "cleanup=true",
                    "power_on=true",
                    "stay_awake=true",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=_NO_WINDOW,
            )
            self._proc = proc
        threading.Thread(target=self._drain, args=(proc, log), daemon=True).start()

    def _codec_rejected(self) -> bool:
        with self._lock:
            text = bytes(self._log).decode("utf-8", "replace").lower()
        return "illegalargumentexception" in text or "codec exception" in text or "0x80001001" in text

    def _sync_clock(self) -> float | None:
        """Сдвиг: локальные секунды = метка сенсора в секундах + сдвиг."""
        if self._suspend is None:
            for _ in range(3):
                if self._stop.is_set():
                    return None
                try:
                    stamped = _run(
                        self.serial,
                        "shell",
                        "cat /proc/uptime; log -t pelengclk m; logcat -v monotonic -d -t 1 -s pelengclk:I",
                        timeout=3,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    continue
                found = suspend_seconds(stamped.stdout)
                if found is not None and found > 0:
                    self._suspend = found
                    break
        if self._suspend is None:
            return None
        best: tuple[float, float] | None = None
        for _ in range(4):
            if self._stop.is_set():
                return None
            t0 = time.monotonic()
            try:
                result = _run(self.serial, "shell", "cat /proc/uptime", timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                continue
            t1 = time.monotonic()
            try:
                uptime = float(result.stdout.split()[0])
            except (ValueError, IndexError):
                continue
            rtt = t1 - t0
            shift = (t0 + t1) / 2 - (uptime - self._suspend)
            if best is None or rtt < best[0]:
                best = (rtt, shift)
        if best is None:
            return None
        return best[1]

    def _release_stream(self, keep_frame: bool = False) -> None:
        with self._life:
            sock = self._sock
            self._sock = None
            proc = self._proc
            self._proc = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
        with self._lock:
            if keep_frame and self._frame is not None:
                now = time.monotonic()
                if self._keep_since is None:
                    self._keep_since = now
                if now - self._keep_since <= KEEP_LAST_S:
                    self._keep_last = True
                else:
                    self._keep_last = False
                    self.stale = True
            else:
                self._frame = None
                self._keep_last = False
                self._keep_since = None
        if _adb() is None or self._stop.is_set():
            return
        _stop_remote(self.serial)

    def _disconnected(self) -> None:
        self._fail("Телефон пропал с кабеля. Жду его снова.")
        self._release_stream()
        self._stop.set()

    def _refuse_usb_mode(self) -> None:
        """Кабель ушёл в «Без передачи данных» или в передачу фото."""
        self._fail(FILE_TRANSFER_REQUIRED)
        self._release_stream()
        self._stop.set()

    def _fail(self, message: str) -> None:
        global last_error
        with self._lock:
            self._frame = None
            self._keep_last = False
            self._keep_since = None
            self.stale = True
        if self._stop.is_set():
            return
        if FILE_TRANSFER_REQUIRED in self.error and FILE_TRANSFER_REQUIRED not in message:
            self._closed = True
            return
        self.error = message
        last_error = message
        self._closed = True

    def _cable_gap(self) -> str:
        """Один пропуск в списке adb не закрывает телефон и не стирает кадр.

        ok — кабель на месте. again — короткая пауза, источник ещё открыт.
        stop — телефон уже закрыт, окно может его убрать.
        """
        if self._stop.is_set() or self._closed:
            return "stop"
        if self._device_online():
            return "ok"
        if self._restarts >= MAX_STREAM_RESTARTS:
            self._disconnected()
            return "stop"
        self._restarts += 1
        self._release_stream(keep_frame=True)
        time.sleep(RESTART_PAUSE)
        if self._stop.is_set() or self._closed:
            return "stop"
        return "again"

    def _pull(self) -> None:
        options = _LOW_OPTIONS
        index = 0
        hold = False
        retries = 0
        while not self._stop.is_set() and index < len(_PROFILES):
            cable = self._cable_gap()
            if cable == "stop":
                return
            if cable == "again":
                continue
            size, fps, bitrate = _PROFILES[index]
            try:
                self._launch(size, fps, bitrate, options)
                action = self._consume(index, hold)
            except _DeviceGone:
                self._release_stream(keep_frame=True)
                cable = self._cable_gap()
                if cable == "stop":
                    return
                continue
            except _CodecRejected:
                self._release_stream(keep_frame=True)
                if self._stop.is_set() or self._closed:
                    return
                cable = self._cable_gap()
                if cable == "stop":
                    return
                if cable == "again":
                    continue
                if index > 0 or retries < 2:
                    if index > 0:
                        index = 0
                        hold = True
                        retries = 0
                    else:
                        retries += 1
                        if options != _SAFE_OPTIONS and retries == 1:
                            options = _SAFE_OPTIONS
                    time.sleep(1.2)
                    continue
                self._fail("Кабель на месте, камера телефона не открылась. Разблокируйте экран.")
                return
            except (EOFError, OSError, av.FFmpegError) as exc:
                if self._stop.is_set() or self._closed:
                    self._release_stream()
                    return
                cable = self._cable_gap()
                if cable == "stop":
                    return
                if cable == "again":
                    continue
                if index > 0:
                    self._release_stream(keep_frame=True)
                    index = 0
                    hold = True
                    self._restarts = 0
                    time.sleep(RESTART_PAUSE)
                    if self._stop.is_set():
                        return
                    continue
                if self._restarts >= MAX_STREAM_RESTARTS:
                    self._release_stream()
                    told = exc.args[0] if exc.args and isinstance(exc.args[0], str) else ""
                    self._fail(told or "Кабель на месте, картинка оборвалась.")
                    return
                self._restarts += 1
                self._release_stream(keep_frame=True)
                time.sleep(RESTART_PAUSE)
                if self._stop.is_set():
                    return
                continue
            retries = 0
            if action == "smaller":
                self._release_stream(keep_frame=True)
                index += 1
                time.sleep(1.2)
                continue
            self._release_stream()
            return

    def _consume(self, index: int, hold: bool = False) -> str:
        decoder = _H264Decoder()
        pending: list[int] = []
        try:
            self._connect()
        except OSError:
            if self._codec_rejected():
                raise _CodecRejected
            raise
        self._last_decoded = time.monotonic()
        codec = int.from_bytes(self._read_exact(4), "big")
        if codec != H264_CODEC:
            self._fail("Телефон не отдал камеру.")
            return "stay"
        shift = None
        synced = False
        shown = 0
        samples: list[float] = []
        deciding = False
        while not self._stop.is_set():
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                if self._codec_rejected():
                    raise _CodecRejected
                raise EOFError
            header = int.from_bytes(self._read_exact(8), "big")
            if header & _SESSION:
                self._read_exact(4)
                decoder = _H264Decoder()
                pending.clear()
                continue
            size = int.from_bytes(self._read_exact(4), "big")
            if size <= 0 or size > 5_000_000:
                raise EOFError
            payload = self._read_exact(size)
            config = bool(header & _CONFIG)
            image = decoder.feed(payload, config)
            if config:
                self._raise_if_stalled()
                continue
            pts = capture_pts_us(header)
            if pts is None:
                self._raise_if_stalled()
                continue
            pending.append(pts)
            if image is None:
                self._raise_if_stalled()
                continue
            frame_pts = pending.pop(0)
            height, width = image.shape[:2]
            if height > width:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            captured = None if shift is None else shift + frame_pts / 1_000_000
            now = time.monotonic()
            with self._lock:
                self._frame = image
                self._capture_mono = captured
                self._seq += 1
                self._keep_last = False
                self._keep_since = None
                self._fresh_at = now
                self.stale = False
            self._last_decoded = now
            self._had_frame = True
            self._restarts = 0
            shown += 1
            if not synced and shown >= 2:
                shift = self._sync_clock()
                synced = True
                deciding = shift is not None and not hold
            if captured is None or not deciding:
                continue
            lag_ms = (time.monotonic() - captured) * 1000
            if not believable_latency_ms(lag_ms):
                continue
            samples.append(lag_ms)
            if len(samples) < _WARMUP_FRAMES + _SAMPLE_FRAMES:
                continue
            measured = median_latency_ms(samples[_WARMUP_FRAMES:])
            deciding = False
            if not believable_latency_ms(measured):
                continue
            self.latency_ms = measured
            if slower_profile(index, measured) is not None:
                return "smaller"
        return "stay"

    def read(self) -> Frame | None:
        if self._closed:
            return None
        now = time.monotonic()
        with self._lock:
            keep = False
            if self._keep_last:
                since = self._keep_since
                if since is not None and now - since <= KEEP_LAST_S:
                    keep = True
                else:
                    self._keep_last = False
                    self.stale = True
                    return None
            if self._frame is None or (self._seq == self._served and not keep):
                return None
            image = self._frame.copy()
            captured = self._capture_mono
            if not keep:
                self._served = self._seq
                self.stale = False
        self._presented_capture = None if keep else captured
        self.rewound = False
        self.index += 1
        return Frame(image, time.monotonic() - self._started, self.camera_id)

    def note_presented(self) -> None:
        """Задержка от метки сенсора до кадра, который уже попал в окно."""
        captured = self._presented_capture
        if captured is None:
            return
        self._present_count += 1
        if self._present_count <= _WARMUP_FRAMES:
            return
        sample = (time.monotonic() - captured) * 1000
        if not believable_latency_ms(sample):
            if self.latency_ms is not None and not believable_latency_ms(self.latency_ms):
                self.latency_ms = None
            return
        self._shown.append(sample)
        if len(self._shown) > 30:
            del self._shown[:-30]
        if len(self._shown) >= 5:
            measured = median_latency_ms(self._shown)
            if believable_latency_ms(measured):
                self.latency_ms = measured

    def close(self) -> None:
        with self._life:
            if self._released:
                return
            self._released = True
            self._closed = True
            self._stop.set()
            sock = self._sock
            self._sock = None
            proc = self._proc
            self._proc = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        if proc is not None and proc.poll() is None:
            proc.terminate()
        thread = self._thread
        awake = self._awake
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
        if awake is not None and awake is not threading.current_thread():
            awake.join(timeout=1)
        if proc is not None and proc.poll() is None:
            proc.kill()
        if _adb() is None:
            return
        try:
            _stop_remote(self.serial)
            _run(self.serial, "forward", "--remove", f"tcp:{self.port}", timeout=4)
        except (OSError, subprocess.TimeoutExpired):
            pass
        time.sleep(0.3)
        self._power_stayon("false")


def _open_one(serial: str, camera_id: str, port: int) -> PhonePreviewSource | None:
    global last_error
    source = PhonePreviewSource(serial, camera_id, port)
    if not source.is_open():
        last_error = source.error or "Кабель на месте, камера телефона не открылась. Разблокируйте экран."
        source.close()
        return None
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        if source.read() is not None:
            source.index = 0
            return source
        if not source.is_open():
            break
        time.sleep(0.05)
    last_error = source.error or "Кабель на месте, картинка не пошла. Разблокируйте экран телефона."
    source.close()
    return None


def open_phones(
    skip: set[str] | None = None,
    used_ports: set[int] | None = None,
    used_ids: set[str] | None = None,
    already: list | None = None,
) -> list[PhonePreviewSource]:
    """Открывает телефоны, которые ещё не снимают.

    Уже открытый серийник не перезапускается: его сервер, порт и процесс
    остаются. Имя камеры у живого телефона подстраивается под порядок
    серийников, без нового подключения. Наружу отдаются только новые источники.
    """
    global last_error
    skip = set(skip or ())
    taken_ports = set(used_ports or ())
    taken_ids = set(used_ids or ())
    live: list = []
    attached: set[str] = set()
    for source in already or []:
        serial = getattr(source, "serial", None)
        if not serial:
            continue
        attached.add(serial)
        skip.add(serial)
        port = getattr(source, "port", None)
        if port is not None:
            taken_ports.add(int(port))
        # Короткий простой кадра источник не закрывает: его не подменяем и не перезапускаем.
        if source.is_open():
            live.append(source)
        elif getattr(source, "camera_id", None):
            taken_ids.add(source.camera_id)
    if _adb() is None:
        if not skip:
            last_error = "На ноутбуке ещё нет программы связи с телефоном по кабелю."
        return []
    if _server_jar() is None:
        if not skip:
            last_error = "На ноутбуке нет сервера камеры для кабеля."
        return []
    if not skip:
        last_error = ""
    serials = _ready_serials()
    if not serials:
        return []
    if attached.issubset(set(serials)):
        for source in live:
            _port, camera_id = phone_slot(source.serial, serials)
            source.camera_id = camera_id
    for source in live:
        if getattr(source, "camera_id", None):
            taken_ids.add(source.camera_id)
        stay = getattr(source, "_power_stayon", None)
        if callable(stay):
            stay("usb")
        else:
            _wake_usb(getattr(source, "serial", ""))
    opened: list[PhonePreviewSource] = []
    for serial in serials:
        if serial in skip:
            continue
        block = cable_camera_block(serial)
        if block:
            last_error = block
            continue
        _wake_usb(serial)
        port, camera_id = phone_slot(serial, serials)
        if port in taken_ports:
            port = LOCAL_PORT
            while port in taken_ports:
                port += 1
        if camera_id in taken_ids:
            number = 1
            while f"phone-{number}" in taken_ids:
                number += 1
            camera_id = f"phone-{number}"
        source = _open_one(serial, camera_id, port)
        if source is None:
            continue
        taken_ports.add(port)
        taken_ids.add(camera_id)
        opened.append(source)
    return opened


def open_phone_usb() -> PhonePreviewSource | None:
    """Первый телефон по порядку серийников. Второй серийник не трогает."""
    global last_error
    if _adb() is None:
        last_error = "На ноутбуке ещё нет программы связи с телефоном по кабелю."
        return None
    if _server_jar() is None:
        last_error = "На ноутбуке нет сервера камеры для кабеля."
        return None
    last_error = ""
    serials = _ready_serials()
    if not serials:
        return None
    port, camera_id = phone_slot(serials[0], serials)
    _wake_usb(serials[0])
    return _open_one(serials[0], camera_id, port)
