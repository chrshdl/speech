import asyncio
import os
import shutil
import socket
import tempfile
import threading
import time
import types

import httpx
import pytest

from speech.backend import wifi
from speech.backend.app import create_app
from speech.backend.messages import WifiConnectError, WifiState

SAE = "NONE WPA-PSK WPA-PSK-SHA256 SAE"
FAST = {"scan": 1, "join": 1, "address": 0.5, "poll": 0.02}


class FakeSupplicant:
    """
    Answers wpa_supplicant's control protocol on a Unix datagram socket,
    with access points by name and password, None for an open one.
    """

    def __init__(self, directory, access_points, key_mgmt=SAE, address="192.168.1.20"):
        self.access_points = access_points
        self.key_mgmt = key_mgmt
        # The address DHCP gives, None for none.
        self.address = address
        # Scans until the first results, as after wpa_supplicant starts.
        self.scans_until_results = 1
        self.networks = {}
        self.next_id = 0
        self.current = None
        self.saved = None
        self.commands = []
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.bind(os.path.join(directory, "wlan0"))
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def add(self, ssid, **settings):
        """A network saved before."""
        network = str(self.next_id)
        self.next_id += 1
        self.networks[network] = {"ssid": ssid.encode().hex(), **settings}
        return network

    def serve(self):
        while True:
            try:
                data, sender = self.socket.recvfrom(4096)
            except OSError:
                return
            command = data.decode()
            self.commands.append(command)
            self.socket.sendto(self.answer(*command.split(" ", 3)).encode(), sender)

    def close(self):
        self.socket.close()

    def ssid(self, network):
        return bytes.fromhex(self.networks[network]["ssid"]).decode()

    def answer(self, command, *args):
        if command == "SCAN":
            if self.scans_until_results:
                self.scans_until_results -= 1
            return "OK\n"
        if command == "SCAN_RESULTS":
            lines = ["bssid / frequency / signal level / flags / ssid"]
            if not self.scans_until_results:
                lines += [
                    "aa:bb:cc:dd:ee:01\t2437\t-70\t[WPA2-PSK-CCMP][ESS]\tHome",
                    "aa:bb:cc:dd:ee:02\t5180\t-50\t[WPA2-PSK-CCMP][ESS]\tHome",
                    "aa:bb:cc:dd:ee:03\t2412\t-80\t[ESS]\tCaf\\xc3\\xa9",
                    "aa:bb:cc:dd:ee:04\t2412\t-60\t[WPA2-PSK-CCMP][ESS]\t",
                    'aa:bb:cc:dd:ee:05\t2412\t-90\t[RSN-SAE-CCMP][ESS]\tSay \\"hi\\"',
                ]
            return "\n".join(lines) + "\n"
        if command == "STATUS":
            if self.current is None:
                return "wpa_state=SCANNING\n"
            lines = [
                "wpa_state=COMPLETED",
                f"id={self.current}",
                f"ssid={self.ssid(self.current)}",
            ]
            if self.address:
                lines.append(f"ip_address={self.address}")
            return "\n".join(lines) + "\n"
        if command == "LIST_NETWORKS":
            lines = ["network id / ssid / bssid / flags"]
            for network, settings in self.networks.items():
                flags = settings.get("flags", "")
                if network == self.current:
                    flags += "[CURRENT]"
                lines.append(f"{network}\t{self.ssid(network)}\tany\t{flags}")
            return "\n".join(lines) + "\n"
        if command == "ADD_NETWORK":
            return f"{self.add('')}\n"
        if command == "SET_NETWORK":
            network, name, value = args[0], args[1], args[2]
            self.networks[network][name] = value
            return "OK\n"
        if command == "GET_CAPABILITY":
            return self.key_mgmt + "\n"
        if command == "SELECT_NETWORK":
            return self.select(args[0])
        if command == "ENABLE_NETWORK":
            self.networks[args[0]].pop("flags", None)
            return "OK\n"
        if command == "REMOVE_NETWORK":
            del self.networks[args[0]]
            if self.current == args[0]:
                self.current = None
            return "OK\n"
        if command == "SAVE_CONFIG":
            self.saved = {n: dict(s) for n, s in self.networks.items()}
            return "OK\n"
        return "UNKNOWN COMMAND\n"

    def select(self, network):
        for other, settings in self.networks.items():
            if other != network:
                settings["flags"] = "[DISABLED]"
        settings = self.networks[network]
        ssid = self.ssid(network)
        if ssid not in self.access_points:
            return "OK\n"  # Never joined.
        password = self.access_points[ssid]
        if password is None:
            joined = settings.get("key_mgmt") == "NONE"
        else:
            joined = settings.get("psk") == wifi.psk(ssid, password)
        if joined:
            self.current = network
        else:
            settings["flags"] = "[TEMP-DISABLED]"
        return "OK\n"


@pytest.fixture
def control_dir():
    # Short, as Unix socket paths are limited to about 100 bytes.
    directory = tempfile.mkdtemp(dir="/tmp")
    yield directory
    shutil.rmtree(directory)


@pytest.fixture
def supplicant(control_dir):
    fake = FakeSupplicant(control_dir, {"Home": "rainbow123", "Café": None})
    yield fake
    fake.close()


def make_wifi(control_dir):
    return wifi.Wifi("wlan0", control_dir, timeouts=FAST)


def test_decode_ssid():
    assert wifi.decode_ssid("Qiu\\xe2\\x80\\x99s Home") == "Qiu’s Home"
    assert wifi.decode_ssid('a\\"b\\\\c') == 'a"b\\c'
    # Invalid UTF-8 and control characters, which a network can send.
    assert wifi.decode_ssid("\\xff\\x00ok") == "��ok"


def test_signal_bars():
    assert [wifi.signal_bars(dbm) for dbm in (-95, -85, -70, -60, -40)] == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_psk_matches_wpa_passphrase():
    # wpa_passphrase test "password", from the IEEE 802.11i test vectors.
    assert wifi.psk("IEEE", "password") == (
        "f42c6fc52df0ebef9ebb4b90b38a5f902e83fe1b135a70e23aed762e9710a12e"
    )


def test_scan(supplicant, control_dir):
    networks = make_wifi(control_dir).scan()
    # One per name with its strongest signal, the strongest first, and no
    # hidden network. The first scan had no results yet.
    assert [(n.ssid, n.secured, n.signal) for n in networks] == [
        ("Home", True, 4),
        ("Café", False, 1),
        ('Say "hi"', True, 0),
    ]
    assert supplicant.commands.count("SCAN") == 1


def test_connect(supplicant, control_dir):
    old = supplicant.add("Old", psk="00" * 32)
    response = make_wifi(control_dir).connect("Home", "rainbow123")
    assert response.error is None
    assert response.status.state == WifiState.CONNECTED
    assert response.status.ssid == "Home"
    assert response.status.ip_address == "192.168.1.20"
    # Saved in place of the network saved before, with the key and not the
    # passphrase for WPA2, and the passphrase for WPA3, which needs it.
    [saved] = supplicant.saved.values()
    assert old not in supplicant.saved
    assert saved["ssid"] == b"Home".hex()
    assert saved["psk"] == wifi.psk("Home", "rainbow123")
    assert saved["sae_password"] == '"rainbow123"'
    assert saved["key_mgmt"] == "WPA-PSK WPA-PSK-SHA256 SAE"


def test_connect_without_sae(supplicant, control_dir):
    supplicant.key_mgmt = "NONE WPA-PSK"
    response = make_wifi(control_dir).connect("Home", "rainbow123")
    assert response.error is None
    [saved] = supplicant.saved.values()
    assert saved["key_mgmt"] == "WPA-PSK"
    assert "sae_password" not in saved


def test_connect_open(supplicant, control_dir):
    response = make_wifi(control_dir).connect("Café")
    assert response.error is None
    assert response.status.ssid == "Café"
    [saved] = supplicant.saved.values()
    assert saved["key_mgmt"] == "NONE"


@pytest.mark.parametrize(
    ("ssid", "password", "error"),
    [
        ("Home", "wrong-password", WifiConnectError.WRONG_PASSWORD),
        ("Far away", "rainbow123", WifiConnectError.NOT_JOINED),
    ],
)
def test_connect_fails(supplicant, control_dir, ssid, password, error):
    old = supplicant.add("Old", psk="00" * 32)
    response = make_wifi(control_dir).connect(ssid, password)
    assert response.error == error
    assert response.status.state == WifiState.DISCONNECTED
    assert response.status.configured
    # The network is forgotten, the one saved before enabled again, and
    # nothing saved.
    assert list(supplicant.networks) == [old]
    assert "flags" not in supplicant.networks[old]
    assert supplicant.saved is None


def test_connect_keeps_disabled_network_disabled(supplicant, control_dir):
    old = supplicant.add("Old", psk="00" * 32, flags="[DISABLED]")
    make_wifi(control_dir).connect("Home", "wrong-password")
    assert supplicant.networks[old]["flags"] == "[DISABLED]"


def test_connect_without_address(supplicant, control_dir):
    supplicant.address = None
    response = make_wifi(control_dir).connect("Home", "rainbow123")
    assert response.error == WifiConnectError.NO_ADDRESS
    assert response.status.state == WifiState.CONNECTING
    # Joined, so the password is right and the network saved.
    assert len(supplicant.saved) == 1


def test_one_connect_at_a_time(supplicant, control_dir):
    control = make_wifi(control_dir)
    control.joining.acquire()
    with pytest.raises(RuntimeError):
        control.connect("Home", "rainbow123")


def test_unavailable(control_dir):
    control = make_wifi(control_dir)
    assert control.status().state == WifiState.UNAVAILABLE
    with pytest.raises(wifi.Unavailable):
        control.scan()


class SlowSupplicant(FakeSupplicant):
    """Holds each SELECT_NETWORK, so a connect lasts."""

    def answer(self, command, *args):
        if command == "SELECT_NETWORK":
            time.sleep(0.5)
        return super().answer(command, *args)


def test_api(control_dir):
    supplicant = SlowSupplicant(control_dir, {"Home": "rainbow123"})
    recognizer = types.SimpleNamespace(
        model_name="tiny", sample_rate=16000, audio_input="CLIENT", language_model=None
    )
    app = create_app(recognizer, wifi=make_wifi(control_dir))

    async def run():
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport, base_url="http://x") as c:
            assert (await c.get("/v1/info")).json()["wifi"]
            assert (await c.get("/v1/wifi")).json() == {
                "state": "DISCONNECTED",
                "ssid": None,
                "ipAddress": None,
                "configured": False,
            }
            scan = (await c.post("/v1/wifi/scan")).json()
            assert scan["networks"][0] == {"ssid": "Home", "secured": True, "signal": 4}
            # Not a WPA password.
            for password in ("short", "x" * 64, "pässwort1"):
                response = await c.post(
                    "/v1/wifi/connect", json={"ssid": "Home", "password": password}
                )
                assert response.status_code == 422
            body = {"ssid": "Home", "password": "rainbow123"}
            first = asyncio.create_task(c.post("/v1/wifi/connect", json=body))
            await asyncio.sleep(0.2)
            # One network is joined at a time, and scanning would disturb it.
            assert (await c.post("/v1/wifi/connect", json=body)).status_code == 409
            assert (await c.post("/v1/wifi/scan")).status_code == 409
            response = (await first).json()
            assert response["error"] is None
            assert response["status"]["state"] == "CONNECTED"

    try:
        asyncio.run(run())
    finally:
        supplicant.close()


def test_api_without_wifi():
    recognizer = types.SimpleNamespace(
        model_name="tiny", sample_rate=16000, audio_input="CLIENT", language_model=None
    )
    app = create_app(recognizer)

    async def run():
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport, base_url="http://x") as c:
            assert not (await c.get("/v1/info")).json()["wifi"]
            assert (await c.get("/v1/wifi")).status_code == 404
            assert (await c.post("/v1/wifi/scan")).status_code == 404

    asyncio.run(run())
