"""
Joins a Wi-Fi network, for a device such as a Raspberry Pi whose Wi-Fi
wpa_supplicant drives, so the user can pick the network and type its
password on the display, see docs/backend-api.md.

It talks to wpa_supplicant through its control interface, a Unix datagram
socket per network interface, which the service's user may use when it is
in the group of wpa_supplicant's ctrl_interface. wpa_supplicant writes its
own config file (update_config=1), and a DHCP client, such as
systemd-networkd's, gets the address once the network is joined.
"""

import hashlib
import itertools
import os
import socket
import tempfile
import threading
import time
import unicodedata

from speech.backend.messages import (
    WifiConnectError,
    WifiConnectResponse,
    WifiNetwork,
    WifiState,
    WifiStatus,
)

CONTROL_DIR = "/run/wpa_supplicant"

# Time limits in seconds. The first scan after starting takes several
# seconds, and joining a network can take some 10 seconds more.
SCAN_TIMEOUT = 10
JOIN_TIMEOUT = 30
ADDRESS_TIMEOUT = 30
POLL = 0.5

# wpa_supplicant's printf escapes in SSIDs, besides \xHH for other bytes.
_ESCAPES = {"n": 0x0A, "r": 0x0D, "t": 0x09, "e": 0x1B, "\\": 0x5C, '"': 0x22}

# The states of wpa_supplicant while it joins a network.
_JOINING = {
    "AUTHENTICATING",
    "ASSOCIATING",
    "ASSOCIATED",
    "4WAY_HANDSHAKE",
    "GROUP_HANDSHAKE",
}


class Unavailable(Exception):
    """wpa_supplicant does not answer, or failed a request."""


class Control:
    def __init__(self, interface, directory=CONTROL_DIR):
        """
        A client of wpa_supplicant's control socket for the network
        interface, such as wlan0, in the directory of its ctrl_interface.
        """
        self.path = os.path.join(directory, interface)
        # wpa_supplicant replies to the sender's address, so each request
        # binds a socket of its own in this directory.
        self.directory = tempfile.mkdtemp(prefix="wpa-")
        self.counter = itertools.count()

    def request(self, command, timeout=5):
        """Sends the command and returns the reply."""
        local = os.path.join(self.directory, str(next(self.counter)))
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            try:
                sock.bind(local)
                sock.settimeout(timeout)
                sock.connect(self.path)
                sock.send(command.encode())
                reply = sock.recv(65536).decode("utf-8", errors="replace")
            except OSError as error:
                raise Unavailable(f"wpa_supplicant: {error}") from None
            finally:
                if os.path.exists(local):
                    os.unlink(local)
        if reply.startswith("FAIL"):
            raise Unavailable(f"wpa_supplicant: {command.split()[0]} {reply.strip()}")
        return reply


def decode_ssid(raw):
    """
    The SSID as wpa_supplicant prints it, with each byte outside printable
    ASCII as \\xHH, decoded as UTF-8. A network can send any bytes, so
    invalid ones and control characters become U+FFFD.
    """
    data = bytearray()
    i = 0
    while i < len(raw):
        if raw[i] == "\\" and i + 1 < len(raw):
            following = raw[i + 1]
            if following == "x" and i + 3 < len(raw):
                try:
                    data.append(int(raw[i + 2 : i + 4], 16))
                    i += 4
                    continue
                except ValueError:
                    pass
            if following in _ESCAPES:
                data.append(_ESCAPES[following])
                i += 2
                continue
        data.extend(raw[i].encode())
        i += 1
    text = data.decode("utf-8", errors="replace")
    return "".join("�" if unicodedata.category(c) == "Cc" else c for c in text)


def signal_bars(dbm):
    """The signal strength in dBm as bars, from 0 to 4."""
    return sum(dbm >= limit for limit in (-88, -78, -67, -55))


def parse_scan_results(text):
    """
    The networks of SCAN_RESULTS, one per SSID with its strongest signal,
    the strongest first. Hidden networks, without an SSID, are left out.
    """
    best = {}
    for line in text.splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) < 5:
            continue
        try:
            dbm = int(fields[2])
        except ValueError:
            continue
        ssid = decode_ssid(fields[4])
        if not ssid:
            continue
        secured = any(tag in fields[3] for tag in ("WPA", "RSN", "WEP", "SAE"))
        if ssid not in best or dbm > best[ssid][0]:
            best[ssid] = (dbm, secured)
    ordered = sorted(best.items(), key=lambda item: item[1][0], reverse=True)
    return [
        WifiNetwork(ssid=ssid, secured=secured, signal=signal_bars(dbm))
        for ssid, (dbm, secured) in ordered
    ]


def parse_pairs(text):
    """The key=value lines of a reply, such as STATUS, as a dict."""
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def parse_networks(text):
    """The saved networks of LIST_NETWORKS, as {id: flags}."""
    networks = {}
    for line in text.splitlines()[1:]:
        fields = line.split("\t")
        if fields[0].isdigit():
            networks[fields[0]] = fields[3] if len(fields) > 3 else ""
    return networks


def psk(ssid, password):
    """The WPA key of the passphrase, as wpa_passphrase derives it."""
    return hashlib.pbkdf2_hmac("sha1", password.encode(), ssid.encode(), 4096, 32).hex()


def interface_address(interface):
    """The IPv4 address of the network interface from the kernel, if any."""
    import fcntl
    import struct

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            request = struct.pack("256s", interface.encode()[:15])
            reply = fcntl.ioctl(sock.fileno(), 0x8915, request)  # SIOCGIFADDR
        except OSError:
            return None
    address = socket.inet_ntoa(reply[20:24])
    # A link-local address is no lease.
    return None if address.startswith("169.254.") else address


class Wifi:
    def __init__(self, interface, directory=CONTROL_DIR, timeouts=None):
        """
        Joins Wi-Fi networks on the network interface, such as wlan0, with
        the wpa_supplicant whose control sockets are in the directory.

        The methods block, for seconds while scanning and joining, so run
        them in a thread. One network is joined at a time.

        Arguments:
            timeouts (dict, optional): Overrides SCAN_TIMEOUT, JOIN_TIMEOUT,
                ADDRESS_TIMEOUT and POLL by their lowercase names.
        """
        self.interface = interface
        self.control = Control(interface, directory)
        timeouts = {
            "scan": SCAN_TIMEOUT,
            "join": JOIN_TIMEOUT,
            "address": ADDRESS_TIMEOUT,
            "poll": POLL,
        } | (timeouts or {})
        self.scan_timeout = timeouts["scan"]
        self.join_timeout = timeouts["join"]
        self.address_timeout = timeouts["address"]
        self.poll = timeouts["poll"]
        self.joining = threading.Lock()

    def request(self, *words):
        return self.control.request(" ".join(words))

    def address(self, status):
        return status.get("ip_address") or interface_address(self.interface)

    def status(self):
        """The WifiStatus, UNAVAILABLE when wpa_supplicant does not answer."""
        try:
            status = parse_pairs(self.request("STATUS"))
            configured = bool(parse_networks(self.request("LIST_NETWORKS")))
        except Unavailable:
            return WifiStatus(state=WifiState.UNAVAILABLE)
        ssid = decode_ssid(status["ssid"]) if status.get("ssid") else None
        wpa_state = status.get("wpa_state")
        address = None
        if wpa_state == "COMPLETED":
            address = self.address(status)
            state = WifiState.CONNECTED if address else WifiState.CONNECTING
        elif wpa_state in _JOINING:
            state = WifiState.CONNECTING
        else:
            state = WifiState.DISCONNECTED
        return WifiStatus(
            state=state, ssid=ssid, ip_address=address, configured=configured
        )

    def scan(self):
        """
        The networks in reach, a list of WifiNetwork, the strongest first.
        Raises Unavailable when wpa_supplicant does not answer.
        """
        try:
            self.request("SCAN")
        except Unavailable as error:
            # A scan is already running, whose results serve as well.
            if "FAIL-BUSY" not in str(error):
                raise
        # The results of an earlier scan show at once, and the first scan
        # after starting takes a while, so wait for results, not a time.
        deadline = time.monotonic() + self.scan_timeout
        while True:
            networks = parse_scan_results(self.request("SCAN_RESULTS"))
            if networks or time.monotonic() >= deadline:
                return networks
            time.sleep(self.poll)

    def connect(self, ssid, password=""):
        """
        Joins the network and returns a WifiConnectResponse. On success the
        network is saved in place of the others; otherwise it is forgotten
        and the networks saved before stay, so a mistyped password never
        loses a network that works. Raises Unavailable when wpa_supplicant
        does not answer, and RuntimeError while another network is joined.
        """
        if not self.joining.acquire(blocking=False):
            raise RuntimeError("A network is being joined.")
        try:
            return self._connect(ssid, password)
        finally:
            self.joining.release()

    def _connect(self, ssid, password):
        before = parse_networks(self.request("LIST_NETWORKS"))
        network = self.request("ADD_NETWORK").strip()
        try:
            self._configure(network, ssid, password)
            # Selecting the network disables the others until it is saved.
            self.request("SELECT_NETWORK", network)
            error = self._join(network, password)
        except Unavailable:
            self._forget(network, before)
            raise
        if error is not None:
            self._forget(network, before)
            return WifiConnectResponse(status=self.status(), error=error)

        for other in before:
            self.request("REMOVE_NETWORK", other)
        self.request("SAVE_CONFIG")
        deadline = time.monotonic() + self.address_timeout
        while (status := self.status()).state != WifiState.CONNECTED:
            if time.monotonic() >= deadline:
                return WifiConnectResponse(
                    status=status, error=WifiConnectError.NO_ADDRESS
                )
            time.sleep(self.poll)
        return WifiConnectResponse(status=status)

    def _configure(self, network, ssid, password):
        def set_network(name, value):
            self.request("SET_NETWORK", network, name, value)

        # As hex, which needs no escaping.
        set_network("ssid", ssid.encode().hex())
        if not password:
            set_network("key_mgmt", "NONE")
            return
        # The key, not the passphrase, so the config file does not hold it.
        set_network("psk", psk(ssid, password))
        key_mgmt = self.request("GET_CAPABILITY", "key_mgmt").split()
        if "SAE" in key_mgmt:
            # Also offer WPA3, which wpa_supplicant uses where the network
            # has it. SAE authenticates with the passphrase itself, by its
            # design. Quoted strings are taken as they are, up to the last
            # quote, and a passphrase is printable ASCII.
            set_network("sae_password", f'"{password}"')
            set_network("key_mgmt", "WPA-PSK WPA-PSK-SHA256 SAE")
            set_network("ieee80211w", "1")
        else:
            set_network("key_mgmt", "WPA-PSK")

    def _join(self, network, password):
        """None once the network is joined, otherwise a WifiConnectError."""
        deadline = time.monotonic() + self.join_timeout
        while time.monotonic() < deadline:
            status = parse_pairs(self.request("STATUS"))
            if status.get("wpa_state") == "COMPLETED" and status.get("id") == network:
                return None
            # wpa_supplicant disables a network for a while after its key
            # failed, such as with a wrong password.
            flags = parse_networks(self.request("LIST_NETWORKS")).get(network, "")
            if "TEMP-DISABLED" in flags:
                if password:
                    return WifiConnectError.WRONG_PASSWORD
                return WifiConnectError.NOT_JOINED
            time.sleep(self.poll)
        return WifiConnectError.NOT_JOINED

    def _forget(self, network, before):
        """Removes the network and enables those enabled before."""
        try:
            self.request("REMOVE_NETWORK", network)
            for other, flags in before.items():
                # Not those the user disabled; [TEMP-DISABLED] ones recover.
                if "[DISABLED]" not in flags:
                    self.request("ENABLE_NETWORK", other)
        except Unavailable:
            pass  # Restarting wpa_supplicant restores the saved config.
