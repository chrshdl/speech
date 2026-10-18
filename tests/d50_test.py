import copy
import json
import os
import subprocess
import sys

import pytest

from speech.backend import d50

TOOL = os.path.join(
    os.path.dirname(__file__), "..", "examples", "songs", "d50_patch.py"
)


def encode_name(name, length):
    return bytes(d50.CHARACTERS.index(c) for c in name.ljust(length))


def encode(patch):
    """A patch's 448 bytes, as stored in the D-50's memory."""
    data = bytearray()
    for tone in ("upper", "lower"):
        for partial in ("partial1", "partial2"):
            data += bytes(patch[tone][partial][p] for p in d50.PARTIAL).ljust(64, b"\0")
        common = patch[tone]["common"]
        data += (
            encode_name(patch[tone]["name"], 10) + bytes(common[p] for p in d50.COMMON)
        ).ljust(64, b"\0")
    settings = bytes(patch["patch"][p] for p in d50.PATCH)
    data += (encode_name(patch["name"], 18) + settings).ljust(64, b"\0")
    return bytes(data)


def bank_dump(patches, device=0):
    """A bank dump of the patches, in data set 1 messages of 256 bytes."""
    memory = b"".join(encode(patch) for patch in patches)
    dump = bytearray()
    for offset in range(0, len(memory), 256):
        address = d50.MEMORY_START + offset
        body = bytes([address >> 14, (address >> 7) & 127, address & 127])
        body += memory[offset : offset + 256]
        dump += bytes([0xF0, 0x41, device, 0x14, 0x12]) + body
        dump += bytes([-sum(body) % 128, 0xF7])
    return bytes(dump)


@pytest.fixture
def fantasia():
    # As a bank dump has it: without where it is from and how it was changed.
    patch = d50.load("fantasia")
    del patch["source"], patch["changes"]
    return patch


def test_fantasia(fantasia):
    assert (fantasia["name"], fantasia["upper"]["name"], fantasia["lower"]["name"]) == (
        "Fantasia",
        "Fantasynth",
        "FantasyPCM",
    )
    # Dual: both tones sound, and its lower tone plays two PCM waves,
    # Bells and Spectrum 2.
    assert fantasia["patch"]["key_mode"] == 1
    lower = fantasia["lower"]
    assert d50.STRUCTURES[lower["common"]["structure"]] == (True, True)
    assert (lower["partial1"]["wg_pcm"] + 1, lower["partial2"]["wg_pcm"] + 1) == (
        13,
        69,
    )


def test_decodes_a_bank(fantasia):
    other = copy.deepcopy(fantasia)
    other["name"] = "Other"
    other["upper"]["partial1"]["tvf_cutoff"] = 99
    patches = d50.bank(bank_dump([fantasia, other] * 32, device=5))
    assert len(patches) == 64
    assert patches[0] == fantasia
    assert patches[1] == other


def test_bad_bank(fantasia):
    dump = bytearray(bank_dump([fantasia] * 64))
    dump[100] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        d50.bank(bytes(dump))
    with pytest.raises(ValueError, match="all 64"):
        d50.bank(bank_dump([fantasia] * 32))
    with pytest.raises(ValueError, match="Not a D-50"):
        d50.bank(b"\xf0\x43\x00\x09\x20\x00\xf7")


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda p: p["patch"].update(key_mode=3), "Key mode"),
        (lambda p: p["patch"].update(reverb_type=10), "Reverb type 11"),
        (lambda p: p["upper"]["common"].update(chorus_type=2), "chorus type 3"),
        (lambda p: p["lower"]["partial1"].update(wg_pcm=19), "PCM wave 20"),
        (lambda p: p["upper"]["partial2"].update(tva_end=1), "never ends"),
        (
            lambda p: p["upper"]["partial1"].update(tvf_cutoff=101),
            "tvf_cutoff is 0 to 100",
        ),
        (lambda p: p["upper"]["partial1"].update(tvf_cutoff=True), "tvf_cutoff"),
        (lambda p: p["upper"]["partial1"].pop("pw"), "exactly the parameters"),
        (lambda p: p.pop("lower"), "needs lower"),
    ],
)
def test_check_refuses(fantasia, change, error):
    change(fantasia)
    with pytest.raises(ValueError, match=error):
        d50.check(fantasia)


def test_muted_partial_is_not_checked(fantasia):
    # Partial 1 of the lower tone muted, its PCM wave never sounds.
    fantasia["lower"]["partial1"]["wg_pcm"] = 19
    fantasia["lower"]["common"]["partial_mute"] = 2
    d50.check(fantasia)


def test_load(tmp_path, fantasia):
    (tmp_path / "bad.json").write_text(json.dumps(fantasia | {"patch": {}}))
    with pytest.raises(ValueError, match="bad.json"):
        d50.load("bad", tmp_path)
    for name in ("missing", "../patches/fantasia", ""):
        with pytest.raises(ValueError, match="No D-50 patch"):
            d50.load(name)


def run_tool(tmp_path, *args):
    return subprocess.run(
        [sys.executable, TOOL, *args],
        capture_output=True,
        text=True,
        check=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    ).stdout


def test_tool_writes_a_patch_file(tmp_path, fantasia):
    other = copy.deepcopy(fantasia) | {"name": "Other"}
    (tmp_path / "bank.syx").write_bytes(bank_dump([other] * 9 + [fantasia] * 55))
    out = run_tool(tmp_path, str(tmp_path / "bank.syx"), "22", "--source", "A test")
    (tmp_path / "fantasia.json").write_text(out)
    assert d50.load("fantasia", tmp_path) == fantasia | {"source": "A test"}
    listing = run_tool(tmp_path, str(tmp_path / "bank.syx")).splitlines()
    assert listing[:2] == ["11 Other", "12 Other"]
    assert listing[9] == "22 Fantasia"
    assert len(listing) == 64
