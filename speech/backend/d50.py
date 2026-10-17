"""
Roland D-50 patches, for songs played with the D-50's voice: the client's
synthesizer in web/assistant/synth.js plays a patch from its parameters, as
the D-50 stores them in its MIDI implementation, in a JSON file in
speech/backend/patches:

    {
      "name": "Fantasia",
      "source": "Roland D-50 factory patch I-11, PN-D50-00",
      "patch": {"key_mode": 1, "tone_balance": 78, ...},
      "upper": {"name": "Fantasynth", "common": {...}, "partial1": {...}, "partial2": {...}},
      "lower": {...}
    }

Each block has the parameters of PATCH, COMMON and PARTIAL, in the order
the D-50 stores them, with the values it stores, such as wg_coarse 36 for
C4 and tone_balance 100 for the upper tone alone. examples/songs/d50_patch.py
writes one from a D-50 bank dump.

The client plays a patch's partials as the D-50's LA chip makes them, but
not everything a D-50 does: the key modes Whole, Dual and Split, the
chorus types 1 and 2, the reverb types 1 to 10, the halls and rooms, and
the PCM waves it has stand-ins for, see PCM_WAVES in synth.js, and not a
TVA end level of 100, which never lets a note end. load() refuses a patch
needing more.
"""

import json
import os

DIRECTORY = os.path.join(os.path.dirname(__file__), "patches")

# The parameters of a patch's blocks, in the order the D-50 stores them,
# with their highest values; the lowest is 0.
PARTIAL = {
    "wg_coarse": 72,
    "wg_fine": 100,
    "wg_keyfollow": 16,
    "wg_lfo_mode": 3,
    "wg_penv_mode": 2,
    "wg_bend_mode": 2,
    "wg_waveform": 1,
    "wg_pcm": 99,
    "pw": 100,
    "pw_velocity": 14,
    "pw_lfo_select": 5,
    "pw_lfo_depth": 100,
    "pw_aftertouch": 14,
    "tvf_cutoff": 100,
    "tvf_resonance": 30,
    "tvf_keyfollow": 14,
    "tvf_bias_point": 127,
    "tvf_bias_level": 14,
    "tvf_env_depth": 100,
    "tvf_env_velocity": 100,
    "tvf_env_depth_kf": 4,
    "tvf_env_time_kf": 4,
    "tvf_t1": 100,
    "tvf_t2": 100,
    "tvf_t3": 100,
    "tvf_t4": 100,
    "tvf_t5": 100,
    "tvf_l1": 100,
    "tvf_l2": 100,
    "tvf_l3": 100,
    "tvf_sus": 100,
    "tvf_end": 1,
    "tvf_lfo_select": 5,
    "tvf_lfo_depth": 100,
    "tvf_aftertouch": 14,
    "tva_level": 100,
    "tva_velocity": 100,
    "tva_bias_point": 127,
    "tva_bias_level": 12,
    "tva_t1": 100,
    "tva_t2": 100,
    "tva_t3": 100,
    "tva_t4": 100,
    "tva_t5": 100,
    "tva_l1": 100,
    "tva_l2": 100,
    "tva_l3": 100,
    "tva_sus": 100,
    "tva_end": 1,
    "tva_velocity_t1": 4,
    "tva_time_kf": 4,
    "tva_lfo_select": 5,
    "tva_lfo_depth": 100,
    "tva_aftertouch": 14,
}
# After the tone's name, of 10 characters.
COMMON = {
    "structure": 6,
    "penv_velocity": 2,
    "penv_time_kf": 4,
    "penv_t1": 50,
    "penv_t2": 50,
    "penv_t3": 50,
    "penv_t4": 50,
    "penv_l0": 100,
    "penv_l1": 100,
    "penv_l2": 100,
    "penv_sus": 100,
    "penv_end": 100,
    "pitch_lfo_depth": 100,
    "pitch_lever": 100,
    "pitch_aftertouch": 100,
    "lfo1_wave": 3,
    "lfo1_rate": 100,
    "lfo1_delay": 100,
    "lfo1_sync": 2,
    "lfo2_wave": 3,
    "lfo2_rate": 100,
    "lfo2_delay": 100,
    "lfo2_sync": 1,
    "lfo3_wave": 3,
    "lfo3_rate": 100,
    "lfo3_delay": 100,
    "lfo3_sync": 1,
    "eq_lf": 15,
    "eq_lg": 24,
    "eq_hf": 21,
    "eq_hq": 8,
    "eq_hg": 24,
    "chorus_type": 7,
    "chorus_rate": 100,
    "chorus_depth": 100,
    "chorus_balance": 100,
    "partial_mute": 3,
    "partial_balance": 100,
}
# After the patch's name, of 18 characters.
PATCH = {
    "key_mode": 8,
    "split_point": 60,
    "portamento_mode": 2,
    "hold_mode": 2,
    "upper_key_shift": 48,
    "lower_key_shift": 48,
    "upper_fine": 100,
    "lower_fine": 100,
    "bender_range": 12,
    "aftertouch_bend": 24,
    "portamento_time": 100,
    "output_mode": 3,
    "reverb_type": 31,
    "reverb_balance": 100,
    "total_volume": 100,
    "tone_balance": 100,
    "chase_mode": 2,
    "chase_level": 100,
    "chase_time": 100,
}

# The characters of names.
CHARACTERS = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890-"
# A patch's bytes: the upper tone's partials and common block, the lower
# tone's, and the patch block, of 64 bytes each.
BLOCK = 64
PATCH_BYTES = 7 * BLOCK
# Where the patches start in the D-50's memory, 02-00-00, and how many.
MEMORY_START = 0x02 << 14
PATCHES = 64

# What the client plays, see the docstring.
KEY_MODES = {0: "Whole", 1: "Dual", 2: "Split"}
CHORUS_TYPES = (0, 1)
REVERB_TYPES = range(10)
PCM_WAVES = {13: "Bells", 69: "Spectrum 2"}
# The structures 1 to 7: whether partials 1 and 2 are PCM.
STRUCTURES = (
    (False, False),
    (False, False),
    (True, False),
    (True, False),
    (False, True),
    (True, True),
    (True, True),
)
# The structures whose partial 2 ring-modulates partial 1, from 0.
RING_STRUCTURES = (1, 3, 4, 6)


def name(data):
    return "".join(CHARACTERS[b] for b in data).rstrip()


def block(data, parameters, start=0):
    return dict(zip(parameters, data[start : start + len(parameters)]))


def decode(data):
    """A patch from its 448 bytes, as stored in the D-50's memory."""
    patch = {
        "name": name(data[6 * BLOCK : 6 * BLOCK + 18]),
        "patch": block(data[6 * BLOCK : 7 * BLOCK], PATCH, 18),
    }
    for tone, start in (("upper", 0), ("lower", 3 * BLOCK)):
        common = data[start + 2 * BLOCK : start + 3 * BLOCK]
        patch[tone] = {
            "name": name(common[:10]),
            "common": block(common, COMMON, 10),
            "partial1": block(data[start : start + BLOCK], PARTIAL),
            "partial2": block(data[start + BLOCK : start + 2 * BLOCK], PARTIAL),
        }
    return patch


def memory(sysex):
    """
    The D-50's memory of patches in a bank dump, its exclusive messages of
    data set 1 from 02-00-00, raising ValueError for a bad one.
    """
    memory = bytearray(PATCHES * PATCH_BYTES)
    seen = 0
    for message in sysex.split(b"\xf0")[1:]:
        end = message.find(b"\xf7")
        if end < 0 or message[:4] != b"\x41" + message[1:2] + b"\x14\x12":
            raise ValueError("Not a D-50 bank dump: a message is not one of its data")
        body = message[4:end]
        if sum(body) % 128:
            raise ValueError("A message of the bank dump has a bad checksum")
        address = (body[0] << 14) | (body[1] << 7) | body[2]
        data = body[3:-1]
        offset = address - MEMORY_START
        if 0 <= offset < len(memory):
            data = data[: len(memory) - offset]
            memory[offset : offset + len(data)] = data
            seen += len(data)
    if seen < len(memory):
        raise ValueError("The bank dump does not have all 64 patches")
    return bytes(memory)


def bank(sysex):
    """The 64 patches of a D-50 bank dump, I-11 to I-88."""
    data = memory(sysex)
    return [
        decode(data[n * PATCH_BYTES : (n + 1) * PATCH_BYTES]) for n in range(PATCHES)
    ]


def check_block(values, parameters, where):
    if set(values) != set(parameters):
        raise ValueError(f"{where} needs exactly the parameters {list(parameters)}")
    for parameter, highest in parameters.items():
        value = values[parameter]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= highest
        ):
            raise ValueError(f"{where}: {parameter} is 0 to {highest}, not {value!r}")


def check(patch):
    """The patch, raising ValueError for one the client cannot play."""
    for key in ("name", "patch", "upper", "lower"):
        if key not in patch:
            raise ValueError(f"A patch needs {key}")
    check_block(patch["patch"], PATCH, "patch")
    settings = patch["patch"]
    if settings["key_mode"] not in KEY_MODES:
        raise ValueError(f"Key mode {settings['key_mode']} is not one of {KEY_MODES}")
    if settings["reverb_type"] not in REVERB_TYPES:
        raise ValueError(
            f"Reverb type {settings['reverb_type'] + 1} is not a hall or room, 1 to 10"
        )
    for tone in ("upper", "lower"):
        common = patch[tone]["common"]
        check_block(common, COMMON, f"{tone} common")
        if common["chorus_type"] not in CHORUS_TYPES:
            raise ValueError(
                f"{tone}: chorus type {common['chorus_type'] + 1} is not 1 or 2"
            )
        for n, pcm in zip((1, 2), STRUCTURES[common["structure"]]):
            partial = patch[tone][f"partial{n}"]
            check_block(partial, PARTIAL, f"{tone} partial {n}")
            # A muted partial does not sound, but rings a ring modulator.
            if (
                not common["partial_mute"] & n
                and common["structure"] not in RING_STRUCTURES
            ):
                continue
            if pcm and partial["wg_pcm"] + 1 not in PCM_WAVES:
                raise ValueError(
                    f"{tone} partial {n}: PCM wave {partial['wg_pcm'] + 1} has no "
                    f"stand-in, only {PCM_WAVES}"
                )
            if partial["tva_end"]:
                raise ValueError(
                    f"{tone} partial {n}: a TVA end level of 100 never ends"
                )
    return patch


def load(patch_name, directory=DIRECTORY):
    """The patch of the name, its file's without .json, raising ValueError for a bad one."""
    path = os.path.join(directory, f"{patch_name}.json")
    if os.path.basename(path) != f"{patch_name}.json" or not os.path.isfile(path):
        raise ValueError(f"No D-50 patch {patch_name!r} in {directory}")
    with open(path, encoding="utf-8") as f:
        try:
            return check(json.load(f))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: {error}") from None
