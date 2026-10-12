"""
Shared steps of preparing a speech dataset for training: normalizing its
transcripts to the characters of the models and converting its audio to
16 kHz FLAC.
"""

import math
import os
import re
import unicodedata

import numpy as np
import soundfile
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
ALLOWED = re.compile(r"[a-z' ]+")
APOSTROPHES = re.compile(r"[’‘`´ʼ]")
DASHES = re.compile(r"[-‐‑‒–—]")
PUNCTUATION = re.compile(r"[.,!?;:\"“”„«»()\[\]{}…¿¡/*&]")


def normalize(sentence):
    """
    Returns the sentence as lowercase letters, apostrophes and single
    spaces, or None if it has other characters, such as digits, that
    would need a guess to write out.
    """
    # Split accented letters and ligatures, then drop the accents.
    text = unicodedata.normalize("NFKD", sentence)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = APOSTROPHES.sub("'", text)
    text = DASHES.sub(" ", text)
    text = PUNCTUATION.sub(" ", text)
    words = []
    for word in text.split():
        # Keep apostrophes inside words and after a plural s, as in
        # "don't" and "the students' books", and drop quoting ones.
        word = word.lstrip("'")
        if not word.endswith("s'"):
            word = word.rstrip("'")
        if word:
            words.append(word)
    text = " ".join(words)
    if not text or not ALLOWED.fullmatch(text):
        return None
    return text


def to_flac(audio, rate, target):
    """
    Writes the audio, samples by channels or mono at the rate, to target as
    16 kHz mono FLAC, through a temporary file so an interrupted write
    never leaves a broken clip, and returns its duration in seconds.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        g = math.gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, rate // g)
    audio = np.clip(audio, -1, 1)
    tmp = target + ".tmp"
    soundfile.write(tmp, audio, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    os.replace(tmp, target)
    return len(audio) / SAMPLE_RATE


def convert(job):
    """
    Converts a clip to 16 kHz mono FLAC and returns its duration in
    seconds, or None if it cannot be read. Converted clips are kept, so
    an interrupted run can continue.
    """
    source, target = job
    try:
        if os.path.exists(target):
            info = soundfile.info(target)
            return info.frames / info.samplerate
        audio, rate = soundfile.read(source, dtype="float32", always_2d=True)
        return to_flac(audio, rate, target)
    except (RuntimeError, soundfile.LibsndfileError):
        return None


def audio_info(path):
    """
    The duration of a clip in seconds and whether it is 16 kHz mono, so
    usable as it is, or None if it cannot be read.
    """
    try:
        info = soundfile.info(path)
    except (RuntimeError, soundfile.LibsndfileError):
        return None
    ready = info.samplerate == SAMPLE_RATE and info.channels == 1
    return info.frames / info.samplerate, ready
