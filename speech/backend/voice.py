"""
Speaks answers with Piper (https://github.com/OHF-Voice/piper1-gpl), a
neural text to speech system that runs on a Raspberry Pi, one sentence at a
time, so the first is heard while the next are synthesized.

Voices are .onnx files with their .onnx.json config beside them, see
`python -m piper.download_voices`.
"""

import json
import threading


class Voice:
    def __init__(self, path):
        """
        Arguments:
            path (str): A Piper voice's .onnx file, with its .onnx.json
                config beside it, read at once, so a wrong path fails here.
        """
        self.path = path
        with open(f"{path}.json") as f:
            self.sample_rate = json.load(f)["audio"]["sample_rate"]
        self._voice = None
        self._loading = threading.Lock()

    def load(self):
        """
        Loads the voice, unless it is loaded, which on a Raspberry Pi 4 takes
        some 9 seconds, all in creating its ONNX Runtime session.
        """
        with self._loading:
            if self._voice is None:
                # Imported here, as the service runs without a voice too.
                from piper import PiperVoice

                self._voice = PiperVoice.load(self.path)

    def sentences(self, text):
        """
        Yields the audio of each sentence of the text, as 16 bit
        little-endian mono samples at sample_rate, loading the voice first
        if it is not loaded.
        """
        self.load()
        for chunk in self._voice.synthesize(text):
            yield chunk.audio_int16_bytes
