import argparse
import math
import queue
import shutil
import sys
import time

import numpy as np
import sounddevice
import torch
from torch import nn

import speech
from speech import loader
from speech.utils import wave

# The window of the training features in milliseconds, see log_specgram.
WINDOW_MS = 20


class Transcriber:
    def __init__(self, model, preproc, sample_rate):
        """
        Transcribes a stream of audio chunk by chunk with greedy CTC
        decoding.
        """
        self.model = model
        self.preproc = preproc
        self.specgram = loader.SpecgramStream(sample_rate, window_size=WINDOW_MS)
        self.state = None
        self.labels = []
        self.prev = model.blank
        # The number of blank frames since the last label.
        self.blanks = 0

        # Each encoded frame covers the feature hop times the time stride
        # of the convolutions.
        stride = math.prod(c.stride[0] for c in model.conv if isinstance(c, nn.Conv2d))
        self.frame_seconds = stride * self.specgram.hop / sample_rate

    def push(self, audio, final=False):
        """
        Adds the next chunk of audio samples. Pass final with the last
        chunk to flush the frames held back for the lookahead.
        """
        frames = self.specgram.push(audio)
        frames = (frames - self.preproc.mean) / self.preproc.std
        frames = torch.from_numpy(frames).float().unsqueeze(0)
        probs, self.state = self.model.stream(frames, self.state, final)

        # Greedy CTC decoding: merge repeats, then drop blanks.
        for p in probs[0].argmax(dim=1).tolist():
            if p != self.prev and p != self.model.blank:
                self.labels.append(p)
            self.blanks = self.blanks + 1 if p == self.model.blank else 0
            self.prev = p

    @property
    def text(self):
        return "".join(self.preproc.decode(self.labels))

    @property
    def pause_seconds(self):
        """The time since the last label."""
        return self.blanks * self.frame_seconds

    def clear(self):
        """Starts a new transcript, keeping the stream going."""
        self.labels = []
        self.blanks = 0


def show(prefix, text, done=False):
    """
    Redraws the current line. Unless done, the text is cut from the
    left to fit the terminal, since a wrapped line cannot be redrawn.
    """
    if not done:
        width = shutil.get_terminal_size().columns - len(prefix) - 1
        if len(text) > width:
            text = "…" + text[-(width - 1) :]
    # \033[K clears the rest of the line from the previous draw.
    print(f"\r{prefix}{text}\033[K", end="\n" if done else "", flush=True)


def model_sample_rate(preproc):
    """
    The sample rate the model was trained on, from the number of
    frequency bins in each feature frame.
    """
    return round((preproc.input_dim - 1) * 2 * 1000 / WINDOW_MS)


def read_inputs(inputs, num):
    """
    Returns (audio file, reference text) pairs. Inputs ending in .json
    are datasets, from which the first num examples are taken.
    """
    examples = []
    for i in inputs:
        if i.endswith(".json"):
            data = loader.read_data_json(i)[:num]
            examples.extend((d["audio"], d["text"]) for d in data)
        else:
            examples.append((i, None))
    return examples


def transcribe_files(model, preproc, inputs, num, chunk_ms, realtime):
    """
    Feeds each audio file to the model in chunks of chunk_ms
    milliseconds, as a microphone would, and compares the result with
    decoding the whole file at once.
    """
    sample_rate = model_sample_rate(preproc)
    for audio_file, reference in read_inputs(inputs, num):
        print(audio_file)
        audio, file_rate = wave.array_from_wave(audio_file)
        if file_rate != sample_rate:
            sys.exit(
                f"{audio_file} is {file_rate} Hz, the model needs {sample_rate} Hz."
            )

        transcriber = Transcriber(model, preproc, sample_rate)
        chunk = int(sample_rate * chunk_ms / 1e3)
        compute = 0.0
        for start in range(0, len(audio), chunk):
            begin = time.time()
            final = start + chunk >= len(audio)
            transcriber.push(audio[start : start + chunk], final)
            compute += time.time() - begin

            seconds = min(start + chunk, len(audio)) / sample_rate
            show(f"{seconds:6.2f}s | ", transcriber.text, done=final)
            if realtime:
                time.sleep(max(chunk_ms / 1e3 - (time.time() - begin), 0))

        # Decoding the whole file at once must give the same result.
        features = preproc.preprocess(audio_file, "")[0]
        features = torch.from_numpy(features).unsqueeze(0)
        with torch.no_grad():
            probs = model.forward_impl(features, softmax=True)
        offline = model.max_decode(probs[0].argmax(dim=1).tolist(), model.blank)

        print(f"  offline:   {''.join(preproc.decode(offline))}")
        if reference is not None:
            cer = speech.compute_cer([(list(reference), list(transcriber.text))])
            print(f"  reference: {reference}")
            print(f"  CER {cer:.3f}")
        rtf = compute / (len(audio) / sample_rate)
        print(f"  compute per second of audio: {rtf:.3f}s")
        print()


def listen(model, preproc, chunk_ms, mic_device, pause):
    """
    Transcribes speech from the microphone as it arrives until Ctrl+C.
    A pause of the given seconds ends the current line.
    """
    sample_rate = model_sample_rate(preproc)
    chunks = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"\n{status}", file=sys.stderr)
        chunks.put(indata[:, 0].copy())

    transcriber = Transcriber(model, preproc, sample_rate)
    heard = 0
    silent = True
    stream = sounddevice.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=int(sample_rate * chunk_ms / 1e3),
        device=mic_device,
        callback=callback,
    )
    name = sounddevice.query_devices(stream.device)["name"]
    print(f"Listening on {name}, press Ctrl+C to stop.")
    with stream:
        try:
            while True:
                # Take everything queued, so a slow step never builds a
                # growing delay.
                audio = [chunks.get()]
                while not chunks.empty():
                    audio.append(chunks.get_nowait())
                audio = np.concatenate(audio)
                transcriber.push(audio)

                # macOS gives only zeros without microphone access.
                heard += len(audio)
                silent = silent and not audio.any()
                if silent and heard >= 2 * sample_rate:
                    print(
                        "\nThe microphone gives only silence. Check that your terminal "
                        "has access in System Settings > Privacy & Security > "
                        "Microphone.",
                        file=sys.stderr,
                    )
                    silent = False

                # Pauses can decode to spaces at the ends of a line.
                text = transcriber.text.strip()
                if text and transcriber.pause_seconds >= pause:
                    show("> ", text, done=True)
                    transcriber.clear()
                else:
                    show("> ", text)
        except KeyboardInterrupt:
            pass

    # Flush the frames held back for the lookahead, then finish the line,
    # or clear it if nothing was said.
    transcriber.push(np.zeros(0, dtype=np.int16), final=True)
    text = transcriber.text.strip()
    show("> " if text else "", text, done=True)


def run(args):
    model, preproc = speech.load(args.model, tag=None if args.last else "best")
    model.to(args.device)
    model.set_eval()

    if args.mic:
        listen(model, preproc, args.chunk_ms, args.mic_device, args.pause)
    else:
        transcribe_files(
            model, preproc, args.inputs, args.num, args.chunk_ms, args.realtime
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe speech chunk by chunk with a streaming CTC model, "
        "from audio files or live from the microphone."
    )
    parser.add_argument("model", help="A path to a stored model.")
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Audio files, or json datasets to take the first examples from.",
    )
    parser.add_argument(
        "--mic", action="store_true", help="Transcribe live from the microphone."
    )
    parser.add_argument(
        "--mic-device",
        help="The microphone's name or index, by default the system default. "
        "See `python -m sounddevice` for the list.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Seconds of silence that end a line when listening.",
    )
    parser.add_argument(
        "--num", type=int, default=3, help="Examples to take from each dataset."
    )
    parser.add_argument(
        "--chunk-ms", type=int, default=100, help="Audio chunk size in milliseconds."
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Wait for each file chunk as if it came from a microphone.",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default="cpu",
        help="The device to run the model on. Streaming does little work per "
        "chunk, so the CPU is usually fastest.",
    )
    parser.add_argument(
        "--last",
        action="store_true",
        help="Last saved model instead of best on dev set.",
    )
    args = parser.parse_args()
    if args.mic == bool(args.inputs):
        parser.error("give either audio inputs or --mic")
    if args.mic_device is not None and args.mic_device.isdigit():
        args.mic_device = int(args.mic_device)

    run(args)
