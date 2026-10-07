import argparse
import queue
import shutil
import sys
import time

import numpy as np
import sounddevice
import torch

import speech
from speech import loader, streaming
from speech.models.ctc_decoder import BEAM_SIZE, PRUNE, decode
from speech.models.word_lm import LM_WEIGHT, UNK_PENALTY, WORD_BONUS
from speech.streaming import Transcriber, model_sample_rate
from speech.utils import wave


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


def transcribe_files(model, preproc, inputs, num, chunk_ms, realtime, new_search):
    """
    Feeds each audio file to the model in chunks of chunk_ms
    milliseconds, as a microphone would, and compares the result with
    decoding the whole file at once. new_search() returns the beam search
    for a file, or None to decode greedily.
    """
    sample_rate = model_sample_rate(preproc)
    for audio_file, reference in read_inputs(inputs, num):
        print(audio_file)
        audio, file_rate = wave.array_from_wave(audio_file)
        if file_rate != sample_rate:
            sys.exit(
                f"{audio_file} is {file_rate} Hz, the model needs {sample_rate} Hz."
            )

        search = new_search()
        transcriber = Transcriber(model, preproc, sample_rate, search)
        chunk = int(sample_rate * chunk_ms / 1e3)
        compute = 0.0
        for start in range(0, len(audio), chunk):
            begin = time.time()
            final = start + chunk >= len(audio)
            transcriber.push(audio[start : start + chunk], final)
            compute += time.time() - begin

            seconds = min(start + chunk, len(audio)) / sample_rate
            show(f"{seconds:6.2f}s | ", transcriber.text(final), done=final)
            if realtime:
                time.sleep(max(chunk_ms / 1e3 - (time.time() - begin), 0))

        # Decoding the whole file at once must give the same result.
        features = preproc.preprocess(audio_file, "")[0]
        features = torch.from_numpy(features).unsqueeze(0)
        with torch.no_grad():
            probs = model.forward_impl(features, softmax=True)[0].numpy()
        if search is None:
            offline = model.max_decode(probs.argmax(axis=1).tolist(), model.blank)
        else:
            offline = decode(
                probs, search.beam_size, search.blank, search.lm, search.prune
            )[0]

        print(f"  offline:   {''.join(preproc.decode(offline))}")
        if reference is not None:
            result = [(list(reference), list(transcriber.text(final=True)))]
            cer = speech.compute_cer(result)
            wer = speech.compute_wer(result)
            print(f"  reference: {reference}")
            print(f"  CER {cer:.3f} WER {wer:.3f}")
        rtf = compute / (len(audio) / sample_rate)
        print(f"  compute per second of audio: {rtf:.3f}s")
        print()


def listen(model, preproc, chunk_ms, mic_device, pause, new_search):
    """
    Transcribes speech from the microphone as it arrives until Ctrl+C.
    A pause of the given seconds ends the current line. new_search()
    returns the beam search, or None to decode greedily.
    """
    sample_rate = model_sample_rate(preproc)
    chunks = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"\n{status}", file=sys.stderr)
        chunks.put(indata[:, 0].copy())

    transcriber = Transcriber(model, preproc, sample_rate, new_search())
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

                line = transcriber.take_line(pause)
                if line is not None:
                    show("> ", line, done=True)
                else:
                    show("> ", transcriber.text().strip())
        except KeyboardInterrupt:
            pass

    # Flush the frames held back for the lookahead, then finish the line,
    # or clear it if nothing was said.
    transcriber.push(np.zeros(0, dtype=np.int16), final=True)
    text = transcriber.text(final=True).strip()
    show("> " if text else "", text, done=True)


def run(args):
    model, preproc = speech.load(args.model, tag=None if args.last else "best")
    model.to(args.device)
    model.set_eval()
    new_search = streaming.search_factory(
        model.blank,
        preproc.char_to_int,
        args.lm,
        args.beam_size,
        args.prune,
        args.lm_weight,
        args.word_bonus,
        args.unk_penalty,
    )

    if args.mic:
        listen(model, preproc, args.chunk_ms, args.mic_device, args.pause, new_search)
    else:
        transcribe_files(
            model,
            preproc,
            args.inputs,
            args.num,
            args.chunk_ms,
            args.realtime,
            new_search,
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
    decoding = parser.add_argument_group("decoding")
    decoding.add_argument(
        "--lm", help="A word LM .npz file from speech.models.word_lm."
    )
    decoding.add_argument(
        "--beam-size",
        type=int,
        help=f"Beam size for the prefix beam search, by default {BEAM_SIZE} with "
        "an LM. Without an LM a beam size of 1 decodes greedily, the default.",
    )
    decoding.add_argument(
        "--prune",
        type=float,
        default=PRUNE,
        help="Skip labels with a lower log probability in a frame.",
    )
    decoding.add_argument(
        "--lm-weight", type=float, default=LM_WEIGHT, help="Scales the LM scores."
    )
    decoding.add_argument(
        "--word-bonus",
        type=float,
        default=WORD_BONUS,
        help="Score added per word, which offsets the LM's preference for fewer words.",
    )
    decoding.add_argument(
        "--unk-penalty",
        type=float,
        default=UNK_PENALTY,
        help="Score added per word outside the LM's vocabulary.",
    )
    args = parser.parse_args()
    if args.mic == bool(args.inputs):
        parser.error("give either audio inputs or --mic")
    if args.mic_device is not None and args.mic_device.isdigit():
        args.mic_device = int(args.mic_device)

    run(args)
