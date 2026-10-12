"""
Prepares My Science Tutor (MyST), children's conversational speech: about
400 hours of third to fifth graders talking with a virtual science tutor,
of which about 45% of the utterances are transcribed. Writes its train, dev
and test sets of the transcribed utterances, whose official splits keep
each student in one of them. MyST is released under CC BY-NC-SA 4.0 for
non-commercial use, and distributed by the LDC as LDC2021S05.

The transcripts are verbatim, with markup for what is not plain speech:
labels for a whole utterance, such as <<DISCARD>>, <<NO_SIGNAL>> and
<<SILENCE>>, tags in angle or square brackets for sounds and other events,
such as <noise> and <laugh>, and double parentheses around what could not
be understood. An utterance with a label, or with speech the transcript
leaves out, is dropped, since the model would learn to skip words it
hears. The tags of sounds are removed. Partial words keep their letters,
and filled pauses, such as "um", stay as they are said. The rest is
normalized as the other datasets, and an utterance with digits or other
characters is dropped. The summary lists every tag found, to check these
rules against the corpus.
"""

import argparse
import collections
import json
import multiprocessing
import os
import re

import tqdm

from speech.utils.prepare import audio_info, convert, normalize

# The corpus's partitions and the sets they become.
SPLITS = {"train": "train", "development": "dev", "test": "test"}
# Utterances longer than this are dropped, as in the MyST paper's recipes.
MAX_SECONDS = 30.0

# A label for the whole utterance, such as <<DISCARD>>.
LABEL = re.compile(r"<<[^<>]*>>")
# Speech the transcript leaves out: double parentheses around what could
# not be understood, and tags for such speech.
OMITTED = re.compile(r"\(\([^()]*\)\)")
# A tag, in angle or square brackets, with an optional slash for one that
# closes a span, such as <whisper> ... </whisper>.
TAG = re.compile(r"[<\[]\s*/?\s*([^<>\[\]]*?)\s*[>\]]")
# Tags of sounds and of how the words are said, removed with their brackets.
# The words of a span, such as a whisper, are kept.
SOUNDS = {
    "noise", "laugh", "laughter", "breath", "breathing", "cough", "sneeze",
    "sniff", "lipsmack", "lip_smack", "click", "silence", "sil", "pause",
    "static", "beep", "music", "sigh", "inhale", "exhale", "background",
    "background_noise", "side_speech", "crosstalk", "whisper", "whispering",
    "singing", "sing", "yawn", "clap", "hum", "humming",
}  # fmt: skip
# Tags of speech the transcript leaves out, which drop the utterance.
UNTRANSCRIBED = {
    "indiscernible", "unintelligible", "unclear", "inaudible", "foreign",
    "unk", "unknown", "babble",
}  # fmt: skip


def clean(transcript):
    """
    Returns the normalized text of a transcript and None, or None and the
    reason it is dropped, and the tags it has.
    """
    tags = [t.lower() for t in TAG.findall(LABEL.sub(" ", transcript))]
    tags += LABEL.findall(transcript)
    if LABEL.search(transcript):
        return None, "label", tags
    if OMITTED.search(transcript):
        tags.append("(( ))")
        return None, "untranscribed speech", tags
    for tag in tags:
        name = tag.replace(" ", "_")
        if name in UNTRANSCRIBED:
            return None, "untranscribed speech", tags
        if name not in SOUNDS:
            return None, "unknown tag", tags
    text = normalize(TAG.sub(" ", transcript))
    if text is None:
        return None, "text", tags
    return text, None, tags


def find_corpus(root):
    """The directory with the partitions, root or its data directory."""
    for candidate in (root, os.path.join(root, "data")):
        if all(os.path.isdir(os.path.join(candidate, p)) for p in SPLITS):
            return candidate
    raise SystemExit(
        f"Found no train, development and test directories in {root} or {root}/data."
    )


def utterances(partition):
    """Yields the audio path and transcript path of each utterance."""
    for folder, _, names in os.walk(partition):
        for name in sorted(names):
            if name.endswith(".flac"):
                audio = os.path.join(folder, name)
                yield audio, os.path.splitext(audio)[0] + ".trn"


def prepare(myst_dir, out_dir, max_seconds=MAX_SECONDS, workers=None):
    """
    Writes <out_dir>/<set>.json for the transcribed utterances of each
    partition. Clips already 16 kHz mono are used where they are, others
    are converted to FLAC in <out_dir>/clips. Returns the clips, hours and
    dropped utterances of each set, and the tags found.
    """
    corpus = find_corpus(myst_dir)
    clip_dir = os.path.join(out_dir, "clips")
    os.makedirs(clip_dir, exist_ok=True)
    summary = {}
    tags = collections.Counter()
    with multiprocessing.Pool(workers) as pool:
        for partition, split in SPLITS.items():
            dropped = collections.Counter()
            texts, audios = [], []
            for audio, trn in utterances(os.path.join(corpus, partition)):
                if not os.path.exists(trn):
                    dropped["not transcribed"] += 1
                    continue
                with open(trn, encoding="utf-8", errors="replace") as fid:
                    text, reason, found = clean(fid.read())
                tags.update(found)
                if text is None:
                    dropped[reason] += 1
                    continue
                texts.append(text)
                audios.append(audio)
            infos = list(
                tqdm.tqdm(
                    pool.imap(audio_info, audios, chunksize=64),
                    total=len(audios),
                    desc=split,
                )
            )
            data, jobs = [], []
            for text, audio, info in zip(texts, audios, infos):
                if info is None:
                    dropped["audio"] += 1
                    continue
                duration, ready = info
                if duration > max_seconds:
                    dropped["too long"] += 1
                    continue
                if not ready:
                    name = os.path.relpath(audio, corpus).replace(os.sep, "_")
                    target = os.path.join(clip_dir, name)
                    jobs.append((len(data), (audio, target)))
                    audio = target
                data.append({"text": text, "duration": duration, "audio": audio})
            converted = pool.imap(convert, [job for _, job in jobs], chunksize=16)
            for (i, _), duration in zip(jobs, converted):
                if duration is None:
                    data[i] = None
                    dropped["audio"] += 1
            data = [d for d in data if d is not None]
            with open(os.path.join(out_dir, f"{split}.json"), "w") as fid:
                fid.writelines(json.dumps(d) + "\n" for d in data)
            hours = sum(d["duration"] for d in data) / 3600
            summary[split] = {
                "clips": len(data),
                "hours": hours,
                "dropped": dict(dropped),
            }
    return summary, tags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "myst_dir",
        help="The extracted LDC2021S05 download, with the train, development and "
        "test directories in it or in its data directory.",
    )
    parser.add_argument(
        "out_dir",
        help="Where to write the sets, and clips that need converting, such as "
        "examples/myst/data.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=MAX_SECONDS,
        help="Drop longer utterances.",
    )
    parser.add_argument(
        "--workers", type=int, help="Processes reading audio, by default one per core."
    )
    args = parser.parse_args()
    summary, tags = prepare(args.myst_dir, args.out_dir, args.max_seconds, args.workers)
    for split, s in summary.items():
        print(
            f"{split}: {s['clips']} clips, {s['hours']:.1f} hours, dropped {s['dropped']}"
        )
    print("Tags found:", dict(tags.most_common()))
