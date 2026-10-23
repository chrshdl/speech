A wake word, "olivia", that starts a capture of `backend.py` without a
tap, as in "olivia, what's the weather in hamburg?". The model is the
depthwise separable CNN (DS-CNN) of [Hello Edge: Keyword Spotting on
Microcontrollers] (Zhang et al., 2017), trained on the Olivia dataset of
[mindorii/kws].

## How it works

`backend.py --wake-word model.onnx --vad silero_vad.onnx` loads the
wake word model, the voice activity detector and the speech model when it
starts, and the service listens to its microphone all along, in stages:
voice activity, then the wake word, then speech recognition.

1. **Ring buffer.** The microphone's audio, in chunks of 100 ms, always
   goes to a ring buffer of the last 2 seconds,
   `--wake-word-lookback`.
2. **Voice activity, while `IDLE`.** [Silero VAD] gives the probability
   of speech of each 32 ms frame. Speech starts when 7 of the last 10
   frames are speech, and ends when at most 3 are, so no single frame
   starts or ends it. Other sounds, such as music, noise or a slammed
   door, never reach the wake word model, which does not run in silence.
3. **Wake word, while speech goes on.** When speech starts, the wake word
   model takes the last second of audio, which holds its start, and then
   the audio as it comes, until a second after the speech ended. Every
   100 ms, the DS-CNN classifies the last second of audio as silence,
   other speech or "olivia", from 10 MFCCs of 40 ms frames every 20 ms.
   The wake word is heard when its probability, averaged over the last 3
   decisions, reaches `--wake-word-threshold`. A decision takes 3 ms on
   a Raspberry Pi 4, see [Speed](#speed). Without `--vad`, the wake word
   model runs on all the audio.
4. **Capture.** The service starts a `WAKE_WORD` capture at once, so the
   client chimes and shows "Listening…" as the wake word is said, and
   transcribes the ring buffer, which holds the wake word and anything
   said right after it, with the speech model, already loaded, so
   nothing said in one breath with the wake word is lost. Live audio
   follows.
5. **Confirmation.** Once 1 second of audio after the wake word was
   heard is transcribed, the wake word is aligned to the speech model's
   output, as intents are, see `speech/intents.py`. The Detector often
   hears it before its end, so the second covers the rest of it. If it
   fits well enough, the transcript starts after it. If not, the wake
   word was heard by mistake, and the capture ends at once with
   `NO_WAKE_WORD`, clearing its card, without an answer, and the client
   skips its end chime. Commercial assistants confirm their wake word
   in the same way, in the cloud. `--wake-word-asr-threshold` sets how
   well it must fit.
6. **End of speech.** The capture ends as a `TAP_TO_TALK` one does, once
   the transcript says something after the wake word and the voice
   activity detector hears silence for `--end-of-speech`, 1 second.
   Without `--vad`, the speech model's own output hears the silence:
   blank frames after text. The service then answers and returns to
   `IDLE`, where it waits for speech to start again. A capture that
   hears nothing after the wake word ends quietly, without a notice.

The wake word model alone would take some other speech for the wake
word. The speech model's confirmation removes most of those false
alarms, so the wake word model can be set to miss fewer wake words, see
[Results](#results).

## Setup

Download the dataset's `olivia.zip` from [mindorii/kws], 734 MB, and
prepare it with a streaming CTC model of characters, which finds where
each clip says the word:

```
uv run examples/olivia/prepare.py olivia.zip \
    examples/commonvoice/models/ctc_streaming_3750h_stride3_epoch2/model-int8.onnx
```

This writes the clips at 16 kHz to `examples/olivia/data/clips` and
`train.json`, `dev.json` and `test.json`, each clip with where the word
starts and ends. The clips were recorded by many people on their own
devices, at 44.1 kHz, most of them saying "olivia" once:

| Set | Clips | Hours | Dropped |
|---|---|---|---|
| train | 1,297 | 0.62 | 91 empty or too short, 120 not heard, 8 longer than a second, 48 longer than 3 seconds |
| dev | 144 | 0.08 | |
| test | 577 | 0.37 | 88 empty or too short, 80 not heard, 4 longer than a second |

- Some clips are silent, cut off or say something else. A clip is kept
  when the CTC model's alignment of "olivia" scores at least -2.5 per
  label, see `--min-score`. The 577 test clips are close to the 550 that
  [mindorii/kws] counts.
- Training clips longer than 3 seconds often say the word several times,
  whose other repetitions would be taken for other speech.
- The clips have no speaker ids. The last 10% of the training clips by
  number are held out as dev, as neighboring numbers likely share a
  speaker.
- The dataset's `olivia_train_noise` holds the training clips with noise
  added, which `train.py` adds itself.

The dataset's few speakers are not enough: a model trained on it alone
missed a third of other voices, see [Results](#results). So
`synthesize.py` says "olivia", and 25 words that sound like it, such as
"oliver", "bolivia" and "trivia", in many voices with [Piper], as
[openWakeWord] trains its models: in the 904 voices of
`en_US-libritts_r-medium` for training, and in the 109 of
`en_GB-vctk-medium`, other speakers and accents, for testing. Each
"olivia" is found and checked as `prepare.py` does.

```
uv run python -m piper.download_voices --download-dir examples/olivia/data/voices \
    en_US-libritts_r-medium en_GB-vctk-medium
A=examples/commonvoice/models/ctc_streaming_3750h_stride3_epoch2/model-int8.onnx
uv run examples/olivia/synthesize.py examples/olivia/data/voices/en_US-libritts_r-medium.onnx \
    $A synth-train --keyword 3000 --words 3000
uv run examples/olivia/synthesize.py examples/olivia/data/voices/en_GB-vctk-medium.onnx \
    $A synth-test --keyword 600 --words 1000 --seed 1
```

| Set | "olivia" | Not heard | Similar words |
|---|---|---|---|
| synth-train, US voices | 2,993 | 7 | 3,000 |
| synth-test, British voices | 492 | 108 | 1,000 |

The voice activity detector is Silero VAD's model of version 5.1.2, under
the MIT license, 2.3 MB:

```
curl -L -o examples/olivia/data/vad/silero_vad.onnx --create-dirs \
    https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx
```

Training also takes other speech, music and noise from
[Speech Commands](../speech_commands/README.md), LibriSpeech's
train-clean-100 and dev-clean, see [LibriSpeech](../librispeech), and
[MUSAN](../musan/README.md). Prepare those first.

## Train

```
D=examples/olivia/data; M=examples/musan/data
uv run examples/olivia/train.py examples/olivia/models/ds_cnn_s \
    --keyword $D/train.json $D/synth-train.json --similar $D/synth-train-words.json \
    --speech examples/librispeech/data/LibriSpeech/train-clean-100.json $M/speech-train.json \
    --music $M/music-train.json --noise $M/noise-train.json
```

This trains DS-CNN-S, 22,595 parameters, for the paper's 30,000 steps of
100 examples, with Adam at learning rates of 0.0005, 0.0001 and 0.00002
for 10,000 steps each. It keeps the model with the lowest loss on a fixed
dev set of 5,000 examples, and writes it to `model.pt` and `model.onnx`.
`--size M` and `--size L` train the paper's larger models.

Each example is a second of audio:

| Kind | Share | Label | What |
|---|---|---|---|
| Wake word | 30% | olivia | A window of a clip that holds all of the word, at a random offset, sped up or slowed down by up to 15%. The recorded and the synthesized clips are drawn as often. |
| Partial | 5% | unknown | A window of a clip that holds at most half of the word, so only the whole word is heard |
| Word | 10% | unknown | A Speech Commands word |
| Similar | 5% | unknown | A word that sounds like the wake word |
| Speech | 25% | unknown | A second of a LibriSpeech utterance, without those that say "olivia", or of MUSAN's speech |
| Music | 10% | unknown | A second of MUSAN's music |
| Silence | 15% | silence | Noise at -70 to -10 dBFS, or nothing |

MUSAN's noise at 0 to 20 dB SNR is mixed into 80% of the examples,
LibriSpeech babble into 10%, music into 20% at 0 to 15 dB, as of a TV
set, reverberation into 20%, and the volume is set to a peak of -40 to
0 dBFS for 80%, with `speech/loader.py`'s augmentations. Without
`--similar` or `--music`, the mix leaves those kinds out.

The CTC alignment marks the word's labels, which start about 0.1
seconds after its sound starts and end about 0.05 before it ends, at the
medians over the training clips against their energy. So the word is
taken to start 0.15 seconds earlier and end 0.1 later.

On the build server, an Intel i5-4690K with 4 cores and no GPU, 4 data
workers make about 7 batches a second, and training takes about 2 hours.
The data, not the model, limits the speed.

## Evaluate

```
A=examples/commonvoice/models/ctc_streaming_3750h_stride3_epoch3/model-int8.onnx
uv run examples/olivia/evaluate.py examples/olivia/models/ds_cnn_s/model.onnx --asr $A \
    --extra-keyword examples/olivia/data/synth-test.json \
    --similar examples/olivia/data/synth-test-words.json \
    --other examples/librispeech/data/LibriSpeech/test-clean.json \
    examples/speech_commands/data/test.json examples/musan/data/music-test.json \
    --other-names LibriSpeech "Speech Commands" Music
```

This streams each test clip, with a second of silence before and after
it, through the Detector's decisions, clean and with background noise at
10 dB SNR, and counts the clips in which it never hears the wake word,
as it does for the synthesized British voices. It counts how many of the
similar words it takes for the wake word. It streams LibriSpeech's
test-clean, Speech Commands' test words, music and background noise,
which never say it, and counts how often it hears it per hour. `--asr`
confirms each detection with the speech model, as `backend.py` does, so
the numbers are those of the service. `--keyword
examples/olivia/data/dev.json` and `--other` with the dev sets tune the
thresholds without the test sets.

## Results

Two DS-CNN-S models, both with the voice activity detector and the
speech model's confirmation, as `backend.py --wake-word --vad` runs them,
with the 3,750 hour stride 3 model after epoch 3, at a smoothing of 3:

- **Baseline:** the recorded clips only, as the paper's recipe, with
  Speech Commands, LibriSpeech and the six noise recordings of Speech
  Commands.
- **Full:** also the synthesized voices and similar words, and MUSAN's
  speech, music and noise, the training above.

On the test sets, the recorded clips clean and at 10 dB SNR, the
synthesized British voices, and the similar words in the British voices:

| Model | Threshold | Missed | Missed at 10 dB | Missed, British voices | Similar words taken | False alarms per hour, LibriSpeech | Speech Commands | Music |
|---|---|---|---|---|---|---|---|---|
| Baseline | 0.7 | 13.3% | 13.7% | 35.6% | 9.3% | 0.37 | 0 | 1.37 |
| Baseline | 0.9 | 16.1% | 15.9% | 46.5% | 7.1% | 0.19 | 0 | 0.91 |
| Full | 0.7 | 14.0% | 16.1% | 43.7% | 3.7% | 0.37 | 0 | 1.14 |
| Full | 0.8 | 16.3% | 17.3% | 52.4% | 2.8% | 0.19 | 0 | 0.69 |
| Full | 0.9 | 18.4% | 21.0% | 64.8% | 1.2% | 0.19 | 0 | 0.23 |

- The full model takes far fewer of the similar words for the wake word,
  less than half of the baseline's at the same misses, and less music. At 0.8 it
  misses as many recorded clips as the baseline at 0.9, with fewer false
  alarms, so `--wake-word-threshold` is 0.8 by default and on the Pi.
- The synthesized voices did not make it hear more voices: it misses more
  of the British ones. They are hard for the speech model too, which did
  not hear the word in 18% of them when they were made, see
  [Setup](#setup), and whose confirmation counts here. More recorded
  speakers would help most.
- About one recorded test clip in six is missed. The dataset's clips are
  of uneven quality, many quiet or distant, and the model was chosen for
  few false alarms. A clear "olivia" close to the speakerphone is heard
  reliably on the Pi.

What each stage does, for the baseline at 0.9 on the test sets:

| Stages | Missed | False alarms per hour, LibriSpeech | Music |
|---|---|---|---|
| Wake word model alone | 12.3% | 2.22 | 11.87 |
| With the speech model's confirmation | 16.1% | 0.19 | 5.94 |
| With the voice activity detector too | 16.1% | 0.19 | 0.91 |

- The confirmation removes most false alarms in speech, nine in ten, at
  the cost of the clips that the speech model does not hear as "olivia".
  `--wake-word-asr-threshold`, -2.0, sets how strict it is: on the dev
  set with the epoch 3 model, -2.5 let 17 of 219 false alarms of the wake
  word model through and -2.0 6, keeping 100% and 98.6% of the wake
  words heard.
- The voice activity detector removes most false alarms in music, 6 in 7,
  and misses no more wake words. It cannot remove those in speech, which
  it hears as speech, such as from a TV set.
- With the stride 3 model after epoch 4, the confirmation let only 3 of
  the same 219 false alarms through at -2.0, and kept every wake word.

### While the radio plays

A speakerphone's echo canceller removes what it plays from its
microphone's audio, such as the radio's music, which then never reaches
the wake word model. But it takes a voice over it apart too: recorded on a
Jabra SPEAK 510 playing the radio, "olivia" lost everything above 4 kHz,
and gaps were cut into the rest, where it removed the music. Said as loud
as without the radio, the wake word model heard it in 1 of 3 with the
radio at 73% of the volume, at probabilities of 0.07 and 0.08 for the
others, and in all 3 without the radio.

The speech model, trained on masked spectrograms, still hears it, so
while a `RadioCard` plays, it hears the wake word instead, behind the voice
activity detector: a `speech.wake_word.SpeechModelDetector` aligns "olivia"
to its output of the last 1.5 seconds every 0.2 seconds, as the
confirmation does, and hears it at `--wake-word-playing-threshold`.
Streaming LibriSpeech's dev-clean, the recorded test clips, and 15
"olivia"s recorded on the Jabra, 9 of them over the radio, 3 at 73%:

| Threshold | Missed, test clips | Missed, on the Jabra | False alarms per hour, LibriSpeech |
|---|---|---|---|
| -2.0 | 4.0% | 0 of 15 | 4.8 |
| -1.5 | 6.9% | 0 of 15 | 1.3 |
| -1.25 | 10.1% | 0 of 15 | 0.6 |
| -1.0 | 17.7% | 2 of 15 | 0 |

-1.25, the default, misses fewer of the test clips than the wake word
model with its confirmation, and the false alarms are of speech, which
reaches it only while the radio plays. It costs as much as transcribing,
about a third of a core of the Pi while someone speaks.

## Speed

On a Raspberry Pi 4 at 1.5 GHz, with the service running, a decision
of the Detector takes:

| Model | Parameters | Per decision | Of one core, at 10 per second |
|---|---|---|---|
| DS-CNN-S | 22,595 | 2.9 ms | 2.9% |
| DS-CNN-M | 135,023 | 5.9 ms | 5.9% |
| DS-CNN-L | 411,243 | 12.3 ms | 12.3% |

On an M1, DS-CNN-S takes 0.34 ms. Silero VAD takes 2 ms of each 32 ms
frame on the Pi, 6% of a core, all along, and the wake word model then
runs only while someone speaks. The speech model stays loaded while
`IDLE`, and takes no compute until the wake word is heard.

[Hello Edge: Keyword Spotting on Microcontrollers]: https://arxiv.org/abs/1711.07128
[Piper]: https://github.com/OHF-Voice/piper1-gpl
[Silero VAD]: https://github.com/snakers4/silero-vad
[openWakeWord]: https://github.com/dscripka/openWakeWord
[mindorii/kws]: https://github.com/mindorii/kws#data
