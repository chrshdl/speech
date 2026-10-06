Background noise for training with the `noise` augmentation.

## Setup

From the top level directory run

```
uv run examples/noise/download.py examples/noise/data
```

This extracts the six background noise recordings, 6.7 minutes in total, of
the [Speech Commands] dataset by Pete Warden into `examples/noise/data`. It
streams the 1.5 GB archive and stops once it has the noise, which comes
first, so it takes seconds.

| Recording | Length |
|---|---|
| `doing_the_dishes.wav` | 95 s |
| `dude_miaowing.wav` | 62 s |
| `exercise_bike.wav` | 61 s |
| `running_tap.wav` | 61 s |
| `pink_noise.wav` | 60 s |
| `white_noise.wav` | 60 s |

Pete Warden recorded them in July 2017, and generated the pink and white
noise. They are part of Speech Commands and released under the
[CC BY 4.0] license, so models trained with them must credit the dataset:

> Warden, P. (2018). Speech Commands: A Dataset for Limited-Vocabulary
> Speech Recognition. arXiv:1804.03209.

## Use

Point a `noise` augmentation in a config's `data` at the directory:

```
"noise" : {"p" : 0.9, "source" : "examples/noise/data", "snr" : [8, 16]}
```

Any directory of WAV or FLAC files works the same way, at any sample rate,
and a bigger collection of noise generalizes better than these six
recordings.

[Speech Commands]: https://arxiv.org/abs/1804.03209
[CC BY 4.0]: https://creativecommons.org/licenses/by/4.0/
