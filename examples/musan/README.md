[MUSAN] by David Snyder, Guoguo Chen and Daniel Povey: music of many
genres, speech in 12 languages, and technical and natural noise, at
16 kHz. `examples/olivia` trains the wake word model with it, so that
music and TV-like sound, speech over music, are not taken for the wake
word.

## Setup

From the top level directory run

```
uv run examples/musan/prepare.py examples/musan/data
```

This streams the 11 GB archive from OpenSLR and extracts the recordings,
about 11 GB, into `examples/musan/data/musan`. `--archive musan.tar.gz`
extracts a downloaded copy instead, such as one from the mirror at
https://openslr.elda.org/resources/17/musan.tar.gz, which was faster in
Europe. It writes `music-train.json`, `music-test.json`,
`speech-train.json`, `speech-test.json`, `noise-train.json` and
`noise-test.json`: a tenth of the recordings, by a hash of their names,
are held out for testing.

MUSAN is released under the [CC BY 4.0] license:

> Snyder, D., Chen, G. and Povey, D. (2015). MUSAN: A Music, Speech, and
> Noise Corpus. arXiv:1510.08484.

[MUSAN]: https://www.openslr.org/17/
[CC BY 4.0]: https://creativecommons.org/licenses/by/4.0/
