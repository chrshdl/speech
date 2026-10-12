The transcribed English speech of [VoxPopuli]: 543 hours of European
Parliament speeches by 1,313 speakers, many of them not native speakers of
English, more spontaneous than read audiobooks. It is released under CC0.
The 3,750 hour model can train on it besides its other sets, see
`examples/commonvoice/README.md`.

## Setup

From the top level directory run

```
uv run examples/voxpopuli/prepare.py examples/voxpopuli/data
```

This downloads the 32 Parquet files of the copy on Hugging Face,
`facebook/voxpopuli`, 99 GB, one at a time, writes their clips to
`examples/voxpopuli/data/clips` as 16 kHz FLAC, and deletes each file once
its clips are written. It then writes `train.json`, `dev.json` and
`test.json`, the official splits, which keep each speaker in one of them.
An interrupted download continues where it stopped, and a run started
again skips the files it finished.

Plan for about 35 GB of clips and 3.3 GB for the file being downloaded, or
99 GB more with `--keep-downloads`. `--train-files 1` prepares only the
first of the 30 train files, about 18 hours, to try it first.

The transcripts are normalized as Common Voice's, and a clip whose
transcript has digits or other characters is dropped. Clips longer than
30 seconds are dropped too, `--max-seconds` sets the limit. In the first
train file, all 6,083 transcripts were gold, checked against the audio,
none had digits, and 58 clips of 17.4 hours were longer than 30 seconds,
1% of the clips and 3% of the hours. Extracting it took 73 seconds on an
M1. `--gold-only` leaves out any transcript not marked as gold.

## License

VoxPopuli is released under [CC0], so it needs no credit, but cite it in
published work:

> Wang, C. et al. (2021). VoxPopuli: A Large-Scale Multilingual Speech
> Corpus for Representation Learning, Semi-Supervised Learning and
> Interpretation. ACL 2021.

[VoxPopuli]: https://github.com/facebookresearch/voxpopuli
[CC0]: https://creativecommons.org/publicdomain/zero/1.0/
