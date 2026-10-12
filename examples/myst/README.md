[My Science Tutor (MyST)] is children's conversational speech: about 400
hours of about 1,300 third to fifth graders, 8 to 11 years old, talking with a
virtual science tutor, of which about 45% of the utterances are
transcribed. It is the largest open corpus of children's speech, which the
other datasets lack, so it helps the model with a child talking to the
assistant. The 3,750 hour model can train on it besides its other sets,
see `examples/commonvoice/README.md`.

## Setup

1. Download the corpus. The [LDC] distributes it as LDC2021S05: sign in or
   create an LDC account, accept the corpus's license and download it. It
   is free for non-commercial use.
2. Extract the download, and run from the top level directory:

   ```
   uv run examples/myst/prepare.py <extracted directory> examples/myst/data
   ```

   It finds the `train`, `development` and `test` directories in the
   given directory or in its `data` directory, and writes `train.json`,
   `dev.json` and `test.json` to `examples/myst/data` for the transcribed
   utterances. The partitions are the official ones, which keep each
   student in one of them. Clips that are 16 kHz mono are used where they
   are; others are converted to 16 kHz FLAC in `examples/myst/data/clips`.

The transcripts are verbatim, with markup for what is not plain speech:

| Markup | Such as | Done |
|---|---|---|
| A label for the whole utterance | `<<DISCARD>>`, `<<NO_SIGNAL>>`, `<<SILENCE>>` | The utterance is dropped |
| Double parentheses | `(( ))`, `((battery))` | Dropped: speech the transcript leaves out or is unsure of |
| A tag of speech left out | `<indiscernible>`, `<unintelligible>` | Dropped |
| A tag of a sound or a way of speaking | `<noise>`, `[laugh]`, `<whisper> … </whisper>` | The tag is removed, its words kept |
| Another tag | | Dropped, to be safe |
| A partial word | `wa-` | Its letters are kept |
| A filled pause | `um`, `uh` | Kept as said |

A model that hears words its transcript leaves out learns to skip words,
which is why those utterances are dropped rather than kept with the gap.
The rest is normalized as Common Voice's, and an utterance with digits or
other characters is dropped, as are utterances longer than 30 seconds,
`--max-seconds`, as in the MyST paper's recipes.

These rules follow the published descriptions of the transcripts, as the
corpus was not available while writing the script. After the run, it
prints the dropped utterances by reason and every tag it found. Check the
tags against the table: a sound counted as another tag should be added to
`SOUNDS` in `prepare.py`, and a tag of speech left out to `UNTRANSCRIBED`.

## License

MyST is released under [CC BY-NC-SA 4.0] for non-commercial use, and
licensed separately for commercial use. A model trained on it is for
non-commercial use, and must credit it:

> Pradhan, S. et al. (2024). My Science Tutor (MyST): A Large Corpus of
> Children's Conversational Speech. LREC-COLING 2024.

[My Science Tutor (MyST)]: https://arxiv.org/abs/2309.13347
[LDC]: https://catalog.ldc.upenn.edu/LDC2021S05
[CC BY-NC-SA 4.0]: https://creativecommons.org/licenses/by-nc-sa/4.0/
