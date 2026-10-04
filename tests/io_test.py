import tempfile

import shared
import torch

import speech.loader
import speech.models


def test_save():
    preproc = speech.loader.Preprocessor("test.json")
    model = speech.models.CTC(
        preproc.input_dim, preproc.vocab_size, shared.model_config
    )

    save_dir = tempfile.mkdtemp()
    speech.save(model, preproc, save_dir)

    s_model, s_preproc = speech.load(save_dir)
    assert isinstance(s_model, speech.models.CTC)
    assert s_preproc.int_to_char == preproc.int_to_char
    assert (s_preproc.mean == preproc.mean).all()
    assert (s_preproc.std == preproc.std).all()

    msd = model.state_dict()
    smsd = s_model.state_dict()
    assert msd.keys() == smsd.keys()
    for k, v in smsd.items():
        assert torch.equal(v, msd[k])
