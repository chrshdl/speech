import tempfile

import shared
import torch

import speech.loader
import speech.models
import speech.utils.io


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


def test_names():
    assert speech.utils.io.get_names("run") == ("run/model", "run/preproc.pyc")
    assert speech.utils.io.get_names("run", "best") == (
        "run/best_model",
        "run/best_preproc.pyc",
    )
    assert speech.utils.io.get_names("run", epoch=0) == (
        "run/model_epoch_0",
        "run/preproc_epoch_0.pyc",
    )


def test_save_epoch(tmp_path):
    preproc = speech.loader.Preprocessor("test.json")
    model = speech.models.CTC(
        preproc.input_dim, preproc.vocab_size, shared.model_config
    )
    speech.save(model, preproc, str(tmp_path), epoch=3)
    assert {p.name for p in tmp_path.iterdir()} == {
        "model_epoch_3",
        "preproc_epoch_3.pyc",
    }
    s_model, _ = speech.load(str(tmp_path), epoch=3)
    for k, v in s_model.state_dict().items():
        assert torch.equal(v, model.state_dict()[k])
