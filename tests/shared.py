import copy

import numpy as np
import torch

model_config = {
    "dropout": 0.0,
    "encoder": {
        "conv": [[32, 5, 32, 2]],
        "rnn": {"dim": 16, "bidirectional": False, "layers": 1},
    },
}


def gen_fake_data(freq_dim, output_dim, max_time=100, max_seq_len=20, batch_size=4):
    data = []
    for i in range(batch_size):
        inputs = np.random.randn(max_time, freq_dim)
        labels = np.random.randint(0, output_dim, max_seq_len)
        data.append((inputs, labels))
    inputs, labels = list(zip(*data))
    return inputs, labels


def randomize_lookahead(model):
    # The lookahead starts as the identity, which ignores future frames.
    # Random weights make tests sensitive to how future frames are used.
    if model.lookahead is not None:
        torch.nn.init.normal_(model.lookahead.conv.weight)


def bidirectional_config():
    # A bidirectional RNN lets padding at the end reach every frame.
    config = copy.deepcopy(model_config)
    config["encoder"]["rnn"]["bidirectional"] = True
    return config


def gen_padded_data(
    freq_dim, output_dim, time_steps=(100, 60, 80), label_lens=(20, 8, 14)
):
    inputs = [np.random.randn(t, freq_dim) for t in time_steps]
    labels = [np.random.randint(0, output_dim, n) for n in label_lens]
    return inputs, labels


def batch_norm_config(config):
    """The config with Deep Speech 2's batch normalization and clipped ReLU."""
    config = copy.deepcopy(config)
    config["encoder"]["batch_norm"] = True
    config["encoder"]["relu_clip"] = 20
    return config


def randomize_batch_norm(model):
    # Batch normalization starts as the identity at inference. Random
    # statistics and weights make tests sensitive to how it is applied.
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.running_mean.normal_()
            m.running_var.uniform_(0.5, 2.0)
            torch.nn.init.uniform_(m.weight, 0.5, 2.0)
            torch.nn.init.normal_(m.bias)
