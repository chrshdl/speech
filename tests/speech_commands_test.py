import importlib.util
import os

# Loaded under its own name, as the examples' scripts share names.
spec = importlib.util.spec_from_file_location(
    "speech_commands_evaluate",
    os.path.join(
        os.path.dirname(__file__), "..", "examples", "speech_commands", "evaluate.py"
    ),
)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def test_command_counts():
    # Each clip's word, and the command closest to it with its score.
    heard = [
        ("left", "left", -0.5),
        ("left", "left", -1.2),
        ("right", "left", -0.8),
        ("right", None, None),
        ("yes", "left", -1.0),
        ("no", "right", -0.4),
        ("stop", None, None),
    ]
    assert evaluate.command_counts(heard, -1.5) == {
        "commands": 4,
        "recognized": 2,
        "wrong": 1,
        "others": 3,
        "others taken": 2,
    }
    # A stricter threshold takes fewer of both.
    counts = evaluate.command_counts(heard, -0.6)
    assert (counts["recognized"], counts["wrong"], counts["others taken"]) == (1, 0, 1)
