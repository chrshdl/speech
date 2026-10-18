import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest
from train_test import make_config, train_run

import train
from speech import loader
from speech.utils import review


@pytest.fixture
def dataset(tmp_path):
    """
    test.json's clips with absolute audio paths, and Whisper's results of
    them: it hears test0 right, and only "hello" of test1's "hello hi".
    """
    with open("test.json") as fid:
        rows = [json.loads(line) for line in fid]
    for row in rows:
        row["audio"] = os.path.abspath(row["audio"])
    path = tmp_path / "train.json"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    whisper = [
        {
            "audio": os.path.abspath("test0.wav"),
            "text": "hello world",
            "duration": 1.101,
            "whisper": "Hello world.",
            "hyp": "hello world",
            "CER": 0.0,
        },
        {
            "audio": os.path.abspath("test1.wav"),
            "text": "hello hi",
            "duration": 1.571,
            "whisper": "Hello.",
            "hyp": "hello",
            "CER": 0.375,
        },
    ]
    (tmp_path / "train-whisper.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in whisper)
    )
    return str(path)


def test_queue_and_verdicts(dataset):
    items = review.queue([dataset])
    assert [r["audio"] for r in items] == [os.path.abspath("test1.wav")]
    assert items[0]["verdict"] is None and items[0]["dataset"] == dataset
    # Above a higher threshold, nothing is to review.
    assert review.queue([dataset], min_cer=0.5) == []
    assert review.excluded([dataset]) == set()

    review.save_verdicts(dataset, {os.path.abspath("test1.wav"): "nok"})
    assert review.queue([dataset])[0]["verdict"] == "nok"
    assert review.excluded([dataset]) == {os.path.abspath("test1.wav")}
    # A dataset without Whisper's results has nothing to review.
    assert review.queue([dataset + ".missing"]) == []


def test_loader_leaves_out_excluded(dataset):
    data = loader.AudioDataset(dataset, None, 1).data
    kept = loader.AudioDataset(
        dataset, None, 1, exclude={os.path.abspath("test1.wav")}
    ).data
    assert len(data) == 8
    assert [d["text"] for d in kept] == ["hello world"] * 4


def request(port, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", data, timeout=5) as r:
        return r.status, r.headers["Content-Type"], r.read()


def test_serve(dataset):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    result = {}
    thread = threading.Thread(
        target=lambda: result.update(nok=review.serve([dataset], port)), daemon=True
    )
    thread.start()
    for _ in range(100):
        try:
            status, kind, page = request(port, "/")
            break
        except OSError:
            time.sleep(0.05)
    assert status == 200 and kind.startswith("text/html") and b"Whisper heard" in page

    _, _, body = request(port, "/api/items")
    items = json.loads(body)
    assert [(r["text"], r["whisper"], r["verdict"]) for r in items] == [
        ("hello hi", "Hello.", None)
    ]
    _, kind, audio = request(port, "/audio/0")
    with open("test1.wav", "rb") as fid:
        assert kind == "audio/wav" and audio == fid.read()
    # Only the clips of the queue are served.
    with pytest.raises(urllib.error.HTTPError, match="404"):
        request(port, "/audio/1")
    with pytest.raises(urllib.error.HTTPError, match="400"):
        request(port, "/api/verdict", {"i": 0, "verdict": "maybe"})

    request(port, "/api/verdict", {"i": 0, "verdict": "ok"})
    assert review.load_verdicts(dataset) == {os.path.abspath("test1.wav"): "ok"}
    request(port, "/api/verdict", {"i": 0, "verdict": None})
    assert review.load_verdicts(dataset) == {}
    request(port, "/api/verdict", {"i": 0, "verdict": "nok"})
    request(port, "/api/continue", {})
    thread.join(5)
    assert not thread.is_alive() and result["nok"] == 1
    assert review.excluded([dataset]) == {os.path.abspath("test1.wav")}


def test_training_pauses_for_review(tmp_path, dataset, monkeypatch, capsys):
    reviews = []

    def serve(sets, port, min_cer, title):
        reviews.append((sets, title))
        # The first review finds test1 mislabeled.
        review.save_verdicts(dataset, {os.path.abspath("test1.wav"): "nok"})

    monkeypatch.setattr(train.review, "serve", serve)
    config = make_config(tmp_path / "run", 3)
    config["data"]["train_set"] = dataset
    config["data"]["review"] = {"sets": [dataset]}
    train_run(config, seed=0)
    # After each epoch but the last, and the next leaves out the NOK clip.
    assert reviews == [
        ([dataset], "Review after epoch 0"),
        ([dataset], "Review after epoch 1"),
    ]
    assert "Leaving out 1 clips reviewed as NOK." in capsys.readouterr().out
