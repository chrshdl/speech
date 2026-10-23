import asyncio
import contextlib
import json
import os
import shutil

import httpx
import numpy as np
import pytest

from speech.backend import radio, volume
from speech.intents import IntentRecognizer

LABELS = "abcdefghijklmnopqrstuvwxyz' "
CHAR_TO_INT = {c: i for i, c in enumerate(LABELS)}
BLANK = len(LABELS)

ARTWORK = "https://is1-ssl.mzstatic.com/image/thumb/Music/a.jpg/100x100bb.jpg"
COVER = "https://is1-ssl.mzstatic.com/image/thumb/Music/a.jpg/600x600bb.jpg"
IMAGE = {"content-type": "image/jpeg"}


def image(player, path):
    """The URL of the image a card shows at the path, through the service."""
    assert path.startswith("/v1/radio/images/")
    return player.images[path.rsplit("/", 1)[1]]


def now(artist, song, kind="now"):
    """What streamABC says plays, as its WebSocket sends it."""
    return json.dumps({"type": kind, "artist": artist, "song": song, "cover": ""})


def client(requests, playing=("Anton", "Halb so schlimm"), found=None):
    """
    An httpx client that answers like streamABC, with what plays, and like
    iTunes, with the songs found, recording the requests.
    """
    if found is None:
        found = [
            {
                "artistName": "Anton",
                "trackName": "Halb so schlimm",
                "artworkUrl100": ARTWORK,
            }
        ]

    def handle(request):
        requests.append(request)
        if request.url.path.endswith((".jpg", ".png")):
            return httpx.Response(
                200, content=b"image " + str(request.url).encode(), headers=IMAGE
            )
        if request.url.host == "api.streamabc.net":
            artist, song = playing
            return httpx.Response(200, text=now(artist, song))
        if isinstance(found, int):
            return httpx.Response(found)
        return httpx.Response(200, json={"resultCount": len(found), "results": found})

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


class FakeWebSocket:
    """
    Connects as websockets' connect does, to a session of messages each
    time, or fails with a session that is an exception. The last session
    stays open.
    """

    def __init__(self, *sessions):
        self.sessions = list(sessions)
        self.urls = []

    @contextlib.asynccontextmanager
    async def __call__(self, url):
        self.urls.append(url)
        session = self.sessions.pop(0)
        if isinstance(session, Exception):
            raise session
        last = not self.sessions

        async def messages():
            for message in session:
                yield message
            if last:
                await asyncio.Event().wait()

        yield messages()


def test_card():
    requests = []

    async def run():
        player = radio.Radio(client=client(requests))
        card = await player.card("abc", {})
        await player.close()
        return player, card

    player, card = asyncio.run(run())
    assert card.model_dump(by_alias=True) == {
        "type": "RadioCard",
        "id": "abc",
        "title": "TOGGO Radio",
        "url": "/v1/radio/stream",
        "sampleRate": 44100,
        "channels": 2,
        "encoding": "LINEAR16",
        "logo": card.logo,
        "song": "Halb so schlimm",
        "artist": "Anton",
        "cover": card.cover,
    }
    # The client fetches the images through the service.
    assert image(player, card.logo) == radio.TOGGO.logo
    assert image(player, card.cover) == COVER
    assert str(requests[0].url) == (
        "https://api.streamabc.net/metadata/channel/srtl_skr4dfdxk6_xndd.json"
    )
    assert requests[1].url.params["term"] == "Anton Halb so schlimm"


@pytest.mark.parametrize(
    ("found", "cover"),
    [
        # Also with a guest and the song's version added.
        (
            [
                {
                    "artistName": "Anton feat. Lena",
                    "trackName": "Halb so schlimm (Radio Edit)",
                    "artworkUrl100": ARTWORK,
                }
            ],
            COVER,
        ),
        # Not another song of the artist, nor the song of another artist.
        (
            [
                {
                    "artistName": "Anton",
                    "trackName": "Ganz schlimm",
                    "artworkUrl100": "x",
                },
                {
                    "artistName": "Bea",
                    "trackName": "Halb so schlimm",
                    "artworkUrl100": "y",
                },
            ],
            None,
        ),
        ([], None),
    ],
)
def test_cover_is_the_song(found, cover):
    async def run():
        player = radio.Radio(client=client([], found=found))
        card = await player.card("abc", {})
        await player.close()
        return player, card

    player, card = asyncio.run(run())
    assert (image(player, card.cover) if card.cover else None) == cover


def test_cover_of_a_clean_release():
    """For children, the cover without a Parental Advisory label, if found."""
    found = [
        {
            "artistName": "Katy Perry",
            "trackName": "Firework",
            "collectionExplicitness": explicitness,
            "artworkUrl100": f"https://covers/{explicitness}/100x100bb.jpg",
        }
        for explicitness in ("explicit", "cleaned", "notExplicit")
    ]
    requests = []

    async def run():
        player = radio.Radio(client=client(requests, found=found))
        cover = await player.cover("Katy Perry", "Firework")
        only = radio.Radio(client=client([], found=found[:1]))
        explicit = await only.cover("Katy Perry", "Firework")
        await player.close()
        await only.close()
        return cover, explicit

    cover, explicit = asyncio.run(run())
    assert cover == "https://covers/cleaned/600x600bb.jpg"
    # Without one, the explicit release's.
    assert explicit == "https://covers/explicit/600x600bb.jpg"
    assert requests[0].url.params["limit"] == "25"


def test_cover_looked_up_once():
    requests = []

    async def run():
        player = radio.Radio(client=client(requests))
        covers = [await player.cover("Anton", "Halb so schlimm") for _ in range(2)]
        await player.close()
        return covers

    assert asyncio.run(run()) == [COVER, COVER]
    assert len(requests) == 1


def test_plays_when_offline():
    """Without what plays, or its cover, the station still plays."""
    requests = []

    async def run():
        player = radio.Radio(client=client(requests, found=503))
        unfound = await player.card("abc", {})
        offline = radio.Radio(client=client([], playing=("", "")))
        unsaid = await offline.card("def", {})
        await player.close()
        await offline.close()
        return unfound, unsaid, player.covers

    unfound, unsaid, covers = asyncio.run(run())
    assert (unfound.song, unfound.cover) == ("Halb so schlimm", None)
    assert (unsaid.song, unsaid.artist, unsaid.cover) == ("", "", None)
    # A failed lookup is not kept, so the next card tries again.
    assert covers == {}
    assert [r.url.host for r in requests].count("itunes.apple.com") == 1


def test_updates(monkeypatch):
    """
    Each song that starts updates the card, also after the WebSocket
    failed, but not what plays next, nor the song again.
    """
    monkeypatch.setattr(radio, "RETRY_SECONDS", 0.01)
    socket = FakeWebSocket(
        [now("Anton", "Halb so schlimm"), now("Bea", "Hey", "next"), now("Bea", "Hey")],
        OSError("refused"),
        [now("Bea", "Hey"), now("", "")],
    )

    async def run():
        player = radio.Radio(client=client([]), websocket=socket)
        card = await player.card("abc", {})
        updates = player.updates(card)
        cards = [await asyncio.wait_for(anext(updates), 5) for _ in range(2)]
        await updates.aclose()
        await player.close()
        return cards

    hey, between = asyncio.run(run())
    assert (
        socket.urls
        == ["wss://api.streamabc.net/metadata/channel/srtl_skr4dfdxk6_xndd"] * 3
    )
    assert (hey.id, hey.artist, hey.song) == ("abc", "Bea", "Hey")
    assert (between.id, between.song, between.cover) == ("abc", "", None)


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("play toggo radio", "PlayRadio"),
        ("toggo radio", "PlayRadio"),
        ("play webradio", "PlayRadio"),
        ("turn on the radio", "PlayRadio"),
        ("stop the radio", "StopRadio"),
        ("turn the radio off", "StopRadio"),
        ("stop", "StopRadio"),
        ("volume up", "VolumeUp"),
        ("turn it down", "VolumeDown"),
    ],
)
def test_intents(text, intent):
    """Told apart, also from turning the volume, as while the radio plays."""
    path = [label for c in text for label in (CHAR_TO_INT[c], BLANK)]
    probs = np.full((len(path), BLANK + 1), 0.03 / BLANK)
    probs[np.arange(len(path)), path] = 0.97
    recognizer = IntentRecognizer(
        [*radio.INTENTS, *volume.INTENTS], CHAR_TO_INT, BLANK, 0.04
    )
    assert recognizer.recognize(probs).intent.name == intent


def test_images_fetched_through_the_service():
    requests = []

    async def run():
        player = radio.Radio(client=client(requests))
        card = await player.card("abc", {})
        key = card.cover.rsplit("/", 1)[1]
        fetched = [await player.image(key) for _ in range(2)]
        unknown = await player.image("nothing")
        await player.close()
        return fetched, unknown

    fetched, unknown = asyncio.run(run())
    assert fetched == [(b"image " + COVER.encode(), "image/jpeg")] * 2
    # Kept, so fetched once.
    assert [str(r.url) for r in requests].count(COVER) == 1
    assert unknown is None


def wave_file(path, seconds=1.0, rate=22050):
    """
    A WAV file at rate, in stereo as the station, of a 440 Hz tone on the
    left and 660 Hz on the right.
    """
    t = np.arange(round(seconds * rate)) / rate
    tones = [0.5 * np.sin(2 * np.pi * hz * t) for hz in (440, 660)]
    samples = np.round(np.stack(tones, axis=1) * 32767).astype("<i2")
    header = (
        b"RIFF"
        + (36 + samples.nbytes).to_bytes(4, "little")
        + b"WAVEfmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (2).to_bytes(2, "little")
        + rate.to_bytes(4, "little")
        + (4 * rate).to_bytes(4, "little")
        + (4).to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data"
        + samples.nbytes.to_bytes(4, "little")
    )
    path.write_bytes(header + samples.tobytes())
    return str(path)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_stream_decodes_with_ffmpeg(tmp_path):
    """The station's audio, decoded to 16 bit stereo at 44.1 kHz."""
    station = radio.Station("Test", wave_file(tmp_path / "station.wav"), "key", "logo")

    async def run():
        player = radio.Radio(station, client=client([]))
        chunks = [chunk async for chunk in player.stream()]
        await player.close()
        return b"".join(chunks)

    samples = np.frombuffer(asyncio.run(run()), "<i2").reshape(-1, 2)
    assert len(samples) == pytest.approx(44100, abs=100)
    # Each channel its own tone, at its level.
    for channel, hz in ((0, 440), (1, 660)):
        spectrum = np.abs(np.fft.rfft(samples[:44100, channel]))
        assert np.argmax(spectrum) == pytest.approx(hz, abs=1)
        peak = np.abs(samples[1000:-1000, channel]).max()
        assert peak == pytest.approx(0.5 * 32767, rel=0.05)


def test_stream_ends_ffmpeg_when_closed(tmp_path):
    """Closed, as when the client goes, the stream ends its ffmpeg at once."""
    started = tmp_path / "started"
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_text(
        f"#!/bin/sh\necho $$ > {started}\n"
        "while true; do printf 'abcd'; sleep 0.01; done\n"
    )
    ffmpeg.chmod(0o755)

    async def run():
        player = radio.Radio(client=client([]), ffmpeg=str(ffmpeg))
        stream = player.stream()
        first = await asyncio.wait_for(anext(stream), 5)
        await stream.aclose()
        await player.close()
        return first

    assert asyncio.run(run()).startswith(b"abcd")
    pid = int(started.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def frames(start, count):
    """Stereo frames that count from start, left the low, right the high bits."""
    index = np.arange(start, start + count)
    return np.stack([index & 0x7FFF, index >> 15], axis=1).astype("<i2").tobytes()


def indexes(audio):
    """The counts of the frames of frames()."""
    pairs = np.frombuffer(audio, "<i2").reshape(-1, 2).astype(np.int64)
    return pairs[:, 0] | (pairs[:, 1] << 15)


async def feed(reader, audio, piece, every):
    """Feeds the audio to the reader in pieces, a piece every so many seconds."""
    for i in range(0, len(audio), piece):
        reader.feed_data(audio[i : i + piece])
        await asyncio.sleep(every)
    reader.feed_eof()


@pytest.mark.parametrize("burst", [4.0, 0.0])
def test_live_skips_the_burst(monkeypatch, burst):
    """
    Of the burst the station sends on connecting, only the newest seconds
    play, and then all the rest, in whole frames, though pieces split them.
    Without a burst, all of it plays.
    """
    rate = 8000
    monkeypatch.setattr(radio, "SAMPLE_RATE", rate)
    monkeypatch.setattr(radio, "KEEP_SECONDS", 0.5)
    monkeypatch.setattr(radio, "SETTLE_SECONDS", 0.2)
    burst_frames = round(burst * rate)
    live_frames = rate

    async def run():
        reader = asyncio.StreamReader()
        # The burst at once, then a second in real time, in pieces of 0.05
        # seconds and a byte, so splitting frames.
        reader.feed_data(frames(0, burst_frames))
        piece = 4 * rate // 20 + 1
        feeding = asyncio.create_task(
            feed(reader, frames(burst_frames, live_frames), piece, 0.05)
        )
        out = [chunk async for chunk in radio.live(reader)]
        await feeding
        return out

    out = asyncio.run(run())
    # Whole frames from the start, though a chunk can end in part of one.
    played = indexes(b"".join(out))
    assert np.array_equal(played, np.arange(played[0], burst_frames + live_frames))
    if burst:
        # The newest half second, of the burst and what came while it settled.
        assert len(out[0]) // 4 == round(0.5 * rate)
        assert burst_frames - 0.5 * rate <= played[0] <= burst_frames
    else:
        assert played[0] == 0
