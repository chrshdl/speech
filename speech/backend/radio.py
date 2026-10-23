"""
Plays web radio when asked, such as "play toggo radio": TOGGO Radio, the
children's station of Super RTL. Its card has the client stream the
station's audio and shows the song playing, with its artist and cover, which
the service follows while the station plays: the station's stream host,
streamABC, pushes each song to a WebSocket as it starts. The covers are
from the iTunes Search API, as streamABC has none for the station, which
needs no key.

The service, not the client, fetches the station's MP3 and decodes it with
ffmpeg, and the client plays the samples as it plays speech, see
web/assistant/speaker.js. So the client needs no internet, as for the
rest, and no codecs: on the Pi, WPE WebKit cannot fetch HTTPS, and its
media elements play to an ALSA device of their own, which Web Audio holds
already. It fetches the logo and the covers through the service, too.

The station's stream, channel key and logo are those of the channel list
of toggo.de's own player,
https://ccstatic.toggo.de/cc-static-files/Webchannels/Webchannel_streams3.json.
"""

import asyncio
import collections
import dataclasses
import hashlib
import json
import logging
import re

import httpx
from rapidfuzz import fuzz
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from speech.backend.messages import RadioCard
from speech.intents import Intent

log = logging.getLogger(__name__)

PLAY = Intent(
    "PlayRadio",
    (
        "play toggo radio",
        "play the toggo radio",
        "toggo radio",
        "turn on toggo radio",
        "play web radio",
        "play webradio",
        "play the web radio",
        "play the radio",
        "play radio",
        "turn on the radio",
        "turn the radio on",
        "start the radio",
    ),
)
STOP = Intent(
    "StopRadio",
    (
        "stop the radio",
        "stop radio",
        "stop the web radio",
        "stop toggo radio",
        "turn off the radio",
        "turn the radio off",
        "radio off",
        "stop the music",
        "stop playing",
        "stop",
    ),
)
INTENTS = (PLAY, STOP)


@dataclasses.dataclass(frozen=True)
class Station:
    name: str
    # The MP3 stream.
    url: str
    # The station's key at streamABC, for what plays.
    channel: str
    logo: str


TOGGO = Station(
    name="TOGGO Radio",
    url="https://radio.toggo.de/live/mp3-192/radio.toggo.de/",
    channel="srtl_skr4dfdxk6_xndd",
    logo="https://staging-api.toggo.de/cc-static-files/Webchannels/"
    "webchannels%20-%20radio@3x1.png",
)

# Where the client streams the radio, and fetches its images by key.
STREAM_URL = "/v1/radio/stream"
IMAGE_URL = "/v1/radio/images/{key}"
# The audio the client streams: 16 bit samples at the rate of the Pi's Web
# Audio, so it plays without resampling, in stereo, in chunks of about
# 0.1 seconds.
SAMPLE_RATE = 44100
CHANNELS = 2
CHUNK = 16384
# On connecting, the station sends the last 20 seconds or so it played at
# once, then plays on in real time. Of that burst, only the newest seconds
# are sent on, so the radio plays this close to live, as the song shown
# changes, and keeps that much against a slow network. The burst is over
# once the audio decoded in the last SETTLE_SECONDS grew by less than half
# as much as the time.
KEEP_SECONDS = 3.0
SETTLE_SECONDS = 0.5
# The images kept, the last fetched, for a card shown again.
MAX_IMAGES = 16

# What plays on a streamABC channel: the song now, as JSON at the URL with
# .json, and pushed to the WebSocket at it with wss: as each song starts.
METADATA_URL = "https://api.streamabc.net/metadata/channel/{channel}"
ITUNES_URL = "https://itunes.apple.com/search"
# The size of the covers, in pixels, which iTunes scales its artwork to.
COVER_SIZE = 600
# The lowest similarity, out of 100, of a song's artist and title on iTunes
# to the station's, such as with "feat." or a remix added.
CLOSE = 80
# The songs found on iTunes to choose from, enough to also find a song's
# clean release, whose cover has no Parental Advisory label, beside its
# explicit ones, which come first.
FOUND = 25
# How long to wait before following what plays again, after the WebSocket
# failed, doubling up to the longest while it keeps failing.
RETRY_SECONDS = 2.0
MAX_RETRY_SECONDS = 60.0
# The covers kept, by artist and song, at most.
MAX_COVERS = 500


def _plain(text):
    """The text in lower case, without punctuation, to compare names."""
    return re.sub(r"[^\w ]+", " ", text.lower()).strip()


class Radio:
    def __init__(self, station=TOGGO, client=None, websocket=connect, ffmpeg="ffmpeg"):
        """
        Plays the station, a Station.

        Arguments:
            client (httpx.AsyncClient, optional): Makes the requests.
            websocket (optional): Connects to a WebSocket, as
                websockets.asyncio.client.connect does.
            ffmpeg (str): The ffmpeg program, which decodes the station.
        """
        self.station = station
        self.client = client or httpx.AsyncClient(timeout=5)
        self.websocket = websocket
        self.ffmpeg = ffmpeg
        # The cover of each song looked up, or None, by artist and song.
        self.covers = {}
        # The URL of each image the cards show, by its key, and the last
        # fetched, its content and type.
        self.images = {}
        self.fetched = collections.OrderedDict()
        # Answers each of INTENTS.
        self.handlers = {PLAY.name: self, STOP.name: Stop()}

    async def card(self, card_id, slots):
        """
        The card that plays the station, with the song playing, or without
        one when the station does not say.
        """
        try:
            response = await self.client.get(
                METADATA_URL.format(channel=self.station.channel) + ".json"
            )
            response.raise_for_status()
            now = response.json()
        except (httpx.HTTPError, ValueError) as error:
            log.warning("Could not get what %s plays: %s", self.station.name, error)
            now = {}
        return await self._card(card_id, now)

    async def updates(self, card):
        """
        Yields the card again, with the same id, each time another song
        plays, until closed, connecting to the WebSocket again after it
        failed.
        """
        url = METADATA_URL.format(channel=self.station.channel).replace(
            "https:", "wss:"
        )
        playing = (card.artist, card.song)
        retry = RETRY_SECONDS
        while True:
            try:
                async with self.websocket(url) as messages:
                    async for message in messages:
                        retry = RETRY_SECONDS
                        now = json.loads(message)
                        # The others say what plays next.
                        if now.get("type") != "now":
                            continue
                        said = (now.get("artist") or "", now.get("song") or "")
                        if said != playing:
                            playing = said
                            yield await self._card(card.id, now)
            except (OSError, WebSocketException, ValueError) as error:
                log.warning("Following %s failed: %s", self.station.name, error)
            await asyncio.sleep(retry)
            retry = min(2 * retry, MAX_RETRY_SECONDS)

    async def _card(self, card_id, now):
        artist = (now.get("artist") or "").strip()
        song = (now.get("song") or "").strip()
        cover = await self.cover(artist, song) if artist and song else None
        return RadioCard(
            id=card_id,
            title=self.station.name,
            url=STREAM_URL,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            logo=self._image(self.station.logo),
            song=song,
            artist=artist,
            cover=self._image(cover) if cover is not None else None,
        )

    def _image(self, url):
        """Where the client fetches the image at the url, through the service."""
        if len(self.images) >= MAX_COVERS:
            self.images.clear()
        key = hashlib.sha256(url.encode()).hexdigest()[:24]
        self.images[key] = url
        return IMAGE_URL.format(key=key)

    async def image(self, key):
        """
        The content and media type of a card's image by its key, or None for
        no such image. Raises an httpx.HTTPError if it cannot be fetched.
        """
        if key not in self.fetched:
            url = self.images.get(key)
            if url is None:
                return None
            response = await self.client.get(url, follow_redirects=True)
            response.raise_for_status()
            media_type = response.headers.get("content-type", "image/jpeg")
            self.fetched[key] = (response.content, media_type)
            if len(self.fetched) > MAX_IMAGES:
                self.fetched.popitem(last=False)
        self.fetched.move_to_end(key)
        return self.fetched[key]

    async def stream(self):
        """
        Yields the station's audio as it plays, 16 bit little-endian samples
        at SAMPLE_RATE, CHANNELS of them interleaved, from KEEP_SECONDS
        before live, see live(), until the station's
        stream ends, or until closed, which ends ffmpeg.
        """
        url = self.station.url
        # Through a short drop of the connection, an option of HTTP only.
        reconnect = []
        if url.startswith(("http:", "https:")):
            reconnect = ["-reconnect", "1", "-reconnect_streamed", "1"]
            reconnect += ["-reconnect_delay_max", "5"]
        process = await asyncio.create_subprocess_exec(
            self.ffmpeg,
            "-nostdin",
            "-loglevel",
            "error",
            *reconnect,
            "-i",
            url,
            "-vn",
            "-ac",
            str(CHANNELS),
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "s16le",
            "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # With -loglevel error, little, read all along so it never blocks.
        stderr = asyncio.create_task(process.stderr.read())
        try:
            async for chunk in live(process.stdout):
                yield chunk
            await process.wait()
            said = (await stderr).decode(errors="replace").strip()
            log.warning(
                "%s ended, ffmpeg exited with %s: %s",
                self.station.name,
                process.returncode,
                said,
            )
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
            stderr.cancel()

    async def cover(self, artist, song):
        """The URL of the song's cover on iTunes, or None if there is none."""
        key = (artist, song)
        if key not in self.covers:
            if len(self.covers) >= MAX_COVERS:
                self.covers.clear()
            try:
                self.covers[key] = await self._find_cover(artist, song)
            except (httpx.HTTPError, ValueError, KeyError) as error:
                # Looked up again for the next card.
                log.warning("Could not find the cover of %r: %s", key, error)
                return None
        return self.covers[key]

    async def _find_cover(self, artist, song):
        response = await self.client.get(
            ITUNES_URL,
            params={
                "term": f"{artist} {song}",
                "media": "music",
                "entity": "song",
                "limit": FOUND,
                "country": "DE",
            },
        )
        response.raise_for_status()
        found = [
            result
            for result in response.json()["results"]
            if fuzz.token_set_ratio(_plain(artist), _plain(result["artistName"]))
            >= CLOSE
            and fuzz.token_set_ratio(_plain(song), _plain(result["trackName"])) >= CLOSE
            and "artworkUrl100" in result
        ]
        if not found:
            return None
        # A release that is not explicit, if there is one, for children.
        clean = [r for r in found if r.get("collectionExplicitness") != "explicit"]
        artwork = (clean or found)[0]["artworkUrl100"]
        return artwork.replace("100x100bb", f"{COVER_SIZE}x{COVER_SIZE}bb")

    async def close(self):
        await self.client.aclose()


class Stop:
    """
    Answers a request to stop the radio: the engine stops what plays, and
    shows nothing for it.
    """

    stops_playing = True

    async def card(self, card_id, slots):
        return None

    async def close(self):
        pass


async def live(audio):
    """
    Yields the audio read from the stream reader as it comes, after the
    burst it starts with, of which only the newest KEEP_SECONDS, whole
    frames. Without a burst, that is all that came in the first
    SETTLE_SECONDS.
    """
    loop = asyncio.get_running_loop()
    frame = 2 * CHANNELS
    per_second = frame * SAMPLE_RATE
    keep = round(KEEP_SECONDS * SAMPLE_RATE) * frame
    pending = bytearray()
    decoded = 0
    start = None
    # How far the audio decoded ran ahead of the time, when.
    history = collections.deque()
    while chunk := await audio.read(CHUNK):
        now = loop.time()
        start = start if start is not None else now
        decoded += len(chunk)
        pending += chunk
        if len(pending) > keep:
            del pending[: (len(pending) - keep) // frame * frame]
        ahead = decoded / per_second - (now - start)
        history.append((now, ahead))
        while now - history[0][0] > SETTLE_SECONDS:
            history.popleft()
        settled = ahead - history[0][1] < SETTLE_SECONDS / 2
        if now - start >= SETTLE_SECONDS and settled:
            break
    if pending:
        yield bytes(pending)
    while chunk := await audio.read(CHUNK):
        yield chunk
