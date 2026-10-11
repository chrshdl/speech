import asyncio
import datetime

import httpx
import pytest

from speech.backend import weather
from speech.backend.messages import WeatherCondition
from speech.intents import IntentRecognizer

PLACE = {"name": "Berlin", "latitude": 52.52, "longitude": 13.41, "country_code": "DE"}


def hours(date, codes, rain, sun, night_rain=0.0):
    """Open-Meteo's hourly forecast of a day, daylight from 8 to 18."""
    hourly = {k: [] for k in ("time", "weather_code", "precipitation")}
    hourly |= {"sunshine_duration": [], "is_day": []}
    for hour in range(24):
        day = 8 <= hour <= 18
        i = hour - 8
        hourly["time"].append(f"{date}T{hour:02}:00")
        hourly["weather_code"].append(codes[i] if day else 80)
        hourly["precipitation"].append(rain[i] if day else night_rain)
        hourly["sunshine_duration"].append(sun[i] * 60 if day else 0)
        hourly["is_day"].append(int(day))
    return hourly


def two_days():
    sunny = hours("2026-10-10", [1] * 11, [0] * 11, [60] * 11, night_rain=1.0)
    snowy = hours("2026-10-11", [73] * 11, [0.5] * 11, [0] * 11)
    return {k: sunny[k] + snowy[k] for k in sunny}


FORECAST = {
    "current": {
        "temperature_2m": 15.4,
        "apparent_temperature": 13.0,
        "relative_humidity_2m": 62,
        "weather_code": 2,
        "wind_speed_10m": 12.8,
        "is_day": 0,
    },
    "hourly": two_days(),
    "daily": {
        "time": ["2026-10-10", "2026-10-11"],
        "temperature_2m_max": [16.7, 12.7],
        "temperature_2m_min": [11.1, 8.4],
    },
}


def client(requests, places=(PLACE,)):
    """An httpx client that answers like Open-Meteo, recording requests."""

    def handle(request):
        requests.append(request)
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(200, json={"results": list(places)})
        return httpx.Response(200, json=FORECAST)

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


def test_card():
    requests = []

    async def run():
        # A place that is not in PLACES is looked up once.
        service = weather.Weather("Somewhere", client(requests))
        cards = [await service.card("id1"), await service.card("id2")]
        await service.close()
        return cards

    first, second = asyncio.run(run())
    assert first.id == "id1"
    assert first.location == "Berlin"
    assert (first.condition, first.description) == (
        WeatherCondition.PARTLY_CLOUDY,
        "Partly cloudy",
    )
    assert not first.is_day
    assert (first.temperature, first.feels_like) == (15.4, 13.0)
    assert (first.humidity, first.wind_speed) == (62, 12.8)
    assert [(d.date, d.condition, d.high, d.low) for d in first.days] == [
        # Rain at night does not make a sunny day rainy.
        (datetime.date(2026, 10, 10), WeatherCondition.CLEAR, 16.7, 11.1),
        (datetime.date(2026, 10, 11), WeatherCondition.SNOW, 12.7, 8.4),
    ]
    hosts = [r.url.host for r in requests]
    assert hosts.count("geocoding-api.open-meteo.com") == 1
    assert requests[0].url.params["name"] == "somewhere"
    assert requests[1].url.params["latitude"] == "52.52"
    assert second.id == "id2"


def test_unknown_place():
    async def run():
        service = weather.Weather("Berlin", client([], places=()))
        return await service.card("id", {"city": "qwxyz"})

    card = asyncio.run(run())
    assert card.type == "NoticeCard"
    assert card.text == "I couldn't find qwxyz."


@pytest.mark.parametrize(
    ("said", "name"),
    [
        ("hamburg", "Hamburg"),
        ("breman", "Bremen"),
        ("München", "Munich"),
        ("cologne", "Köln"),
    ],
)
def test_place(said, name):
    async def run():
        # These are all in PLACES, or close to one, so none is looked up.
        service = weather.Weather("Berlin", client([], places=()))
        return await service.place(said)

    assert asyncio.run(run()).name == name


def test_unknown_weather_code():
    assert weather.condition(1234) == (WeatherCondition.CLOUDY, "Unknown")


def test_intent_fits_the_model_labels():
    # The English models' labels: lowercase letters, apostrophe and space.
    # The recognizer rejects samples and slot values with other characters.
    labels = "abcdefghijklmnopqrstuvwxyz' "
    char_to_int = {c: i for i, c in enumerate(labels)}
    recognizer = IntentRecognizer([weather.INTENT], char_to_int, len(labels), 0.04)
    assert len(recognizer.samples) == len(weather.INTENT.samples)


def test_places():
    assert weather.PLACES["hamburg"].name == "Hamburg"
    # Places can be said by other names.
    assert weather.PLACES["cologne"] == weather.PLACES["koln"]
    assert weather.PLACES["frankfurt"].name == "Frankfurt am Main"
    assert len(weather.PLACES) > 500


def test_card_for_city():
    requests = []

    async def run():
        service = weather.Weather("Berlin", client(requests))
        card = await service.card("id", {"city": "munchen"})
        await service.close()
        return card

    card = asyncio.run(run())
    assert card.location == "Munich"
    # The coordinates are known, so the place is not looked up.
    assert [r.url.host for r in requests] == ["api.open-meteo.com"]
    assert requests[0].url.params["latitude"] == "48.13743"


def day(codes, rain, sun):
    return weather._hours_by_day(hours("2026-10-10", codes, rain, sun))["2026-10-10"]


@pytest.mark.parametrize(
    ("codes", "rain", "sun", "expected"),
    [
        # Berlin on 10 October 2026: a shower in the morning, sunny after.
        (
            [2, 3, 61, 80, 3, 2, 3, 2, 1, 1, 1],
            [0, 0, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0],
            [0, 9, 0, 2, 60, 60, 60, 60, 60, 60, 60],
            WeatherCondition.PARTLY_CLOUDY,
        ),
        # And on the 11th: cloudy, a little rain in the afternoon.
        (
            [3, 3, 3, 3, 3, 3, 3, 3, 61, 61, 61],
            [0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.1, 0.3],
            [0, 0, 0, 0, 7, 26, 22, 60, 20, 4, 0],
            WeatherCondition.DRIZZLE,
        ),
        ([1] * 11, [0] * 11, [60] * 9 + [0, 0], WeatherCondition.CLEAR),
        ([3] * 11, [0] * 11, [5] * 11, WeatherCondition.CLOUDY),
        ([45] * 11, [0] * 11, [0] * 11, WeatherCondition.FOG),
        ([63] * 11, [1.0] * 11, [0] * 11, WeatherCondition.RAIN),
        ([95] * 6 + [63] * 5, [2.0] * 11, [0] * 11, WeatherCondition.THUNDERSTORM),
    ],
)
def test_day_condition(codes, rain, sun, expected):
    assert weather.day_condition(day(codes, rain, sun)) == expected
