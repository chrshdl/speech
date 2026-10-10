import asyncio
import datetime

import httpx
import pytest

from speech.backend import weather
from speech.backend.messages import WeatherCondition

PLACE = {"name": "Berlin", "latitude": 52.52, "longitude": 13.41}
FORECAST = {
    "current": {
        "temperature_2m": 15.4,
        "apparent_temperature": 13.0,
        "relative_humidity_2m": 62,
        "weather_code": 2,
        "wind_speed_10m": 12.8,
        "is_day": 0,
    },
    "daily": {
        "time": ["2026-10-10", "2026-10-11"],
        "weather_code": [80, 71],
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
        service = weather.Weather("Berlin", client(requests))
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
        (datetime.date(2026, 10, 10), WeatherCondition.RAIN, 16.7, 11.1),
        (datetime.date(2026, 10, 11), WeatherCondition.SNOW, 12.7, 8.4),
    ]
    # The place is looked up once.
    hosts = [r.url.host for r in requests]
    assert hosts.count("geocoding-api.open-meteo.com") == 1
    assert requests[1].url.params["latitude"] == "52.52"
    assert second.id == "id2"


def test_unknown_place():
    async def run():
        service = weather.Weather("Nowhere", client([], places=()))
        with pytest.raises(LookupError, match="Nowhere"):
            await service.card("id")

    asyncio.run(run())


def test_unknown_weather_code():
    assert weather.condition(1234) == (WeatherCondition.CLOUDY, "Unknown")


def test_intent_samples_fit_the_model_labels():
    # The English models' labels: lowercase letters, apostrophe and space.
    labels = set("abcdefghijklmnopqrstuvwxyz' ")
    for sample in weather.INTENT.samples:
        assert set(sample) <= labels, sample
