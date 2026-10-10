"""
Answers questions for the weather with a card of the current weather and
the forecast, from Open-Meteo (https://open-meteo.com), which needs no
API key.
"""

import datetime
import re

import httpx
from rapidfuzz import fuzz, process

from speech.backend import places
from speech.backend.messages import (
    NoticeCard,
    WeatherCard,
    WeatherCondition,
    WeatherDay,
)
from speech.intents import Intent

# The places the weather can be asked for by name.
PLACES = places.load()
# The lowest similarity, out of 100, of a misspelled name to one of
# PLACES, such as "breman" to "bremen".
CLOSE = 80
# Words said after a place, such as "what's the weather in hamburg today".
TRAILING = re.compile(r"\s+(today|tomorrow|now|right now|please)$")
# Primes Whisper for the words of the intent, so it writes "weather" rather
# than "whether".
PROMPT = "What's the weather in Hamburg?"

INTENT = Intent(
    "GetWeather",
    (
        "what's the weather",
        "what is the weather",
        "how is the weather",
        "how's the weather",
        "what's the weather like",
        "what's the weather like today",
        "how's the weather today",
        "how's the weather tomorrow",
        "what's the weather tomorrow",
        "tell me the weather",
        "what's the weather forecast",
        "weather forecast",
        "weather",
        "what's the weather in {city}",
        "what is the weather in {city}",
        "how is the weather in {city}",
        "how's the weather in {city}",
        "what's the weather like in {city}",
        "what's the weather in {city} today",
        "what's the weather in {city} tomorrow",
        "tell me the weather in {city}",
        "weather in {city}",
    ),
    {"city": tuple(PLACES)},
)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Today and the next three days.
DAYS = 4

# The WMO weather codes Open-Meteo reports, see its documentation.
CONDITIONS = {
    0: (WeatherCondition.CLEAR, "Clear"),
    1: (WeatherCondition.CLEAR, "Mainly clear"),
    2: (WeatherCondition.PARTLY_CLOUDY, "Partly cloudy"),
    3: (WeatherCondition.CLOUDY, "Overcast"),
    45: (WeatherCondition.FOG, "Fog"),
    48: (WeatherCondition.FOG, "Freezing fog"),
    51: (WeatherCondition.DRIZZLE, "Light drizzle"),
    53: (WeatherCondition.DRIZZLE, "Drizzle"),
    55: (WeatherCondition.DRIZZLE, "Heavy drizzle"),
    56: (WeatherCondition.DRIZZLE, "Freezing drizzle"),
    57: (WeatherCondition.DRIZZLE, "Freezing drizzle"),
    61: (WeatherCondition.RAIN, "Light rain"),
    63: (WeatherCondition.RAIN, "Rain"),
    65: (WeatherCondition.RAIN, "Heavy rain"),
    66: (WeatherCondition.RAIN, "Freezing rain"),
    67: (WeatherCondition.RAIN, "Freezing rain"),
    71: (WeatherCondition.SNOW, "Light snow"),
    73: (WeatherCondition.SNOW, "Snow"),
    75: (WeatherCondition.SNOW, "Heavy snow"),
    77: (WeatherCondition.SNOW, "Snow grains"),
    80: (WeatherCondition.RAIN, "Rain showers"),
    81: (WeatherCondition.RAIN, "Rain showers"),
    82: (WeatherCondition.RAIN, "Heavy rain showers"),
    85: (WeatherCondition.SNOW, "Snow showers"),
    86: (WeatherCondition.SNOW, "Heavy snow showers"),
    95: (WeatherCondition.THUNDERSTORM, "Thunderstorm"),
    96: (WeatherCondition.THUNDERSTORM, "Thunderstorm with hail"),
    99: (WeatherCondition.THUNDERSTORM, "Thunderstorm with hail"),
}


def condition(code):
    """The condition and its description for a WMO weather code."""
    return CONDITIONS.get(code, (WeatherCondition.CLOUDY, "Unknown"))


class Weather:
    def __init__(self, location, client=None):
        """
        Arguments:
            location (str): The place to report the weather for when none
                is asked for, such as "Berlin".
            client (httpx.AsyncClient, optional): Makes the requests.
        """
        self.location = location
        self.client = client or httpx.AsyncClient(timeout=5)
        # The places looked up with Open-Meteo, by name.
        self.found = {}

    async def card(self, card_id, slots=None):
        """
        Returns a WeatherCard for the city slot, or else for the location,
        or a NoticeCard if the place is unknown. Raises an httpx.HTTPError
        if offline.
        """
        city = (slots or {}).get("city")
        place = await self.place(city or self.location)
        if place is None:
            return NoticeCard(
                id=card_id,
                title="Sorry, I don't know that place",
                text=f"I couldn't find {city or self.location}.",
            )
        return await self._forecast(card_id, place)

    async def place(self, name):
        """
        The Place of a name, said or written: one of PLACES, or one close
        to it, as for a misspelled transcript, or else one Open-Meteo
        finds. None if there is none.
        """
        said = places.spoken(name) or ""
        said = TRAILING.sub("", said)
        if said in PLACES:
            return PLACES[said]
        close = process.extractOne(
            said, list(PLACES), scorer=fuzz.ratio, score_cutoff=CLOSE
        )
        if close is not None:
            return PLACES[close[0]]
        if said not in self.found:
            self.found[said] = await self._find(said)
        return self.found[said]

    async def _find(self, name):
        response = await self.client.get(
            GEOCODING_URL, params={"name": name, "count": 1, "format": "json"}
        )
        response.raise_for_status()
        results = response.json().get("results")
        if not results:
            return None
        result = results[0]
        return places.Place(
            result["name"],
            result.get("country_code", ""),
            result["latitude"],
            result["longitude"],
            result.get("population", 0),
        )

    async def _forecast(self, card_id, place):
        response = await self.client.get(
            FORECAST_URL,
            params={
                "latitude": place.latitude,
                "longitude": place.longitude,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "weather_code,wind_speed_10m,is_day",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                "timezone": "auto",
                "forecast_days": DAYS,
            },
        )
        response.raise_for_status()
        forecast = response.json()
        current = forecast["current"]
        daily = forecast["daily"]
        now, description = condition(current["weather_code"])
        days = [
            WeatherDay(
                date=datetime.date.fromisoformat(date),
                condition=condition(code)[0],
                high=high,
                low=low,
            )
            for date, code, high, low in zip(
                daily["time"],
                daily["weather_code"],
                daily["temperature_2m_max"],
                daily["temperature_2m_min"],
                strict=True,
            )
        ]
        return WeatherCard(
            id=card_id,
            location=place.name,
            condition=now,
            description=description,
            is_day=bool(current["is_day"]),
            temperature=current["temperature_2m"],
            feels_like=current["apparent_temperature"],
            humidity=current["relative_humidity_2m"],
            wind_speed=current["wind_speed_10m"],
            days=days,
        )

    async def close(self):
        await self.client.aclose()
