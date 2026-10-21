"""
Answers questions for the weather with a card of the current weather and
the forecast, from Open-Meteo (https://open-meteo.com), which needs no
API key, and says it in a few sentences.
"""

import dataclasses
import datetime

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


# The daylight hours with this much rain or snow, in mm, are wet, and a day
# with WET_HOURS of them and WET_DAY mm in all is rainy or snowy, light
# below LIGHT mm.
WET = 0.1
WET_HOURS = 2
WET_DAY = 0.5
LIGHT = 1.0
# The share of daylight with sunshine of a clear and a partly cloudy day.
SUNNY = 0.75
PARTLY_SUNNY = 0.3

# Conditions with rain or snow, the most severe last.
PRECIPITATION = [
    WeatherCondition.DRIZZLE,
    WeatherCondition.RAIN,
    WeatherCondition.SNOW,
    WeatherCondition.THUNDERSTORM,
]


@dataclasses.dataclass
class Hour:
    weather_code: int
    precipitation: float
    # Seconds of sunshine in the hour.
    sunshine: float


def _hours_by_day(hourly):
    """The daylight hours of each day of Open-Meteo's hourly forecast."""
    days = {}
    for time, code, rain, sun, day in zip(
        hourly["time"],
        hourly["weather_code"],
        hourly["precipitation"],
        hourly["sunshine_duration"],
        hourly["is_day"],
        strict=True,
    ):
        if day:
            days.setdefault(time[:10], []).append(Hour(code, rain or 0.0, sun or 0.0))
    return days


def day_condition(hours):
    """
    The condition of a day as people see it, from its daylight hours.

    Open-Meteo's daily weather code is the day's most severe, so a shower
    at night makes a sunny day rainy. Rather, a day is rainy or snowy when
    it rains or snows WET_DAY mm in at least WET_HOURS of daylight, and else as
    sunny as its share of daylight with sunshine. A few drops are no rain.
    """
    if not hours:
        return WeatherCondition.CLOUDY
    wet = [h for h in hours if h.precipitation >= WET]
    if len(wet) >= WET_HOURS and sum(h.precipitation for h in wet) >= WET_DAY:
        kinds = [condition(h.weather_code)[0] for h in wet]
        kinds = [k for k in kinds if k in PRECIPITATION] or [WeatherCondition.RAIN]
        kind = max(set(kinds), key=lambda k: (kinds.count(k), PRECIPITATION.index(k)))
        if kind == WeatherCondition.RAIN and sum(h.precipitation for h in wet) < LIGHT:
            return WeatherCondition.DRIZZLE
        return kind
    sunshine = sum(h.sunshine for h in hours) / (3600 * len(hours))
    if sunshine >= SUNNY:
        return WeatherCondition.CLEAR
    if sunshine >= PARTLY_SUNNY:
        return WeatherCondition.PARTLY_CLOUDY
    fog = sum(condition(h.weather_code)[0] == WeatherCondition.FOG for h in hours)
    return WeatherCondition.FOG if fog > len(hours) / 2 else WeatherCondition.CLOUDY


# The weather now, after "it's 12 degrees", by day and by night.
NOW = {
    WeatherCondition.CLEAR: ("and sunny", "and clear"),
    WeatherCondition.PARTLY_CLOUDY: ("and partly cloudy",) * 2,
    WeatherCondition.CLOUDY: ("and cloudy",) * 2,
    WeatherCondition.FOG: ("and foggy",) * 2,
    WeatherCondition.DRIZZLE: ("and drizzling",) * 2,
    WeatherCondition.RAIN: ("and raining",) * 2,
    WeatherCondition.SNOW: ("and snowing",) * 2,
    WeatherCondition.THUNDERSTORM: ("with a thunderstorm",) * 2,
}
# A day's weather, after "today will be".
DAY = {
    WeatherCondition.CLEAR: "sunny",
    WeatherCondition.PARTLY_CLOUDY: "partly cloudy",
    WeatherCondition.CLOUDY: "cloudy",
    WeatherCondition.FOG: "foggy",
    WeatherCondition.DRIZZLE: "a little rainy",
    WeatherCondition.RAIN: "rainy",
    WeatherCondition.SNOW: "snowy",
    WeatherCondition.THUNDERSTORM: "stormy",
}


def _degrees(value, unit=True):
    """A temperature as it is said, such as "minus 3 degrees"."""
    degrees = round(value) or 0  # Not "-0".
    said = f"minus {-degrees}" if degrees < 0 else str(degrees)
    if unit:
        said += " degree" if abs(degrees) == 1 else " degrees"
    return said


def summary(card):
    """
    The weather of a WeatherCard in a few short sentences, to be said: the
    weather now, and today's and tomorrow's. The first is the shortest, so
    it is heard soonest.
    """
    now = NOW[card.condition][0 if card.is_day else 1]
    sentences = [f"In {card.location} it's {_degrees(card.temperature)} {now}."]
    for name, day in zip(("Today", "Tomorrow"), card.days):
        sentences.append(
            f"{name} will be {DAY[day.condition]}, with a high of "
            f"{_degrees(day.high, unit=False)} and a low of {_degrees(day.low)}."
        )
    return " ".join(sentences)


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

    def speech(self, card):
        """What to say for a card of card(): the summary of the weather."""
        return summary(card) if isinstance(card, WeatherCard) else None

    async def place(self, name):
        """
        The Place of a name, said or written: one of PLACES, or one close
        to it, such as a misspelled --weather-location, or else one
        Open-Meteo finds. None if there is none.
        """
        said = places.spoken(name) or ""
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
                "hourly": "weather_code,precipitation,sunshine_duration,is_day",
                "daily": "temperature_2m_max,temperature_2m_min",
                "timezone": "auto",
                "forecast_days": DAYS,
            },
        )
        response.raise_for_status()
        forecast = response.json()
        current = forecast["current"]
        daily = forecast["daily"]
        now, description = condition(current["weather_code"])
        hours = _hours_by_day(forecast["hourly"])
        days = [
            WeatherDay(
                date=datetime.date.fromisoformat(date),
                condition=day_condition(hours.get(date, [])),
                high=high,
                low=low,
            )
            for date, high, low in zip(
                daily["time"],
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
