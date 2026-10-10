"""
Answers questions for the weather with a card of the current weather and
the forecast, from Open-Meteo (https://open-meteo.com), which needs no
API key.
"""

import datetime

import httpx

from speech.backend.messages import WeatherCard, WeatherCondition, WeatherDay
from speech.intents import Intent

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
    ),
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
            location (str): The place to report the weather for, such as
                "Berlin", which Open-Meteo looks up when first needed.
            client (httpx.AsyncClient, optional): Makes the requests.
        """
        self.location = location
        self.client = client or httpx.AsyncClient(timeout=5)
        self.place = None

    async def card(self, card_id):
        """Returns a WeatherCard. Raises an httpx.HTTPError if offline."""
        if self.place is None:
            self.place = await self._find(self.location)
        response = await self.client.get(
            FORECAST_URL,
            params={
                "latitude": self.place["latitude"],
                "longitude": self.place["longitude"],
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
            location=self.place["name"],
            condition=now,
            description=description,
            is_day=bool(current["is_day"]),
            temperature=current["temperature_2m"],
            feels_like=current["apparent_temperature"],
            humidity=current["relative_humidity_2m"],
            wind_speed=current["wind_speed_10m"],
            days=days,
        )

    async def _find(self, name):
        response = await self.client.get(
            GEOCODING_URL, params={"name": name, "count": 1, "format": "json"}
        )
        response.raise_for_status()
        results = response.json().get("results")
        if not results:
            raise LookupError(f"Open-Meteo does not know the place {name!r}.")
        return results[0]

    async def close(self):
        await self.client.aclose()
