from typing import Optional
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(city: str, region: Optional[str] = None, country: Optional[str] = None) -> dict:
    """Return simulated weather for a location."""
    location = ", ".join([p for p in [city, region, country] if p])

    return {
        "location": location,
        "temperature_c": 22,
        "condition": "partly cloudy",
        "source": "simulated"
    }

if __name__ == "__main__":
    mcp.run(transport="streamable-http")