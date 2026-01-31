from mcp.server.fastmcp import FastMCP 
mcp=FastMCP("Weather")

@mcp.tool()
async def get_weather(city: str) -> dict:
    """Fetches weather information for a specific city."""
    # Simulate an asynchronous API call to a weather service
    return f"The weather in {city} is 22°C and partly cloudy."

if __name__ == "__main__":
    mcp.run(transport="streamable-http")

