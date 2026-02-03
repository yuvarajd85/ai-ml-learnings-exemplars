from mcp.server.fastmcp import FastMCP
import math

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

@mcp.tool()
def divide(a: int, b: int) -> float:
    """Divide a by b."""
    if b == 0:
        raise ZeroDivisionError("Division by zero is not allowed")
    return a / b

@mcp.tool()
def sqrt(x: float) -> float:
    """Return the square root of a non-negative number."""
    if x < 0:
        raise ValueError("Square root of negative number is not supported")
    return math.sqrt(x)

if __name__ == "__main__":
    mcp.run(transport="stdio")