from mcp.server.fastmcp import FastMCP 

mcp=FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtracts two numbers."""
    return a - b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

@mcp.tool()
def divide(a: int, b: int) -> float:
    """Divides two numbers."""
    if b == 0:
        return 0
    return a / b

#the transport ="stdio" argument is used to specify the communication method for the MCP server.
#use standard ip/op (stdin and stdout) to receive and respond to tool function


if __name__ == "__main__":
    mcp.run(transport="stdio")