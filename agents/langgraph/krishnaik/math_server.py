from mcp.server.fastmcp import FastMCP


mcp = FastMCP("math")



@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together.
    
    Args:
        a: First number to add
        b: Second number to add
        
    Returns:
        The sum of a and b
    """
    return a + b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract the second number from the first.
    
    Args:
        a: The number to subtract from
        b: The number to subtract
        
    Returns:
        The result of a - b
    """
    return a - b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together.
    
    Args:
        a: First number to multiply
        b: Second number to multiply
        
    Returns:
        The product of a and b
    """
    return a * b

@mcp.tool()
def divide(a: int, b: int) -> float:
    """Divide the first number by the second.
    
    Args:
        a: The dividend
        b: The divisor (must not be zero)
        
    Returns:
        The result of a divided by b as a float
        
    Raises:
        ZeroDivisionError: If b is zero
    """
    return a / b




if __name__ == "__main__":
    print("Starting math MCP server...")
    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        print("\nShutting down math MCP server...")
