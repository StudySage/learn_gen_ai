# Getting Started with LLM Agents and MCP Support

A step by step example to implement a MCP server with basic math functions and its client using LangChain and OpenAI APIs. You can use the LLM client to do math queries like "what is the result of (3+5)*2". 

**Prerequisites (as mentioned in the article):**

1.  Create and activate a Python virtual environment:
    ```bash
    python3 -m venv MCP_Demo
    source MCP_Demo/bin/activate
    ```
2.  Install the necessary libraries:
    ```bash
    pip install langchain-mcp-adapters langgraph langchain-openai
    ```
3.  Set your OpenAI API key (replace `<your_api_key>` with your actual key):
    ```bash
    export OPENAI_API_KEY=<your_api_key>
    ```

---

**1. MCP Server Code (`math_server.py`)**

This script defines the MCP server and the math tools (`add`, `multiply`).

```python
# math_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Adds two integers.""" # Added a docstring for clarity, though not in original
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiplies two integers.""" # Added a docstring for clarity, though not in original
    return a * b

if __name__ == "__main__":
    print("Starting Math MCP Server via stdio...")
    mcp.run(transport="stdio")
    print("Math MCP Server stopped.")
```

**To run the server:**

```bash
python3 math_server.py
```
It will wait for the client to connect via standard input/output.

---

**2. Client Code (`client.py`)**

This script sets up the LangGraph agent, connects to the `math_server.py` using MCP, loads the tools, and runs a query.

* **Important:** Make sure `math_server.py` is runnable from where you execute this script, or update the path in `StdioServerParameters`. The example uses a relative path, assuming they are in the same directory.

```python
# client.py
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio
import os # Import os to check for API key

# Ensure the API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set.")

model = ChatOpenAI(model="gpt-4o") # Or specify your preferred model

# Parameters to run the math_server.py script via stdio
server_params = StdioServerParameters(
    command="python3", # Use python3 explicitly if needed
    args=["math_server.py"],  # Assumes math_server.py is in the same directory.
                              # Otherwise, provide the absolute path:
                              # args=["/path/to/your/math_server.py"]
)

async def run_agent():
    print("Attempting to connect to MCP server...")
    async with stdio_client(server_params) as (read, write):
        print("MCP stdio client connected. Initializing session...")
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("MCP session initialized. Loading tools...")
            # Load tools defined in the MCP server
            tools = await load_mcp_tools(session)
            print(f"Tools loaded: {[tool.name for tool in tools]}")

            # Create a ReAct agent using LangGraph
            print("Creating LangGraph agent...")
            agent_executor = create_react_agent(model, tools)
            print("Agent created. Invoking agent...")

            # Define the input for the agent
            # The input format depends on the agent type (create_react_agent expects messages)
            inputs = {"messages": [("human", "what's (3 + 5) x 12?")]}

            # Invoke the agent asynchronously
            result = await agent_executor.ainvoke(inputs)
            print("Agent invocation complete.")
            return result

if __name__ == "__main__":
    print("Running MCP client...")
    # Use asyncio.run to execute the async function
    final_result = asyncio.run(run_agent())
    print("\n--- Agent Result ---")
    # The result structure might vary slightly based on LangChain/LangGraph versions,
    # but typically the final answer is in the 'messages' list.
    print(final_result)
    # Print the last message content for clarity
    if final_result and 'messages' in final_result and len(final_result['messages']) > 0:
         print("\nFinal Answer:", final_result['messages'][-1].content)
    print("--------------------")

```

**To run the client (after starting the server):**

Open a *new* terminal window (while the server is running in the first one), navigate to the same directory, activate the virtual environment, and run:

```bash
python3 client.py
```

This will start the client, which in turn runs `math_server.py` as a subprocess, connects to it via MCP, uses the LLM and the loaded math tools to figure out the steps for "(3 + 5) x 12", executes the steps using the tools, and finally prints the result object containing the conversation history and the final answer.