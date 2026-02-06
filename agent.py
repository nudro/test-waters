"""
ReAct Agent for Vessel Tracking
Uses LangChain with OpenAI to orchestrate vessel tracking tools.
"""

import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from tools import get_all_tools


def load_api_key():
    """
    Load OpenAI API key from secure location.
    """
    key_path = Path("/home/nudro/Documents/keys/maritime-oai")
    
    if not key_path.exists():
        raise FileNotFoundError(
            f"API key file not found at {key_path}. "
            "Please ensure the key file exists."
        )
    
    with open(key_path, 'r') as f:
        key = f.read().strip()
    
    if not key:
        raise ValueError("API key file is empty")
    
    # Set as environment variable
    os.environ['OPENAI_API_KEY'] = key
    
    return key


def initialize_agent(model_name: str = "o1", temperature: float = 0.1):
    """
    Initialize the ReAct agent with all tools.
    
    Args:
        model_name: OpenAI model to use (default: o1)
        temperature: Model temperature (ignored for o1 models, default: 0.1)
    
    Returns:
        Configured agent graph
    """
    # Load API key
    load_api_key()
    
    # Initialize LLM
    # Note: o1 models don't support temperature parameter
    llm_kwargs = {"model": model_name}
    if not model_name.startswith("o1"):
        llm_kwargs["temperature"] = temperature
    
    llm = ChatOpenAI(**llm_kwargs)
    
    # Get all tools
    tools = get_all_tools()
    
    # Create agent using new LangChain 1.2+ API
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""You are a helpful assistant that can use tools to answer questions about vessel tracking and motion analysis.

You have access to tools for:
- Calculating motion metrics (centroid, velocity, speed, acceleration, direction)
- Querying vessel tracking data by track ID
- Processing videos for vessel detection and tracking
- Assigning track IDs via GUI

Always use the appropriate tools to answer user questions. Be precise and helpful.""",
        debug=False,  # Disable verbose debug output
    )
    
    return agent


def get_agent(model_name: str = "o1", temperature: float = 0.1):
    """
    Get a configured agent instance.
    
    Args:
        model_name: OpenAI model to use (default: o1)
        temperature: Model temperature (ignored for o1 models)
    
    Returns:
        Agent graph instance
    """
    return initialize_agent(model_name=model_name, temperature=temperature)


def run_query(agent_graph, query: str):
    """
    Run a query through the agent.
    
    Args:
        agent_graph: The agent graph instance
        query: User query string
    
    Returns:
        Agent response
    """
    try:
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Invoke the agent graph with the query
        result = agent_graph.invoke({"messages": [HumanMessage(content=query)]})
        
        # Extract the response from the result
        # The result structure may vary, so we handle different formats
        if isinstance(result, dict):
            if "messages" in result:
                # Get the last assistant message (AI response)
                messages = result["messages"]
                if messages:
                    # Find the last AI message (skip tool messages)
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) or (hasattr(msg, 'type') and msg.type == 'ai'):
                            if hasattr(msg, 'content') and msg.content:
                                return msg.content
                        elif hasattr(msg, 'content') and msg.content and not (hasattr(msg, 'type') and msg.type == 'tool'):
                            # Fallback for other message types (but not tool messages)
                            content = msg.content if hasattr(msg, 'content') else str(msg)
                            if content and content.strip():
                                return content
            if "output" in result:
                return result["output"]
            if "response" in result:
                return result["response"]
        
        # Fallback: return string representation
        return str(result)
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}"


if __name__ == "__main__":
    # Test agent initialization
    print("Initializing agent...")
    try:
        agent = get_agent()
        print("Agent initialized successfully!")
        
        # Test query
        test_query = "List all available tools"
        print(f"\nTest query: {test_query}")
        response = run_query(agent, test_query)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
