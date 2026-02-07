#!/usr/bin/env python3
"""
GraphWeaver LangChain Agent - Claude-powered autonomous FK discovery.

This agent uses Claude to:
1. Reason about what to explore
2. Decide which tables/columns to analyze
3. Call tools to discover FK relationships
4. Build knowledge graph
5. Generate insights
6. Use text and KG embeddings for semantic search

The agent THINKS using Claude and makes decisions, not just runs a script.

MODES:
- Interactive (default): Chat with the agent using new streaming API
- SDK Streaming (--sdk): Token-by-token streaming with Anthropic SDK
- Invoke (--invoke): Non-streaming invoke mode
- Autonomous (--auto): Run full discovery autonomously

FEATURES:
- Database exploration & FK discovery
- Neo4j knowledge graph building & analysis
- Text & KG embeddings with semantic search
- Business rules execution & lineage tracking (Marquez)
- RDF/SPARQL support (Apache Jena Fuseki)
- LTN rule learning
- Dynamic tool creation at runtime

UPDATED: Migrated from deprecated create_react_agent (langgraph.prebuilt) 
to new create_agent API (langchain.agents) with proper streaming support.
"""
import os
import sys
from typing import Optional, Any, TypedDict

# Force unbuffered output for streaming mode
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=False, write_through=True)

# =============================================================================
# New LangChain Imports - Using create_agent instead of create_react_agent
# =============================================================================
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    wrap_model_call,
    wrap_tool_call,
    before_model,
    after_model,
    ModelRequest,
    ModelResponse,
    AgentMiddleware,
)
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# =============================================================================
# Import tools, system prompt, and ALL_TOOLS from agent_tools module
# Re-export everything that streamlit_app.py and other consumers need
# =============================================================================
from agent_tools import (
    SYSTEM_PROMPT,
    ALL_TOOLS,
    # Re-export for streamlit_app.py and other consumers
    DataSourceConfig,
    Neo4jConfig,
    PostgreSQLConnector,
    Neo4jClient,
    get_pg,
    get_neo4j,
    get_pg_config,
    get_text_embedder,
    get_kg_embedder,
    get_fuseki,
    get_sparql,
    get_marquez,
    get_registry,
    get_rule_learner,
    get_rule_generator,
    get_rules_config,
    set_rules_config,
)

# =============================================================================
# Import streaming mode and tool dicts from agent_streaming module
# Re-export for streamlit_app.py
# =============================================================================
from agent_streaming import (
    run_interactive_streaming,
    STREAMING_TOOLS,
    STREAMING_TOOL_FUNCTIONS,
)


# =============================================================================
# Middleware for Error Handling (using NEW @wrap_tool_call decorator)
# =============================================================================

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        import traceback
        error_msg = f"Tool error: {type(e).__name__}: {e}"
        print(f"[TOOL ERROR] {error_msg}")
        print(traceback.format_exc())
        return ToolMessage(
            content=error_msg,
            tool_call_id=request.tool_call["id"]
        )


# =============================================================================
# Agent Creation using NEW create_agent API
# =============================================================================

def create_graphweaver_agent(verbose: bool = True):
    """Create the GraphWeaver agent using the NEW create_agent API.
    
    This replaces the deprecated create_react_agent from langgraph.prebuilt.
    
    Key differences from old API:
    - Uses langchain.agents.create_agent instead of langgraph.prebuilt.create_react_agent
    - System prompt is passed via system_prompt parameter (str or SystemMessage)
    - Middleware replaces checkpointer for customization
    - state_schema must be TypedDict (AgentState or subclass) if provided
    """

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable required")

    # Create the model with configuration
    model = ChatAnthropic(
        model="claude-opus-4-5-20251101",
        temperature=0.1,
        max_tokens=4096,
    )

    # Create agent with NEW API
    agent = create_agent(
        model=model,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[handle_tool_errors],
    )

    return agent


def extract_response(result) -> str:
    """Extract text response from agent result."""
    if not isinstance(result, dict):
        return str(result)

    messages = result.get("messages", [])
    if not messages:
        return str(result)

    last_msg = messages[-1]
    content = getattr(last_msg, 'content', None)

    if content is None:
        return str(last_msg)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get('type') == 'text':
                text_parts.append(block.get('text', ''))
            elif hasattr(block, 'text'):
                text_parts.append(block.text)
        return '\n'.join(text_parts) if text_parts else str(content)

    return str(content)


# =============================================================================
# Autonomous Discovery Mode
# =============================================================================

def run_autonomous_discovery(verbose: bool = True) -> dict:
    """Run fully autonomous FK discovery using Claude."""

    print("\n" + "=" * 60)
    print("  GraphWeaver Agent - Claude-Powered FK Discovery")
    print("  (Using NEW create_agent API)")
    print("=" * 60 + "\n")

    agent = create_graphweaver_agent(verbose=verbose)

    instruction = """Discover all foreign key relationships in this database.

Work autonomously:
1. Connect and explore all tables
2. Identify and validate FK candidates  
3. Build the Neo4j graph with confirmed FKs
4. Generate embeddings for semantic search
5. Analyze and report insights

Go!"""

    result = agent.invoke(
        {"messages": [{"role": "user", "content": instruction}]},
        config={"recursion_limit": 100}
    )
    response = extract_response(result)

    print("\n" + "=" * 60)
    print("  FINAL REPORT")
    print("=" * 60 + "\n")
    print(response)

    return {"output": response}


# =============================================================================
# Interactive Mode with NEW Streaming API
# =============================================================================

def run_interactive_with_streaming():
    """Run agent in interactive mode with streaming using NEW create_agent API.
    
    Uses the .stream() method with stream_mode="values" for step-by-step output.
    """

    agent = create_graphweaver_agent(verbose=True)
    history = []

    print("\n" + "=" * 60)
    print("  🕸️  GraphWeaver Agent - NEW API with Streaming")
    print("=" * 60)
    print("\nI can help you discover FK relationships in your database.")
    print("\nTry saying:")
    print("  • 'connect and show me the tables'")
    print("  • 'find all foreign keys'")
    print("  • 'is orders.customer_id a FK to customers?'")
    print("  • 'build the graph and analyze it'")
    print("  • 'load business rules from file and execute them'")
    print("  • 'generate embeddings and search for customer columns'")
    print("  • 'find tables similar to orders'")
    print("  • 'sync graph to RDF and run SPARQL queries'")
    print("  • 'learn rules with LTN and generate validation rules'")
    print("  • 'create a tool that generates an ERD'")
    print("\nType 'quit' to exit.\n")
    sys.stdout.flush()

    while True:
        try:
            sys.stdout.write("\033[92mYou:\033[0m ")
            sys.stdout.flush()
            user_input = sys.stdin.readline()

            if not user_input:
                print("\nEnd of input.")
                break

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            messages = history + [{"role": "user", "content": user_input}]

            print("\n\033[96m🤖 Agent:\033[0m ", end="")
            sys.stdout.flush()

            # Stream response using NEW API
            full_response = ""
            tool_calls_seen = set()

            for chunk in agent.stream(
                {"messages": messages},
                stream_mode="values",
                config={"recursion_limit": 100}
            ):
                if "messages" in chunk and chunk["messages"]:
                    latest_message = chunk["messages"][-1]

                    # Handle text content
                    content = getattr(latest_message, 'content', '')
                    if isinstance(content, str) and content:
                        new_content = content[len(full_response):]
                        if new_content:
                            print(new_content, end="", flush=True)
                            full_response = content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text = block.get('text', '')
                                new_content = text[len(full_response):]
                                if new_content:
                                    print(new_content, end="", flush=True)
                                    full_response = text

                    # Handle tool calls
                    tool_calls = getattr(latest_message, 'tool_calls', None)
                    if tool_calls:
                        for tc in tool_calls:
                            tc_id = tc.get('id', '')
                            if tc_id and tc_id not in tool_calls_seen:
                                tool_calls_seen.add(tc_id)
                                print(f"\n\n🔧 **{tc.get('name', 'unknown')}**", flush=True)

            print("\n")
            sys.stdout.flush()

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": full_response})

            if len(history) > 20:
                history = history[-20:]

        except EOFError:
            print("\nEnd of input.")
            break
        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()

    print("\n👋 Goodbye!")


# =============================================================================
# Interactive Mode (Non-streaming using invoke)
# =============================================================================

def run_interactive_invoke():
    """Run agent in interactive mode using invoke (non-streaming)."""

    agent = create_graphweaver_agent(verbose=True)
    history = []

    print("\n" + "=" * 60)
    print("  GraphWeaver Agent - Invoke Mode (non-streaming)")
    print("=" * 60)
    print("\nType 'quit' to exit.\n")
    sys.stdout.flush()

    while True:
        try:
            sys.stdout.write("You: ")
            sys.stdout.flush()
            user_input = sys.stdin.readline()

            if not user_input:
                break

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            print("\nThinking...", flush=True)

            messages = history + [{"role": "user", "content": user_input}]

            result = agent.invoke(
                {"messages": messages},
                config={"recursion_limit": 100}
            )

            response_text = extract_response(result)

            print(f"\nAgent: {response_text}\n")
            sys.stdout.flush()

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response_text})

            if len(history) > 20:
                history = history[-20:]

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()

    print("Goodbye!")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GraphWeaver Agent - FK Discovery & Knowledge Graph (NEW API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  (default)    Interactive chat using NEW create_agent API with streaming
  --sdk        Interactive chat with Anthropic SDK streaming (raw tokens)
  --invoke     Interactive chat using invoke (non-streaming)
  --auto       Run autonomous FK discovery then exit

Migration Notes:
  This version uses langchain.agents.create_agent instead of the 
  deprecated langgraph.prebuilt.create_react_agent.
  
  Key changes:
  - System prompt via system_prompt parameter (str or SystemMessage)
  - Middleware for customization (@wrap_tool_call, @before_model, etc.)
  - Custom state must be TypedDict extending AgentState
  - Streaming via .stream(stream_mode="values")

Examples:
  python agent.py              # NEW API with streaming
  python agent.py --sdk        # Anthropic SDK streaming
  python agent.py --invoke     # Non-streaming invoke
  python agent.py --auto       # Autonomous discovery
        """
    )
    parser.add_argument("--auto", "-a", action="store_true",
                       help="Run autonomous discovery then exit")
    parser.add_argument("--sdk", "-s", action="store_true",
                       help="Use Anthropic SDK streaming mode (raw tokens)")
    parser.add_argument("--invoke", "-i", action="store_true",
                       help="Use invoke mode (non-streaming)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Less verbose output")
    args = parser.parse_args()

    if args.auto:
        run_autonomous_discovery(verbose=not args.quiet)
    elif args.sdk:
        run_interactive_streaming()
    elif args.invoke:
        run_interactive_invoke()
    else:
        # Default: NEW API with streaming
        run_interactive_with_streaming()


if __name__ == "__main__":
    main()