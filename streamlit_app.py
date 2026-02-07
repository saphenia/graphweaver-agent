#!/usr/bin/env python3
# =============================================================================
# FILE: streamlit_app.py
# =============================================================================
"""
Streamlit Chat Interface for GraphWeaver Agent

REFACTORED: Imports tools and agent from agent.py (which re-exports from agent_tools.py)
No more duplicate tool definitions!

Run with: DEBUG=1 streamlit run streamlit_app.py
"""
import os
import sys
import streamlit as st
from typing import List, Dict
import json
import re
import anthropic

# =============================================================================
# Path setup
# =============================================================================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mcp_servers"))

# =============================================================================
# Import EVERYTHING from the agent - no duplication!
# =============================================================================
from agent import (
    # System prompt
    SYSTEM_PROMPT,
    
    # Agent creation
    create_graphweaver_agent,
    
    # All tools (for SDK streaming)
    ALL_TOOLS,
    STREAMING_TOOLS,
    STREAMING_TOOL_FUNCTIONS,
    
    # State accessors (we'll wrap these for Streamlit session state)
    DataSourceConfig,
    Neo4jConfig,
    PostgreSQLConnector,
    Neo4jClient,
)

# Import LangChain message types
from langchain.messages import HumanMessage, AIMessage

# =============================================================================
# Debug setup
# =============================================================================
os.environ['PYTHONUNBUFFERED'] = '1'
DEBUG_MODE = os.environ.get("DEBUG", "0").lower() in ("1", "true", "yes")


# =============================================================================
# Streamlit Session State Wrappers
# 
# These override the module-level singletons in agent_tools.py
# to use Streamlit's session_state (required because Streamlit reruns the script)
# =============================================================================

def get_pg() -> PostgreSQLConnector:
    """Get PostgreSQL connector from session state."""
    if "pg_connector" not in st.session_state:
        config = DataSourceConfig(
            host=st.session_state.get("pg_host", os.environ.get("POSTGRES_HOST", "localhost")),
            port=int(st.session_state.get("pg_port", os.environ.get("POSTGRES_PORT", "5432"))),
            database=st.session_state.get("pg_database", os.environ.get("POSTGRES_DB", "orders")),
            username=st.session_state.get("pg_username", os.environ.get("POSTGRES_USER", "saphenia")),
            password=st.session_state.get("pg_password", os.environ.get("POSTGRES_PASSWORD", "secret")),
        )
        st.session_state.pg_connector = PostgreSQLConnector(config)
    return st.session_state.pg_connector


def get_neo4j() -> Neo4jClient:
    """Get Neo4j client from session state."""
    if "neo4j_client" not in st.session_state:
        config = Neo4jConfig(
            uri=st.session_state.get("neo4j_uri", os.environ.get("NEO4J_URI", "bolt://localhost:7687")),
            username=st.session_state.get("neo4j_user", os.environ.get("NEO4J_USER", "neo4j")),
            password=st.session_state.get("neo4j_password", os.environ.get("NEO4J_PASSWORD", "password")),
        )
        st.session_state.neo4j_client = Neo4jClient(config)
    return st.session_state.neo4j_client


# =============================================================================
# Patch the agent_tools module to use Streamlit session state
# =============================================================================
import agent_tools as agent_module

def _patched_get_pg():
    """Patched get_pg that uses Streamlit session state."""
    return get_pg()

def _patched_get_neo4j():
    """Patched get_neo4j that uses Streamlit session state."""
    return get_neo4j()

# Apply patches - tools will now use Streamlit session state
agent_module.get_pg = _patched_get_pg
agent_module.get_neo4j = _patched_get_neo4j


# =============================================================================
# Agent Creation (uses the agent's create function)
# =============================================================================

def get_agent():
    """Get or create the GraphWeaver agent."""
    if "agent" not in st.session_state:
        api_key = st.session_state.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        
        # Set the API key in environment for the agent
        os.environ["ANTHROPIC_API_KEY"] = api_key
        
        st.session_state.agent = create_graphweaver_agent(verbose=True)
    
    return st.session_state.agent


# =============================================================================
# SDK Tool Execution (uses the agent's tool functions directly)
# =============================================================================

def execute_tool_by_name(tool_name: str, tool_input: Dict) -> str:
    """Execute a tool by name using the agent's tool functions."""
    fn = STREAMING_TOOL_FUNCTIONS.get(tool_name)
    if fn:
        try:
            return fn(**tool_input)
        except Exception as e:
            import traceback
            return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"
    return f"ERROR: Unknown tool: {tool_name}"


# =============================================================================
# Streaming with Anthropic SDK (token-by-token)
# =============================================================================

def stream_with_anthropic_sdk(messages: List[Dict], message_placeholder, status_placeholder) -> str:
    """TRUE token-by-token streaming using Anthropic SDK directly."""
    
    api_key = st.session_state.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "ERROR: No API key configured"
    
    client = anthropic.Anthropic(api_key=api_key)
    
    full_response = ""
    api_messages = messages.copy()
    iteration = 0
    max_iterations = 20
    tool_was_called = False
    
    # Detect if user is asking for a tool action
    user_msg = api_messages[-1]["content"] if api_messages else ""
    if isinstance(user_msg, str):
        user_lower = user_msg.lower()
        needs_tool = any(phrase in user_lower for phrase in [
            "generate text embeddings", "generate kg embeddings", "generate embeddings",
            "create embeddings", "text embedding", "kg embedding", "create vector indexes",
            "connect datasets", "list tables", "show tables", "graph stats",
            "run fk discovery", "discover fk", "foreign key",
        ])
    else:
        needs_tool = False
    
    while iteration < max_iterations:
        iteration += 1
        
        if iteration > 1:
            status_placeholder.info(f"🔄 Processing... (iteration {iteration})")
        
        current_tool_id = None
        current_tool_name = None
        current_tool_input = ""
        tool_use_blocks = []
        
        try:
            api_params = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 4096,
                "system": SYSTEM_PROMPT,
                "messages": api_messages,
                "tools": STREAMING_TOOLS,  # Use the agent's tool definitions!
            }
            
            # Force tool usage when user clearly wants a tool
            if iteration == 1 and needs_tool and not tool_was_called:
                api_params["tool_choice"] = {"type": "any"}
            
            with client.messages.stream(**api_params) as stream:
                for event in stream:
                    if event.type == "content_block_start":
                        if hasattr(event.content_block, 'type'):
                            if event.content_block.type == "tool_use":
                                tool_was_called = True
                                current_tool_id = event.content_block.id
                                current_tool_name = event.content_block.name
                                current_tool_input = ""
                                full_response += f"\n\n🔧 **{current_tool_name}**\n"
                                message_placeholder.markdown(full_response + "▌")
                    
                    elif event.type == "content_block_delta":
                        if hasattr(event.delta, 'text'):
                            full_response += event.delta.text
                            message_placeholder.markdown(full_response + "▌")
                        elif hasattr(event.delta, 'partial_json'):
                            current_tool_input += event.delta.partial_json
                    
                    elif event.type == "content_block_stop":
                        if current_tool_id and current_tool_name:
                            try:
                                tool_input = json.loads(current_tool_input) if current_tool_input else {}
                            except json.JSONDecodeError:
                                tool_input = {}
                            
                            tool_use_blocks.append({
                                "id": current_tool_id,
                                "name": current_tool_name,
                                "input": tool_input
                            })
                            current_tool_id = None
                            current_tool_name = None
                            current_tool_input = ""
                
                final_message = stream.get_final_message()
            
            # Check for hallucination
            if final_message.stop_reason != "tool_use":
                hallucination_patterns = [
                    r"## Text Embeddings Generated", r"## KG Embeddings Generated",
                    r"Tables processed:\s*\d+", r"Columns processed:\s*\d+",
                    r"Total embeddings generated:\s*\d+", r"✓ Embeddings created",
                ]
                
                is_hallucination = any(
                    re.search(pattern, full_response, re.IGNORECASE) 
                    for pattern in hallucination_patterns
                )
                
                if is_hallucination and needs_tool and not tool_was_called and iteration == 1:
                    full_response = ""
                    api_messages_retry = messages.copy()
                    api_messages_retry.append({
                        "role": "assistant",
                        "content": "I need to actually call the tool, not describe what it would do."
                    })
                    api_messages_retry.append({
                        "role": "user", 
                        "content": "You generated fake output. Please ACTUALLY CALL the tool. Do not write any text - just call the tool."
                    })
                    api_messages = api_messages_retry
                    continue
                
                break
            
            # Add assistant message
            api_messages.append({
                "role": "assistant",
                "content": final_message.content
            })
            
            # Execute tools
            tool_results = []
            for tool_block in tool_use_blocks:
                tool_name = tool_block["name"]
                tool_input = tool_block["input"]
                tool_id = tool_block["id"]
                
                status_placeholder.info(f"⏳ Executing: {tool_name}...")
                
                result = execute_tool_by_name(tool_name, tool_input)
                
                preview = result[:500] + "..." if len(result) > 500 else result
                full_response += f"\n```\n{preview}\n```\n"
                message_placeholder.markdown(full_response + "▌")
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result
                })
            
            api_messages.append({
                "role": "user",
                "content": tool_results
            })
            
            status_placeholder.empty()
            
        except anthropic.APIError as e:
            full_response += f"\n\n❌ **API Error:** {e}"
            message_placeholder.markdown(full_response)
            break
        except Exception as e:
            full_response += f"\n\n❌ **Error:** {type(e).__name__}: {e}"
            message_placeholder.markdown(full_response)
            break
    
    message_placeholder.markdown(full_response)
    status_placeholder.empty()
    return full_response


# =============================================================================
# Streaming with LangChain Agent (step-by-step)
# =============================================================================

def stream_agent_response(messages: List[Dict], message_placeholder) -> str:
    """Stream response from LangChain agent (step-by-step updates)."""
    
    agent = get_agent()
    if agent is None:
        return "ERROR: Agent not initialized. Check API key."
    
    full_response = ""
    tool_calls_seen = set()
    
    try:
        # Convert to LangChain format
        lc_messages = []
        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        
        # Stream response
        for chunk in agent.stream(
            {"messages": lc_messages},
            stream_mode="values",
            config={"recursion_limit": 100}
        ):
            if "messages" in chunk and chunk["messages"]:
                latest_message = chunk["messages"][-1]
                
                content = getattr(latest_message, 'content', '')
                if isinstance(content, str) and content:
                    new_content = content[len(full_response):]
                    if new_content:
                        full_response = content
                        message_placeholder.markdown(full_response + "▌")
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            text = block.get('text', '')
                            new_content = text[len(full_response):]
                            if new_content:
                                full_response = text
                                message_placeholder.markdown(full_response + "▌")
                
                tool_calls = getattr(latest_message, 'tool_calls', None)
                if tool_calls:
                    for tc in tool_calls:
                        tc_id = tc.get('id', '')
                        if tc_id and tc_id not in tool_calls_seen:
                            tool_calls_seen.add(tc_id)
                            full_response += f"\n\n🔧 **{tc.get('name', 'unknown')}**\n"
                            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        return full_response
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {type(e).__name__}: {e}\n\n```\n{traceback.format_exc()}\n```"
        message_placeholder.markdown(error_msg)
        return error_msg


# =============================================================================
# Streamlit UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="GraphWeaver Agent",
        page_icon="🕸️",
        layout="wide",
    )
    
    st.title("🕸️ GraphWeaver Agent")
    st.caption("FK Discovery • Knowledge Graph • Semantic Search • Business Rules • RDF/SPARQL")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        if DEBUG_MODE:
            st.success("🔍 DEBUG MODE: ON")
        
        # API Key
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=st.session_state.get("anthropic_api_key", os.environ.get("ANTHROPIC_API_KEY", "")),
        )
        if api_key:
            st.session_state.anthropic_api_key = api_key
            if "agent" in st.session_state:
                del st.session_state.agent
        
        st.divider()
        
        # Streaming mode
        st.subheader("🎯 Streaming Mode")
        streaming_mode = st.radio(
            "Choose streaming method:",
            ["Anthropic SDK (token-by-token)", "LangChain Agent (step-by-step)"],
            index=0,
            help="SDK mode gives real-time character output. Agent mode updates after each tool call."
        )
        st.session_state.streaming_mode = streaming_mode
        
        st.divider()
        
        # PostgreSQL config
        st.subheader("🗄️ PostgreSQL")
        pg_host = st.text_input("Host", value=st.session_state.get("pg_host", os.environ.get("POSTGRES_HOST", "localhost")))
        pg_port = st.number_input("Port", value=int(st.session_state.get("pg_port", os.environ.get("POSTGRES_PORT", "5432"))))
        pg_database = st.text_input("Database", value=st.session_state.get("pg_database", os.environ.get("POSTGRES_DB", "orders")))
        pg_username = st.text_input("Username", value=st.session_state.get("pg_username", os.environ.get("POSTGRES_USER", "saphenia")))
        pg_password = st.text_input("Password", type="password", value=st.session_state.get("pg_password", os.environ.get("POSTGRES_PASSWORD", "secret")))
        
        st.session_state.pg_host = pg_host
        st.session_state.pg_port = pg_port
        st.session_state.pg_database = pg_database
        st.session_state.pg_username = pg_username
        st.session_state.pg_password = pg_password
        
        # Clear cached connector if config changes
        if "pg_connector" in st.session_state:
            del st.session_state.pg_connector
        
        st.divider()
        
        # Neo4j config
        st.subheader("🔵 Neo4j")
        neo4j_uri = st.text_input("URI", value=st.session_state.get("neo4j_uri", os.environ.get("NEO4J_URI", "bolt://localhost:7687")))
        neo4j_user = st.text_input("Neo4j User", value=st.session_state.get("neo4j_user", os.environ.get("NEO4J_USER", "neo4j")))
        neo4j_password = st.text_input("Neo4j Password", type="password", value=st.session_state.get("neo4j_password", os.environ.get("NEO4J_PASSWORD", "password")))
        
        st.session_state.neo4j_uri = neo4j_uri
        st.session_state.neo4j_user = neo4j_user
        st.session_state.neo4j_password = neo4j_password
        
        # Clear cached client if config changes
        if "neo4j_client" in st.session_state:
            del st.session_state.neo4j_client
        
        st.divider()
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        if st.button("🔍 Discover FKs", use_container_width=True):
            st.session_state.quick_action = "Run FK discovery on the database"
        if st.button("📊 List Tables", use_container_width=True):
            st.session_state.quick_action = "List all database tables"
        if st.button("🧠 Generate Embeddings", use_container_width=True):
            st.session_state.quick_action = "Generate text embeddings for semantic search"
        if st.button("📈 Graph Stats", use_container_width=True):
            st.session_state.quick_action = "Show Neo4j graph statistics"
        if st.button("🔗 Sync to RDF", use_container_width=True):
            st.session_state.quick_action = "Sync the Neo4j graph to RDF/Fuseki"
        
        st.divider()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.streaming_messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "streaming_messages" not in st.session_state:
        st.session_state.streaming_messages = []
    
    # Check API key
    if not st.session_state.get("anthropic_api_key") and not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("⚠️ Please enter your Anthropic API key in the sidebar to start chatting.")
        st.stop()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle quick action or user input
    if "quick_action" in st.session_state and st.session_state.quick_action:
        prompt = st.session_state.quick_action
        st.session_state.quick_action = None
    else:
        prompt = st.chat_input("Ask me anything about your database...")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                st.session_state.streaming_messages.append({"role": "user", "content": prompt})
                message_placeholder = st.empty()
                status_placeholder = st.empty()
                
                # Choose streaming method
                if "SDK" in st.session_state.get("streaming_mode", ""):
                    response = stream_with_anthropic_sdk(
                        st.session_state.streaming_messages.copy(),
                        message_placeholder,
                        status_placeholder
                    )
                else:
                    response = stream_agent_response(
                        st.session_state.streaming_messages.copy(),
                        message_placeholder
                    )
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.streaming_messages.append({"role": "assistant", "content": response})
                
                # Trim history
                if len(st.session_state.streaming_messages) > 20:
                    st.session_state.streaming_messages = st.session_state.streaming_messages[-20:]
                    
            except Exception as e:
                import traceback
                error_msg = f"Error: {type(e).__name__}: {e}\n\n```\n{traceback.format_exc()}\n```"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})


if __name__ == "__main__":
    main()