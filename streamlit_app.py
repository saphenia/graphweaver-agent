#!/usr/bin/env python3
"""
GraphWeaver Agent - Streamlit UI

This UI imports the LangGraph agent from agent.py - single source of truth for tools.
No duplicate tool definitions!
"""
import os
import sys
import streamlit as st
from typing import Dict, List

# =============================================================================
# Path Setup & Debug
# =============================================================================
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mcp_servers"))

from debug_logger import DebugLogger, debug

DEBUG_MODE = os.environ.get("DEBUG", "1").lower() in ("1", "true", "yes")
if DEBUG_MODE:
    DebugLogger.enable(verbose=True, log_file="agent_debug.log")

# =============================================================================
# Import agent from agent.py - SINGLE SOURCE OF TRUTH
# =============================================================================
from agent import create_agent, ALL_TOOLS, SYSTEM_PROMPT

from langchain_core.messages import HumanMessage, AIMessage


# =============================================================================
# LangGraph Streaming for Streamlit
# =============================================================================

def stream_agent_response(agent, user_message: str, chat_history: List[Dict], message_placeholder) -> str:
    """Stream response from LangGraph agent with Streamlit UI updates."""
    full_response = ""
    
    # Convert chat history to LangGraph message format
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    
    # Add current user message
    messages.append(HumanMessage(content=user_message))
    
    debug.section("LANGGRAPH AGENT CALL")
    debug.agent(f"User message: {user_message[:100]}...")
    debug.agent(f"History length: {len(chat_history)}")
    
    # Config with higher recursion limit for complex multi-step operations
    config = {"recursion_limit": 100}
    
    try:
        # Stream events from LangGraph agent
        for event in agent.stream({"messages": messages}, config=config):
            
            # Handle agent output (text and tool calls)
            if "agent" in event:
                for msg in event["agent"].get("messages", []):
                    content = getattr(msg, 'content', None)
                    
                    if isinstance(content, str):
                        # Plain text response
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")
                        
                    elif isinstance(content, list):
                        # Mixed content (text blocks + tool use blocks)
                        for block in content:
                            if hasattr(block, 'text'):
                                full_response += block.text
                                message_placeholder.markdown(full_response + "▌")
                            elif hasattr(block, 'name'):
                                # Tool call
                                tool_name = block.name
                                full_response += f"\n\n🔧 **Calling: {tool_name}**\n"
                                message_placeholder.markdown(full_response + "▌")
                                debug.tool(f"Tool call: {tool_name}")
            
            # Handle tool results
            if "tools" in event:
                for msg in event["tools"].get("messages", []):
                    tool_content = getattr(msg, 'content', '')
                    if tool_content:
                        full_response += f"\n{tool_content}\n"
                        message_placeholder.markdown(full_response + "▌")
                        debug.tool(f"Tool result: {tool_content[:200]}...")
        
        # Final render without cursor
        message_placeholder.markdown(full_response)
        debug.agent("Agent finished")
        return full_response
        
    except Exception as e:
        import traceback
        error_msg = f"**Error:** {e}\n\n```\n{traceback.format_exc()}\n```"
        debug.error(f"Agent error: {e}", e)
        message_placeholder.markdown(error_msg)
        return error_msg


# =============================================================================
# Streamlit UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="GraphWeaver Agent",
        page_icon="🕸️",
        layout="wide"
    )
    
    st.title("🕸️ GraphWeaver Agent")
    st.caption("Database FK Discovery, Knowledge Graphs & Data Lineage - Powered by LangGraph")
    
    # ==========================================================================
    # Sidebar Configuration
    # ==========================================================================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Anthropic API Key",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            type="password",
            help="Required for Claude. Set ANTHROPIC_API_KEY env var or enter here."
        )
        if api_key:
            st.session_state.anthropic_api_key = api_key
            os.environ["ANTHROPIC_API_KEY"] = api_key  # For agent.py to pick up
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Quick commands
        st.markdown("### 🛠️ Quick Commands")
        st.markdown("Click to copy, paste into chat:")
        
        quick_commands = [
            "test database connection",
            "list database tables",
            "run fk discovery", 
            "import lineage to graph",
            "get graph stats",
            "sync graph to rdf",
            "generate text embeddings",
        ]
        for cmd in quick_commands:
            st.code(cmd, language=None)
        
        st.divider()
        
        # Tool count
        st.markdown(f"### 📦 {len(ALL_TOOLS)} Tools Available")
        with st.expander("View all tools"):
            for tool in ALL_TOOLS:
                st.markdown(f"- `{tool.name}`")
    
    # ==========================================================================
    # Chat Interface
    # ==========================================================================
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask GraphWeaver Agent..."):
        
        # Validate API key
        effective_api_key = st.session_state.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not effective_api_key:
            st.error("⚠️ Please enter your Anthropic API key in the sidebar.")
            return
        
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and stream response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Create agent (uses API key from env)
                agent = create_agent()
                
                # Stream response
                response = stream_agent_response(
                    agent, 
                    prompt, 
                    st.session_state.messages[:-1],  # History without current message
                    message_placeholder
                )
                
                # Save response to history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except ValueError as e:
                if "API" in str(e):
                    st.error("⚠️ Please set your Anthropic API key.")
                else:
                    st.error(f"Error: {e}")
            except Exception as e:
                import traceback
                error_msg = f"Error: {e}\n\n{traceback.format_exc()}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
