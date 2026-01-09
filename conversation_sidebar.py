#!/usr/bin/env python3
"""
conversation_sidebar.py

Streamlit sidebar UI component for conversation history.
Provides a Claude-like interface for managing multiple conversations.

Usage:
    from conversation_sidebar import render_conversation_sidebar, sync_session_state
    
    # In your main() function:
    with st.sidebar:
        render_conversation_sidebar()
    
    # After processing messages, sync state:
    sync_session_state()
"""

import streamlit as st
from datetime import datetime
from typing import Optional
from conversation_memory import get_memory, Conversation


def _format_time(iso_string: str) -> str:
    """Format ISO timestamp to a human-readable string."""
    try:
        dt = datetime.fromisoformat(iso_string)
        now = datetime.now()
        diff = now - dt
        
        if diff.days == 0:
            if diff.seconds < 60:
                return "Just now"
            elif diff.seconds < 3600:
                mins = diff.seconds // 60
                return f"{mins}m ago"
            else:
                hours = diff.seconds // 3600
                return f"{hours}h ago"
        elif diff.days == 1:
            return "Yesterday"
        elif diff.days < 7:
            return f"{diff.days}d ago"
        else:
            return dt.strftime("%b %d")
    except:
        return ""


def _truncate_title(title: str, max_length: int = 28) -> str:
    """Truncate title for display."""
    if len(title) <= max_length:
        return title
    return title[:max_length-3] + "..."


def init_session_state():
    """Initialize session state for conversation management."""
    memory = get_memory()
    
    # Initialize conversation memory in session state
    if "conv_memory_initialized" not in st.session_state:
        st.session_state.conv_memory_initialized = True
        
        # Get or create current conversation
        conv = memory.get_or_create_current()
        
        # Sync session state with conversation
        st.session_state.messages = conv.messages.copy()
        st.session_state.streaming_messages = conv.streaming_messages.copy()
        st.session_state.current_conv_id = conv.id
    
    # Track editing state
    if "editing_conv_id" not in st.session_state:
        st.session_state.editing_conv_id = None
    
    if "show_delete_confirm" not in st.session_state:
        st.session_state.show_delete_confirm = None


def sync_session_state():
    """Sync session state messages back to conversation memory."""
    memory = get_memory()
    conv = memory.get_current_conversation()
    
    if conv and "messages" in st.session_state:
        conv.messages = st.session_state.messages.copy()
        conv.streaming_messages = st.session_state.get("streaming_messages", []).copy()
        conv.updated_at = datetime.now().isoformat()
        memory._save()


def switch_to_conversation(conv_id: str):
    """Switch to a different conversation."""
    memory = get_memory()
    
    # Save current conversation first
    sync_session_state()
    
    # Switch conversation
    conv = memory.switch_conversation(conv_id)
    
    if conv:
        # Update session state
        st.session_state.messages = conv.messages.copy()
        st.session_state.streaming_messages = conv.streaming_messages.copy()
        st.session_state.current_conv_id = conv.id
        
        # Clear agent if needed (to reset context)
        if "agent" in st.session_state:
            del st.session_state.agent


def create_new_conversation():
    """Create a new conversation and switch to it."""
    memory = get_memory()
    
    # Save current conversation first
    sync_session_state()
    
    # Create new conversation
    conv = memory.create_conversation()
    
    # Update session state
    st.session_state.messages = []
    st.session_state.streaming_messages = []
    st.session_state.current_conv_id = conv.id
    
    # Clear agent to reset context
    if "agent" in st.session_state:
        del st.session_state.agent


def delete_conversation(conv_id: str):
    """Delete a conversation."""
    memory = get_memory()
    
    was_current = (conv_id == st.session_state.get("current_conv_id"))
    
    memory.delete_conversation(conv_id)
    
    # If we deleted the current conversation, load the new current one
    if was_current:
        conv = memory.get_or_create_current()
        st.session_state.messages = conv.messages.copy()
        st.session_state.streaming_messages = conv.streaming_messages.copy()
        st.session_state.current_conv_id = conv.id


def render_conversation_sidebar():
    """Render the conversation history sidebar section."""
    init_session_state()
    memory = get_memory()
    
    st.header("💬 Conversations")
    
    # New conversation button
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
            create_new_conversation()
            st.rerun()
    with col2:
        conversation_count = memory.get_conversation_count()
        st.markdown(f"<div style='text-align:center;padding-top:8px;color:#888;'>{conversation_count}</div>", 
                   unsafe_allow_html=True)
    
    st.divider()
    
    # List all conversations
    conversations = memory.get_all_conversations()
    current_id = st.session_state.get("current_conv_id")
    
    if not conversations:
        st.caption("No conversations yet. Start chatting!")
        return
    
    for conv in conversations:
        is_current = conv.id == current_id
        is_editing = st.session_state.editing_conv_id == conv.id
        show_delete = st.session_state.show_delete_confirm == conv.id
        
        # Container for conversation item
        with st.container():
            if is_editing:
                # Edit mode - show text input for rename
                col1, col2 = st.columns([4, 1])
                with col1:
                    new_title = st.text_input(
                        "Rename",
                        value=conv.title,
                        key=f"rename_{conv.id}",
                        label_visibility="collapsed"
                    )
                with col2:
                    if st.button("✓", key=f"save_{conv.id}"):
                        memory.rename_conversation(conv.id, new_title)
                        st.session_state.editing_conv_id = None
                        st.rerun()
            
            elif show_delete:
                # Delete confirmation
                st.warning(f"Delete '{_truncate_title(conv.title, 20)}'?", icon="⚠️")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes", key=f"confirm_del_{conv.id}", type="primary"):
                        delete_conversation(conv.id)
                        st.session_state.show_delete_confirm = None
                        st.rerun()
                with col2:
                    if st.button("No", key=f"cancel_del_{conv.id}"):
                        st.session_state.show_delete_confirm = None
                        st.rerun()
            
            else:
                # Normal display mode
                # Conversation button with title and time
                btn_type = "primary" if is_current else "secondary"
                title_display = _truncate_title(conv.title)
                time_display = _format_time(conv.updated_at)
                msg_count = len(conv.messages)
                
                col1, col2, col3 = st.columns([5, 1, 1])
                
                with col1:
                    # Main conversation button
                    btn_label = f"{'▶ ' if is_current else ''}{title_display}"
                    if st.button(
                        btn_label,
                        key=f"conv_{conv.id}",
                        use_container_width=True,
                        type=btn_type if is_current else "secondary"
                    ):
                        if not is_current:
                            switch_to_conversation(conv.id)
                            st.rerun()
                
                with col2:
                    # Edit button
                    if st.button("✏️", key=f"edit_{conv.id}", help="Rename"):
                        st.session_state.editing_conv_id = conv.id
                        st.rerun()
                
                with col3:
                    # Delete button
                    if st.button("🗑️", key=f"del_{conv.id}", help="Delete"):
                        st.session_state.show_delete_confirm = conv.id
                        st.rerun()
                
                # Show metadata for current conversation
                if is_current:
                    st.caption(f"📝 {msg_count} messages • {time_display}")


def render_conversation_actions():
    """Render additional conversation actions (export, clear, etc.)."""
    memory = get_memory()
    conv = memory.get_current_conversation()
    
    if not conv:
        return
    
    st.divider()
    st.subheader("📋 Current Chat")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_current"):
            memory.clear_current_conversation()
            st.session_state.messages = []
            st.session_state.streaming_messages = []
            st.rerun()
    
    with col2:
        if st.button("📤 Export", use_container_width=True, key="export_current"):
            st.session_state.show_export = True
    
    # Export dialog
    if st.session_state.get("show_export"):
        export_format = st.selectbox(
            "Export format",
            ["Markdown", "JSON"],
            key="export_format"
        )
        
        fmt = "markdown" if export_format == "Markdown" else "json"
        export_data = memory.export_conversation(conv.id, fmt)
        
        if export_data:
            st.download_button(
                label=f"⬇️ Download {export_format}",
                data=export_data,
                file_name=f"conversation_{conv.id}.{'md' if fmt == 'markdown' else 'json'}",
                mime="text/markdown" if fmt == "markdown" else "application/json",
                use_container_width=True,
                key="download_export"
            )
        
        if st.button("Close", key="close_export"):
            st.session_state.show_export = False
            st.rerun()


def get_current_messages():
    """Get messages for the current conversation (for use with agent)."""
    return st.session_state.get("messages", [])


def get_current_streaming_messages():
    """Get streaming messages for the current conversation (for agent context)."""
    return st.session_state.get("streaming_messages", [])


def add_message(role: str, content: str):
    """Add a message to the current conversation."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "streaming_messages" not in st.session_state:
        st.session_state.streaming_messages = []
    
    st.session_state.messages.append({"role": role, "content": content})
    st.session_state.streaming_messages.append({"role": role, "content": content})
    
    # Keep streaming messages limited
    if len(st.session_state.streaming_messages) > 20:
        st.session_state.streaming_messages = st.session_state.streaming_messages[-20:]
    
    # Auto-update conversation title
    memory = get_memory()
    conv = memory.get_current_conversation()
    if conv and conv.title == "New Conversation" and role == "user" and len(st.session_state.messages) == 1:
        conv.title = Conversation._generate_title(content)
    
    # Sync to memory
    sync_session_state()
