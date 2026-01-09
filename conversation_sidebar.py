#!/usr/bin/env python3
"""
conversation_sidebar.py

Streamlit sidebar UI component for conversation history.
"""

import streamlit as st
from datetime import datetime
from typing import Optional
from conversation_memory import get_memory, Conversation


def _format_time(iso_string: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_string)
        now = datetime.now()
        diff = now - dt
        
        if diff.days == 0:
            if diff.seconds < 60:
                return "Just now"
            elif diff.seconds < 3600:
                return f"{diff.seconds // 60}m ago"
            else:
                return f"{diff.seconds // 3600}h ago"
        elif diff.days == 1:
            return "Yesterday"
        elif diff.days < 7:
            return f"{diff.days}d ago"
        else:
            return dt.strftime("%b %d")
    except:
        return ""


def _truncate_title(title: str, max_length: int = 28) -> str:
    if len(title) <= max_length:
        return title
    return title[:max_length-3] + "..."


def init_session_state():
    memory = get_memory()
    
    if "conv_memory_initialized" not in st.session_state:
        st.session_state.conv_memory_initialized = True
        conv = memory.get_or_create_current()
        st.session_state.messages = conv.messages.copy()
        st.session_state.streaming_messages = conv.streaming_messages.copy()
        st.session_state.current_conv_id = conv.id
    
    if "editing_conv_id" not in st.session_state:
        st.session_state.editing_conv_id = None
    
    if "show_delete_confirm" not in st.session_state:
        st.session_state.show_delete_confirm = None


def sync_session_state():
    memory = get_memory()
    conv = memory.get_current_conversation()
    
    if conv and "messages" in st.session_state:
        conv.messages = st.session_state.messages.copy()
        conv.streaming_messages = st.session_state.get("streaming_messages", []).copy()
        conv.updated_at = datetime.now().isoformat()
        memory._save()


def switch_to_conversation(conv_id: str):
    memory = get_memory()
    sync_session_state()
    conv = memory.switch_conversation(conv_id)
    
    if conv:
        st.session_state.messages = conv.messages.copy()
        st.session_state.streaming_messages = conv.streaming_messages.copy()
        st.session_state.current_conv_id = conv.id
        if "agent" in st.session_state:
            del st.session_state.agent


def create_new_conversation():
    memory = get_memory()
    sync_session_state()
    conv = memory.create_conversation()
    st.session_state.messages = []
    st.session_state.streaming_messages = []
    st.session_state.current_conv_id = conv.id
    if "agent" in st.session_state:
        del st.session_state.agent


def delete_conversation(conv_id: str):
    memory = get_memory()
    was_current = (conv_id == st.session_state.get("current_conv_id"))
    memory.delete_conversation(conv_id)
    
    if was_current:
        conv = memory.get_or_create_current()
        st.session_state.messages = conv.messages.copy()
        st.session_state.streaming_messages = conv.streaming_messages.copy()
        st.session_state.current_conv_id = conv.id


def render_conversation_sidebar():
    init_session_state()
    memory = get_memory()
    
    st.header("💬 Conversations")
    
    # Single "New Chat" button
    if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
        create_new_conversation()
        st.rerun()
    
    st.divider()
    
    conversations = memory.get_all_conversations()
    current_id = st.session_state.get("current_conv_id")
    
    if not conversations:
        st.caption("No conversations yet. Start chatting!")
        return
    
    for conv in conversations:
        is_current = conv.id == current_id
        is_editing = st.session_state.editing_conv_id == conv.id
        show_delete = st.session_state.show_delete_confirm == conv.id
        
        with st.container():
            if is_editing:
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
                title_display = _truncate_title(conv.title)
                time_display = _format_time(conv.updated_at)
                msg_count = len(conv.messages)
                
                col1, col2, col3 = st.columns([5, 1, 1])
                
                with col1:
                    btn_label = f"{'▶ ' if is_current else ''}{title_display}"
                    if st.button(
                        btn_label,
                        key=f"conv_{conv.id}",
                        use_container_width=True,
                        type="primary" if is_current else "secondary"
                    ):
                        if not is_current:
                            switch_to_conversation(conv.id)
                            st.rerun()
                
                with col2:
                    if st.button("✏️", key=f"edit_{conv.id}", help="Rename"):
                        st.session_state.editing_conv_id = conv.id
                        st.rerun()
                
                with col3:
                    if st.button("🗑️", key=f"del_{conv.id}", help="Delete"):
                        st.session_state.show_delete_confirm = conv.id
                        st.rerun()
                
                if is_current:
                    st.caption(f"📝 {msg_count} messages • {time_display}")


def get_current_messages():
    return st.session_state.get("messages", [])


def get_current_streaming_messages():
    return st.session_state.get("streaming_messages", [])


def add_message(role: str, content: str):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "streaming_messages" not in st.session_state:
        st.session_state.streaming_messages = []
    
    st.session_state.messages.append({"role": role, "content": content})
    st.session_state.streaming_messages.append({"role": role, "content": content})
    
    if len(st.session_state.streaming_messages) > 20:
        st.session_state.streaming_messages = st.session_state.streaming_messages[-20:]
    
    memory = get_memory()
    conv = memory.get_current_conversation()
    if conv and conv.title == "New Conversation" and role == "user" and len(st.session_state.messages) == 1:
        conv.title = Conversation._generate_title(content)
    
    sync_session_state()
