#!/usr/bin/env python3
"""
conversation_memory.py

Persistent conversation memory for GraphWeaver Agent Streamlit app.
Provides functionality similar to Claude's sidebar conversation history.

Features:
- Save/load conversations to JSON file
- Create, rename, delete conversations
- Switch between conversations
- Auto-generate titles from first message
- Timestamp tracking
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class Conversation:
    """A single conversation with messages and metadata."""
    id: str
    title: str
    messages: List[Dict[str, str]]
    streaming_messages: List[Dict[str, str]]
    created_at: str
    updated_at: str
    
    @classmethod
    def create_new(cls, title: str = "New Conversation") -> "Conversation":
        """Create a new conversation with a unique ID."""
        now = datetime.now().isoformat()
        return cls(
            id=str(uuid.uuid4())[:8],
            title=title,
            messages=[],
            streaming_messages=[],
            created_at=now,
            updated_at=now,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """Create from dictionary."""
        return cls(**data)
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.messages.append({"role": role, "content": content})
        self.updated_at = datetime.now().isoformat()
        
        # Auto-generate title from first user message if still default
        if self.title == "New Conversation" and role == "user" and len(self.messages) == 1:
            self.title = self._generate_title(content)
    
    def add_streaming_message(self, role: str, content: str):
        """Add a message to the streaming messages (for agent context)."""
        self.streaming_messages.append({"role": role, "content": content})
        # Keep streaming messages limited
        if len(self.streaming_messages) > 20:
            self.streaming_messages = self.streaming_messages[-20:]
    
    @staticmethod
    def _generate_title(text: str, max_length: int = 40) -> str:
        """Generate a title from the first user message."""
        # Clean up the text
        text = text.strip()
        # Remove common prefixes
        for prefix in ["please ", "can you ", "could you ", "i want to ", "i'd like to "]:
            if text.lower().startswith(prefix):
                text = text[len(prefix):]
                break
        # Capitalize first letter
        text = text[0].upper() + text[1:] if text else text
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length-3].rsplit(' ', 1)[0] + "..."
        return text or "New Conversation"


class ConversationMemory:
    """Manages multiple conversations with persistent storage."""
    
    def __init__(self, storage_path: str = None):
        """Initialize conversation memory.
        
        Args:
            storage_path: Path to JSON file for storage. 
                         Defaults to ~/.graphweaver/conversations.json
        """
        if storage_path is None:
            storage_dir = Path.home() / ".graphweaver"
            storage_dir.mkdir(exist_ok=True)
            storage_path = storage_dir / "conversations.json"
        
        self.storage_path = Path(storage_path)
        self.conversations: Dict[str, Conversation] = {}
        self.current_conversation_id: Optional[str] = None
        
        # Load existing conversations
        self._load()
    
    def _load(self):
        """Load conversations from storage file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                for conv_data in data.get("conversations", []):
                    conv = Conversation.from_dict(conv_data)
                    self.conversations[conv.id] = conv
                
                self.current_conversation_id = data.get("current_conversation_id")
                
                # Validate current conversation exists
                if self.current_conversation_id not in self.conversations:
                    self.current_conversation_id = None
                    
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Warning: Could not load conversations: {e}")
                self.conversations = {}
                self.current_conversation_id = None
    
    def _save(self):
        """Save conversations to storage file."""
        data = {
            "conversations": [conv.to_dict() for conv in self.conversations.values()],
            "current_conversation_id": self.current_conversation_id,
        }
        
        # Ensure directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_conversation(self, title: str = "New Conversation") -> Conversation:
        """Create a new conversation and set it as current."""
        conv = Conversation.create_new(title)
        self.conversations[conv.id] = conv
        self.current_conversation_id = conv.id
        self._save()
        return conv
    
    def get_current_conversation(self) -> Optional[Conversation]:
        """Get the current active conversation."""
        if self.current_conversation_id is None:
            return None
        return self.conversations.get(self.current_conversation_id)
    
    def get_or_create_current(self) -> Conversation:
        """Get current conversation or create a new one if none exists."""
        conv = self.get_current_conversation()
        if conv is None:
            conv = self.create_conversation()
        return conv
    
    def switch_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Switch to a different conversation."""
        if conversation_id in self.conversations:
            self.current_conversation_id = conversation_id
            self._save()
            return self.conversations[conversation_id]
        return None
    
    def rename_conversation(self, conversation_id: str, new_title: str) -> bool:
        """Rename a conversation."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id].title = new_title
            self.conversations[conversation_id].updated_at = datetime.now().isoformat()
            self._save()
            return True
        return False
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            
            # If we deleted the current conversation, clear it
            if self.current_conversation_id == conversation_id:
                self.current_conversation_id = None
                # Switch to another conversation if available
                if self.conversations:
                    self.current_conversation_id = list(self.conversations.keys())[0]
            
            self._save()
            return True
        return False
    
    def get_all_conversations(self) -> List[Conversation]:
        """Get all conversations sorted by updated_at (most recent first)."""
        return sorted(
            self.conversations.values(),
            key=lambda c: c.updated_at,
            reverse=True
        )
    
    def add_message_to_current(self, role: str, content: str):
        """Add a message to the current conversation."""
        conv = self.get_or_create_current()
        conv.add_message(role, content)
        self._save()
    
    def add_streaming_message_to_current(self, role: str, content: str):
        """Add a streaming message to the current conversation."""
        conv = self.get_or_create_current()
        conv.add_streaming_message(role, content)
        self._save()
    
    def clear_current_conversation(self):
        """Clear all messages from the current conversation."""
        conv = self.get_current_conversation()
        if conv:
            conv.messages = []
            conv.streaming_messages = []
            conv.updated_at = datetime.now().isoformat()
            self._save()
    
    def get_conversation_count(self) -> int:
        """Get the total number of conversations."""
        return len(self.conversations)
    
    def export_conversation(self, conversation_id: str, format: str = "json") -> Optional[str]:
        """Export a conversation to a string format."""
        conv = self.conversations.get(conversation_id)
        if not conv:
            return None
        
        if format == "json":
            return json.dumps(conv.to_dict(), indent=2)
        elif format == "markdown":
            lines = [f"# {conv.title}\n"]
            lines.append(f"*Created: {conv.created_at}*\n")
            lines.append(f"*Updated: {conv.updated_at}*\n\n")
            lines.append("---\n\n")
            for msg in conv.messages:
                role = "**User:**" if msg["role"] == "user" else "**Assistant:**"
                lines.append(f"{role}\n\n{msg['content']}\n\n---\n\n")
            return "".join(lines)
        
        return None


# Singleton instance for use in Streamlit
_memory_instance: Optional[ConversationMemory] = None

def get_memory() -> ConversationMemory:
    """Get the singleton ConversationMemory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ConversationMemory()
    return _memory_instance


def reset_memory():
    """Reset the memory instance (useful for testing)."""
    global _memory_instance
    _memory_instance = None
