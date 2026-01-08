#!/usr/bin/env python3
# =============================================================================
# FILE: debug_logger.py
# PATH: /home/gp/Downloads/graphweaver-agent/debug_logger.py
# =============================================================================
"""
Terminal Debug Logger for GraphWeaver Agent

Usage:
    DEBUG=1 streamlit run streamlit_app.py
"""

import os
import sys
import json
import time
import traceback
import functools
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from contextlib import contextmanager


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_GREEN = "\033[42m"


class DebugLogger:
    _enabled = False
    _verbose = True
    _show_timestamps = True
    _show_stacktrace = True
    _indent_level = 0
    _log_file = None
    
    _categories = {
        "agent": True,
        "tool": True,
        "api": True,
        "neo4j": True,
        "postgres": True,
        "embedding": True,
        "error": True,
        "timing": True,
    }
    
    @classmethod
    def enable(cls, verbose: bool = True, log_file: Optional[str] = None):
        cls._enabled = True
        cls._verbose = verbose
        if log_file:
            cls._log_file = open(log_file, "a", encoding="utf-8")
        print(f"{Colors.BG_GREEN}{Colors.WHITE}{Colors.BOLD} DEBUG MODE ENABLED {Colors.RESET}", flush=True)
        print(f"{Colors.DIM}All agent activity will be logged to terminal{Colors.RESET}\n", flush=True)
    
    @classmethod
    def disable(cls):
        cls._enabled = False
        if cls._log_file:
            cls._log_file.close()
            cls._log_file = None
    
    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled or os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
    
    @classmethod
    def _format_timestamp(cls) -> str:
        if cls._show_timestamps:
            return f"{Colors.DIM}[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}]{Colors.RESET} "
        return ""
    
    @classmethod
    def _get_indent(cls) -> str:
        return "  " * cls._indent_level
    
    @classmethod
    def _write(cls, msg: str):
        print(msg, flush=True)
        if cls._log_file:
            import re
            clean = re.sub(r'\033\[[0-9;]*m', '', msg)
            cls._log_file.write(clean + "\n")
            cls._log_file.flush()
    
    @classmethod
    def log(cls, category: str, message: str, data: Any = None, color: str = Colors.WHITE):
        if not cls.is_enabled():
            return
        if not cls._categories.get(category, True):
            return
        
        indent = cls._get_indent()
        ts = cls._format_timestamp()
        cat_str = f"{Colors.BOLD}[{category.upper()}]{Colors.RESET}"
        
        cls._write(f"{ts}{indent}{cat_str} {color}{message}{Colors.RESET}")
        
        if data is not None and cls._verbose:
            cls._log_data(data)
    
    @classmethod
    def _log_data(cls, data: Any, max_length: int = 2000):
        indent = cls._get_indent() + "    "
        
        if isinstance(data, dict):
            try:
                formatted = json.dumps(data, indent=2, default=str)
                if len(formatted) > max_length:
                    formatted = formatted[:max_length] + "\n... (truncated)"
                for line in formatted.split("\n"):
                    cls._write(f"{Colors.DIM}{indent}{line}{Colors.RESET}")
            except:
                cls._write(f"{Colors.DIM}{indent}{str(data)[:max_length]}{Colors.RESET}")
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data[:20]):
                cls._write(f"{Colors.DIM}{indent}[{i}] {str(item)[:200]}{Colors.RESET}")
            if len(data) > 20:
                cls._write(f"{Colors.DIM}{indent}... and {len(data) - 20} more items{Colors.RESET}")
        elif isinstance(data, str):
            if len(data) > max_length:
                data = data[:max_length] + "... (truncated)"
            for line in data.split("\n")[:50]:
                cls._write(f"{Colors.DIM}{indent}{line}{Colors.RESET}")
        else:
            cls._write(f"{Colors.DIM}{indent}{str(data)[:max_length]}{Colors.RESET}")
    
    @classmethod
    @contextmanager
    def indent(cls):
        cls._indent_level += 1
        try:
            yield
        finally:
            cls._indent_level -= 1
    
    @classmethod
    def section(cls, title: str):
        if not cls.is_enabled():
            return
        cls._write(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
        cls._write(f"{Colors.BOLD}{Colors.CYAN}{title}{Colors.RESET}")
        cls._write(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    
    @classmethod
    def subsection(cls, title: str):
        if not cls.is_enabled():
            return
        indent = cls._get_indent()
        cls._write(f"\n{indent}{Colors.BOLD}{Colors.YELLOW}--- {title} ---{Colors.RESET}")
    
    @classmethod
    def agent(cls, message: str, data: Any = None):
        cls.log("agent", message, data, Colors.MAGENTA)
    
    @classmethod
    def tool(cls, message: str, data: Any = None):
        cls.log("tool", message, data, Colors.GREEN)
    
    @classmethod
    def api(cls, message: str, data: Any = None):
        cls.log("api", message, data, Colors.BLUE)
    
    @classmethod
    def neo4j(cls, message: str, data: Any = None):
        cls.log("neo4j", message, data, Colors.CYAN)
    
    @classmethod
    def postgres(cls, message: str, data: Any = None):
        cls.log("postgres", message, data, Colors.YELLOW)
    
    @classmethod
    def embedding(cls, message: str, data: Any = None):
        cls.log("embedding", message, data, Colors.WHITE)
    
    @classmethod
    def error(cls, message: str, exception: Optional[Exception] = None):
        cls.log("error", message, None, Colors.RED)
        if exception and cls._show_stacktrace:
            tb = traceback.format_exception(type(exception), exception, exception.__traceback__)
            for line in tb:
                cls._write(f"{Colors.RED}{cls._get_indent()}    {line.rstrip()}{Colors.RESET}")


debug = DebugLogger


def debug_tool(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not DebugLogger.is_enabled():
            return func(*args, **kwargs)
        
        func_name = func.__name__
        DebugLogger.tool(f"⚡ TOOL CALL: {func_name}")
        
        with DebugLogger.indent():
            if args:
                DebugLogger.tool("Args:", list(args))
            if kwargs:
                DebugLogger.tool("Kwargs:", kwargs)
            
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                
                DebugLogger.tool(f"✓ TOOL RESULT ({duration*1000:.1f}ms):")
                with DebugLogger.indent():
                    DebugLogger._log_data(result)
                
                return result
            except Exception as e:
                duration = time.time() - start
                DebugLogger.error(f"✗ TOOL ERROR ({duration*1000:.1f}ms): {type(e).__name__}: {e}", e)
                raise
    
    return wrapper


class APIStreamLogger:
    def __init__(self):
        self.current_tool = None
        self.tool_input_buffer = ""
        self.text_buffer = ""
        self.start_time = None
    
    def on_stream_start(self, model: str, messages: list):
        if not DebugLogger.is_enabled():
            return
        
        self.start_time = time.time()
        DebugLogger.section("ANTHROPIC API CALL")
        DebugLogger.api(f"Model: {model}")
        DebugLogger.api(f"Messages count: {len(messages)}")
        
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    DebugLogger.api(f"Last user message: {content[:300]}...")
                elif isinstance(content, list):
                    DebugLogger.api(f"Last user message: (tool results)")
                break
    
    def on_content_block_start(self, block_type: str, block_id: str = None, name: str = None):
        if not DebugLogger.is_enabled():
            return
        
        if block_type == "tool_use":
            self.current_tool = name
            self.tool_input_buffer = ""
            DebugLogger.subsection(f"TOOL USE: {name}")
        elif block_type == "text":
            DebugLogger.subsection("AGENT THINKING")
    
    def on_text_delta(self, text: str):
        if not DebugLogger.is_enabled():
            return
        self.text_buffer += text
        print(f"{Colors.MAGENTA}{text}{Colors.RESET}", end="", flush=True)
    
    def on_input_json_delta(self, partial_json: str):
        if not DebugLogger.is_enabled():
            return
        self.tool_input_buffer += partial_json
    
    def on_content_block_stop(self, block_type: str):
        if not DebugLogger.is_enabled():
            return
        
        if block_type == "tool_use" and self.tool_input_buffer:
            try:
                parsed = json.loads(self.tool_input_buffer)
                DebugLogger.tool(f"Tool inputs for {self.current_tool}:", parsed)
            except:
                DebugLogger.tool(f"Tool inputs (raw): {self.tool_input_buffer[:500]}")
            self.current_tool = None
            self.tool_input_buffer = ""
        elif block_type == "text":
            if self.text_buffer:
                print(flush=True)
                self.text_buffer = ""
    
    def on_tool_result(self, tool_name: str, result: str):
        if not DebugLogger.is_enabled():
            return
        DebugLogger.tool(f"🔧 TOOL RESULT: {tool_name}")
        with DebugLogger.indent():
            DebugLogger._log_data(result)
    
    def on_stream_end(self, stop_reason: str):
        if not DebugLogger.is_enabled():
            return
        duration = time.time() - self.start_time if self.start_time else 0
        DebugLogger.api(f"Stream ended: {stop_reason} ({duration:.2f}s)")
    
    def on_error(self, error: Exception):
        DebugLogger.error(f"API Error: {error}", error)


if os.environ.get("DEBUG", "").lower() in ("1", "true", "yes"):
    DebugLogger.enable()
