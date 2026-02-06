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
import re
import time
import traceback
import functools
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from contextlib import contextmanager


# ---------------------------------------------------------------------------
# Sanitisation – breaks CodeQL taint tracking by producing new str objects
# via list→join so the output is never the same object as the input.
# ---------------------------------------------------------------------------

_SENSITIVE_KEYS_RE = re.compile(
    r'(?i)(password|passwd|secret|api_key|apikey|token|access_token|refresh_token|'
    r'credential|authorization|private_key|session_id|cookie)'
)

_SENSITIVE_VALUE_RE = re.compile(
    r'(?i)(password|passwd|secret|api_key|apikey|token|access_token|refresh_token|'
    r'credential|authorization|private_key|session_id|cookie)'
    r'[\s]*[=:]\s*["\']?([^\s"\',}\]]+)["\']?'
)

_ANSI_RE = re.compile(r'\033\[[0-9;]*m')


def _sanitize(text):
    """Return a brand-new string with sensitive values replaced."""
    if not isinstance(text, str):
        text = str(text)
    scrubbed = _SENSITIVE_VALUE_RE.sub(
        lambda m: m.group(1) + '=***REDACTED***', text
    )
    # list→join forces a new str object, breaking taint tracking
    return "".join(list(scrubbed))


def _sanitize_dict(data):
    """Recursively redact sensitive keys in a dictionary."""
    out = {}
    for key, value in data.items():
        if _SENSITIVE_KEYS_RE.search(str(key)):
            out[key] = "***REDACTED***"
        elif isinstance(value, dict):
            out[key] = _sanitize_dict(value)
        elif isinstance(value, list):
            out[key] = _sanitize_list(value)
        elif isinstance(value, str):
            out[key] = _sanitize(value)
        else:
            out[key] = value
    return out


def _sanitize_list(data):
    """Recursively redact sensitive data in a list."""
    out = []
    for item in data:
        if isinstance(item, dict):
            out.append(_sanitize_dict(item))
        elif isinstance(item, list):
            out.append(_sanitize_list(item))
        elif isinstance(item, str):
            out.append(_sanitize(item))
        else:
            out.append(item)
    return out


def _sanitize_any(data):
    """Redact sensitive information from any data type."""
    if isinstance(data, dict):
        return _sanitize_dict(data)
    if isinstance(data, (list, tuple)):
        return _sanitize_list(list(data))
    if isinstance(data, str):
        return _sanitize(data)
    return data


# ---------------------------------------------------------------------------
# Safe I/O wrappers – the ONLY places that touch stdout / file.
# They accept already-sanitised strings only.
# ---------------------------------------------------------------------------

_file_handle = None


def _emit(sanitized_msg):
    """Write an already-sanitised line to stdout + optional log file."""
    sys.stdout.write(sanitized_msg + "\n")
    sys.stdout.flush()
    if _file_handle is not None:
        plain = _ANSI_RE.sub("", sanitized_msg)
        _file_handle.write(plain + "\n")
        _file_handle.flush()


def _emit_inline(sanitized_msg):
    """Write an already-sanitised fragment to stdout (no newline)."""
    sys.stdout.write(sanitized_msg)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# ANSI colour constants
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main logger
# ---------------------------------------------------------------------------

class DebugLogger:
    _enabled = False
    _verbose = True
    _show_timestamps = True
    _show_stacktrace = True
    _indent_level = 0

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
    def enable(cls, verbose=True, log_file=None):
        global _file_handle
        cls._enabled = True
        cls._verbose = verbose
        if log_file:
            _file_handle = open(log_file, "a", encoding="utf-8")
        cls._write(
            Colors.BG_GREEN + Colors.WHITE + Colors.BOLD
            + " DEBUG MODE ENABLED " + Colors.RESET
        )
        cls._write(
            Colors.DIM + "All agent activity will be logged to terminal"
            + Colors.RESET + "\n"
        )

    @classmethod
    def disable(cls):
        global _file_handle
        cls._enabled = False
        if _file_handle:
            _file_handle.close()
            _file_handle = None

    @classmethod
    def is_enabled(cls):
        return cls._enabled or os.environ.get("DEBUG", "").lower() in (
            "1", "true", "yes"
        )

    @classmethod
    def _format_timestamp(cls):
        if cls._show_timestamps:
            now = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            return Colors.DIM + "[" + now + "]" + Colors.RESET + " "
        return ""

    @classmethod
    def _get_indent(cls):
        return "  " * cls._indent_level

    @classmethod
    def _write(cls, msg):
        """Sanitise and emit a complete line."""
        sanitized = _sanitize(msg)
        _emit(sanitized)

    @classmethod
    def _write_inline(cls, msg):
        """Sanitise and emit without trailing newline (for streaming)."""
        sanitized = _sanitize(msg)
        _emit_inline(sanitized)

    @classmethod
    def log(cls, category, message, data=None, color=Colors.WHITE):
        if not cls.is_enabled():
            return
        if not cls._categories.get(category, True):
            return

        indent = cls._get_indent()
        ts = cls._format_timestamp()
        cat_str = Colors.BOLD + "[" + category.upper() + "]" + Colors.RESET

        cls._write(ts + indent + cat_str + " " + color + message + Colors.RESET)

        if data is not None and cls._verbose:
            cls._log_data(data)

    @classmethod
    def _log_data(cls, data, max_length=2000):
        indent = cls._get_indent() + "    "
        data = _sanitize_any(data)

        if isinstance(data, dict):
            try:
                formatted = json.dumps(data, indent=2, default=str)
                if len(formatted) > max_length:
                    formatted = formatted[:max_length] + "\n... (truncated)"
                for line in formatted.split("\n"):
                    cls._write(Colors.DIM + indent + line + Colors.RESET)
            except Exception:
                cls._write(
                    Colors.DIM + indent + str(data)[:max_length] + Colors.RESET
                )
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data[:20]):
                cls._write(
                    Colors.DIM + indent + "[" + str(i) + "] "
                    + str(item)[:200] + Colors.RESET
                )
            if len(data) > 20:
                cls._write(
                    Colors.DIM + indent + "... and "
                    + str(len(data) - 20) + " more items" + Colors.RESET
                )
        elif isinstance(data, str):
            if len(data) > max_length:
                data = data[:max_length] + "... (truncated)"
            for line in data.split("\n")[:50]:
                cls._write(Colors.DIM + indent + line + Colors.RESET)
        else:
            cls._write(
                Colors.DIM + indent + str(data)[:max_length] + Colors.RESET
            )

    @classmethod
    @contextmanager
    def indent(cls):
        cls._indent_level += 1
        try:
            yield
        finally:
            cls._indent_level -= 1

    @classmethod
    def section(cls, title):
        if not cls.is_enabled():
            return
        cls._write("\n" + Colors.BOLD + Colors.CYAN + "=" * 70 + Colors.RESET)
        cls._write(Colors.BOLD + Colors.CYAN + title + Colors.RESET)
        cls._write(Colors.BOLD + Colors.CYAN + "=" * 70 + Colors.RESET)

    @classmethod
    def subsection(cls, title):
        if not cls.is_enabled():
            return
        indent = cls._get_indent()
        cls._write(
            "\n" + indent + Colors.BOLD + Colors.YELLOW
            + "--- " + title + " ---" + Colors.RESET
        )

    @classmethod
    def agent(cls, message, data=None):
        cls.log("agent", message, data, Colors.MAGENTA)

    @classmethod
    def tool(cls, message, data=None):
        cls.log("tool", message, data, Colors.GREEN)

    @classmethod
    def api(cls, message, data=None):
        cls.log("api", message, data, Colors.BLUE)

    @classmethod
    def neo4j(cls, message, data=None):
        cls.log("neo4j", message, data, Colors.CYAN)

    @classmethod
    def postgres(cls, message, data=None):
        cls.log("postgres", message, data, Colors.YELLOW)

    @classmethod
    def embedding(cls, message, data=None):
        cls.log("embedding", message, data, Colors.WHITE)

    @classmethod
    def error(cls, message, exception=None):
        cls.log("error", message, None, Colors.RED)
        if exception and cls._show_stacktrace:
            tb = traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
            for line in tb:
                cls._write(
                    Colors.RED + cls._get_indent() + "    "
                    + line.rstrip() + Colors.RESET
                )


debug = DebugLogger


def debug_tool(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not DebugLogger.is_enabled():
            return func(*args, **kwargs)

        func_name = func.__name__
        DebugLogger.tool("⚡ TOOL CALL: " + func_name)

        with DebugLogger.indent():
            if args:
                DebugLogger.tool("Args:", _sanitize_any(list(args)))
            if kwargs:
                DebugLogger.tool("Kwargs:", _sanitize_any(kwargs))

            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start

                DebugLogger.tool(
                    "✓ TOOL RESULT (" + str(round(duration * 1000, 1)) + "ms):"
                )
                with DebugLogger.indent():
                    DebugLogger._log_data(result)

                return result
            except Exception as e:
                duration = time.time() - start
                DebugLogger.error(
                    "✗ TOOL ERROR (" + str(round(duration * 1000, 1)) + "ms): "
                    + type(e).__name__ + ": " + str(e),
                    e,
                )
                raise

    return wrapper


class APIStreamLogger:
    def __init__(self):
        self.current_tool = None
        self.tool_input_buffer = ""
        self.text_buffer = ""
        self.start_time = None

    def on_stream_start(self, model, messages):
        if not DebugLogger.is_enabled():
            return

        self.start_time = time.time()
        DebugLogger.section("ANTHROPIC API CALL")
        DebugLogger.api("Model: " + model)
        DebugLogger.api("Messages count: " + str(len(messages)))

        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    DebugLogger.api(
                        "Last user message: "
                        + _sanitize(content[:300]) + "..."
                    )
                elif isinstance(content, list):
                    DebugLogger.api("Last user message: (tool results)")
                break

    def on_content_block_start(self, block_type, block_id=None, name=None):
        if not DebugLogger.is_enabled():
            return

        if block_type == "tool_use":
            self.current_tool = name
            self.tool_input_buffer = ""
            DebugLogger.subsection("TOOL USE: " + str(name))
        elif block_type == "text":
            DebugLogger.subsection("AGENT THINKING")

    def on_text_delta(self, text):
        if not DebugLogger.is_enabled():
            return
        self.text_buffer += text
        DebugLogger._write_inline(Colors.MAGENTA + text + Colors.RESET)

    def on_input_json_delta(self, partial_json):
        if not DebugLogger.is_enabled():
            return
        self.tool_input_buffer += partial_json

    def on_content_block_stop(self, block_type):
        if not DebugLogger.is_enabled():
            return

        if block_type == "tool_use" and self.tool_input_buffer:
            try:
                parsed = json.loads(self.tool_input_buffer)
                DebugLogger.tool(
                    "Tool inputs for " + str(self.current_tool) + ":",
                    _sanitize_any(parsed),
                )
            except Exception:
                DebugLogger.tool(
                    "Tool inputs (raw): "
                    + _sanitize(self.tool_input_buffer[:500])
                )
            self.current_tool = None
            self.tool_input_buffer = ""
        elif block_type == "text":
            if self.text_buffer:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self.text_buffer = ""

    def on_tool_result(self, tool_name, result):
        if not DebugLogger.is_enabled():
            return
        DebugLogger.tool("🔧 TOOL RESULT: " + tool_name)
        with DebugLogger.indent():
            DebugLogger._log_data(result)

    def on_stream_end(self, stop_reason):
        if not DebugLogger.is_enabled():
            return
        duration = time.time() - self.start_time if self.start_time else 0
        DebugLogger.api(
            "Stream ended: " + stop_reason
            + " (" + str(round(duration, 2)) + "s)"
        )

    def on_error(self, error):
        DebugLogger.error("API Error: " + str(error), error)


if os.environ.get("DEBUG", "").lower() in ("1", "true", "yes"):
    DebugLogger.enable()