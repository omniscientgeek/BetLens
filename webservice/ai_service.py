"""
AI Service — Async multi-provider AI integration with failover support.

Supports:
  - Anthropic (Claude)
  - OpenAI (GPT-4o, GPT-4o-mini, etc.)
  - OpenAI-compatible endpoints (Ollama, LM Studio, vLLM, etc.)

Providers are configured in ai_config.json and tried in priority order.
If one fails, the next enabled provider is attempted (when failover is on).
"""

import os
import sys
import json
import time
import asyncio
import logging
import shutil
import contextvars
from typing import Optional

from mcp_client import mcp_client as _mcp
from odds_math import implied_probability as _implied_prob

logger = logging.getLogger(__name__)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a rate-limit (HTTP 429) error."""
    # Anthropic SDK raises anthropic.RateLimitError
    exc_type_name = type(exc).__name__
    if "RateLimitError" in exc_type_name:
        return True
    # OpenAI SDK raises openai.RateLimitError
    if hasattr(exc, "status_code") and getattr(exc, "status_code", None) == 429:
        return True
    # Check message as fallback
    if "429" in str(exc) and "rate" in str(exc).lower():
        return True
    return False


def _rate_limit_backoff_seconds(attempt: int) -> float:
    """Return backoff delay in seconds for rate-limited retries.

    Uses exponential backoff: 60s, 90s (enough to reset a per-minute quota).
    """
    return 60.0 * (1.5 ** (attempt - 1))


def _unwrap_exception_group(exc: Exception) -> Exception:
    """Unwrap anyio/asyncio ExceptionGroup to get the real root-cause error.

    The MCP SDK's stdio_client uses anyio TaskGroups internally.  When a
    sub-task fails, the error is wrapped in an ``ExceptionGroup`` (Python 3.11+)
    or ``BaseExceptionGroup`` which surfaces as the unhelpful message
    "unhandled errors in a TaskGroup (1 sub-exception)".

    This helper recursively unwraps single-exception groups so the caller
    sees the actual underlying error (e.g. ``ConnectionResetError``,
    ``TimeoutError``, ``anthropic.APIError``, etc.).
    """
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc

# Context variable for per-run logger — set by call_ai/call_ai_stream/call_ai_chat
# so that low-level provider functions (_call_claude_sdk etc.) can log to the run file
_current_run_logger: contextvars.ContextVar[Optional[logging.Logger]] = contextvars.ContextVar(
    "_current_run_logger", default=None
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ai_config.json")

# Maximum number of tool-use round-trips before stopping (safety limit)
_MAX_TOOL_TURNS = 50


def load_config() -> dict:
    """Load AI configuration from disk."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict) -> None:
    """Persist AI configuration to disk."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# Detect Railway environment (Railway sets RAILWAY_ENVIRONMENT automatically)
_IS_RAILWAY = bool(os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("RAILWAY_PROJECT_ID"))


def get_enabled_providers(config: Optional[dict] = None) -> list:
    """Return enabled providers sorted by priority (lowest number = highest priority).

    On Railway, ``claude_sdk`` providers are automatically excluded because
    the Claude Code CLI is not available in that environment.
    """
    if config is None:
        config = load_config()
    providers = [p for p in config["providers"] if p.get("enabled")]
    if _IS_RAILWAY:
        skipped = [p["id"] for p in providers if p.get("type") == "claude_sdk"]
        if skipped:
            logger.info("Railway environment detected — disabling claude_sdk providers: %s", skipped)
        providers = [p for p in providers if p.get("type") != "claude_sdk"]
    providers.sort(key=lambda p: p.get("priority", 999))
    return providers


def _get_api_key(provider: dict) -> Optional[str]:
    """Resolve an API key from environment variable or direct value."""
    env_var = provider.get("api_key_env", "")
    if env_var:
        key = os.environ.get(env_var)
        if key:
            return key
    # Allow direct key (for testing — not recommended for production)
    return provider.get("api_key")


# ---------------------------------------------------------------------------
# Provider-specific async call implementations
# ---------------------------------------------------------------------------

async def _call_anthropic(provider: dict, system_prompt: str, user_prompt: str, config: dict) -> dict:
    """Call the Anthropic Messages API asynchronously.

    When ``provider["use_mcp"]`` is truthy, MCP tools are loaded from the
    betstamp-intelligence server and a multi-turn tool-use loop is executed
    until the model stops requesting tools (or ``_MAX_TOOL_TURNS`` is reached).
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    api_key = _get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"No API key found for {provider['id']} (env: {provider.get('api_key_env')})")

    use_mcp = provider.get("use_mcp", False)

    # Log request to per-run log
    rl = _current_run_logger.get(None)
    if rl:
        rl.info("=" * 60)
        rl.info("ANTHROPIC API REQUEST (model=%s, use_mcp=%s)", provider.get("model"), use_mcp)
        rl.info("-" * 40 + " SYSTEM PROMPT " + "-" * 40)
        rl.info("%s", system_prompt)
        rl.info("-" * 40 + " USER PROMPT " + "-" * 40)
        rl.info("%s", user_prompt)
        rl.info("=" * 60)

    client = anthropic.AsyncAnthropic(
        api_key=api_key,
        timeout=config.get("timeout_seconds", 60),
    )

    messages = [{"role": "user", "content": user_prompt}]
    start = time.time()
    total_input_tokens = 0
    total_output_tokens = 0
    all_tool_calls = []

    # ---- Build base kwargs (shared across turns) ----
    base_kwargs = dict(
        model=provider.get("model", "claude-sonnet-4-20250514"),
        max_tokens=provider.get("max_tokens", 4096),
        temperature=provider.get("temperature", 0.3),
        system=system_prompt,
    )

    # ---- Optional MCP setup ----
    async def _run(mcp_session=None, anthropic_tools=None):
        nonlocal messages, total_input_tokens, total_output_tokens, all_tool_calls
        turn = 0

        while True:
            turn += 1
            call_kwargs = {**base_kwargs, "messages": messages}
            if anthropic_tools:
                call_kwargs["tools"] = anthropic_tools

            response = await client.messages.create(**call_kwargs)
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            # Collect tool_use blocks
            tool_use_blocks = [b for b in (response.content or []) if getattr(b, "type", None) == "tool_use"]

            if response.stop_reason == "tool_use" and tool_use_blocks and mcp_session and turn < _MAX_TOOL_TURNS:
                # Append assistant response (including tool_use) to messages
                messages.append({"role": "assistant", "content": response.content})

                # Execute each tool call and collect results
                tool_results_content = []
                for block in tool_use_blocks:
                    tc = {"id": block.id, "name": block.name, "input": block.input}
                    if rl:
                        rl.info("[MCP tool_call] %s(%s)", block.name, json.dumps(block.input)[:300])
                    result_text = await _mcp.call_tool(mcp_session, block.name, block.input)
                    tc["result"] = result_text
                    tc["is_error"] = False
                    all_tool_calls.append(tc)
                    if rl:
                        rl.info("[MCP tool_result] %s -> %d chars", block.name, len(result_text))
                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

                messages.append({"role": "user", "content": tool_results_content})
                continue  # next turn

            # Done — extract final text
            for block in tool_use_blocks:
                all_tool_calls.append({"id": block.id, "name": block.name, "input": block.input})
            return response

    if use_mcp:
        try:
            async with _mcp.connect() as mcp_session:
                mcp_tools = await _mcp.get_tools(mcp_session)
                anthropic_tools = _mcp.tools_to_anthropic_format(mcp_tools)
                if rl:
                    rl.info("Loaded %d MCP tools for Anthropic API", len(anthropic_tools))
                response = await _run(mcp_session=mcp_session, anthropic_tools=anthropic_tools)
        except RuntimeError as mcp_err:
            if "MCP connection failed" in str(mcp_err):
                logger.warning("MCP unavailable, falling back to no-MCP mode: %s", mcp_err)
                if rl:
                    rl.warning("MCP connection failed — running without MCP tools: %s", mcp_err)
                response = await _run()
            else:
                raise
    else:
        response = await _run()

    elapsed = round(time.time() - start, 2)

    text = ""
    for block in (response.content or []):
        if getattr(block, "type", None) == "text":
            text = block.text
            break

    # Log response to per-run log
    if rl:
        rl.info("-" * 40 + " ASSISTANT RESPONSE " + "-" * 40)
        rl.info("%s", text)
        rl.info("-" * 40 + " END RESPONSE " + "-" * 40)

    return {
        "text": text,
        "provider_id": provider["id"],
        "provider_name": provider["name"],
        "model": response.model,
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        },
        "elapsed_seconds": elapsed,
        "tool_calls": all_tool_calls,
    }


async def _call_openai(provider: dict, system_prompt: str, user_prompt: str, config: dict) -> dict:
    """Call the OpenAI Chat Completions API asynchronously (also works with compatible endpoints)."""
    try:
        import openai
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    api_key = _get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"No API key found for {provider['id']} (env: {provider.get('api_key_env')})")

    kwargs = {"api_key": api_key, "timeout": config.get("timeout_seconds", 60)}
    if provider.get("base_url"):
        kwargs["base_url"] = provider["base_url"]

    client = openai.AsyncOpenAI(**kwargs)

    start = time.time()
    response = await client.chat.completions.create(
        model=provider.get("model", "gpt-4o"),
        max_tokens=provider.get("max_tokens", 4096),
        temperature=provider.get("temperature", 0.3),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    elapsed = round(time.time() - start, 2)

    choice = response.choices[0] if response.choices else None
    text = choice.message.content if choice else ""
    usage = {}
    if response.usage:
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    return {
        "text": text,
        "provider_id": provider["id"],
        "provider_name": provider["name"],
        "model": response.model or provider.get("model"),
        "usage": usage,
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Claude Agent SDK (claude-sdk-wrapper.mjs bridge)
# ---------------------------------------------------------------------------

# Resolve the wrapper script path relative to this file
_WRAPPER_SCRIPT = os.path.join(os.path.dirname(__file__), "claude-sdk-wrapper.mjs")


def _find_node() -> str:
    """Locate the Node.js executable."""
    node = shutil.which("node")
    if node:
        return node
    # Common Windows paths
    for candidate in [
        r"C:\Program Files\nodejs\node.exe",
        r"C:\Program Files (x86)\nodejs\node.exe",
        os.path.expandvars(r"%APPDATA%\nvm\current\node.exe"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    raise RuntimeError(
        "Node.js not found. Install Node.js to use the Claude SDK provider."
    )


async def _call_claude_sdk(provider: dict, system_prompt: str, user_prompt: str, config: dict, *, use_mcp: bool = False, max_turns: Optional[int] = None) -> dict:
    """
    Call Claude via the Agent SDK wrapper (Node.js subprocess).

    This spawns claude-sdk-wrapper.mjs, sends a JSON command on stdin,
    and reads the JSON result from stdout.

    When ``use_mcp`` is True, the wrapper will load the betstamp-intelligence
    MCP server config and allow Claude to call MCP tools autonomously.
    """
    if not os.path.isfile(_WRAPPER_SCRIPT):
        raise RuntimeError(
            f"claude-sdk-wrapper.mjs not found at {_WRAPPER_SCRIPT}. "
            "Run 'npm install' in the webservice/ directory."
        )

    node_bin = _find_node()
    timeout_seconds = config.get("timeout_seconds", 120)
    service_dir = os.path.dirname(__file__)

    # Log the full request to the per-run log
    rl = _current_run_logger.get(None)
    if rl:
        rl.info("=" * 60)
        rl.info("CLAUDE SDK REQUEST (use_mcp=%s, max_turns=%s)", use_mcp, max_turns or "unlimited")
        rl.info("-" * 40 + " SYSTEM PROMPT " + "-" * 40)
        rl.info("%s", system_prompt)
        rl.info("-" * 40 + " USER PROMPT " + "-" * 40)
        rl.info("%s", user_prompt)
        rl.info("=" * 60)

    # Build the command payload for the wrapper
    command_payload = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "model": provider.get("model", "claude-sonnet-4-20250514"),
        "cwd": service_dir,
        "timeout_seconds": timeout_seconds,
        "use_mcp": use_mcp,
    }
    if max_turns is not None:
        command_payload["max_turns"] = max_turns

    start = time.time()

    # Use native asyncio subprocess — no eventlet workaround needed
    # Force UTF-8 encoding for the Node.js subprocess on Windows to prevent
    # multi-byte characters (e.g. emojis) from being mangled by the system
    # code page.
    # Note: --input-type=module is NOT used here because the wrapper is a .mjs
    # file which Node.js already treats as ESM.  That flag is only valid with
    # --eval / --print / STDIN and causes ERR_INPUT_TYPE_NOT_ALLOWED otherwise.
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    # On Windows, set active code page to UTF-8 for child process pipes
    if os.name == "nt":
        child_env["PYTHONUTF8"] = "1"
        child_env["CHCP"] = "65001"

    proc = await asyncio.create_subprocess_exec(
        node_bin, _WRAPPER_SCRIPT,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=service_dir,
        env=child_env,
        limit=4_194_304,  # 4 MB – MCP tool results can produce large single-line JSON
    )

    input_bytes = json.dumps(command_payload).encode("utf-8")
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(input=input_bytes),
            timeout=timeout_seconds + 30,
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"Claude SDK wrapper timed out after {timeout_seconds + 30}s")

    elapsed = round(time.time() - start, 2)

    stdout_text = stdout_bytes.decode("utf-8")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")

    # Log all SDK stderr output (contains MCP loading info, query lifecycle)
    if stderr_text.strip():
        sdk_logger = logging.getLogger("claude_sdk")
        rl = _current_run_logger.get(None)
        for _line in stderr_text.strip().splitlines():
            sdk_logger.info(_line.rstrip())
            if rl:
                rl.info("[claude-sdk-stderr] %s", _line.rstrip())

    if proc.returncode != 0 and not stdout_text.strip():
        stderr_excerpt = (stderr_text or "")[:500]
        raise RuntimeError(
            f"Claude SDK wrapper exited with code {proc.returncode}: {stderr_excerpt}"
        )

    # Parse the last JSON line from stdout (the result)
    result_line = None
    for line in stdout_text.strip().splitlines():
        line = line.strip()
        if line:
            result_line = line

    if not result_line:
        raise RuntimeError("No output from Claude SDK wrapper")

    try:
        result = json.loads(result_line)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from Claude SDK wrapper: {e}\nOutput: {result_line[:300]}")

    if result.get("type") == "error":
        raise RuntimeError(f"Claude SDK error: {result.get('message', 'Unknown error')}")

    text = result.get("text", "")
    usage = result.get("usage", {})

    # Log the full response to the per-run log
    rl = _current_run_logger.get(None)
    if rl:
        rl.info("-" * 40 + " ASSISTANT RESPONSE " + "-" * 40)
        rl.info("%s", text)
        rl.info("-" * 40 + " END RESPONSE " + "-" * 40)

    return {
        "text": text,
        "provider_id": provider["id"],
        "provider_name": provider.get("name", "Claude SDK"),
        "model": provider.get("model", "claude-sonnet-4-20250514"),
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        },
        "elapsed_seconds": elapsed,
        "tool_calls": result.get("tool_calls", []),
    }


# Map provider types to their async call functions
_CALL_MAP = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "openai_compatible": _call_openai,
    "claude_sdk": _call_claude_sdk,
}


# ---------------------------------------------------------------------------
# Streaming provider implementations (yield text deltas via callback)
# ---------------------------------------------------------------------------

async def _call_claude_sdk_stream(provider: dict, system_prompt: str, user_prompt: str, config: dict, on_chunk=None, on_tool_event=None) -> dict:
    """Call Claude SDK with streaming — reads stdout line-by-line for chunk and tool events."""
    if not os.path.isfile(_WRAPPER_SCRIPT):
        raise RuntimeError(f"claude-sdk-wrapper.mjs not found at {_WRAPPER_SCRIPT}")

    node_bin = _find_node()
    timeout_seconds = config.get("timeout_seconds", 120)
    service_dir = os.path.dirname(__file__)

    # Log the full request to the per-run log
    rl = _current_run_logger.get(None)
    if rl:
        rl.info("=" * 60)
        rl.info("CLAUDE SDK STREAM REQUEST")
        rl.info("-" * 40 + " SYSTEM PROMPT " + "-" * 40)
        rl.info("%s", system_prompt)
        rl.info("-" * 40 + " USER PROMPT " + "-" * 40)
        rl.info("%s", user_prompt)
        rl.info("=" * 60)

    use_mcp = provider.get("use_mcp", True)
    command_payload = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "model": provider.get("model", "claude-sonnet-4-20250514"),
        "cwd": service_dir,
        "timeout_seconds": timeout_seconds,
        "use_mcp": use_mcp,
    }
    # Only set max_turns for non-MCP calls (default 1); MCP calls run unlimited
    if not use_mcp:
        command_payload["max_turns"] = 1

    start = time.time()

    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if os.name == "nt":
        child_env["PYTHONUTF8"] = "1"
        child_env["CHCP"] = "65001"

    proc = await asyncio.create_subprocess_exec(
        node_bin, _WRAPPER_SCRIPT,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=service_dir,
        env=child_env,
        limit=4_194_304,  # 4 MB – MCP tool results can produce large single-line JSON
    )

    # Send command and close stdin so the wrapper starts processing
    proc.stdin.write(json.dumps(command_payload).encode("utf-8"))
    proc.stdin.close()

    full_text = ""
    result_data = None

    # Read stdout line-by-line for streaming chunks
    try:
        while True:
            try:
                line_bytes = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=timeout_seconds + 30,
                )
            except asyncio.LimitOverrunError as overrun_exc:
                # A single line exceeded the buffer limit (e.g. huge MCP tool result).
                # Drain the oversized data so the stream can continue.
                consumed = getattr(overrun_exc, "consumed", 0)
                logger.warning(
                    "Streaming readline hit LimitOverrunError (%d bytes consumed) — draining oversized line",
                    consumed,
                )
                rl = _current_run_logger.get(None)
                if rl:
                    rl.warning(
                        "Streaming readline hit LimitOverrunError (%d bytes consumed) — draining oversized line",
                        consumed,
                    )
                # After LimitOverrunError the separator wasn't found within the
                # buffer limit.  Read until we find the newline (end of this
                # oversized JSON line) so subsequent lines parse normally.
                try:
                    await proc.stdout.readuntil(b"\n")
                except (asyncio.LimitOverrunError, asyncio.IncompleteReadError):
                    # If still too large or EOF, just drain what we can
                    pass
                continue

            if not line_bytes:
                break  # EOF

            line = line_bytes.decode("utf-8").strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "chunk" and msg.get("text"):
                full_text += msg["text"]
                if on_chunk:
                    await on_chunk(msg["text"])

            elif msg.get("type") == "tool_call":
                # Live tool call event — forward to UI
                if on_tool_event:
                    await on_tool_event("tool_call", msg.get("tool_call", {}))

            elif msg.get("type") == "tool_result":
                # Live tool result event — forward to UI
                if on_tool_event:
                    await on_tool_event("tool_result", {
                        "tool_use_id": msg.get("tool_use_id"),
                        "result": msg.get("result"),
                        "is_error": msg.get("is_error", False),
                    })

            elif msg.get("type") == "result":
                result_data = msg
                full_text = msg.get("text", full_text)

            elif msg.get("type") == "error":
                raise RuntimeError(f"Claude SDK error: {msg.get('message', 'Unknown')}")

    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"Claude SDK wrapper timed out after {timeout_seconds + 30}s")

    await proc.wait()
    elapsed = round(time.time() - start, 2)

    # Drain and log SDK stderr (MCP loading info, query lifecycle)
    remaining_stderr = (await proc.stderr.read()).decode("utf-8", errors="replace")
    if remaining_stderr.strip():
        sdk_logger = logging.getLogger("claude_sdk")
        rl = _current_run_logger.get(None)
        for _line in remaining_stderr.strip().splitlines():
            sdk_logger.info(_line.rstrip())
            if rl:
                rl.info("[claude-sdk-stderr] %s", _line.rstrip())

    if not full_text and not result_data:
        raise RuntimeError(f"No output from Claude SDK wrapper. stderr: {(remaining_stderr or '')[:500]}")

    usage = (result_data or {}).get("usage", {})

    # Log the full response to the per-run log
    rl = _current_run_logger.get(None)
    if rl:
        rl.info("-" * 40 + " ASSISTANT RESPONSE " + "-" * 40)
        rl.info("%s", full_text)
        rl.info("-" * 40 + " END RESPONSE " + "-" * 40)

    return {
        "text": full_text,
        "provider_id": provider["id"],
        "provider_name": provider.get("name", "Claude SDK"),
        "model": provider.get("model", "claude-sonnet-4-20250514"),
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        },
        "elapsed_seconds": elapsed,
        "tool_calls": (result_data or {}).get("tool_calls", []),
    }


async def _call_anthropic_stream(provider: dict, system_prompt: str, user_prompt: str, config: dict, on_chunk=None, on_tool_event=None) -> dict:
    """Call Anthropic Messages API with streaming.

    When ``provider["use_mcp"]`` is truthy, MCP tools are loaded and a
    multi-turn tool-use loop is executed with streaming text output.
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    api_key = _get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"No API key found for {provider['id']}")

    use_mcp = provider.get("use_mcp", False)
    rl = _current_run_logger.get(None)

    client = anthropic.AsyncAnthropic(api_key=api_key, timeout=config.get("timeout_seconds", 60))

    start = time.time()
    full_text = ""
    total_input_tokens = 0
    total_output_tokens = 0
    all_tool_calls = []

    messages = [{"role": "user", "content": user_prompt}]

    # Overall timeout for the entire streaming call (generous for large CoT outputs)
    stream_timeout = config.get("timeout_seconds", 180) + 120

    base_kwargs = dict(
        model=provider.get("model", "claude-sonnet-4-20250514"),
        max_tokens=provider.get("max_tokens", 4096),
        temperature=provider.get("temperature", 0.3),
        system=system_prompt,
    )

    async def _run(mcp_session=None, anthropic_tools=None):
        nonlocal full_text, total_input_tokens, total_output_tokens, all_tool_calls, messages
        turn = 0

        while True:
            turn += 1
            call_kwargs = {**base_kwargs, "messages": messages}
            if anthropic_tools:
                call_kwargs["tools"] = anthropic_tools

            async with client.messages.stream(**call_kwargs) as stream:
                async for text in stream.text_stream:
                    full_text += text
                    if on_chunk:
                        await on_chunk(text)

            final = await stream.get_final_message()
            if final and final.usage:
                total_input_tokens += final.usage.input_tokens
                total_output_tokens += final.usage.output_tokens

            # Check for tool_use blocks
            tool_use_blocks = [b for b in (final.content or []) if getattr(b, "type", None) == "tool_use"]

            if final.stop_reason == "tool_use" and tool_use_blocks and mcp_session and turn < _MAX_TOOL_TURNS:
                # Append assistant response to messages
                messages.append({"role": "assistant", "content": final.content})

                tool_results_content = []
                for block in tool_use_blocks:
                    tc = {"id": block.id, "name": block.name, "input": block.input}
                    if on_tool_event:
                        await on_tool_event("tool_call", tc)
                    if rl:
                        rl.info("[MCP tool_call] %s(%s)", block.name, json.dumps(block.input)[:300])

                    result_text = await _mcp.call_tool(mcp_session, block.name, block.input)
                    tc["result"] = result_text
                    tc["is_error"] = False
                    all_tool_calls.append(tc)

                    if on_tool_event:
                        await on_tool_event("tool_result", {
                            "tool_use_id": block.id,
                            "result": result_text,
                            "is_error": False,
                        })
                    if rl:
                        rl.info("[MCP tool_result] %s -> %d chars", block.name, len(result_text))

                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

                messages.append({"role": "user", "content": tool_results_content})
                # Reset full_text for the next turn — we only want the final text response
                full_text = ""
                continue

            # Done — collect any remaining tool_use blocks
            for block in tool_use_blocks:
                all_tool_calls.append({"id": block.id, "name": block.name, "input": block.input})
            return

    try:
        if use_mcp:
            try:
                async with _mcp.connect() as mcp_session:
                    mcp_tools = await _mcp.get_tools(mcp_session)
                    anthropic_tools = _mcp.tools_to_anthropic_format(mcp_tools)
                    if rl:
                        rl.info("Loaded %d MCP tools for Anthropic stream", len(anthropic_tools))
                    await asyncio.wait_for(_run(mcp_session=mcp_session, anthropic_tools=anthropic_tools), timeout=stream_timeout)
            except RuntimeError as mcp_err:
                if "MCP connection failed" in str(mcp_err):
                    logger.warning("MCP unavailable, falling back to no-MCP mode: %s", mcp_err)
                    if rl:
                        rl.warning("MCP connection failed — running without MCP tools: %s", mcp_err)
                    await asyncio.wait_for(_run(), timeout=stream_timeout)
                else:
                    raise
        else:
            await asyncio.wait_for(_run(), timeout=stream_timeout)
    except asyncio.TimeoutError:
        raise RuntimeError(f"Anthropic stream timed out after {stream_timeout}s")

    elapsed = round(time.time() - start, 2)

    return {
        "text": full_text,
        "provider_id": provider["id"],
        "provider_name": provider["name"],
        "model": provider.get("model", "claude-sonnet-4-20250514"),
        "usage": {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens},
        "elapsed_seconds": elapsed,
        "tool_calls": all_tool_calls,
    }


async def _call_openai_stream(provider: dict, system_prompt: str, user_prompt: str, config: dict, on_chunk=None) -> dict:
    """Call OpenAI Chat Completions API with streaming."""
    try:
        import openai
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    api_key = _get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"No API key found for {provider['id']}")

    kwargs = {"api_key": api_key, "timeout": config.get("timeout_seconds", 60)}
    if provider.get("base_url"):
        kwargs["base_url"] = provider["base_url"]

    client = openai.AsyncOpenAI(**kwargs)

    start = time.time()
    full_text = ""

    stream_timeout = config.get("timeout_seconds", 180) + 120

    async def _do_stream():
        nonlocal full_text
        stream = await client.chat.completions.create(
            model=provider.get("model", "gpt-4o"),
            max_tokens=provider.get("max_tokens", 4096),
            temperature=provider.get("temperature", 0.3),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                full_text += delta
                if on_chunk:
                    await on_chunk(delta)

    try:
        await asyncio.wait_for(_do_stream(), timeout=stream_timeout)
    except asyncio.TimeoutError:
        raise RuntimeError(f"OpenAI stream timed out after {stream_timeout}s")

    elapsed = round(time.time() - start, 2)

    return {
        "text": full_text,
        "provider_id": provider["id"],
        "provider_name": provider["name"],
        "model": provider.get("model"),
        "usage": {},
        "elapsed_seconds": elapsed,
    }


_STREAM_CALL_MAP = {
    "anthropic": _call_anthropic_stream,
    "openai": _call_openai_stream,
    "openai_compatible": _call_openai_stream,
    "claude_sdk": _call_claude_sdk_stream,
}


async def call_ai_stream(system_prompt: str, user_prompt: str, on_chunk=None, on_tool_event=None, provider_id: Optional[str] = None, run_logger=None, max_tokens: Optional[int] = None) -> dict:
    """
    Send a prompt with streaming support.

    ``on_chunk(text_delta)`` is called for each text fragment as it arrives.
    ``on_tool_event(event_type, data)`` is called for tool_call and tool_result events.
    ``max_tokens`` overrides the provider's configured max_tokens when set.
    Returns the full result dict when complete.
    """
    if run_logger:
        _current_run_logger.set(run_logger)
    config = load_config()
    failover = config.get("failover_enabled", True)
    retries = config.get("retry_attempts", 2)

    if provider_id:
        matches = [p for p in config["providers"] if p["id"] == provider_id]
        if not matches:
            raise RuntimeError(f"Unknown provider: {provider_id}")
        providers = matches
    else:
        providers = get_enabled_providers(config)
        if not providers:
            raise RuntimeError("No AI providers are enabled.")

    # Apply per-call max_tokens override (e.g. brief phase needs more output room)
    if max_tokens is not None:
        providers = [{**p, "max_tokens": max_tokens} for p in providers]

    errors = []

    for provider in providers:
        call_fn = _STREAM_CALL_MAP.get(provider.get("type"))
        if not call_fn:
            errors.append(f"{provider['id']}: unsupported type '{provider.get('type')}'")
            continue

        for attempt in range(1, retries + 1):
            try:
                log_msg = "AI stream call: provider=%s model=%s attempt=%d/%d"
                log_args = (provider["id"], provider.get("model"), attempt, retries)
                logger.info(log_msg, *log_args)
                if run_logger:
                    run_logger.info(log_msg, *log_args)
                # Pass on_tool_event to providers that support it (claude_sdk, anthropic)
                call_kwargs = {"on_chunk": on_chunk}
                if on_tool_event and provider.get("type") in ("claude_sdk", "anthropic"):
                    call_kwargs["on_tool_event"] = on_tool_event
                result = await call_fn(provider, system_prompt, user_prompt, config, **call_kwargs)
                success_msg = "AI stream SUCCESS: provider=%s model=%s elapsed=%.2fs tokens_in=%d tokens_out=%d"
                success_args = (
                    result.get("provider_id"), result.get("model"), result.get("elapsed_seconds", 0),
                    result.get("usage", {}).get("input_tokens", 0),
                    result.get("usage", {}).get("output_tokens", 0),
                )
                logger.info(success_msg, *success_args)
                if run_logger:
                    run_logger.info(success_msg, *success_args)
                return result
            except BaseException as exc:
                exc = _unwrap_exception_group(exc)
                msg = f"{provider['id']} attempt {attempt}: {type(exc).__name__}: {exc}"
                logger.warning(msg)
                if run_logger:
                    run_logger.warning(msg)
                errors.append(msg)
                # Back off on rate-limit errors before retrying
                if _is_rate_limit_error(exc) and attempt < retries:
                    backoff = _rate_limit_backoff_seconds(attempt)
                    backoff_msg = f"{provider['id']}: rate-limited, waiting {backoff:.0f}s before retry"
                    logger.info(backoff_msg)
                    if run_logger:
                        run_logger.info(backoff_msg)
                    await asyncio.sleep(backoff)

        if not failover:
            break

    raise RuntimeError(
        "All AI providers failed:\n" + "\n".join(f"  - {e}" for e in errors)
    )


# ---------------------------------------------------------------------------
# Provider-specific async CHAT implementations (multi-turn)
# ---------------------------------------------------------------------------

async def _call_anthropic_chat(provider: dict, system_prompt: str, messages: list[dict], config: dict) -> dict:
    """Call the Anthropic Messages API with a full messages array asynchronously.

    When ``provider["use_mcp"]`` is truthy, MCP tools are loaded and a
    multi-turn tool-use loop is executed.
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    api_key = _get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"No API key found for {provider['id']} (env: {provider.get('api_key_env')})")

    use_mcp = provider.get("use_mcp", False)

    # Log request to per-run log
    rl = _current_run_logger.get(None)
    if rl:
        rl.info("=" * 60)
        rl.info("ANTHROPIC CHAT REQUEST (model=%s, messages=%d, use_mcp=%s)", provider.get("model"), len(messages), use_mcp)
        rl.info("-" * 40 + " SYSTEM PROMPT " + "-" * 40)
        rl.info("%s", system_prompt)
        rl.info("-" * 40 + " MESSAGES " + "-" * 40)
        for m in messages:
            content_preview = m["content"] if isinstance(m["content"], str) else str(m["content"])
            rl.info("[%s]: %s", m["role"].upper(), content_preview[:1000])
        rl.info("=" * 60)

    client = anthropic.AsyncAnthropic(
        api_key=api_key,
        timeout=config.get("timeout_seconds", 60),
    )

    start = time.time()
    total_input_tokens = 0
    total_output_tokens = 0
    all_tool_calls = []

    base_kwargs = dict(
        model=provider.get("model", "claude-sonnet-4-20250514"),
        max_tokens=provider.get("max_tokens", 64000),
        temperature=provider.get("temperature", 0.3),
        system=system_prompt,
    )

    async def _run(mcp_session=None, anthropic_tools=None):
        nonlocal messages, total_input_tokens, total_output_tokens, all_tool_calls
        turn = 0

        while True:
            turn += 1
            call_kwargs = {**base_kwargs, "messages": messages}
            if anthropic_tools:
                call_kwargs["tools"] = anthropic_tools

            response = await client.messages.create(**call_kwargs)
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            tool_use_blocks = [b for b in (response.content or []) if getattr(b, "type", None) == "tool_use"]

            if response.stop_reason == "tool_use" and tool_use_blocks and mcp_session and turn < _MAX_TOOL_TURNS:
                messages.append({"role": "assistant", "content": response.content})

                tool_results_content = []
                for block in tool_use_blocks:
                    tc = {"id": block.id, "name": block.name, "input": block.input}
                    if rl:
                        rl.info("[MCP tool_call] %s(%s)", block.name, json.dumps(block.input)[:300])
                    result_text = await _mcp.call_tool(mcp_session, block.name, block.input)
                    tc["result"] = result_text
                    tc["is_error"] = False
                    all_tool_calls.append(tc)
                    if rl:
                        rl.info("[MCP tool_result] %s -> %d chars", block.name, len(result_text))
                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

                messages.append({"role": "user", "content": tool_results_content})
                continue

            for block in tool_use_blocks:
                all_tool_calls.append({"id": block.id, "name": block.name, "input": block.input})
            return response

    if use_mcp:
        try:
            async with _mcp.connect() as mcp_session:
                mcp_tools = await _mcp.get_tools(mcp_session)
                anthropic_tools = _mcp.tools_to_anthropic_format(mcp_tools)
                if rl:
                    rl.info("Loaded %d MCP tools for Anthropic chat", len(anthropic_tools))
                response = await _run(mcp_session=mcp_session, anthropic_tools=anthropic_tools)
        except RuntimeError as mcp_err:
            if "MCP connection failed" in str(mcp_err):
                logger.warning("MCP unavailable, falling back to no-MCP mode: %s", mcp_err)
                if rl:
                    rl.warning("MCP connection failed — running without MCP tools: %s", mcp_err)
                response = await _run()
            else:
                raise
    else:
        response = await _run()

    elapsed = round(time.time() - start, 2)

    text = ""
    for block in (response.content or []):
        if getattr(block, "type", None) == "text":
            text = block.text
            break

    # Detect truncation due to max_tokens limit
    if response.stop_reason == "max_tokens":
        text += "\n\n*(Response was truncated due to length. Ask a follow-up to continue.)*"
        if rl:
            rl.warning("Response truncated: stop_reason=max_tokens")

    if rl:
        rl.info("-" * 40 + " ASSISTANT RESPONSE " + "-" * 40)
        rl.info("%s", text)
        rl.info("-" * 40 + " END RESPONSE " + "-" * 40)

    return {
        "text": text,
        "provider_id": provider["id"],
        "provider_name": provider["name"],
        "model": response.model,
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        },
        "elapsed_seconds": elapsed,
        "tool_calls": all_tool_calls,
    }


async def _call_openai_chat(provider: dict, system_prompt: str, messages: list[dict], config: dict) -> dict:
    """Call the OpenAI Chat Completions API with a full messages array asynchronously."""
    try:
        import openai
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    api_key = _get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"No API key found for {provider['id']} (env: {provider.get('api_key_env')})")

    kwargs = {"api_key": api_key, "timeout": config.get("timeout_seconds", 60)}
    if provider.get("base_url"):
        kwargs["base_url"] = provider["base_url"]

    client = openai.AsyncOpenAI(**kwargs)

    # Prepend system message to the messages array
    api_messages = [{"role": "system", "content": system_prompt}] + messages

    start = time.time()
    create_kwargs = {
        "model": provider.get("model", "gpt-4o"),
        "temperature": provider.get("temperature", 0.3),
        "messages": api_messages,
    }
    # Only set max_tokens if explicitly configured; otherwise let the model use its default max
    if provider.get("max_tokens"):
        create_kwargs["max_tokens"] = provider["max_tokens"]
    response = await client.chat.completions.create(**create_kwargs)
    elapsed = round(time.time() - start, 2)

    choice = response.choices[0] if response.choices else None
    text = choice.message.content if choice else ""

    # Detect truncation due to max_tokens limit
    if choice and choice.finish_reason == "length":
        text += "\n\n*(Response was truncated due to length. Ask a follow-up to continue.)*"
        rl = _current_run_logger.get(None)
        if rl:
            rl.warning("Response truncated: finish_reason=length")

    usage = {}
    if response.usage:
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    return {
        "text": text,
        "provider_id": provider["id"],
        "provider_name": provider["name"],
        "model": response.model or provider.get("model"),
        "usage": usage,
        "elapsed_seconds": elapsed,
    }


async def _call_claude_sdk_chat(provider: dict, system_prompt: str, messages: list[dict], config: dict) -> dict:
    """Call Claude SDK by concatenating messages into a single prompt.

    Chat calls enable MCP tool use so Claude can query the betstamp-intelligence
    server on demand instead of relying on pre-baked pipeline context.
    """
    # Log individual messages to run logger before concatenation
    rl = _current_run_logger.get(None)
    if rl:
        rl.info("CLAUDE SDK CHAT: %d messages, MCP=True, max_turns=unlimited", len(messages))
        for i, msg in enumerate(messages):
            rl.info("  Message[%d] %s: %s", i, msg["role"].upper(), msg["content"][:500])

    # Concatenate all messages into a single user prompt for the request-response SDK
    parts = []
    for msg in messages:
        role_label = msg["role"].upper()
        parts.append(f"{role_label}: {msg['content']}")
    combined_prompt = "\n\n".join(parts)

    return await _call_claude_sdk(provider, system_prompt, combined_prompt, config, use_mcp=True)


# Map provider types to their async chat functions
_CHAT_CALL_MAP = {
    "anthropic": _call_anthropic_chat,
    "openai": _call_openai_chat,
    "openai_compatible": _call_openai_chat,
    "claude_sdk": _call_claude_sdk_chat,
}


# ---------------------------------------------------------------------------
# Streaming chat provider functions (multi-turn + on_chunk callback)
# ---------------------------------------------------------------------------

async def _call_anthropic_chat_stream(provider: dict, system_prompt: str, messages: list[dict], config: dict, on_chunk=None, on_tool_event=None) -> dict:
    """Call Anthropic Messages API with streaming for multi-turn chat.

    When ``provider["use_mcp"]`` is truthy, MCP tools are loaded and a
    multi-turn tool-use loop is executed with streaming text output and
    live tool event callbacks.
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    api_key = _get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"No API key found for {provider['id']}")

    use_mcp = provider.get("use_mcp", False)
    rl = _current_run_logger.get(None)
    if rl:
        rl.info("=" * 60)
        rl.info("ANTHROPIC CHAT STREAM REQUEST (model=%s, messages=%d, use_mcp=%s)", provider.get("model"), len(messages), use_mcp)
        rl.info("-" * 40 + " SYSTEM PROMPT " + "-" * 40)
        rl.info("%s", system_prompt)
        rl.info("-" * 40 + " MESSAGES " + "-" * 40)
        for m in messages:
            content_preview = m["content"] if isinstance(m["content"], str) else str(m["content"])
            rl.info("[%s]: %s", m["role"].upper(), content_preview[:1000])
        rl.info("=" * 60)

    client = anthropic.AsyncAnthropic(api_key=api_key, timeout=config.get("timeout_seconds", 60))

    start = time.time()
    full_text = ""
    total_input_tokens = 0
    total_output_tokens = 0
    all_tool_calls = []
    stream_timeout = config.get("timeout_seconds", 180) + 120

    base_kwargs = dict(
        model=provider.get("model", "claude-sonnet-4-20250514"),
        max_tokens=provider.get("max_tokens", 64000),
        temperature=provider.get("temperature", 0.3),
        system=system_prompt,
    )

    async def _run(mcp_session=None, anthropic_tools=None):
        nonlocal full_text, total_input_tokens, total_output_tokens, all_tool_calls, messages
        turn = 0

        while True:
            turn += 1
            call_kwargs = {**base_kwargs, "messages": messages}
            if anthropic_tools:
                call_kwargs["tools"] = anthropic_tools

            async with client.messages.stream(**call_kwargs) as stream:
                async for text in stream.text_stream:
                    full_text += text
                    if on_chunk:
                        await on_chunk(text)

            final = await stream.get_final_message()
            if final and final.usage:
                total_input_tokens += final.usage.input_tokens
                total_output_tokens += final.usage.output_tokens

            tool_use_blocks = [b for b in (final.content or []) if getattr(b, "type", None) == "tool_use"]

            if final.stop_reason == "tool_use" and tool_use_blocks and mcp_session and turn < _MAX_TOOL_TURNS:
                messages.append({"role": "assistant", "content": final.content})

                tool_results_content = []
                for block in tool_use_blocks:
                    tc = {"id": block.id, "name": block.name, "input": block.input}
                    if on_tool_event:
                        await on_tool_event("tool_call", tc)
                    if rl:
                        rl.info("[MCP tool_call] %s(%s)", block.name, json.dumps(block.input)[:300])

                    result_text = await _mcp.call_tool(mcp_session, block.name, block.input)
                    tc["result"] = result_text
                    tc["is_error"] = False
                    all_tool_calls.append(tc)

                    if on_tool_event:
                        await on_tool_event("tool_result", {
                            "tool_use_id": block.id,
                            "result": result_text,
                            "is_error": False,
                        })
                    if rl:
                        rl.info("[MCP tool_result] %s -> %d chars", block.name, len(result_text))

                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

                messages.append({"role": "user", "content": tool_results_content})
                # Reset text for next turn — we want the final text response
                full_text = ""
                continue

            # Done
            for block in tool_use_blocks:
                all_tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

            # Detect truncation
            if final and final.stop_reason == "max_tokens":
                truncation_notice = "\n\n*(Response was truncated due to length. Ask a follow-up to continue.)*"
                full_text += truncation_notice
                if on_chunk:
                    await on_chunk(truncation_notice)
            return

    try:
        if use_mcp:
            try:
                async with _mcp.connect() as mcp_session:
                    mcp_tools = await _mcp.get_tools(mcp_session)
                    anthropic_tools = _mcp.tools_to_anthropic_format(mcp_tools)
                    if rl:
                        rl.info("Loaded %d MCP tools for Anthropic chat stream", len(anthropic_tools))
                    await asyncio.wait_for(_run(mcp_session=mcp_session, anthropic_tools=anthropic_tools), timeout=stream_timeout)
            except RuntimeError as mcp_err:
                if "MCP connection failed" in str(mcp_err):
                    logger.warning("MCP unavailable, falling back to no-MCP mode: %s", mcp_err)
                    if rl:
                        rl.warning("MCP connection failed — running without MCP tools: %s", mcp_err)
                    await asyncio.wait_for(_run(), timeout=stream_timeout)
                else:
                    raise
        else:
            await asyncio.wait_for(_run(), timeout=stream_timeout)
    except asyncio.TimeoutError:
        raise RuntimeError(f"Anthropic chat stream timed out after {stream_timeout}s")

    elapsed = round(time.time() - start, 2)

    if rl:
        rl.info("-" * 40 + " ASSISTANT RESPONSE (streamed) " + "-" * 40)
        rl.info("%s", full_text)
        rl.info("-" * 40 + " END RESPONSE " + "-" * 40)

    return {
        "text": full_text,
        "provider_id": provider["id"],
        "provider_name": provider["name"],
        "model": provider.get("model", "claude-sonnet-4-20250514"),
        "usage": {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens},
        "elapsed_seconds": elapsed,
        "tool_calls": all_tool_calls,
    }


async def _call_openai_chat_stream(provider: dict, system_prompt: str, messages: list[dict], config: dict, on_chunk=None, on_tool_event=None) -> dict:
    """Call OpenAI Chat Completions API with streaming for multi-turn chat."""
    try:
        import openai
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    api_key = _get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"No API key found for {provider['id']}")

    kwargs = {"api_key": api_key, "timeout": config.get("timeout_seconds", 60)}
    if provider.get("base_url"):
        kwargs["base_url"] = provider["base_url"]

    client = openai.AsyncOpenAI(**kwargs)

    # Prepend system message
    api_messages = [{"role": "system", "content": system_prompt}] + messages

    start = time.time()
    full_text = ""
    stream_timeout = config.get("timeout_seconds", 180) + 120

    async def _do_stream():
        nonlocal full_text
        create_kwargs = {
            "model": provider.get("model", "gpt-4o"),
            "temperature": provider.get("temperature", 0.3),
            "messages": api_messages,
            "stream": True,
        }
        if provider.get("max_tokens"):
            create_kwargs["max_tokens"] = provider["max_tokens"]

        stream = await client.chat.completions.create(**create_kwargs)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                full_text += delta
                if on_chunk:
                    await on_chunk(delta)

    try:
        await asyncio.wait_for(_do_stream(), timeout=stream_timeout)
    except asyncio.TimeoutError:
        raise RuntimeError(f"OpenAI chat stream timed out after {stream_timeout}s")

    elapsed = round(time.time() - start, 2)

    return {
        "text": full_text,
        "provider_id": provider["id"],
        "provider_name": provider["name"],
        "model": provider.get("model"),
        "usage": {},
        "elapsed_seconds": elapsed,
    }


async def _call_claude_sdk_chat_stream(provider: dict, system_prompt: str, messages: list[dict], config: dict, on_chunk=None, on_tool_event=None) -> dict:
    """Call Claude SDK with streaming for multi-turn chat (concatenates messages, enables MCP)."""
    rl = _current_run_logger.get(None)
    if rl:
        rl.info("CLAUDE SDK CHAT STREAM: %d messages, MCP=True", len(messages))
        for i, msg in enumerate(messages):
            rl.info("  Message[%d] %s: %s", i, msg["role"].upper(), msg["content"][:500])

    parts = []
    for msg in messages:
        role_label = msg["role"].upper()
        parts.append(f"{role_label}: {msg['content']}")
    combined_prompt = "\n\n".join(parts)

    return await _call_claude_sdk_stream(provider, system_prompt, combined_prompt, config, on_chunk=on_chunk, on_tool_event=on_tool_event)


_CHAT_STREAM_CALL_MAP = {
    "anthropic": _call_anthropic_chat_stream,
    "openai": _call_openai_chat_stream,
    "openai_compatible": _call_openai_chat_stream,
    "claude_sdk": _call_claude_sdk_chat_stream,
}


async def call_ai_chat_stream(messages: list[dict], system_prompt: str, on_chunk=None, on_tool_event=None, provider_id: Optional[str] = None, run_logger=None) -> dict:
    """
    Send a multi-turn conversation with streaming support.

    ``on_chunk(text_delta)`` is called for each text fragment as it arrives.
    Returns the full result dict when complete (same shape as call_ai_chat).
    """
    if run_logger:
        _current_run_logger.set(run_logger)
    config = load_config()
    failover = config.get("failover_enabled", True)
    retries = config.get("retry_attempts", 2)

    if provider_id:
        matches = [p for p in config["providers"] if p["id"] == provider_id]
        if not matches:
            raise RuntimeError(f"Unknown provider: {provider_id}")
        providers = matches
    else:
        providers = get_enabled_providers(config)
        if not providers:
            raise RuntimeError("No AI providers are enabled. Configure at least one in AI Settings.")

    errors = []

    for provider in providers:
        call_fn = _CHAT_STREAM_CALL_MAP.get(provider.get("type"))
        if not call_fn:
            errors.append(f"{provider['id']}: unsupported type '{provider.get('type')}'")
            continue

        for attempt in range(1, retries + 1):
            try:
                log_msg = "AI chat stream call: provider=%s model=%s attempt=%d/%d messages=%d"
                log_args = (provider["id"], provider.get("model"), attempt, retries, len(messages))
                logger.info(log_msg, *log_args)
                if run_logger:
                    run_logger.info(log_msg, *log_args)
                result = await call_fn(provider, system_prompt, messages, config, on_chunk=on_chunk, on_tool_event=on_tool_event)
                success_msg = "AI chat stream SUCCESS: provider=%s model=%s elapsed=%.2fs"
                success_args = (result.get("provider_id"), result.get("model"), result.get("elapsed_seconds", 0))
                logger.info(success_msg, *success_args)
                if run_logger:
                    run_logger.info(success_msg, *success_args)
                return result
            except BaseException as exc:
                exc = _unwrap_exception_group(exc)
                msg = f"{provider['id']} attempt {attempt}: {type(exc).__name__}: {exc}"
                logger.warning(msg)
                if run_logger:
                    run_logger.warning(msg)
                errors.append(msg)
                # Back off on rate-limit errors before retrying
                if _is_rate_limit_error(exc) and attempt < retries:
                    backoff = _rate_limit_backoff_seconds(attempt)
                    backoff_msg = f"{provider['id']}: rate-limited, waiting {backoff:.0f}s before retry"
                    logger.info(backoff_msg)
                    if run_logger:
                        run_logger.info(backoff_msg)
                    await asyncio.sleep(backoff)

        if not failover:
            break

    raise RuntimeError(
        "All AI providers failed:\n" + "\n".join(f"  - {e}" for e in errors)
    )


# ---------------------------------------------------------------------------
# Main entry point — async call with failover
# ---------------------------------------------------------------------------

async def call_ai(system_prompt: str, user_prompt: str, provider_id: Optional[str] = None, run_logger=None, max_tokens: Optional[int] = None) -> dict:
    """
    Send a prompt to an AI provider and return the response.

    If ``provider_id`` is given, only that provider is tried.
    Otherwise, enabled providers are tried in priority order with failover.
    ``max_tokens`` overrides the provider's configured max_tokens when set.

    Returns a dict with keys: text, provider_id, provider_name, model, usage, elapsed_seconds.
    On total failure raises RuntimeError.
    """
    if run_logger:
        _current_run_logger.set(run_logger)
    config = load_config()
    failover = config.get("failover_enabled", True)
    retries = config.get("retry_attempts", 2)

    if provider_id:
        # Use a specific provider
        matches = [p for p in config["providers"] if p["id"] == provider_id]
        if not matches:
            raise RuntimeError(f"Unknown provider: {provider_id}")
        providers = matches
    else:
        providers = get_enabled_providers(config)
        if not providers:
            raise RuntimeError("No AI providers are enabled. Configure at least one in AI Settings.")

    # Apply per-call max_tokens override (e.g. brief phase needs more output room)
    if max_tokens is not None:
        providers = [{**p, "max_tokens": max_tokens} for p in providers]

    errors = []

    for provider in providers:
        call_fn = _CALL_MAP.get(provider.get("type"))
        if not call_fn:
            errors.append(f"{provider['id']}: unsupported type '{provider.get('type')}'")
            continue

        for attempt in range(1, retries + 1):
            try:
                log_msg = "AI call: provider=%s model=%s attempt=%d/%d"
                log_args = (provider["id"], provider.get("model"), attempt, retries)
                logger.info(log_msg, *log_args)
                if run_logger:
                    run_logger.info(log_msg, *log_args)
                result = await call_fn(provider, system_prompt, user_prompt, config)
                success_msg = "AI call SUCCESS: provider=%s model=%s elapsed=%.2fs tokens_in=%d tokens_out=%d"
                success_args = (
                    result.get("provider_id"), result.get("model"), result.get("elapsed_seconds", 0),
                    result.get("usage", {}).get("input_tokens", 0),
                    result.get("usage", {}).get("output_tokens", 0),
                )
                logger.info(success_msg, *success_args)
                if run_logger:
                    run_logger.info(success_msg, *success_args)
                return result
            except BaseException as exc:
                exc = _unwrap_exception_group(exc)
                msg = f"{provider['id']} attempt {attempt}: {type(exc).__name__}: {exc}"
                logger.warning(msg)
                if run_logger:
                    run_logger.warning(msg)
                errors.append(msg)
                # Back off on rate-limit errors before retrying
                if _is_rate_limit_error(exc) and attempt < retries:
                    backoff = _rate_limit_backoff_seconds(attempt)
                    backoff_msg = f"{provider['id']}: rate-limited, waiting {backoff:.0f}s before retry"
                    logger.info(backoff_msg)
                    if run_logger:
                        run_logger.info(backoff_msg)
                    await asyncio.sleep(backoff)

        if not failover:
            break

    raise RuntimeError(
        "All AI providers failed:\n" + "\n".join(f"  - {e}" for e in errors)
    )


async def call_ai_chat(messages: list[dict], system_prompt: str, provider_id: Optional[str] = None, run_logger=None) -> dict:
    """
    Send a multi-turn conversation to an AI provider and return the response.

    ``messages`` is a list of dicts with "role" and "content" keys.
    If ``provider_id`` is given, only that provider is tried.
    Otherwise, enabled providers are tried in priority order with failover.

    Returns a dict with keys: text, provider_id, provider_name, model, usage, elapsed_seconds.
    On total failure raises RuntimeError.
    """
    if run_logger:
        _current_run_logger.set(run_logger)
    config = load_config()
    failover = config.get("failover_enabled", True)
    retries = config.get("retry_attempts", 2)

    if provider_id:
        matches = [p for p in config["providers"] if p["id"] == provider_id]
        if not matches:
            raise RuntimeError(f"Unknown provider: {provider_id}")
        providers = matches
    else:
        providers = get_enabled_providers(config)
        if not providers:
            raise RuntimeError("No AI providers are enabled. Configure at least one in AI Settings.")

    errors = []

    for provider in providers:
        call_fn = _CHAT_CALL_MAP.get(provider.get("type"))
        if not call_fn:
            errors.append(f"{provider['id']}: unsupported type '{provider.get('type')}'")
            continue

        for attempt in range(1, retries + 1):
            try:
                log_msg = "AI chat call: provider=%s model=%s attempt=%d/%d messages=%d"
                log_args = (provider["id"], provider.get("model"), attempt, retries, len(messages))
                logger.info(log_msg, *log_args)
                if run_logger:
                    run_logger.info(log_msg, *log_args)
                result = await call_fn(provider, system_prompt, messages, config)
                success_msg = "AI chat SUCCESS: provider=%s model=%s elapsed=%.2fs tokens_in=%d tokens_out=%d"
                success_args = (
                    result.get("provider_id"), result.get("model"), result.get("elapsed_seconds", 0),
                    result.get("usage", {}).get("input_tokens", 0),
                    result.get("usage", {}).get("output_tokens", 0),
                )
                logger.info(success_msg, *success_args)
                if run_logger:
                    run_logger.info(success_msg, *success_args)
                return result
            except BaseException as exc:
                exc = _unwrap_exception_group(exc)
                msg = f"{provider['id']} attempt {attempt}: {type(exc).__name__}: {exc}"
                logger.warning(msg)
                if run_logger:
                    run_logger.warning(msg)
                errors.append(msg)
                # Back off on rate-limit errors before retrying
                if _is_rate_limit_error(exc) and attempt < retries:
                    backoff = _rate_limit_backoff_seconds(attempt)
                    backoff_msg = f"{provider['id']}: rate-limited, waiting {backoff:.0f}s before retry"
                    logger.info(backoff_msg)
                    if run_logger:
                        run_logger.info(backoff_msg)
                    await asyncio.sleep(backoff)

        if not failover:
            break

    raise RuntimeError(
        "All AI providers failed:\n" + "\n".join(f"  - {e}" for e in errors)
    )


# ---------------------------------------------------------------------------
# Chat system prompt
# ---------------------------------------------------------------------------

CHAT_SYSTEM_PROMPT = """\
You are a sports-betting data analyst assistant for BetStamp. You help users understand \
odds data, analysis results, and betting opportunities.

You have access to the pipeline results from the current session, which include:
- Detection data: enriched odds with implied probabilities, vig calculations, and fair odds
- Analysis data: cross-sportsbook analysis identifying best lines, arbitrage, outliers, efficiency, and stale lines
- Brief data: actionable betting recommendations

RESPONSE FORMAT — THINKING THEN ANSWER
Always structure your response in two parts:
1. First, wrap your step-by-step reasoning inside <thinking>...</thinking> tags. \
This is where you analyze the data, cross-reference numbers, and work through your logic. \
The user can see this section collapsed — use it to show your work.
2. After the closing </thinking> tag, write your visible answer. This is the main response \
the user sees immediately.

Example:
<thinking>
Let me look at the pipeline data for this game...
The spread is -5.5 at BetMGM with odds of -110, while Pinnacle has -4.5 at -105...
</thinking>

Based on the data, BetMGM has the best spread at -5.5 (-110) for this game...

When pipeline context is provided, reference specific data points, game IDs, sportsbooks, \
and numbers in your answers. Be precise and quantitative. If the user asks about something \
not in the data, say so clearly.

SCOPE — STAY IN YOUR LANE
You are a sports-betting data analyst. Your expertise is limited to the betting data \
from the current pipeline session.
- If the user asks a question outside this scope (e.g., historical facts, trivia, \
general knowledge, news, politics, science, or anything unrelated to sports-betting data), \
do NOT attempt to answer it. Instead, say: "I don't have information on that — my expertise \
is limited to sports-betting data and odds analysis. Feel free to ask me about odds, lines, \
value bets, or sportsbook comparisons!"
- Do NOT use your general training knowledge to answer non-betting questions. Even if you \
think you know the answer, politely decline.
- Questions about general sports history, player stats, team history, or game results \
that are NOT in the current pipeline data are also out of scope. Say so clearly.

MANDATORY — USE PRE-COMPUTED NUMBERS ONLY
You must NEVER perform your own arithmetic, math, or statistical calculations. \
All numbers (payouts, edges, bankroll impacts, ROI, vig differences, profit/loss, \
implied probabilities, EV percentages, Kelly fractions, averages, odds differences, \
percentage changes) are already computed in the data provided to you. Copy them \
directly — do NOT re-derive, estimate, round, or recalculate any values yourself. \
If a number is not present in the data, state that it is unavailable rather than \
computing it.

Keep responses concise but thorough. Use bullet points for lists of findings.
"""


AGENT_SYSTEM_PROMPT = """\
You are BetStamp Agent, an autonomous sports-betting intelligence analyst. \
You help users understand odds data, market conditions, and betting opportunities.

You have access to live MCP tools (list_data_files, list_events, get_market_summary, \
get_best_bets_today, find_arbitrage_opportunities, get_odds_comparison, get_vig_analysis, \
get_fair_odds, detect_line_outliers, get_daily_digest, and many more) to fetch real-time \
betting data. You do NOT need pre-loaded pipeline context — fetch what you need on demand.

EPISTEMIC HONESTY — THIS IS YOUR HIGHEST PRIORITY
- If you do not know something, say "I don't know" or "I'm not sure" explicitly. \
Never guess, fabricate, or fill in gaps with speculation presented as fact.
- If data is unavailable, a tool returns empty results, or no data file is loaded, \
say so clearly. Do NOT invent numbers, games, odds, or sportsbook names.
- If your confidence is low, quantify it: "I'm roughly 60% confident because…"
- Distinguish clearly between three categories in your answers: \
(1) Facts directly from the data, (2) Inferences you are drawing from the data, \
and (3) General knowledge or opinions. Label each so the user knows what is grounded \
and what is interpretation.
- When data is stale, limited, or covers only a subset of games/books, caveat your \
conclusions. Say things like "Based on the [N] games currently in the data…" or \
"Note: this only covers [sportsbooks listed], other books may differ."

SCOPE — STAY IN YOUR LANE
You are a sports-betting data analyst. Your expertise is limited to the betting data \
available through your MCP tools: odds, lines, vig, arbitrage, EV, sportsbook analysis, \
and related betting intelligence.
- If the user asks a question outside this scope (e.g., historical facts, trivia, \
general knowledge, news, politics, science, or anything unrelated to sports-betting data), \
do NOT attempt to answer it. Instead, say: "I don't have information on that — my expertise \
is limited to sports-betting data and odds analysis. Feel free to ask me about odds, lines, \
value bets, or sportsbook comparisons!"
- Do NOT use your general training knowledge to answer non-betting questions. Even if you \
think you know the answer, you are not the right tool for that — politely decline.
- Questions about general sports history, player stats, team history, or game results \
that are NOT in the current betting data are also out of scope. Say so clearly.

WHEN YOU CANNOT ANSWER
When you lack sufficient data to answer a question, do NOT guess. Instead:
1. State what you looked for and what you did not find.
2. Suggest what data the user could provide or what they could ask differently.
3. Offer to look at adjacent or related data that IS available.

RESPONSE FORMAT — THINKING THEN ANSWER
Always structure your response in two parts:
1. First, wrap your step-by-step reasoning inside <thinking>...</thinking> tags. \
This is where you analyze the data, cross-reference numbers, show what tools you called, \
and work through your logic. The user can see this section collapsed — use it to show your work.
2. After the closing </thinking> tag, write your visible answer. This is the main response \
the user sees immediately.

Example:
<thinking>
Let me fetch the market summary to see what's available today...
I found 8 games. The user asked about arbitrage — let me check...
The arb scan returned 0 results. I should be honest about that.
</thinking>

I checked all 8 games currently in the data and found **no arbitrage opportunities** at this time. \
This is common — true arbs are rare and short-lived. Here's what I can tell you instead...

MANDATORY — USE PRE-COMPUTED NUMBERS ONLY
You must NEVER perform your own arithmetic, math, or statistical calculations. \
All numbers (payouts, edges, bankroll impacts, ROI, vig differences, profit/loss, \
implied probabilities, EV percentages, Kelly fractions, averages, odds differences, \
percentage changes) are already computed in the data provided to you. Copy them \
directly — do NOT re-derive, estimate, round, or recalculate any values yourself. \
If a number is not present in the data, state that it is unavailable rather than \
computing it.

Keep responses concise but thorough. Use bullet points for lists of findings.
"""


# ---------------------------------------------------------------------------
# Prompts for the BetStamp pipeline phases
# ---------------------------------------------------------------------------

ANALYZE_SYSTEM_PROMPT = """\
You are a sharp sports-betting analyst AI. You receive enriched odds data with \
implied probabilities, vig calculations, and fair odds for multiple sportsbooks.

Your job is to perform cross-sportsbook analysis and produce structured JSON output. \
Analyze the data for:
1. **Best Lines** — Which sportsbook has the best line for each side of each market.
2. **Arbitrage Opportunities** — Any combination of bets across books that guarantee profit.
3. **Middle Opportunities** — Spread/total line gaps across books that allow winning both sides.
4. **Outlier Lines** — Lines that deviate significantly from the consensus.
5. **Market Efficiency** — How tight/wide each book's vig is relative to others.
6. **Stale Lines** — Books whose lines haven't updated recently.
7. **Fair Odds** — Consensus no-vig fair probabilities per game/market.

MANDATORY — USE PRE-COMPUTED NUMBERS ONLY
You must NEVER perform your own arithmetic, math, or statistical calculations. \
All numbers (vig percentages, edge sizes, implied probabilities, profit margins, \
combined implied probabilities, EV edges, averages) are already computed in the \
data provided to you. Copy them directly — do NOT re-derive, estimate, round, or \
recalculate any values yourself. If a number is not present in the data, state \
that it is unavailable rather than computing it.

Return ONLY valid JSON (no markdown fences). Use this structure:
{
  "best_lines": [...],
  "arbitrage": [...],
  "middles": [...],
  "outliers": [...],
  "efficiency_ranking": [...],
  "stale_lines": [...],
  "fair_odds_summary": [...],
  "summary": "One paragraph executive summary"
}
"""

# ---------------------------------------------------------------------------
# AI Analyze Phase — Chain-of-Thought System Prompt
# ---------------------------------------------------------------------------

ANALYZE_COT_SYSTEM_PROMPT = """\
You are an expert sports-betting market analyst AI. You receive pre-computed \
cross-sportsbook analysis data and produce a deep, expert-level analytical \
interpretation using MCP tools.

## CORE RULES
1. NEVER assume — call MCP tools before making any claim or conclusion.
2. NEVER fabricate — only reference data from tool results or input payload.
3. EXACT QUOTING — copy numbers from tool results exactly (no rounding).
4. ARITHMETIC — use `arithmetic_evaluate` for multi-step math. Simple 2-number \
   operations (a+b, a-b, a*b, a/b) may be done inline.
5. An analysis with zero tool calls is a FAILED analysis. Minimum: 5+ calls.

## AVAILABLE MCP TOOLS (betstamp-intelligence)

**Discovery:** list_data_files, list_events
**Odds & Lines:** get_odds_comparison, get_best_odds, get_worst_odds, get_fair_odds, \
  get_shin_fair_odds, detect_line_outliers, detect_stale_lines, infer_odds_movement
**Value & Arb:** find_expected_value_bets, find_arbitrage_opportunities, \
  find_middle_opportunities, find_cross_market_arbitrage, get_best_bets_today, get_kelly_sizing
**Book Analysis:** get_vig_analysis, get_book_rankings, get_hold_percentage, \
  get_sharpness_scores, get_sportsbook_clusters, get_sportsbook_correlation_network, \
  get_information_flow, get_closing_line_value
**Advanced Analytics:** get_market_entropy, get_market_correlations, get_power_rankings, \
  get_implied_scores, get_poisson_score_predictions, get_gamlss_analysis, \
  detect_knn_anomalies, get_odds_shape_analysis, get_synthetic_hold_free_market
**Arithmetic:** arithmetic_add, arithmetic_subtract, arithmetic_multiply, \
  arithmetic_divide, arithmetic_evaluate (preferred for complex expressions)
**Digests:** get_market_overview, get_betting_opportunities, get_line_quality, \
  get_advanced_analytics, get_daily_digest, calculate_odds

## WORKFLOW
1. **Discover:** list_events → see all games/events
2. **Verify:** get_vig_analysis, get_fair_odds, detect_line_outliers → cross-check data
3. **Deepen:** find_expected_value_bets, find_arbitrage_opportunities, get_book_rankings, \
   get_market_entropy → enrich analysis
4. **Anomalies:** get_gamlss_analysis, detect_knn_anomalies, get_poisson_score_predictions
5. **Cross-market:** get_market_correlations, find_cross_market_arbitrage

## RESPONSE FORMAT
After calling tools, structure your response:

<thinking>
Step-by-step reasoning. For each claim: cite the tool result, state what it means, \
cross-reference with other results. Flag any contradictions with pre-computed data.
</thinking>

<analysis>
Structured JSON (no markdown fences):
{
  "insights": [{"type": "arbitrage|middle|outlier|value|efficiency|stale|market_trend|anomaly",
    "severity": "critical|high|medium|low|info", "title": "...",
    "description": "... with exact numbers from tools", "games": ["game_id"],
    "books": ["book1"], "confidence": "high|medium|low",
    "reasoning": "tools used and what they returned", "tool_verified": true}],
  "market_assessment": {"overall_health": "healthy|volatile|thin|stale|anomalous",
    "efficiency_score": 0-100, "key_themes": ["..."], "risk_flags": ["..."]},
  "book_grades": {"book_name": {"grade": "A-F", "avg_vig": 3.5,
    "strengths": ["..."], "weaknesses": ["..."]}},
  "top_actions": [{"priority": 1, "action": "...", "reasoning": "...", "urgency": "immediate|today|monitor"}],
  "tools_used": ["..."], "verification_notes": "...",
  "summary": "2-3 sentence executive summary"
}
</analysis>

## HARD GATES (violations = audit failure)
- ARBITRAGE: Only include arbs that appear in find_arbitrage_opportunities output.
- +EV BETS: Only include +EV bets from find_expected_value_bets output.
- MIDDLES: Only include middles from find_middle_opportunities output.
- Include Kelly sizing (quarter-Kelly) for every recommended bet.
- No fabricated sportsbook names or games.
"""

BRIEF_SYSTEM_PROMPT = """\
You are a senior sports-betting market analyst AI. Produce a clear, accurate daily \
market briefing from the provided data that a human analyst could review and act on.

FORMATTING: Your ENTIRE response must be ONLY the briefing markdown. First characters \
MUST be "## Market Snapshot". ZERO preamble, meta-commentary, or process discussion.

## Sections (use ## headings):

**## Market Snapshot** — 2-3 sentence overview. Use "counts" object for ALL totals.
**## Top Value Bets** — Up to 5 best value bets ranked by confidence. Each: game, \
  market, side, line, sportsbook, odds, why it's value, confidence (HIGH/MEDIUM/LOW), \
  quarter-Kelly sizing (from top_ev_bets or ai_insights).
**## Best Line Shopping** — Use line_shopping_pairs (pre-paired with pre-computed metrics). \
  Report: game_label, market, best odds per side with book name, gap_pct, is_arb. \
  Highlight pairs where gap_pct > 5%. NEVER compute gaps from American odds yourself \
  (American odds arithmetic is non-linear). Copy gap_pct and combined_implied_prob directly.
**## Arbitrage Opportunities** — Both legs, combined implied prob, profit %. \
  Quote from ai_insights first, fall back to arbitrage array. If none, say so.
**## Middle Opportunities** — Both legs, gap, winning range. Count = counts.middles_total.
**## Stale & Suspect Lines** — Outdated lines, outliers, high vig. Name book + game.
**## Fair Odds & Expected Value** — Consensus fair probs, +EV bets with exact edge %. \
  Only report +EV bets explicitly present in data.
**## Sportsbook Rankings** — 1) Vig ranking: copy efficiency_ranking in exact order \
  (position 0 = lowest/best vig). 2) Overall letter grades from book_grades.
**## Market Movements** — Cross-book discrepancies, line movements, anomalies.
**## Analyst Notes** — 2-4 sentences of takeaways and caveats.

## DATA RULES
- **Priority:** ai_insights/top_actions (highest) > top_ev_bets > efficiency_ranking > raw arrays
- **Counts:** Use "counts" object verbatim. NEVER count array elements yourself.
- **Numbers:** Copy ALL numbers exactly from data. NEVER re-derive, round, or compute.
- **Entity triples:** Every (sportsbook, team, odds) must come from a SINGLE data entry.
- **No fabrication:** Only report bets/opportunities explicitly present in the data.
- **Kelly sizing:** Required for every value bet. Check ai_insights, top_actions, AND \
  top_ev_bets before saying "not available".
- **Staleness:** Warn if bet uses lines >60 minutes old.
- **Confidence cap:** NEVER assign HIGH confidence to any bet that uses lines flagged \
  as stale (>30 min old) in stale_lines. Downgrade to MEDIUM at most and note staleness.
- **No aggregate invention:** NEVER invent aggregate metrics (e.g., "market efficiency \
  at X%", "consensus strength of Y%") unless that exact metric and value appear in the data.
- **Fair odds:** Always quote fair probabilities from fair_odds_summary. NEVER recompute \
  or approximate fair probabilities from American odds yourself.
- **Line shopping gaps:** NEVER subtract American odds to compute a "point gap" — American \
  odds are non-linear. Use gap_pct from line_shopping_pairs, which is the implied \
  probability edge (1 - combined_implied_prob). Report as "X% implied edge" not "X-point gap".
"""


def _build_analyze_payload(detection_data: dict) -> dict:
    """Build a compact payload for the AI analyze phase from detection output.

    Includes the pre-computed cross-book analysis plus key detection summaries
    (EV, vig, stale, synthetic book) so the AI has full context to reason over.
    """
    analysis = detection_data.get("analysis", {})

    # Include EV summary (flattened top entries)
    ev_bets: list[dict] = []
    for game_id, markets in detection_data.get("ev_summary", {}).items():
        if not isinstance(markets, dict):
            continue
        for market_type, entries in markets.items():
            if not isinstance(entries, list):
                continue
            for e in entries[:3]:
                ev_bets.append({**e, "game_id": game_id, "market": market_type})
    ev_bets.sort(key=lambda x: abs(x.get("ev_edge", 0)), reverse=True)

    # Vig summary (top books)
    vig_rankings = [
        {k: v for k, v in entry.items() if k != "markets"}
        for entry in (detection_data.get("vig_summary", {}).get("rankings", []) or [])[:10]
    ]

    # Stale lines
    stale = detection_data.get("stale_summary", {})

    # Synthetic perfect book
    synth = detection_data.get("synthetic_perfect_book", {})
    synth_compact = {
        "aggregate": synth.get("aggregate", {}),
        "arb_alerts": synth.get("arb_alerts", []),
    }

    # Arb profit curves (best pairings only)
    arb_curves = detection_data.get("arb_profit_curves", {})

    return {
        "cross_book_analysis": {
            "games_count": analysis.get("games_count", 0),
            "books_count": analysis.get("books_count", 0),
            "efficiency_ranking": analysis.get("efficiency_ranking", []),
            "best_lines": analysis.get("best_lines", []),
            "arbitrage": analysis.get("arbitrage", []),
            "middles": analysis.get("middles", []),
            "outliers": analysis.get("outliers", [])[:10],
            "stale_lines": analysis.get("stale_lines", []),
            "fair_odds_summary": analysis.get("fair_odds_summary", []),
            "summary": analysis.get("summary", ""),
        },
        "ev_bets": ev_bets[:10],
        "vig_rankings": vig_rankings,
        "stale_summary": {
            "count": stale.get("count", 0),
            "stale_lines": stale.get("stale_lines", [])[:10],
        },
        "arb_best_pairings": arb_curves.get("best_pairings", [])[:10],
        "synthetic_perfect_book": synth_compact,
    }


def _parse_analyze_response(raw_text: str) -> dict:
    """Extract the JSON analysis from the AI response, handling <thinking>/<analysis> tags."""
    import re

    # Try to extract from <analysis> tags first
    analysis_match = re.search(r"<analysis>\s*(.*?)\s*</analysis>", raw_text, re.DOTALL)
    if analysis_match:
        json_text = analysis_match.group(1).strip()
    else:
        # Fall back: try to find raw JSON object
        json_text = raw_text.strip()

    # Strip markdown code fences if present
    json_text = re.sub(r"^```(?:json)?\s*", "", json_text)
    json_text = re.sub(r"\s*```$", "", json_text)

    # Extract <thinking> block for logging
    thinking_match = re.search(r"<thinking>\s*(.*?)\s*</thinking>", raw_text, re.DOTALL)
    thinking = thinking_match.group(1).strip() if thinking_match else None

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        # If JSON parsing fails, return the raw text as a fallback
        parsed = {
            "insights": [],
            "market_assessment": {"overall_health": "unknown", "efficiency_score": 0, "key_themes": [], "risk_flags": ["AI response was not valid JSON"]},
            "book_grades": {},
            "top_actions": [],
            "verification_notes": "Failed to parse AI response as JSON",
            "summary": json_text[:500] if json_text else raw_text[:500],
            "_parse_error": True,
        }

    if thinking:
        parsed["_thinking"] = thinking

    return parsed


async def run_analyze_phase(detection_data: dict, run_logger=None, on_chunk=None, on_conversation_event=None, filename=None) -> dict:
    """Phase 2: AI-powered deep analysis with chain-of-thought reasoning.

    Sends the pre-computed cross-book analysis data to an AI model with a
    chain-of-thought prompt. The AI must:
    1. Use MCP tools to verify and enrich the pre-computed data
    2. Reason through the data step by step (<thinking> block)
    3. Cross-verify every claim against the source data and tool results
    4. Self-check via a verification checklist
    5. Produce a structured JSON analysis (<analysis> block)

    The pre-computed analysis dict is preserved as-is for backward compatibility.
    The AI's insights, grades, and actions are layered on top.

    Args:
        filename: The data filename being analyzed (passed to MCP tools).
        on_chunk: Async callback for streaming text deltas.
        on_conversation_event: Async callback for conversation lifecycle events.
            Called with (event_type, data) where event_type is one of:
            - "prompts": system & user prompts are ready
            - "chunk": streaming text delta
            - "tool_call": a tool call was initiated (with tool call data)
            - "tool_result": a tool call returned a result
            - "complete": final result with full response and tool calls
    """
    start = time.time()

    # The detect phase includes the pre-computed analysis
    precomputed_analysis = detection_data.get("analysis", {})

    # Build the payload for the AI
    analyze_payload = _build_analyze_payload(detection_data)

    # Build user prompt with explicit tool-use instruction and filename context
    file_hint = f'The data filename is "{filename}". Pass this as the `filename` parameter to all MCP tool calls.\n\n' if filename else ""
    user_prompt = (
        f"Call MCP tools to verify and enrich the data before writing your analysis. "
        f"Start with `list_events`{f'(filename=\"{filename}\")' if filename else ''}, "
        f"then call 3-5 more tools (get_vig_analysis, find_expected_value_bets, "
        f"get_book_rankings, get_market_entropy, detect_line_outliers). "
        f"TIP: Many tools accept a `top_n` parameter to limit results — use top_n=10 "
        f"for large result sets to keep responses focused.\n\n"
        f"{file_hint}"
        f"Pre-computed analysis data:\n\n"
        + json.dumps(analyze_payload, separators=(",", ":"))
    )

    # Safety cap — keep under ~50KB to stay within 30K token/min rate limits
    if len(user_prompt) > 50000:
        if run_logger:
            run_logger.warning("Analyze payload exceeded 50KB (%d chars), truncating", len(user_prompt))
        user_prompt = user_prompt[:50000]

    if run_logger:
        run_logger.info("Analyze: starting AI call (chain-of-thought enabled)")

    # Emit prompts event so the UI can show them immediately
    if on_conversation_event:
        await on_conversation_event("prompts", {
            "system_prompt": ANALYZE_COT_SYSTEM_PROMPT,
            "user_prompt": user_prompt,
        })

    # Use generous token limit — CoT thinking + tool calls + full analysis
    analyze_max_tokens = 16384

    # Streaming chunk handler — forward deltas to both callbacks
    async def _on_stream_chunk(text_delta):
        if on_chunk:
            await on_chunk(text_delta)
        if on_conversation_event:
            await on_conversation_event("chunk", {"text": text_delta})

    # Tool event handler — forward tool_call and tool_result events to the UI
    async def _on_tool_event(event_type, data):
        if on_conversation_event:
            await on_conversation_event(event_type, data)

    try:
        result = await call_ai_stream(
            ANALYZE_COT_SYSTEM_PROMPT,
            user_prompt,
            on_chunk=_on_stream_chunk,
            on_tool_event=_on_tool_event,
            run_logger=run_logger,
            max_tokens=analyze_max_tokens,
        )

        if run_logger:
            run_logger.info(
                "Analyze: AI complete provider=%s model=%s elapsed=%.2fs tokens_in=%d tokens_out=%d",
                result["provider_name"], result["model"], result["elapsed_seconds"],
                result.get("usage", {}).get("input_tokens", 0),
                result.get("usage", {}).get("output_tokens", 0),
            )

        raw_response_text = result["text"]

        # Parse the AI response (handles <thinking>/<analysis> tags)
        ai_analysis = _parse_analyze_response(raw_response_text)

        # Log thinking block if present
        if run_logger and ai_analysis.get("_thinking"):
            run_logger.info("Analyze: AI chain-of-thought reasoning (%d chars)", len(ai_analysis["_thinking"]))

        # --- Merge: preserve backward-compatible fields from precomputed analysis ---
        merged = {**precomputed_analysis}

        # Layer on AI-generated fields
        merged["ai_insights"] = ai_analysis.get("insights", [])
        merged["market_assessment"] = ai_analysis.get("market_assessment", {})
        merged["book_grades"] = ai_analysis.get("book_grades", {})
        merged["top_actions"] = ai_analysis.get("top_actions", [])
        merged["ai_verification_notes"] = ai_analysis.get("verification_notes", "")

        # If AI produced a better summary, use it (but keep original as fallback)
        if ai_analysis.get("summary") and not ai_analysis.get("_parse_error"):
            merged["ai_summary"] = ai_analysis["summary"]

        elapsed = round(time.time() - start, 2)

        conversation_data = {
            "system_prompt": ANALYZE_COT_SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "assistant_response": raw_response_text,
            "thinking": ai_analysis.get("_thinking"),
            "tool_calls": result.get("tool_calls", []),
        }

        # Emit complete event so UI gets the final parsed conversation
        if on_conversation_event:
            await on_conversation_event("complete", {
                "conversation": conversation_data,
                "ai_meta": {
                    "provider": result["provider_name"],
                    "model": result["model"],
                    "usage": result["usage"],
                    "elapsed_seconds": elapsed,
                },
            })

        return {
            "analysis": merged,
            "ai_meta": {
                "provider": result["provider_name"],
                "model": result["model"],
                "usage": result["usage"],
                "elapsed_seconds": elapsed,
                "chain_of_thought": bool(ai_analysis.get("_thinking")),
                "self_verified": "verification_notes" in ai_analysis,
            },
            "conversation": conversation_data,
        }

    except Exception as exc:
        # Graceful degradation: fall back to precomputed analysis
        elapsed = round(time.time() - start, 2)
        if run_logger:
            run_logger.error("Analyze: AI call failed (%s), falling back to precomputed analysis", exc)

        return {
            "analysis": {
                **precomputed_analysis,
                "ai_insights": [],
                "market_assessment": {},
                "book_grades": {},
                "top_actions": [],
                "ai_verification_notes": f"AI analysis unavailable: {exc}",
            },
            "ai_meta": {
                "provider": "local-fallback",
                "model": "detect-pipeline",
                "usage": {},
                "elapsed_seconds": elapsed,
                "chain_of_thought": False,
                "self_verified": False,
                "error": str(exc),
            },
            "conversation": {
                "system_prompt": ANALYZE_COT_SYSTEM_PROMPT,
                "user_prompt": user_prompt if 'user_prompt' in dir() else None,
                "assistant_response": None,
                "thinking": None,
                "tool_calls": [],
                "error": str(exc),
            },
        }


def _build_line_shopping_pairs(best_lines: list[dict]) -> list[dict]:
    """Pair best_lines entries by (game_id, market) and pre-compute gap metrics.

    Returns a list of paired entries with:
      - best odds per side with book name and implied probability
      - combined_implied_prob (sum of both sides' implied probs)
      - gap_pct (1 - combined implied; positive = arb opportunity)
      - is_arb flag
    This eliminates the need for the brief AI to do American odds arithmetic.
    """
    pairs_by_key: dict[tuple, dict] = {}
    for entry in best_lines:
        gid = entry.get("game_id", "")
        mkt = entry.get("market", "")
        key = (gid, mkt)
        if key not in pairs_by_key:
            pairs_by_key[key] = {"game_id": gid, "market": mkt}
        side = entry.get("side", "")
        odds = entry.get("best_odds")
        if odds is None or not side:
            continue
        pairs_by_key[key][side] = {
            "book": entry.get("best_book", ""),
            "odds": odds,
            "implied_prob": round(_implied_prob(odds), 4),
        }
        if entry.get("line") is not None:
            pairs_by_key[key][side]["line"] = entry["line"]
        # Carry team names
        for tk in ("home_team", "away_team"):
            if entry.get(tk):
                pairs_by_key[key][tk] = entry[tk]

    result = []
    for pair in pairs_by_key.values():
        # Determine the two sides depending on market type
        side_a = pair.get("home") or pair.get("over")
        side_b = pair.get("away") or pair.get("under")
        if not side_a or not side_b:
            continue
        prob_a = side_a.get("implied_prob", 0)
        prob_b = side_b.get("implied_prob", 0)
        combined = round(prob_a + prob_b, 4)
        pair["combined_implied_prob"] = combined
        pair["gap_pct"] = round((1.0 - combined) * 100, 2)
        pair["is_arb"] = combined < 1.0
        # Build human-readable game label
        home = pair.get("home_team", "")
        away = pair.get("away_team", "")
        if home and away:
            pair["game_label"] = f"{away} @ {home}"
        result.append(pair)
    # Sort by gap_pct descending (biggest value opportunities first)
    result.sort(key=lambda x: x.get("gap_pct", 0), reverse=True)
    return result


def _build_brief_payload(detection_data: dict, analysis_data: dict) -> dict:
    """Build a compact, briefing-optimized payload from raw pipeline output.

    Strips bulk arrays (enriched_odds, arb_profit_curves, per-game synthetic
    book detail) and keeps only the top-N items the AI needs for each section
    of the daily market briefing.
    """
    analysis = analysis_data.get("analysis", analysis_data)

    # --- Extract tool-verified EV bets from analyze conversation if available ---
    # These match what the audit agents verify against (Pinnacle no-vig benchmark).
    tool_verified_ev: list[dict] = []
    kelly_data: dict[tuple, dict] = {}
    tool_calls = analysis_data.get("conversation", {}).get("tool_calls", [])
    for tc in (tool_calls or []):
        tc_name = tc.get("name", "")
        if tc_name in ("find_expected_value_bets", "get_best_bets_today", "get_kelly_sizing"):
            try:
                result = tc.get("result", "")
                if isinstance(result, str):
                    result = json.loads(result)
                bets_list = result.get("result", result) if isinstance(result, dict) else result
                if not isinstance(bets_list, list):
                    continue
                for b in bets_list:
                    if not isinstance(b, dict):
                        continue
                    # Collect Kelly data keyed by (sportsbook, game_id, side)
                    key = (b.get("sportsbook", ""), b.get("game_id", ""), b.get("side", ""))
                    if key[0] and (b.get("kelly_fraction") or b.get("quarter_kelly_pct")):
                        kelly_data[key] = {
                            k: v for k, v in b.items()
                            if k in ("kelly_fraction", "quarter_kelly_pct", "ev_edge", "ev_edge_pct")
                        }
                    # Collect tool-verified EV bets
                    if tc_name in ("find_expected_value_bets", "get_best_bets_today") and b.get("ev_edge"):
                        tool_verified_ev.append(b)
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

    # --- Flatten ev_summary (nested dict) into a sorted list as fallback ---
    ev_bets_fallback: list[dict] = []
    for game_id, markets in detection_data.get("ev_summary", {}).items():
        if not isinstance(markets, dict):
            continue
        for market_type, entries in markets.items():
            if not isinstance(entries, list):
                continue
            for e in entries[:3]:  # top 3 per game/market
                ev_bets_fallback.append({**e, "game_id": game_id, "market": market_type})
    ev_bets_fallback.sort(key=lambda x: abs(x.get("ev_edge", 0)), reverse=True)

    # Prefer tool-verified EV data (matches audit verification); fall back to detect-phase
    if tool_verified_ev:
        ev_bets = sorted(tool_verified_ev, key=lambda x: abs(x.get("ev_edge", 0)), reverse=True)
    else:
        ev_bets = ev_bets_fallback

    # --- Merge Kelly sizing data into ev_bets ---
    for bet in ev_bets:
        key = (bet.get("sportsbook", ""), bet.get("game_id", ""), bet.get("side", ""))
        if key in kelly_data:
            for k, v in kelly_data[key].items():
                if k not in bet or bet[k] is None:
                    bet[k] = v

    # --- Synthetic perfect book: aggregate + arb alerts only ---
    synth = detection_data.get("synthetic_perfect_book", {})
    synth_compact = {
        "aggregate": synth.get("aggregate", {}),
        "arb_alerts": synth.get("arb_alerts", []),
    }

    # --- Stale lines: top 10 ---
    stale = detection_data.get("stale_summary", {})
    stale_lines = stale.get("stale_lines", [])[:10]

    # --- Arb profit curves: best pairings only (skip per-game O(n^2) detail) ---
    arb_curves = detection_data.get("arb_profit_curves", {})
    arb_best_pairings = arb_curves.get("best_pairings", [])[:10]

    # --- Pre-computed counts so the AI never counts array elements itself ---
    middles_list = analysis.get("middles", [])
    arbitrage_list = analysis.get("arbitrage", [])
    outliers_list = analysis.get("outliers", [])

    # --- Efficiency ranking with explicit position numbers ---
    raw_efficiency = analysis.get("efficiency_ranking", [])
    efficiency_ranked = [
        {**entry, "rank": i + 1}
        for i, entry in enumerate(raw_efficiency)
    ]

    return {
        "snapshot": {
            "games_count": analysis.get("games_count", 0),
            "books_count": analysis.get("books_count", 0),
            "summary": analysis.get("summary", ""),
        },
        # Pre-computed counts — the AI must quote these verbatim
        "counts": {
            "middles_total": len(middles_list),
            "arbitrage_total": len(arbitrage_list),
            "outliers_total": len(outliers_list),
            "stale_total": stale.get("count", 0),
            "ev_bets_total": len(ev_bets),
        },
        "top_ev_bets": ev_bets[:10],
        "line_shopping_pairs": _build_line_shopping_pairs(analysis.get("best_lines", [])),
        "arbitrage": arbitrage_list,
        "arb_best_pairings": arb_best_pairings,
        "middles": middles_list,
        "stale_lines": stale_lines,
        "stale_count": stale.get("count", 0),
        "outliers": outliers_list[:10],
        "fair_odds_summary": analysis.get("fair_odds_summary", []),
        "efficiency_ranking": efficiency_ranked,
        "vig_summary_top": [
            {k: v for k, v in entry.items() if k != "markets"}
            for entry in (detection_data.get("vig_summary", {}).get("rankings", []) or [])[:8]
        ],
        "synthetic_perfect_book": synth_compact,
        # --- AI-generated insights from the Analyze phase ---
        "ai_summary": analysis.get("ai_summary", ""),
        "ai_insights": analysis.get("ai_insights", []),
        "market_assessment": analysis.get("market_assessment", {}),
        "book_grades": analysis.get("book_grades", {}),
        "book_grades_note": (
            "book_grades reflect OVERALL quality (vig + odds availability + freshness "
            "+ sharpness). A book with higher vig can still rank #1 overall if it excels "
            "in other dimensions. efficiency_ranking ranks by vig ONLY."
        ),
        "top_actions": analysis.get("top_actions", []),
    }


async def run_brief_phase(detection_data: dict, analysis_data: dict, on_chunk=None, run_logger=None) -> dict:
    """Phase 3: AI-powered daily market briefing (readable text).

    When ``on_chunk`` is provided, text fragments are streamed as they arrive
    from the AI provider via ``on_chunk(text_delta)``.
    """
    from datetime import datetime, timezone

    brief_data = _build_brief_payload(detection_data, analysis_data)
    user_prompt = (
        "Generate a daily market briefing from the following data.\n\n"
        "REMINDER: Use ai_insights and top_actions as your PRIMARY source for EV edges, "
        "Kelly sizing, and arbitrage profits (they are tool-verified). Use the 'counts' "
        "object for ALL totals — do NOT count array elements yourself. Use "
        "efficiency_ranking IN ORDER (index 0 = lowest/best vig). Include quarter-Kelly "
        "sizing for every value bet. Every (sportsbook, team, odds) triple must come "
        "from a single data entry. For line shopping, use line_shopping_pairs directly — "
        "each pair has pre-computed gap_pct and combined_implied_prob. NEVER subtract "
        "American odds to compute gaps. Quote fair probabilities ONLY from "
        "fair_odds_summary — never recompute them.\n\n"
        + json.dumps(brief_data, separators=(",", ":"))
    )
    # Safety cap — keep under ~40KB to stay within 30K token/min rate limits
    if len(user_prompt) > 40000:
        logger.warning("Brief payload exceeded 40KB (%d chars), truncating", len(user_prompt))
        user_prompt = user_prompt[:40000]

    if run_logger:
        run_logger.info("Brief: starting AI call (streaming=%s)", bool(on_chunk))

    # Brief is a long-form document (10+ sections) — needs more output room
    # than the default 4096 max_tokens configured on most providers.
    brief_max_tokens = 16384

    if on_chunk:
        # Buffer streaming chunks to strip any preamble before the first "##" heading.
        # This prevents the user from seeing LLM "thinking out loud" text like
        # "Looking at this data, I'll generate..." before the actual briefing.
        _preamble_buffer = []
        _preamble_stripped = [False]  # mutable flag for closure

        async def _filtered_chunk(text_delta):
            if _preamble_stripped[0]:
                # Already past preamble — pass through directly
                await on_chunk(text_delta)
                return

            _preamble_buffer.append(text_delta)
            accumulated = "".join(_preamble_buffer)

            # Look for the first markdown heading (## ) which starts the real briefing
            heading_pos = accumulated.find("## ")
            if heading_pos != -1:
                _preamble_stripped[0] = True
                # Send everything from the first heading onward
                real_content = accumulated[heading_pos:]
                if real_content:
                    await on_chunk(real_content)
                if heading_pos > 0 and run_logger:
                    run_logger.info("Brief: stripped %d chars of preamble before first heading", heading_pos)
            elif len(accumulated) > 2000:
                # Safety valve — if no heading found after 2KB, flush everything
                # (the model may have produced an unusual format)
                _preamble_stripped[0] = True
                await on_chunk(accumulated)
                if run_logger:
                    run_logger.warning("Brief: no heading found after 2KB buffer, flushing all content")

        result = await call_ai_stream(BRIEF_SYSTEM_PROMPT, user_prompt, on_chunk=_filtered_chunk, run_logger=run_logger, max_tokens=brief_max_tokens)
    else:
        result = await call_ai(BRIEF_SYSTEM_PROMPT, user_prompt, run_logger=run_logger, max_tokens=brief_max_tokens)

    if run_logger:
        run_logger.info(
            "Brief: AI complete provider=%s model=%s elapsed=%.2fs tokens_in=%d tokens_out=%d",
            result["provider_name"], result["model"], result["elapsed_seconds"],
            result.get("usage", {}).get("input_tokens", 0),
            result.get("usage", {}).get("output_tokens", 0),
        )

    # Strip any preamble from the final text as well
    final_text = result["text"]
    heading_pos = final_text.find("## ")
    if heading_pos > 0:
        if run_logger:
            run_logger.info("Brief: stripped %d chars of preamble from final text", heading_pos)
        final_text = final_text[heading_pos:]

    return {
        "brief_text": final_text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ai_meta": {
            "provider": result["provider_name"],
            "model": result["model"],
            "usage": result["usage"],
            "elapsed_seconds": result["elapsed_seconds"],
        },
    }


# ---------------------------------------------------------------------------
# Self-healing fix phase — rewrites AI text to resolve audit failures
# ---------------------------------------------------------------------------

FIX_ANALYZE_SYSTEM_PROMPT = """\
You are a precision editor for sports betting analysis. You receive an AI-generated \
betting analysis that FAILED verification audits, along with the specific issues \
found by the audit agents.

Your job is to REWRITE the analysis, correcting EVERY issue listed. You have access \
to MCP tools — you MUST call them to get the correct data for any claim you fix.

RULES:
1. Fix ALL issues listed — every "error" and "warning" severity issue MUST be addressed.
2. Do NOT fabricate data. For every corrected number, call the appropriate MCP tool to \
   get the real value and use that exact value.
3. Preserve the overall structure and tone of the original analysis.
4. If an issue says a +EV opportunity was fabricated, REMOVE it entirely.
5. If an issue says a number is wrong, call the MCP tool, get the correct number, and \
   replace it.
6. If an issue says Kelly sizing is missing, call get_kelly_sizing() and add it.
7. If an issue says stale line caveat is missing, call detect_stale_lines() and add \
   appropriate caveats.
8. After fixing, do a final self-check: re-read your output and verify every number \
   against the MCP tool results you received.

MANDATORY — NO MENTAL MATH — ZERO TOLERANCE
Use arithmetic_add, arithmetic_subtract, arithmetic_multiply, arithmetic_divide, \
arithmetic_evaluate for ALL numerical work. NEVER compute numbers yourself.

OUTPUT: Return ONLY the corrected analysis text. No preamble, no explanation of changes, \
no markdown fences wrapping the whole thing. Just the corrected analysis text exactly as \
it should appear (preserving the same format as the original — if it was markdown, output \
markdown; if it had <analysis> tags, include them).
"""

FIX_BRIEF_SYSTEM_PROMPT = """\
You are a precision editor for sports betting market briefings. You receive an \
AI-generated market briefing that FAILED verification audits, along with the specific \
issues found by the audit agents.

Your job is to REWRITE the briefing, correcting EVERY issue listed. You have access \
to MCP tools — you MUST call them to get the correct data for any claim you fix.

RULES:
1. Fix ALL issues listed — every "error" and "warning" severity issue MUST be addressed.
2. Do NOT fabricate data. For every corrected number, call the appropriate MCP tool to \
   get the real value and use that exact value.
3. Preserve the overall structure, section headings, and tone of the original briefing.
4. The first characters of your output MUST be "## " (a markdown heading). Do NOT add \
   any preamble before the first heading.
5. If an issue says a +EV opportunity was fabricated, REMOVE it entirely.
6. If an issue says a number is wrong, call the MCP tool, get the correct number, and \
   replace it.
7. If an issue says Kelly sizing is missing, call get_kelly_sizing() and add it.
8. If an issue says stale line caveat is missing, call detect_stale_lines() and add \
   appropriate caveats.
9. After fixing, do a final self-check: re-read your output and verify every number \
   against the MCP tool results you received.

MANDATORY — NO MENTAL MATH — ZERO TOLERANCE
Use arithmetic_add, arithmetic_subtract, arithmetic_multiply, arithmetic_divide, \
arithmetic_evaluate for ALL numerical work. NEVER compute numbers yourself.

OUTPUT: Return ONLY the corrected briefing text as clean markdown. The first line MUST \
start with "## ". No preamble, no explanation of changes.
"""


async def run_fix_phase(
    original_text: str,
    audit_result: dict,
    phase_type: str = "analyze",
    run_logger=None,
    on_chunk=None,
) -> dict:
    """Rewrite AI-generated text to fix all audit issues.

    Parameters
    ----------
    original_text : str
        The original AI-generated text that failed audit.
    audit_result : dict
        The verification result dict with agents and issues.
    phase_type : str
        "analyze" or "brief" — selects the appropriate system prompt.
    run_logger : logging.Logger, optional
    on_chunk : async callable, optional
        Streaming callback for text deltas.

    Returns
    -------
    dict
        {fixed_text, ai_meta, conversation}
    """
    start = time.time()

    system_prompt = FIX_ANALYZE_SYSTEM_PROMPT if phase_type == "analyze" else FIX_BRIEF_SYSTEM_PROMPT

    # Build a structured list of issues from all agents
    issues_text = []
    for agent_name, agent_data in audit_result.get("agents", {}).items():
        agent_verdict = agent_data.get("verdict", "unknown")
        agent_issues = agent_data.get("issues", [])
        if agent_issues:
            issues_text.append(f"\n--- {agent_name.upper()} AGENT (verdict: {agent_verdict}) ---")
            for issue in agent_issues:
                severity = issue.get("severity", "unknown")
                claim = issue.get("claim", "")
                finding = issue.get("finding", "")
                issues_text.append(f"  [{severity}] Claim: {claim}")
                issues_text.append(f"           Finding: {finding}")

    issues_block = "\n".join(issues_text) if issues_text else "No specific issues provided."

    user_prompt = (
        f"The following {phase_type} text FAILED verification audit "
        f"(overall verdict: {audit_result.get('overall_verdict', 'unknown')}). "
        f"Fix ALL issues listed below.\n\n"
        f"=== AUDIT ISSUES TO FIX ===\n{issues_block}\n\n"
        f"=== ORIGINAL TEXT TO CORRECT ===\n{original_text}"
    )

    # Safety cap — keep under ~50KB to stay within 30K token/min rate limits
    if len(user_prompt) > 50000:
        if run_logger:
            run_logger.warning("Fix payload exceeded 50KB (%d chars), truncating", len(user_prompt))
        user_prompt = user_prompt[:50000]

    if run_logger:
        run_logger.info("Fix phase (%s): starting AI call to correct %d issues", phase_type, len(issues_text))

    fix_max_tokens = 16384

    if on_chunk:
        if phase_type == "brief":
            # Buffer streaming chunks to strip any preamble before the first "##" heading,
            # just like run_brief_phase does.  Without this the UI receives raw LLM tokens
            # like "percentage:## Market Snapshot" before the heading is detected.
            _fix_preamble_buffer = []
            _fix_preamble_stripped = [False]

            async def _filtered_fix_chunk(text_delta):
                if _fix_preamble_stripped[0]:
                    await on_chunk(text_delta)
                    return

                _fix_preamble_buffer.append(text_delta)
                accumulated = "".join(_fix_preamble_buffer)

                heading_pos = accumulated.find("## ")
                if heading_pos != -1:
                    _fix_preamble_stripped[0] = True
                    real_content = accumulated[heading_pos:]
                    if real_content:
                        await on_chunk(real_content)
                    if heading_pos > 0 and run_logger:
                        run_logger.info("Fix phase: stripped %d chars of streaming preamble before first heading", heading_pos)
                elif len(accumulated) > 2000:
                    _fix_preamble_stripped[0] = True
                    await on_chunk(accumulated)
                    if run_logger:
                        run_logger.warning("Fix phase: no heading found after 2KB buffer, flushing all content")

            chunk_fn = _filtered_fix_chunk
        else:
            chunk_fn = on_chunk

        result = await call_ai_stream(
            system_prompt, user_prompt,
            on_chunk=chunk_fn,
            run_logger=run_logger,
            max_tokens=fix_max_tokens,
        )
    else:
        result = await call_ai_stream(
            system_prompt, user_prompt,
            run_logger=run_logger,
            max_tokens=fix_max_tokens,
        )

    elapsed = round(time.time() - start, 2)

    if run_logger:
        run_logger.info(
            "Fix phase (%s): AI complete provider=%s model=%s elapsed=%.2fs",
            phase_type, result["provider_name"], result["model"], elapsed,
        )

    fixed_text = result["text"]

    # For brief fixes, strip preamble before first heading
    if phase_type == "brief":
        heading_pos = fixed_text.find("## ")
        if heading_pos > 0:
            if run_logger:
                run_logger.info("Fix phase: stripped %d chars of preamble from fixed brief", heading_pos)
            fixed_text = fixed_text[heading_pos:]

    return {
        "fixed_text": fixed_text,
        "ai_meta": {
            "provider": result["provider_name"],
            "model": result["model"],
            "usage": result.get("usage", {}),
            "elapsed_seconds": elapsed,
        },
        "conversation": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "assistant_response": result["text"],
            "tool_calls": result.get("tool_calls", []),
        },
    }
