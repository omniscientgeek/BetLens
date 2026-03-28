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

logger = logging.getLogger(__name__)

# Context variable for per-run logger — set by call_ai/call_ai_stream/call_ai_chat
# so that low-level provider functions (_call_claude_sdk etc.) can log to the run file
_current_run_logger: contextvars.ContextVar[Optional[logging.Logger]] = contextvars.ContextVar(
    "_current_run_logger", default=None
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ai_config.json")


def load_config() -> dict:
    """Load AI configuration from disk."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict) -> None:
    """Persist AI configuration to disk."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def get_enabled_providers(config: Optional[dict] = None) -> list:
    """Return enabled providers sorted by priority (lowest number = highest priority)."""
    if config is None:
        config = load_config()
    providers = [p for p in config["providers"] if p.get("enabled")]
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
    """Call the Anthropic Messages API asynchronously."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    api_key = _get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"No API key found for {provider['id']} (env: {provider.get('api_key_env')})")

    # Log request to per-run log
    rl = _current_run_logger.get(None)
    if rl:
        rl.info("=" * 60)
        rl.info("ANTHROPIC API REQUEST (model=%s)", provider.get("model"))
        rl.info("-" * 40 + " SYSTEM PROMPT " + "-" * 40)
        rl.info("%s", system_prompt)
        rl.info("-" * 40 + " USER PROMPT " + "-" * 40)
        rl.info("%s", user_prompt)
        rl.info("=" * 60)

    client = anthropic.AsyncAnthropic(
        api_key=api_key,
        timeout=config.get("timeout_seconds", 60),
    )

    start = time.time()
    response = await client.messages.create(
        model=provider.get("model", "claude-sonnet-4-20250514"),
        max_tokens=provider.get("max_tokens", 4096),
        temperature=provider.get("temperature", 0.3),
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    elapsed = round(time.time() - start, 2)

    text = response.content[0].text if response.content else ""

    # Log response to per-run log
    if rl:
        rl.info("-" * 40 + " ASSISTANT RESPONSE " + "-" * 40)
        rl.info("%s", text)
        rl.info("-" * 40 + " END RESPONSE " + "-" * 40)

    # Extract any tool_use blocks from the response
    tool_calls = []
    for block in (response.content or []):
        if getattr(block, "type", None) == "tool_use":
            tool_calls.append({
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })

    return {
        "text": text,
        "provider_id": provider["id"],
        "provider_name": provider["name"],
        "model": response.model,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "elapsed_seconds": elapsed,
        "tool_calls": tool_calls,
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


async def _call_claude_sdk(provider: dict, system_prompt: str, user_prompt: str, config: dict, *, use_mcp: bool = False, max_turns: int = 1) -> dict:
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
        rl.info("CLAUDE SDK REQUEST (use_mcp=%s, max_turns=%d)", use_mcp, max_turns)
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
        "max_turns": max_turns,
        "cwd": service_dir,
        "timeout_seconds": timeout_seconds,
        "use_mcp": use_mcp,
    }

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

async def _call_claude_sdk_stream(provider: dict, system_prompt: str, user_prompt: str, config: dict, on_chunk=None) -> dict:
    """Call Claude SDK with streaming — reads stdout line-by-line for chunk events."""
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
        "max_turns": provider.get("max_turns", 25) if use_mcp else 1,
        "cwd": service_dir,
        "timeout_seconds": timeout_seconds,
        "use_mcp": use_mcp,
    }

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
    )

    # Send command and close stdin so the wrapper starts processing
    proc.stdin.write(json.dumps(command_payload).encode("utf-8"))
    proc.stdin.close()

    full_text = ""
    result_data = None

    # Read stdout line-by-line for streaming chunks
    try:
        while True:
            line_bytes = await asyncio.wait_for(
                proc.stdout.readline(),
                timeout=timeout_seconds + 30,
            )
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


async def _call_anthropic_stream(provider: dict, system_prompt: str, user_prompt: str, config: dict, on_chunk=None) -> dict:
    """Call Anthropic Messages API with streaming."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    api_key = _get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"No API key found for {provider['id']}")

    client = anthropic.AsyncAnthropic(api_key=api_key, timeout=config.get("timeout_seconds", 60))

    start = time.time()
    full_text = ""
    input_tokens = 0
    output_tokens = 0

    # Overall timeout for the entire streaming call (generous for large CoT outputs)
    stream_timeout = config.get("timeout_seconds", 180) + 120

    async def _do_stream():
        nonlocal full_text, input_tokens, output_tokens
        async with client.messages.stream(
            model=provider.get("model", "claude-sonnet-4-20250514"),
            max_tokens=provider.get("max_tokens", 4096),
            temperature=provider.get("temperature", 0.3),
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            async for text in stream.text_stream:
                full_text += text
                if on_chunk:
                    await on_chunk(text)

        final = await stream.get_final_message()
        if final and final.usage:
            input_tokens = final.usage.input_tokens
            output_tokens = final.usage.output_tokens
        return final

    try:
        final_message = await asyncio.wait_for(_do_stream(), timeout=stream_timeout)
    except asyncio.TimeoutError:
        raise RuntimeError(f"Anthropic stream timed out after {stream_timeout}s")

    elapsed = round(time.time() - start, 2)

    # Extract any tool_use blocks from the final message
    tool_calls = []
    if final_message and final_message.content:
        for block in final_message.content:
            if getattr(block, "type", None) == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

    return {
        "text": full_text,
        "provider_id": provider["id"],
        "provider_name": provider["name"],
        "model": provider.get("model", "claude-sonnet-4-20250514"),
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "elapsed_seconds": elapsed,
        "tool_calls": tool_calls,
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


async def call_ai_stream(system_prompt: str, user_prompt: str, on_chunk=None, provider_id: Optional[str] = None, run_logger=None, max_tokens: Optional[int] = None) -> dict:
    """
    Send a prompt with streaming support.

    ``on_chunk(text_delta)`` is called for each text fragment as it arrives.
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
                result = await call_fn(provider, system_prompt, user_prompt, config, on_chunk=on_chunk)
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
            except Exception as exc:
                msg = f"{provider['id']} attempt {attempt}: {exc}"
                logger.warning(msg)
                if run_logger:
                    run_logger.warning(msg)
                errors.append(msg)

        if not failover:
            break

    raise RuntimeError(
        "All AI providers failed:\n" + "\n".join(f"  - {e}" for e in errors)
    )


# ---------------------------------------------------------------------------
# Provider-specific async CHAT implementations (multi-turn)
# ---------------------------------------------------------------------------

async def _call_anthropic_chat(provider: dict, system_prompt: str, messages: list[dict], config: dict) -> dict:
    """Call the Anthropic Messages API with a full messages array asynchronously."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    api_key = _get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"No API key found for {provider['id']} (env: {provider.get('api_key_env')})")

    # Log request to per-run log
    rl = _current_run_logger.get(None)
    if rl:
        rl.info("=" * 60)
        rl.info("ANTHROPIC CHAT REQUEST (model=%s, messages=%d)", provider.get("model"), len(messages))
        rl.info("-" * 40 + " SYSTEM PROMPT " + "-" * 40)
        rl.info("%s", system_prompt)
        rl.info("-" * 40 + " MESSAGES " + "-" * 40)
        for m in messages:
            rl.info("[%s]: %s", m["role"].upper(), m["content"][:1000])
        rl.info("=" * 60)

    client = anthropic.AsyncAnthropic(
        api_key=api_key,
        timeout=config.get("timeout_seconds", 60),
    )

    start = time.time()
    response = await client.messages.create(
        model=provider.get("model", "claude-sonnet-4-20250514"),
        max_tokens=provider.get("max_tokens", 64000),
        temperature=provider.get("temperature", 0.3),
        system=system_prompt,
        messages=messages,
    )
    elapsed = round(time.time() - start, 2)

    text = response.content[0].text if response.content else ""

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
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "elapsed_seconds": elapsed,
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
        rl.info("CLAUDE SDK CHAT: %d messages, MCP=True, max_turns=10", len(messages))
        for i, msg in enumerate(messages):
            rl.info("  Message[%d] %s: %s", i, msg["role"].upper(), msg["content"][:500])

    # Concatenate all messages into a single user prompt for the request-response SDK
    parts = []
    for msg in messages:
        role_label = msg["role"].upper()
        parts.append(f"{role_label}: {msg['content']}")
    combined_prompt = "\n\n".join(parts)

    return await _call_claude_sdk(provider, system_prompt, combined_prompt, config, use_mcp=True, max_turns=10)


# Map provider types to their async chat functions
_CHAT_CALL_MAP = {
    "anthropic": _call_anthropic_chat,
    "openai": _call_openai_chat,
    "openai_compatible": _call_openai_chat,
    "claude_sdk": _call_claude_sdk_chat,
}


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
            except Exception as exc:
                msg = f"{provider['id']} attempt {attempt}: {exc}"
                logger.warning(msg)
                if run_logger:
                    run_logger.warning(msg)
                errors.append(msg)

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
            except Exception as exc:
                msg = f"{provider['id']} attempt {attempt}: {exc}"
                logger.warning(msg)
                if run_logger:
                    run_logger.warning(msg)
                errors.append(msg)

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

When pipeline context is provided, reference specific data points, game IDs, sportsbooks, \
and numbers in your answers. Be precise and quantitative. If the user asks about something \
not in the data, say so clearly.

MANDATORY — NO MENTAL MATH — ZERO TOLERANCE
You must NEVER perform arithmetic, math, or statistical calculations yourself — \
not even simple ones like adding two numbers or computing a percentage. Every \
number you derive MUST come from calling an MCP arithmetic tool. If you produce \
a calculated number without a tool call it is assumed WRONG and will be flagged \
as an error. Do NOT estimate, round in your head, or "quickly" compute anything.

Arithmetic MCP tools (you MUST use these for ALL math):
- arithmetic_add(a, b) — addition
- arithmetic_subtract(a, b) — subtraction (a - b)
- arithmetic_multiply(a, b) — multiplication (a * b)
- arithmetic_divide(a, b) — division (a / b)
- arithmetic_modulo(a, b) — remainder (a % b)
- arithmetic_evaluate(expression) — multi-step, e.g. "(100 * 0.25) + 50"

This applies to ALL numerical work: payouts, edges, bankroll impacts, ROI, vig \
differences, profit/loss, implied probabilities, EV percentages, Kelly fractions, \
averages, odds differences, percentage changes — ANY derivation of a number.

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

MANDATORY — NO MENTAL MATH — ZERO TOLERANCE
You must NEVER perform arithmetic, math, or statistical calculations yourself — \
not even simple ones. Every derived number in your output MUST come from an MCP \
arithmetic tool call. If you produce a calculated number without a tool call it \
is assumed WRONG. Do NOT estimate, round in your head, or "quickly" compute anything.

Arithmetic MCP tools (you MUST use these for ALL math):
- arithmetic_add(a, b), arithmetic_subtract(a, b), arithmetic_multiply(a, b)
- arithmetic_divide(a, b), arithmetic_modulo(a, b)
- arithmetic_evaluate(expression) — multi-step, e.g. "(100 * 0.25) + 50"

This applies to: vig percentages, edge sizes, implied probabilities, profit \
margins, combined implied probabilities, EV edges, averages — every number.

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
cross-sportsbook analysis data (efficiency rankings, best lines, arbitrage, \
middles, outliers, stale lines, fair odds) and your job is to produce a deep, \
expert-level analytical interpretation.

## MANDATORY: THINK STEP BY STEP

You MUST use explicit chain-of-thought reasoning. Structure your response as:

<thinking>
Work through your analysis step by step here. For EVERY claim you make:
1. Identify the specific data points you are referencing
2. State what they mean in context
3. Reason about implications, cross-referencing multiple data sources
4. Consider alternative explanations or caveats
5. Only then form your conclusion

Be thorough. Check your reasoning as you go.
</thinking>

<analysis>
Your final structured JSON analysis goes here (no markdown fences).
</analysis>

## MANDATORY: SELF-VERIFICATION CHECKLIST

Before writing your <analysis> block, you MUST complete this checklist inside \
your <thinking> block:

[ ] CROSS-CHECK: Every arbitrage opportunity — verify both legs exist in the \
    source data with the exact odds cited. Recompute combined implied probability.
[ ] CROSS-CHECK: Every "best line" — confirm no other book in the data offers \
    better odds for that side.
[ ] CROSS-CHECK: Every outlier — verify it actually deviates from consensus by \
    the amount claimed.
[ ] CROSS-CHECK: Every middle — verify the line gap exists between the cited books.
[ ] CROSS-CHECK: Efficiency rankings — verify the ordering matches vig data.
[ ] SANITY CHECK: No fabricated sportsbook names — only use books present in data.
[ ] SANITY CHECK: No fabricated games — only reference games present in data.
[ ] SANITY CHECK: Fair odds probabilities sum to ~100% per market (before margin).
[ ] FINAL REVIEW: Re-read your analysis and flag anything you are less than 80% \
    confident about. Mark uncertain items with "confidence": "low".

If ANY check fails, fix it before writing the <analysis> block.

## Analysis Sections

Your <analysis> JSON must include:

{
  "insights": [
    {
      "type": "arbitrage|middle|outlier|value|efficiency|stale|market_trend",
      "severity": "critical|high|medium|low|info",
      "title": "Short descriptive title",
      "description": "Detailed explanation with specific numbers",
      "games": ["game_id1"],
      "books": ["book1", "book2"],
      "confidence": "high|medium|low",
      "reasoning": "Why this matters / how you verified it"
    }
  ],
  "market_assessment": {
    "overall_health": "healthy|volatile|thin|stale|anomalous",
    "efficiency_score": 0-100,
    "key_themes": ["theme1", "theme2"],
    "risk_flags": ["flag1"]
  },
  "book_grades": {
    "book_name": {
      "grade": "A|B|C|D|F",
      "avg_vig": 3.5,
      "strengths": ["..."],
      "weaknesses": ["..."]
    }
  },
  "top_actions": [
    {
      "priority": 1,
      "action": "What to do",
      "reasoning": "Why",
      "urgency": "immediate|today|monitor"
    }
  ],
  "verification_notes": "Summary of self-check results and any caveats",
  "summary": "2-3 sentence executive summary of the most important findings"
}

## Rules
- Reference ONLY data that is actually present in the input. Never fabricate.
- Every number you cite must trace back to the source data.
- If data is insufficient for a section, say so explicitly with confidence: "low".
- Prioritize actionable insights over exhaustive listing.
- Grade sportsbooks relative to each other, not absolute standards.
"""

BRIEF_SYSTEM_PROMPT = """\
You are a senior sports-betting market analyst AI. You receive raw odds data and \
cross-sportsbook analysis results. Your job is to produce a clear, accurate daily \
market briefing that a human analyst could review and act on.

Write the briefing as clean, readable text using markdown formatting. Structure it \
with the following sections using ## headings:

## Market Snapshot
A 2-3 sentence executive overview of today's market conditions. State the total number \
of games and sportsbooks covered, and characterize overall market health (normal, \
volatile, thin, or stale data concerns).

## Top Value Bets
List up to 5 of the best value bets you can identify, ranked by confidence. For each, \
state the game, market (spread/moneyline/total), the specific side and line, which \
sportsbook has the best odds, the odds themselves, and why this is a value play \
(reference fair odds, vig, or consensus). Rate confidence as HIGH, MEDIUM, or LOW \
with a brief justification.

## Best Line Shopping
For key games, show which sportsbook offers the best odds on each side. Highlight \
cases where the best line is meaningfully better than the next-best (>5 cents in \
American odds). This tells the reader where to place each bet for maximum value.

## Arbitrage Opportunities
List any genuine arbitrage opportunities across sportsbooks. For each, specify both \
legs (side, book, odds), the combined implied probability, and the estimated profit \
percentage. If none exist, say so clearly.

## Middle Opportunities
List any middle opportunities where spread or total lines differ across sportsbooks \
enough to allow winning both sides. For each, state the game, both legs (side, line, \
book, odds), the gap size, and the range of outcomes where both bets win. If none \
exist, say so clearly.

## Stale & Suspect Lines
Flag any lines that appear significantly outdated, are outliers from consensus, or \
carry unusually high vig. Name the sportsbook, game, and explain the concern.

## Fair Odds & Expected Value
Summarize the consensus no-vig fair probabilities for each game. Highlight any \
sportsbook lines that offer positive expected value (+EV) against the consensus \
fair odds. Include the EV edge percentage and the relevant fair probability.

## Sportsbook Rankings
Rank all sportsbooks in the data by average vig percentage. For each, note whether \
they are sharp (low vig, < 3%), fair, or to be avoided, with a brief reason.

## Market Movements
Note any notable cross-book discrepancies, line movements, or anomalies worth watching.

## Analyst Notes
2-4 sentences of overall takeaways, things to watch, or caveats a human reviewer \
should keep in mind before acting on this briefing.

MANDATORY — NO MENTAL MATH — ZERO TOLERANCE
You must NEVER perform arithmetic, math, or statistical calculations yourself — \
not even simple ones like adding two numbers or computing a percentage. Every \
number you derive MUST come from calling an MCP arithmetic tool. If you produce \
a calculated number without a tool call it is assumed WRONG and will be flagged \
as an error. Do NOT estimate, round in your head, or "quickly" compute anything.

Arithmetic MCP tools (you MUST use these for ALL math):
- arithmetic_add(a, b), arithmetic_subtract(a, b), arithmetic_multiply(a, b)
- arithmetic_divide(a, b), arithmetic_modulo(a, b)
- arithmetic_evaluate(expression) — multi-step, e.g. "(100 * 0.25) + 50"

This applies to: profit margins, vig percentages, EV edges, implied probabilities, \
payout amounts, combined implied probabilities, Kelly fractions, ROI, averages, \
odds differences, percentage changes — ANY derivation of a number in the briefing.

AI Analysis Summary:
The data may include fields from a prior AI Analyze step: "ai_summary", "ai_insights", \
"market_assessment", "book_grades", and "top_actions". When these are present, use them \
as the primary basis for your briefing — they contain pre-verified chain-of-thought \
conclusions. Summarize and reformat their findings into the briefing sections above \
rather than re-deriving conclusions from the raw numbers. Where the AI analysis is \
missing or empty, fall back to the raw detection data as before.

Guidelines:
- Be precise with numbers. Always cite specific odds, lines, and books.
- Only include value bets where a quantifiable edge exists.
- Only flag genuine arbitrage — do not fabricate opportunities.
- If data is insufficient for any section, state that clearly rather than guessing.
- Write for a professional audience. Be concise but thorough.
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
            "outliers": analysis.get("outliers", [])[:15],
            "stale_lines": analysis.get("stale_lines", []),
            "fair_odds_summary": analysis.get("fair_odds_summary", []),
            "summary": analysis.get("summary", ""),
        },
        "ev_bets": ev_bets[:15],
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


async def run_analyze_phase(detection_data: dict, run_logger=None, on_chunk=None, on_conversation_event=None) -> dict:
    """Phase 2: AI-powered deep analysis with chain-of-thought reasoning.

    Sends the pre-computed cross-book analysis data to an AI model with a
    chain-of-thought prompt. The AI must:
    1. Reason through the data step by step (<thinking> block)
    2. Cross-verify every claim against the source data
    3. Self-check via a verification checklist
    4. Produce a structured JSON analysis (<analysis> block)

    The pre-computed analysis dict is preserved as-is for backward compatibility.
    The AI's insights, grades, and actions are layered on top.

    Args:
        on_chunk: Async callback for streaming text deltas.
        on_conversation_event: Async callback for conversation lifecycle events.
            Called with (event_type, data) where event_type is one of:
            - "prompts": system & user prompts are ready
            - "chunk": streaming text delta
            - "complete": final result with full response and tool calls
    """
    start = time.time()

    # The detect phase includes the pre-computed analysis
    precomputed_analysis = detection_data.get("analysis", {})

    # Build the payload for the AI
    analyze_payload = _build_analyze_payload(detection_data)
    user_prompt = (
        "Analyze the following betting market data. Think step by step, verify your work, "
        "and produce your structured analysis.\n\n"
        + json.dumps(analyze_payload, separators=(",", ":"))
    )

    # Safety cap
    if len(user_prompt) > 80000:
        if run_logger:
            run_logger.warning("Analyze payload exceeded 80KB (%d chars), truncating", len(user_prompt))
        user_prompt = user_prompt[:80000]

    if run_logger:
        run_logger.info("Analyze: starting AI call (chain-of-thought enabled)")

    # Emit prompts event so the UI can show them immediately
    if on_conversation_event:
        await on_conversation_event("prompts", {
            "system_prompt": ANALYZE_COT_SYSTEM_PROMPT,
            "user_prompt": user_prompt,
        })

    # Use generous token limit — CoT thinking + full analysis
    analyze_max_tokens = 16384

    # Streaming chunk handler — forward deltas to both callbacks
    async def _on_stream_chunk(text_delta):
        if on_chunk:
            await on_chunk(text_delta)
        if on_conversation_event:
            await on_conversation_event("chunk", {"text": text_delta})

    try:
        result = await call_ai_stream(
            ANALYZE_COT_SYSTEM_PROMPT,
            user_prompt,
            on_chunk=_on_stream_chunk,
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


def _build_brief_payload(detection_data: dict, analysis_data: dict) -> dict:
    """Build a compact, briefing-optimized payload from raw pipeline output.

    Strips bulk arrays (enriched_odds, arb_profit_curves, per-game synthetic
    book detail) and keeps only the top-N items the AI needs for each section
    of the daily market briefing.
    """
    analysis = analysis_data.get("analysis", analysis_data)

    # --- Flatten ev_summary (nested dict) into a sorted top-10 list ---
    ev_bets: list[dict] = []
    for game_id, markets in detection_data.get("ev_summary", {}).items():
        if not isinstance(markets, dict):
            continue
        for market_type, entries in markets.items():
            if not isinstance(entries, list):
                continue
            for e in entries[:3]:  # top 3 per game/market
                ev_bets.append({**e, "game_id": game_id, "market": market_type})
    ev_bets.sort(key=lambda x: abs(x.get("ev_edge", 0)), reverse=True)

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

    return {
        "snapshot": {
            "games_count": analysis.get("games_count", 0),
            "books_count": analysis.get("books_count", 0),
            "summary": analysis.get("summary", ""),
        },
        "top_ev_bets": ev_bets[:10],
        "best_lines": analysis.get("best_lines", []),
        "arbitrage": analysis.get("arbitrage", []),
        "arb_best_pairings": arb_best_pairings,
        "middles": analysis.get("middles", []),
        "stale_lines": stale_lines,
        "stale_count": stale.get("count", 0),
        "outliers": analysis.get("outliers", [])[:10],
        "fair_odds_summary": analysis.get("fair_odds_summary", []),
        "efficiency_ranking": analysis.get("efficiency_ranking", []),
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
        + json.dumps(brief_data, separators=(",", ":"))
    )
    # Safety cap — should not trigger with properly summarized data
    if len(user_prompt) > 60000:
        logger.warning("Brief payload exceeded 60KB (%d chars), truncating", len(user_prompt))
        user_prompt = user_prompt[:60000]

    if run_logger:
        run_logger.info("Brief: starting AI call (streaming=%s)", bool(on_chunk))

    # Brief is a long-form document (10+ sections) — needs more output room
    # than the default 4096 max_tokens configured on most providers.
    brief_max_tokens = 16384

    if on_chunk:
        result = await call_ai_stream(BRIEF_SYSTEM_PROMPT, user_prompt, on_chunk=on_chunk, run_logger=run_logger, max_tokens=brief_max_tokens)
    else:
        result = await call_ai(BRIEF_SYSTEM_PROMPT, user_prompt, run_logger=run_logger, max_tokens=brief_max_tokens)

    if run_logger:
        run_logger.info(
            "Brief: AI complete provider=%s model=%s elapsed=%.2fs tokens_in=%d tokens_out=%d",
            result["provider_name"], result["model"], result["elapsed_seconds"],
            result.get("usage", {}).get("input_tokens", 0),
            result.get("usage", {}).get("output_tokens", 0),
        )

    return {
        "brief_text": result["text"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ai_meta": {
            "provider": result["provider_name"],
            "model": result["model"],
            "usage": result["usage"],
            "elapsed_seconds": result["elapsed_seconds"],
        },
    }
