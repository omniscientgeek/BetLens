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

    command_payload = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "model": provider.get("model", "claude-sonnet-4-20250514"),
        "max_turns": 1,
        "cwd": service_dir,
        "timeout_seconds": timeout_seconds,
        "use_mcp": False,
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

    elapsed = round(time.time() - start, 2)
    final = await stream.get_final_message()
    if final and final.usage:
        input_tokens = final.usage.input_tokens
        output_tokens = final.usage.output_tokens

    return {
        "text": full_text,
        "provider_id": provider["id"],
        "provider_name": provider["name"],
        "model": final.model if final else provider.get("model"),
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "elapsed_seconds": elapsed,
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


async def call_ai_stream(system_prompt: str, user_prompt: str, on_chunk=None, provider_id: Optional[str] = None, run_logger=None) -> dict:
    """
    Send a prompt with streaming support.

    ``on_chunk(text_delta)`` is called for each text fragment as it arrives.
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

async def call_ai(system_prompt: str, user_prompt: str, provider_id: Optional[str] = None, run_logger=None) -> dict:
    """
    Send a prompt to an AI provider and return the response.

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
        # Use a specific provider
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
3. **Outlier Lines** — Lines that deviate significantly from the consensus.
4. **Market Efficiency** — How tight/wide each book's vig is relative to others.
5. **Stale Lines** — Books whose lines haven't updated recently.

Return ONLY valid JSON (no markdown fences). Use this structure:
{
  "best_lines": [...],
  "arbitrage": [...],
  "outliers": [...],
  "efficiency_ranking": [...],
  "stale_lines": [...],
  "summary": "One paragraph executive summary"
}
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

## Arbitrage Opportunities
List any genuine arbitrage or middle opportunities across sportsbooks. For each, \
specify both legs (side, book, odds) and the estimated profit percentage. If none \
exist, say so clearly.

## Stale & Suspect Lines
Flag any lines that appear significantly outdated, are outliers from consensus, or \
carry unusually high vig. Name the sportsbook, game, and explain the concern.

## Sportsbook Rankings
Rank all sportsbooks in the data by average vig percentage. For each, note whether \
they are sharp (low vig, < 3%), fair, or to be avoided, with a brief reason.

## Market Movements
Note any notable cross-book discrepancies, line movements, or anomalies worth watching.

## Analyst Notes
2-4 sentences of overall takeaways, things to watch, or caveats a human reviewer \
should keep in mind before acting on this briefing.

Guidelines:
- Be precise with numbers. Always cite specific odds, lines, and books.
- Only include value bets where a quantifiable edge exists.
- Only flag genuine arbitrage — do not fabricate opportunities.
- If data is insufficient for any section, state that clearly rather than guessing.
- Write for a professional audience. Be concise but thorough.
"""


async def run_analyze_phase(detection_data: dict, run_logger=None) -> dict:
    """Phase 2: Local cross-sportsbook analysis (no AI call).

    Performs comprehensive cross-sportsbook calculations:
    1. No-Vig / Fair Odds (already in enriched data from detect phase)
    2. Vig / Efficiency Ranking per sportsbook
    3. Best Line Shopping — best odds for each game+market+side
    4. Arbitrage Detection — combined implied prob < 100%
    5. Middles Detection — spread/total line gaps allowing both-side wins
    6. Outlier / Anomaly Detection — lines deviating from consensus
    7. Stale Lines Detection — outdated last_updated timestamps
    """
    from odds_math import implied_probability
    from datetime import datetime, timezone as tz

    start = time.time()

    # Pull the enriched odds and vig summary from detection
    odds = detection_data.get("enriched_odds", detection_data.get("odds", []))
    vig_summary = detection_data.get("vig_summary", {})

    # ── Group odds by game ──────────────────────────────────────────────
    games: dict[str, dict] = {}
    all_books: set[str] = set()
    for row in odds:
        gid = row.get("game_id", "")
        book = row.get("sportsbook", "")
        all_books.add(book)
        if gid not in games:
            games[gid] = {
                "home": row.get("home_team", ""),
                "away": row.get("away_team", ""),
                "rows": [],
            }
        games[gid]["rows"].append(row)

    # ── 1. Vig / Efficiency Ranking (per sportsbook) ───────────────────
    book_vigs: dict[str, list[float]] = {}
    for row in odds:
        book = row.get("sportsbook", "")
        markets = row.get("markets", {})
        for mkt in markets.values():
            v = mkt.get("vig")
            if v is not None:
                book_vigs.setdefault(book, []).append(v)

    efficiency = []
    for book, vigs in book_vigs.items():
        avg = sum(vigs) / len(vigs) if vigs else 0
        efficiency.append({
            "book": book,
            "avg_vig": round(avg, 6),
            "avg_vig_pct": f"{round(avg * 100, 2)}%",
            "markets_counted": len(vigs),
        })
    efficiency.sort(key=lambda x: x["avg_vig"])

    # ── 2. Best Line Shopping ──────────────────────────────────────────
    best_lines = []
    for gid, gdata in games.items():
        rows = gdata["rows"]
        # Spread: home side & away side
        spread_rows = [(r["sportsbook"], r["markets"]["spread"]) for r in rows if "spread" in r.get("markets", {})]
        if spread_rows:
            best_home = max(spread_rows, key=lambda x: x[1]["home_odds"])
            best_away = max(spread_rows, key=lambda x: x[1]["away_odds"])
            best_lines.append({
                "game_id": gid, "market": "spread", "side": "home",
                "line": best_home[1].get("home_line"),
                "best_odds": best_home[1]["home_odds"],
                "best_book": best_home[0],
                "home_team": gdata["home"],
            })
            best_lines.append({
                "game_id": gid, "market": "spread", "side": "away",
                "line": best_away[1].get("away_line"),
                "best_odds": best_away[1]["away_odds"],
                "best_book": best_away[0],
                "away_team": gdata["away"],
            })

        # Moneyline: home & away
        ml_rows = [(r["sportsbook"], r["markets"]["moneyline"]) for r in rows if "moneyline" in r.get("markets", {})]
        if ml_rows:
            best_home = max(ml_rows, key=lambda x: x[1]["home_odds"])
            best_away = max(ml_rows, key=lambda x: x[1]["away_odds"])
            best_lines.append({
                "game_id": gid, "market": "moneyline", "side": "home",
                "best_odds": best_home[1]["home_odds"],
                "best_book": best_home[0],
                "home_team": gdata["home"],
            })
            best_lines.append({
                "game_id": gid, "market": "moneyline", "side": "away",
                "best_odds": best_away[1]["away_odds"],
                "best_book": best_away[0],
                "away_team": gdata["away"],
            })

        # Totals: over & under
        total_rows = [(r["sportsbook"], r["markets"]["total"]) for r in rows if "total" in r.get("markets", {})]
        if total_rows:
            best_over = max(total_rows, key=lambda x: x[1]["over_odds"])
            best_under = max(total_rows, key=lambda x: x[1]["under_odds"])
            best_lines.append({
                "game_id": gid, "market": "total", "side": "over",
                "line": best_over[1].get("line"),
                "best_odds": best_over[1]["over_odds"],
                "best_book": best_over[0],
            })
            best_lines.append({
                "game_id": gid, "market": "total", "side": "under",
                "line": best_under[1].get("line"),
                "best_odds": best_under[1]["under_odds"],
                "best_book": best_under[0],
            })

    # ── 3. Arbitrage Detection ─────────────────────────────────────────
    arbitrage = []
    for gid, gdata in games.items():
        rows = gdata["rows"]

        # Check spread arb
        spread_rows = [(r["sportsbook"], r["markets"]["spread"]) for r in rows if "spread" in r.get("markets", {})]
        if len(spread_rows) >= 2:
            best_home = max(spread_rows, key=lambda x: x[1]["home_odds"])
            best_away = max(spread_rows, key=lambda x: x[1]["away_odds"])
            # Only valid arb if lines are the same (or both sides at same spread number)
            prob_home = implied_probability(best_home[1]["home_odds"])
            prob_away = implied_probability(best_away[1]["away_odds"])
            combined = prob_home + prob_away
            if combined < 1.0:
                profit_pct = round((1.0 - combined) * 100, 3)
                arbitrage.append({
                    "game_id": gid, "market": "spread",
                    "home_team": gdata["home"], "away_team": gdata["away"],
                    "leg_1": {"side": "home", "line": best_home[1].get("home_line"),
                              "odds": best_home[1]["home_odds"], "book": best_home[0],
                              "implied_prob": round(prob_home, 4)},
                    "leg_2": {"side": "away", "line": best_away[1].get("away_line"),
                              "odds": best_away[1]["away_odds"], "book": best_away[0],
                              "implied_prob": round(prob_away, 4)},
                    "combined_implied": round(combined, 4),
                    "profit_pct": profit_pct,
                })

        # Check moneyline arb
        ml_rows = [(r["sportsbook"], r["markets"]["moneyline"]) for r in rows if "moneyline" in r.get("markets", {})]
        if len(ml_rows) >= 2:
            best_home = max(ml_rows, key=lambda x: x[1]["home_odds"])
            best_away = max(ml_rows, key=lambda x: x[1]["away_odds"])
            prob_home = implied_probability(best_home[1]["home_odds"])
            prob_away = implied_probability(best_away[1]["away_odds"])
            combined = prob_home + prob_away
            if combined < 1.0:
                profit_pct = round((1.0 - combined) * 100, 3)
                arbitrage.append({
                    "game_id": gid, "market": "moneyline",
                    "home_team": gdata["home"], "away_team": gdata["away"],
                    "leg_1": {"side": "home", "odds": best_home[1]["home_odds"],
                              "book": best_home[0], "implied_prob": round(prob_home, 4)},
                    "leg_2": {"side": "away", "odds": best_away[1]["away_odds"],
                              "book": best_away[0], "implied_prob": round(prob_away, 4)},
                    "combined_implied": round(combined, 4),
                    "profit_pct": profit_pct,
                })

        # Check total arb
        total_rows = [(r["sportsbook"], r["markets"]["total"]) for r in rows if "total" in r.get("markets", {})]
        if len(total_rows) >= 2:
            best_over = max(total_rows, key=lambda x: x[1]["over_odds"])
            best_under = max(total_rows, key=lambda x: x[1]["under_odds"])
            prob_over = implied_probability(best_over[1]["over_odds"])
            prob_under = implied_probability(best_under[1]["under_odds"])
            combined = prob_over + prob_under
            if combined < 1.0:
                profit_pct = round((1.0 - combined) * 100, 3)
                arbitrage.append({
                    "game_id": gid, "market": "total",
                    "home_team": gdata["home"], "away_team": gdata["away"],
                    "leg_1": {"side": "over", "line": best_over[1].get("line"),
                              "odds": best_over[1]["over_odds"], "book": best_over[0],
                              "implied_prob": round(prob_over, 4)},
                    "leg_2": {"side": "under", "line": best_under[1].get("line"),
                              "odds": best_under[1]["under_odds"], "book": best_under[0],
                              "implied_prob": round(prob_under, 4)},
                    "combined_implied": round(combined, 4),
                    "profit_pct": profit_pct,
                })

    arbitrage.sort(key=lambda x: x["profit_pct"], reverse=True)

    # ── 4. Middles Detection ───────────────────────────────────────────
    middles = []
    for gid, gdata in games.items():
        rows = gdata["rows"]

        # Spread middles: find pairs where one book's home_line differs enough
        spread_rows = [(r["sportsbook"], r["markets"]["spread"]) for r in rows if "spread" in r.get("markets", {})]
        if len(spread_rows) >= 2:
            for i in range(len(spread_rows)):
                for j in range(i + 1, len(spread_rows)):
                    book_a, mkt_a = spread_rows[i]
                    book_b, mkt_b = spread_rows[j]
                    # Middle exists when: book_a home_line < book_b away_line
                    # i.e., line gap allows winning both
                    home_line_a = mkt_a.get("home_line", 0)
                    away_line_b = mkt_b.get("away_line", 0)
                    home_line_b = mkt_b.get("home_line", 0)
                    away_line_a = mkt_a.get("away_line", 0)

                    # Check: bet home at book_a (line = home_line_a) and away at book_b (line = away_line_b)
                    # Middle if |home_line_a| < away_line_b (gap exists)
                    gap = away_line_b - abs(home_line_a) if home_line_a < 0 else away_line_b + home_line_a
                    # Simpler: middle exists when the absolute spread differs
                    if abs(home_line_a) != abs(home_line_b):
                        spread_gap = abs(home_line_a) - abs(home_line_b)
                        if abs(spread_gap) >= 1.0:
                            # Determine direction: bet the smaller spread side
                            if abs(home_line_a) < abs(home_line_b):
                                middles.append({
                                    "game_id": gid, "market": "spread",
                                    "home_team": gdata["home"], "away_team": gdata["away"],
                                    "leg_1": {"side": "home", "line": home_line_a,
                                              "odds": mkt_a["home_odds"], "book": book_a},
                                    "leg_2": {"side": "away", "line": away_line_b,
                                              "odds": mkt_b["away_odds"], "book": book_b},
                                    "middle_gap": abs(spread_gap),
                                    "middle_range": f"Result lands between {home_line_a} and {home_line_b}",
                                })
                            else:
                                middles.append({
                                    "game_id": gid, "market": "spread",
                                    "home_team": gdata["home"], "away_team": gdata["away"],
                                    "leg_1": {"side": "home", "line": home_line_b,
                                              "odds": mkt_b["home_odds"], "book": book_b},
                                    "leg_2": {"side": "away", "line": away_line_a,
                                              "odds": mkt_a["away_odds"], "book": book_a},
                                    "middle_gap": abs(spread_gap),
                                    "middle_range": f"Result lands between {home_line_b} and {home_line_a}",
                                })

        # Total middles: different O/U lines across books
        total_rows = [(r["sportsbook"], r["markets"]["total"]) for r in rows if "total" in r.get("markets", {})]
        if len(total_rows) >= 2:
            for i in range(len(total_rows)):
                for j in range(i + 1, len(total_rows)):
                    book_a, mkt_a = total_rows[i]
                    book_b, mkt_b = total_rows[j]
                    line_a = mkt_a.get("line", 0)
                    line_b = mkt_b.get("line", 0)
                    if line_a != line_b:
                        gap = abs(line_a - line_b)
                        if gap >= 1.0:
                            # Bet over on lower line, under on higher line
                            if line_a < line_b:
                                middles.append({
                                    "game_id": gid, "market": "total",
                                    "home_team": gdata["home"], "away_team": gdata["away"],
                                    "leg_1": {"side": "over", "line": line_a,
                                              "odds": mkt_a["over_odds"], "book": book_a},
                                    "leg_2": {"side": "under", "line": line_b,
                                              "odds": mkt_b["under_odds"], "book": book_b},
                                    "middle_gap": gap,
                                    "middle_range": f"Total lands between {line_a} and {line_b}",
                                })
                            else:
                                middles.append({
                                    "game_id": gid, "market": "total",
                                    "home_team": gdata["home"], "away_team": gdata["away"],
                                    "leg_1": {"side": "over", "line": line_b,
                                              "odds": mkt_b["over_odds"], "book": book_b},
                                    "leg_2": {"side": "under", "line": line_a,
                                              "odds": mkt_a["under_odds"], "book": book_a},
                                    "middle_gap": gap,
                                    "middle_range": f"Total lands between {line_b} and {line_a}",
                                })

    # Deduplicate middles (same game+market pairs appear twice in nested loop)
    seen_middles = set()
    unique_middles = []
    for m in middles:
        key = (m["game_id"], m["market"], m["leg_1"]["book"], m["leg_2"]["book"])
        rev_key = (m["game_id"], m["market"], m["leg_2"]["book"], m["leg_1"]["book"])
        if key not in seen_middles and rev_key not in seen_middles:
            seen_middles.add(key)
            unique_middles.append(m)
    middles = sorted(unique_middles, key=lambda x: x["middle_gap"], reverse=True)

    # ── 5. Outlier / Anomaly Detection ─────────────────────────────────
    outliers = []
    OUTLIER_THRESHOLD = 15  # American odds points deviation from consensus

    for gid, gdata in games.items():
        rows = gdata["rows"]
        # Compute consensus (average) for each market's odds
        for market_name in ("spread", "moneyline", "total"):
            market_rows = [(r["sportsbook"], r["markets"][market_name])
                           for r in rows if market_name in r.get("markets", {})]
            if len(market_rows) < 3:
                continue

            if market_name in ("spread", "moneyline"):
                odds_keys = [("home_odds", "home"), ("away_odds", "away")]
            else:
                odds_keys = [("over_odds", "over"), ("under_odds", "under")]

            for odds_key, side_label in odds_keys:
                values = [m[1][odds_key] for m in market_rows]
                avg_odds = sum(values) / len(values)

                for book, mkt in market_rows:
                    deviation = abs(mkt[odds_key] - avg_odds)
                    if deviation >= OUTLIER_THRESHOLD:
                        outliers.append({
                            "game_id": gid,
                            "home_team": gdata["home"],
                            "away_team": gdata["away"],
                            "market": market_name,
                            "side": side_label,
                            "sportsbook": book,
                            "odds": mkt[odds_key],
                            "consensus_avg": round(avg_odds, 1),
                            "deviation": round(deviation, 1),
                            "type": "odds_outlier",
                        })

            # Line outliers for spread and total
            if market_name == "spread":
                lines = [m[1].get("home_line", 0) for m in market_rows]
                avg_line = sum(lines) / len(lines)
                for book, mkt in market_rows:
                    line_dev = abs(mkt.get("home_line", 0) - avg_line)
                    if line_dev >= 1.0:
                        outliers.append({
                            "game_id": gid,
                            "home_team": gdata["home"],
                            "away_team": gdata["away"],
                            "market": "spread",
                            "sportsbook": book,
                            "line": mkt.get("home_line"),
                            "consensus_line": round(avg_line, 1),
                            "deviation": round(line_dev, 1),
                            "type": "line_outlier",
                        })
            elif market_name == "total":
                lines = [m[1].get("line", 0) for m in market_rows]
                avg_line = sum(lines) / len(lines)
                for book, mkt in market_rows:
                    line_dev = abs(mkt.get("line", 0) - avg_line)
                    if line_dev >= 1.0:
                        outliers.append({
                            "game_id": gid,
                            "home_team": gdata["home"],
                            "away_team": gdata["away"],
                            "market": "total",
                            "sportsbook": book,
                            "line": mkt.get("line"),
                            "consensus_line": round(avg_line, 1),
                            "deviation": round(line_dev, 1),
                            "type": "line_outlier",
                        })

    outliers.sort(key=lambda x: x["deviation"], reverse=True)

    # ── 6. Stale Lines Detection ───────────────────────────────────────
    stale_lines = []
    STALE_THRESHOLD_MINUTES = 30

    for gid, gdata in games.items():
        rows = gdata["rows"]
        timestamps = []
        for r in rows:
            ts_str = r.get("last_updated", "")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    timestamps.append((r["sportsbook"], ts, r))
                except (ValueError, TypeError):
                    pass

        if not timestamps:
            continue

        newest = max(ts for _, ts, _ in timestamps)
        for book, ts, row in timestamps:
            age_minutes = (newest - ts).total_seconds() / 60
            if age_minutes >= STALE_THRESHOLD_MINUTES:
                stale_lines.append({
                    "game_id": gid,
                    "home_team": gdata["home"],
                    "away_team": gdata["away"],
                    "sportsbook": book,
                    "last_updated": ts.isoformat(),
                    "newest_update": newest.isoformat(),
                    "age_vs_newest_minutes": round(age_minutes, 1),
                    "concern": (
                        f"{book} is {round(age_minutes, 0):.0f} min behind the newest "
                        f"update for this game — lines may be stale"
                    ),
                })

    stale_lines.sort(key=lambda x: x["age_vs_newest_minutes"], reverse=True)

    # ── 7. Fair Odds Summary (consensus no-vig lines) ──────────────────
    fair_odds_summary = []
    for gid, gdata in games.items():
        rows = gdata["rows"]
        game_fair: dict = {"game_id": gid, "home_team": gdata["home"], "away_team": gdata["away"]}
        for market_name in ("spread", "moneyline", "total"):
            market_rows = [r["markets"][market_name]
                           for r in rows if market_name in r.get("markets", {})]
            if not market_rows:
                continue
            if market_name in ("spread", "moneyline"):
                fair_home = [m.get("home_fair_prob") or m.get("home_fair_odds") for m in market_rows
                             if m.get("home_fair_prob") is not None or m.get("home_fair_odds") is not None]
                fair_away = [m.get("away_fair_prob") or m.get("away_fair_odds") for m in market_rows
                             if m.get("away_fair_prob") is not None or m.get("away_fair_odds") is not None]
                if fair_home and fair_away:
                    avg_h = sum(fair_home) / len(fair_home)
                    avg_a = sum(fair_away) / len(fair_away)
                    game_fair[f"{market_name}_home_fair_prob"] = round(avg_h, 4)
                    game_fair[f"{market_name}_away_fair_prob"] = round(avg_a, 4)
            else:
                fair_over = [m.get("over_fair_prob") or m.get("over_fair_odds") for m in market_rows
                             if m.get("over_fair_prob") is not None or m.get("over_fair_odds") is not None]
                fair_under = [m.get("under_fair_prob") or m.get("under_fair_odds") for m in market_rows
                              if m.get("under_fair_prob") is not None or m.get("under_fair_odds") is not None]
                if fair_over and fair_under:
                    avg_o = sum(fair_over) / len(fair_over)
                    avg_u = sum(fair_under) / len(fair_under)
                    game_fair["total_over_fair_prob"] = round(avg_o, 4)
                    game_fair["total_under_fair_prob"] = round(avg_u, 4)
        fair_odds_summary.append(game_fair)

    # ── Build final analysis ───────────────────────────────────────────
    summary_parts = [f"Analyzed {len(games)} games across {len(all_books)} sportsbooks."]
    if arbitrage:
        summary_parts.append(f"Found {len(arbitrage)} arbitrage opportunity(ies).")
    if middles:
        summary_parts.append(f"Found {len(middles)} middle opportunity(ies).")
    if outliers:
        summary_parts.append(f"Detected {len(outliers)} outlier(s).")
    if stale_lines:
        summary_parts.append(f"Detected {len(stale_lines)} stale line(s).")

    analysis = {
        "games_count": len(games),
        "books_count": len(all_books),
        "efficiency_ranking": efficiency,
        "best_lines": best_lines,
        "arbitrage": arbitrage,
        "middles": middles,
        "outliers": outliers,
        "stale_lines": stale_lines,
        "fair_odds_summary": fair_odds_summary,
        "summary": " ".join(summary_parts),
    }

    elapsed = round(time.time() - start, 2)

    if run_logger:
        run_logger.info(
            "Analyze: games=%d books=%d arbs=%d outliers=%d stale=%d elapsed=%.2fs",
            len(games), len(all_books), len(arbitrage), len(outliers), len(stale_lines), elapsed,
        )

    return {
        "analysis": analysis,
        "ai_meta": {
            "provider": "local",
            "model": "detect-pipeline",
            "usage": {},
            "elapsed_seconds": elapsed,
        },
    }


async def run_brief_phase(detection_data: dict, analysis_data: dict, on_chunk=None, run_logger=None) -> dict:
    """Phase 3: AI-powered daily market briefing (readable text).

    When ``on_chunk`` is provided, text fragments are streamed as they arrive
    from the AI provider via ``on_chunk(text_delta)``.
    """
    from datetime import datetime, timezone

    user_prompt = (
        "Generate a daily market briefing from the following data.\n\n"
        "=== DETECTION DATA (enriched odds with implied probabilities, vig, fair odds) ===\n"
        + json.dumps(detection_data, indent=2)[:15000]
        + "\n\n=== CROSS-SPORTSBOOK ANALYSIS ===\n"
        + json.dumps(analysis_data, indent=2)[:15000]
    )

    if run_logger:
        run_logger.info("Brief: starting AI call (streaming=%s)", bool(on_chunk))

    if on_chunk:
        result = await call_ai_stream(BRIEF_SYSTEM_PROMPT, user_prompt, on_chunk=on_chunk, run_logger=run_logger)
    else:
        result = await call_ai(BRIEF_SYSTEM_PROMPT, user_prompt, run_logger=run_logger)

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
