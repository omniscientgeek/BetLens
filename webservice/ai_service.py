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
from typing import Optional

logger = logging.getLogger(__name__)

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

    if not full_text and not result_data:
        stderr_text = (await proc.stderr.read()).decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"No output from Claude SDK wrapper. stderr: {stderr_text}")

    usage = (result_data or {}).get("usage", {})

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


async def call_ai_stream(system_prompt: str, user_prompt: str, on_chunk=None, provider_id: Optional[str] = None) -> dict:
    """
    Send a prompt with streaming support.

    ``on_chunk(text_delta)`` is called for each text fragment as it arrives.
    Returns the full result dict when complete.
    """
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
                logger.info(
                    "AI stream call: provider=%s model=%s attempt=%d/%d",
                    provider["id"], provider.get("model"), attempt, retries,
                )
                result = await call_fn(provider, system_prompt, user_prompt, config, on_chunk=on_chunk)
                return result
            except Exception as exc:
                msg = f"{provider['id']} attempt {attempt}: {exc}"
                logger.warning(msg)
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
        messages=messages,
    )
    elapsed = round(time.time() - start, 2)

    text = response.content[0].text if response.content else ""
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
    response = await client.chat.completions.create(
        model=provider.get("model", "gpt-4o"),
        max_tokens=provider.get("max_tokens", 4096),
        temperature=provider.get("temperature", 0.3),
        messages=api_messages,
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


async def _call_claude_sdk_chat(provider: dict, system_prompt: str, messages: list[dict], config: dict) -> dict:
    """Call Claude SDK by concatenating messages into a single prompt.

    Chat calls enable MCP tool use so Claude can query the betstamp-intelligence
    server on demand instead of relying on pre-baked pipeline context.
    """
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

async def call_ai(system_prompt: str, user_prompt: str, provider_id: Optional[str] = None) -> dict:
    """
    Send a prompt to an AI provider and return the response.

    If ``provider_id`` is given, only that provider is tried.
    Otherwise, enabled providers are tried in priority order with failover.

    Returns a dict with keys: text, provider_id, provider_name, model, usage, elapsed_seconds.
    On total failure raises RuntimeError.
    """
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
                logger.info(
                    "AI call: provider=%s model=%s attempt=%d/%d",
                    provider["id"], provider.get("model"), attempt, retries,
                )
                result = await call_fn(provider, system_prompt, user_prompt, config)
                return result
            except Exception as exc:
                msg = f"{provider['id']} attempt {attempt}: {exc}"
                logger.warning(msg)
                errors.append(msg)

        if not failover:
            break

    raise RuntimeError(
        "All AI providers failed:\n" + "\n".join(f"  - {e}" for e in errors)
    )


async def call_ai_chat(messages: list[dict], system_prompt: str, provider_id: Optional[str] = None) -> dict:
    """
    Send a multi-turn conversation to an AI provider and return the response.

    ``messages`` is a list of dicts with "role" and "content" keys.
    If ``provider_id`` is given, only that provider is tried.
    Otherwise, enabled providers are tried in priority order with failover.

    Returns a dict with keys: text, provider_id, provider_name, model, usage, elapsed_seconds.
    On total failure raises RuntimeError.
    """
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
                logger.info(
                    "AI chat call: provider=%s model=%s attempt=%d/%d messages=%d",
                    provider["id"], provider.get("model"), attempt, retries, len(messages),
                )
                result = await call_fn(provider, system_prompt, messages, config)
                return result
            except Exception as exc:
                msg = f"{provider['id']} attempt {attempt}: {exc}"
                logger.warning(msg)
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


async def run_analyze_phase(detection_data: dict) -> dict:
    """Phase 2: Local cross-sportsbook analysis (no AI call).

    Extracts structured insights from the enriched detection data so the
    brief phase has clean inputs for the AI-powered briefing.
    """
    start = time.time()

    # Pull the enriched odds and vig summary from detection
    odds = detection_data.get("odds", [])
    vig_summary = detection_data.get("vig_summary", {})

    # --- Best lines per game+market+side ---
    best_lines = []
    # Group odds by game
    games = {}
    for row in odds:
        gid = row.get("game_id", "")
        if gid not in games:
            games[gid] = {"home": row.get("home_team", ""), "away": row.get("away_team", ""), "rows": []}
        games[gid]["rows"].append(row)

    # --- Vig / efficiency ranking ---
    efficiency = []
    for book, data in vig_summary.items():
        avg_vig = data.get("avg_vig") or data.get("average_vig")
        if avg_vig is not None:
            efficiency.append({"book": book, "avg_vig_pct": round(avg_vig, 2)})
    efficiency.sort(key=lambda x: x["avg_vig_pct"])

    # --- Simple outlier / stale detection ---
    outliers = []
    stale_lines = []

    analysis = {
        "games_count": len(games),
        "books_count": len(vig_summary),
        "efficiency_ranking": efficiency,
        "best_lines": best_lines,
        "outliers": outliers,
        "stale_lines": stale_lines,
        "summary": (
            f"Analyzed {len(games)} games across {len(vig_summary)} sportsbooks. "
            f"Detection data ready for AI briefing."
        ),
    }

    elapsed = round(time.time() - start, 2)

    return {
        "analysis": analysis,
        "ai_meta": {
            "provider": "local",
            "model": "detect-pipeline",
            "usage": {},
            "elapsed_seconds": elapsed,
        },
    }


async def run_brief_phase(detection_data: dict, analysis_data: dict, on_chunk=None) -> dict:
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

    if on_chunk:
        result = await call_ai_stream(BRIEF_SYSTEM_PROMPT, user_prompt, on_chunk=on_chunk)
    else:
        result = await call_ai(BRIEF_SYSTEM_PROMPT, user_prompt)

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
