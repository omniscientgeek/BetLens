"""
AI Service — Multi-provider AI integration with failover support.

Supports:
  - Anthropic (Claude)
  - OpenAI (GPT-4o, GPT-4o-mini, etc.)
  - OpenAI-compatible endpoints (Ollama, LM Studio, vLLM, etc.)

Providers are configured in ai_config.json and tried in priority order.
If one fails, the next enabled provider is attempted (when failover is on).
"""

import os
import json
import time
import logging
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
# Provider-specific call implementations
# ---------------------------------------------------------------------------

def _call_anthropic(provider: dict, system_prompt: str, user_prompt: str, config: dict) -> dict:
    """Call the Anthropic Messages API."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    api_key = _get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"No API key found for {provider['id']} (env: {provider.get('api_key_env')})")

    client = anthropic.Anthropic(
        api_key=api_key,
        timeout=config.get("timeout_seconds", 60),
    )

    start = time.time()
    response = client.messages.create(
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


def _call_openai(provider: dict, system_prompt: str, user_prompt: str, config: dict) -> dict:
    """Call the OpenAI Chat Completions API (also works with compatible endpoints)."""
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

    client = openai.OpenAI(**kwargs)

    start = time.time()
    response = client.chat.completions.create(
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


# Map provider types to their call functions
_CALL_MAP = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "openai_compatible": _call_openai,
}


# ---------------------------------------------------------------------------
# Main entry point — call with failover
# ---------------------------------------------------------------------------

def call_ai(system_prompt: str, user_prompt: str, provider_id: Optional[str] = None) -> dict:
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
                result = call_fn(provider, system_prompt, user_prompt, config)
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
You are a sports-betting briefing AI. You receive raw odds data and analysis results.

Your job is to generate a concise, actionable briefing for a bettor. Write in clear, \
direct language. Highlight the most important findings:
- Top value bets and why
- Any arbitrage or middle opportunities
- Sportsbooks to avoid (high vig, stale lines)
- Key market movements or anomalies

Keep it under 500 words. Use bullet points for actionable items. \
End with a confidence-rated summary (Low / Medium / High confidence for each recommendation).
"""


def run_analyze_phase(detection_data: dict) -> dict:
    """Phase 2: AI-powered cross-sportsbook analysis."""
    user_prompt = (
        "Here is the enriched odds data from the detection phase. "
        "Analyze it and return your structured JSON findings.\n\n"
        + json.dumps(detection_data, indent=2)[:30000]  # Cap prompt size
    )
    result = call_ai(ANALYZE_SYSTEM_PROMPT, user_prompt)

    # Try to parse the AI response as JSON for structured output
    analysis = result["text"]
    try:
        analysis = json.loads(result["text"])
    except (json.JSONDecodeError, TypeError):
        pass  # Keep as raw text if not valid JSON

    return {
        "analysis": analysis,
        "ai_meta": {
            "provider": result["provider_name"],
            "model": result["model"],
            "usage": result["usage"],
            "elapsed_seconds": result["elapsed_seconds"],
        },
    }


def run_brief_phase(detection_data: dict, analysis_data: dict) -> dict:
    """Phase 3: AI-powered actionable briefing."""
    user_prompt = (
        "Here is the odds data and analysis. Generate a concise actionable briefing.\n\n"
        "=== DETECTION DATA ===\n"
        + json.dumps(detection_data, indent=2)[:15000]
        + "\n\n=== ANALYSIS ===\n"
        + json.dumps(analysis_data, indent=2)[:15000]
    )
    result = call_ai(BRIEF_SYSTEM_PROMPT, user_prompt)

    return {
        "brief": result["text"],
        "ai_meta": {
            "provider": result["provider_name"],
            "model": result["model"],
            "usage": result["usage"],
            "elapsed_seconds": result["elapsed_seconds"],
        },
    }
