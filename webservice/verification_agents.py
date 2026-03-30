"""
Verification Agents — Three-agent system that double-checks AI responses.

Agents:
  1. Reasoning Agent   — Verifies logical consistency of analysis
  2. Factual Agent     — Cross-checks claims against MCP betting data tools
  3. Betting Agent     — Validates betting recommendations for soundness

All three run concurrently via asyncio.gather(). Each parent agent extracts
individual claims then fans out parallel sub-agents (one per claim) for
faster verification. Results are attached to the AI response for frontend
display as verification badges.
"""

import json
import hashlib
import re
import time
import asyncio
import logging
from typing import Optional

from ai_service import call_ai, call_ai_chat, call_ai_chat_stream

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audit result cache — avoids re-verifying identical text
# ---------------------------------------------------------------------------
# Keyed by SHA-256 hash of the text being verified. Each entry stores the
# full verification result and a timestamp for TTL expiration.

_audit_cache: dict[str, dict] = {}
AUDIT_CACHE_TTL = 1800  # 30 minutes — matches pipeline cache TTL


def _audit_cache_key(text: str) -> str:
    """Deterministic cache key from the text being audited."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_cached_audit(text: str) -> Optional[dict]:
    """Return cached verification result if present and not expired."""
    key = _audit_cache_key(text)
    entry = _audit_cache.get(key)
    if entry is None:
        return None
    if time.time() - entry["cached_at"] > AUDIT_CACHE_TTL:
        del _audit_cache[key]
        return None
    return entry["result"]


def _store_audit_cache(text: str, result: dict) -> None:
    """Store a verification result in the cache."""
    key = _audit_cache_key(text)
    _audit_cache[key] = {
        "result": result,
        "cached_at": time.time(),
    }


def clear_audit_cache() -> int:
    """Clear the entire audit cache. Returns the number of entries removed."""
    count = len(_audit_cache)
    _audit_cache.clear()
    return count


def get_audit_cache_stats() -> dict:
    """Return cache statistics for diagnostics."""
    now = time.time()
    active = sum(1 for e in _audit_cache.values() if now - e["cached_at"] <= AUDIT_CACHE_TTL)
    return {
        "total_entries": len(_audit_cache),
        "active_entries": active,
        "expired_entries": len(_audit_cache) - active,
        "ttl_seconds": AUDIT_CACHE_TTL,
    }


# ---------------------------------------------------------------------------
# Verification result structure helpers
# ---------------------------------------------------------------------------

def _error_result(agent_name: str, error_msg: str) -> dict:
    """Return a fallback verification result when an agent fails."""
    return {
        "agent": agent_name,
        "verdict": "error",
        "confidence": 0.0,
        "issues": [
            {
                "severity": "error",
                "claim": "Agent execution",
                "finding": f"Verification agent failed: {error_msg}",
            }
        ],
        "summary": f"Verification could not be completed: {error_msg}",
        "ai_meta": None,
    }


def _parse_agent_response(raw_text: str, agent_name: str) -> dict:
    """Parse JSON from an agent's AI response.

    Handles:
      - Direct JSON
      - JSON wrapped in ```json ... ``` markdown fences
      - Graceful fallback with the raw text as summary
    """
    text = raw_text.strip()

    # Try direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "verdict" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1).strip())
            if isinstance(parsed, dict) and "verdict" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group(0))
            if isinstance(parsed, dict) and "verdict" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # Fallback: couldn't parse structured response
    logger.warning(
        "Could not parse JSON from %s agent response, using fallback", agent_name
    )
    return {
        "verdict": "warn",
        "confidence": 0.5,
        "issues": [
            {
                "severity": "warning",
                "claim": "Response format",
                "finding": "Agent returned unstructured response — manual review recommended.",
            }
        ],
        "summary": text[:300],
    }


def _smart_truncate(result_str: str, max_chars: int) -> str:
    """Truncate a tool result string, preserving JSON ``summary`` if present.

    Many MCP tools (GAMLSS, KNN, Poisson, etc.) return large JSON payloads
    but include a compact ``summary`` key with the verifiable numbers.  When
    the full result exceeds *max_chars*, we try to extract and keep just the
    summary so audit agents can still verify claims against it.
    """
    if len(result_str) <= max_chars:
        return result_str
    # Try to parse and preserve the summary section
    try:
        data = json.loads(result_str)
        if isinstance(data, dict) and "summary" in data:
            summary = json.dumps({"summary": data["summary"]}, indent=2)
            if len(summary) <= max_chars:
                return summary + "\n... [full result truncated, summary preserved]"
    except (json.JSONDecodeError, TypeError):
        pass
    return result_str[:max_chars] + "\n... [truncated]"


def _truncate_tool_results(tool_calls: list, max_result_len: int = 4000) -> list:
    """Truncate tool call results to keep payload sizes reasonable.

    Default limit lowered from 8000 to 4000 chars — audit agents receive
    pre-fetched reference data so individual tool results need less detail.
    Uses summary-aware truncation so large results (e.g. GAMLSS) keep their
    verifiable summary section.
    """
    truncated = []
    for tc in tool_calls:
        entry = {**tc}
        if isinstance(entry.get("result"), str) and len(entry["result"]) > max_result_len:
            entry["result"] = _smart_truncate(entry["result"], max_result_len)
        truncated.append(entry)
    return truncated


# ---------------------------------------------------------------------------
# Reference data extraction — pre-fetch tool results to avoid duplicate calls
# ---------------------------------------------------------------------------

# MCP tool names to EXCLUDE from reference data extraction.
# All other successful MCP tool results from the analyze phase are automatically
# included as reference data for audit agents.  This ensures tools like
# get_gamlss_analysis, detect_knn_anomalies, get_poisson_score_predictions, etc.
# are available for verification without requiring manual updates here.
_REFERENCE_TOOL_EXCLUDE = {
    "arithmetic_add",
    "arithmetic_subtract",
    "arithmetic_multiply",
    "arithmetic_divide",
    "arithmetic_modulo",
    "arithmetic_evaluate",
    "list_data_files",
    "calculate_odds",
}

# Max chars per reference tool result — keeps total reference block manageable
_REF_MAX_CHARS = 6000


def build_reference_data(
    tool_calls: list[dict],
    max_chars_per_tool: int = _REF_MAX_CHARS,
) -> str:
    """Extract MCP tool results from a prior conversation for audit reference.

    Scans the ``tool_calls`` list (from the analyze phase conversation) and
    collects the FIRST successful result for every MCP tool **not** in
    ``_REFERENCE_TOOL_EXCLUDE``.  This is dynamic — any new tool the analyze
    phase calls (GAMLSS, KNN, Poisson, Shin, etc.) is automatically available
    as reference data without code changes.

    Large results are smart-truncated: the ``summary`` JSON key is preserved
    when possible so audit agents can still verify aggregate numbers.

    Returns a formatted string suitable for inclusion in audit agent prompts.
    """
    collected: dict[str, str] = {}

    for tc in (tool_calls or []):
        raw_name = tc.get("name", "")
        # Strip MCP namespace prefix: "mcp__betstamp-intelligence__tool_name" → "tool_name"
        short_name = raw_name.split("__")[-1] if "__" in raw_name else raw_name

        if short_name in _REFERENCE_TOOL_EXCLUDE:
            continue
        if short_name in collected:
            continue  # keep first (analyze phase result, not a retry)
        if tc.get("is_error"):
            continue

        result = tc.get("result", "")
        if isinstance(result, dict):
            result = json.dumps(result, separators=(",", ":"))
        result = str(result)

        result = _smart_truncate(result, max_chars_per_tool)

        collected[short_name] = result

    if not collected:
        return ""

    lines = ["=== PRE-FETCHED REFERENCE DATA (from prior MCP tool calls) ===",
             "Use this data to verify claims FIRST. Only call MCP tools for data NOT included below.\n"]
    for tool_name, result in collected.items():
        lines.append(f"--- {tool_name} ---")
        lines.append(result)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sub-agent parallelism constants
# ---------------------------------------------------------------------------

MAX_CONCURRENT_SUB_AGENTS = 10  # semaphore limit per parent agent
MIN_CLAIMS_FOR_PARALLEL = 2     # below this, fall back to single-call


# ---------------------------------------------------------------------------
# System prompts for each verification agent
# ---------------------------------------------------------------------------

REASONING_SYSTEM_PROMPT = """\
You are a logical reasoning verification agent for sports betting analysis.

TASK: Verify LOGICAL CONSISTENCY of claims against actual data. Use the \
PRE-FETCHED REFERENCE DATA in the user prompt FIRST — only call MCP tools \
for data not already provided.

ARITHMETIC: Use MCP arithmetic tools (arithmetic_add, arithmetic_subtract, \
arithmetic_multiply, arithmetic_divide) for ALL math — even simple operations. \
NEVER compute numbers yourself, not even basic math. For multi-step calculations, \
chain multiple arithmetic tool calls.

FAIR ODDS: get_fair_odds() = consensus (canonical baseline). Pinnacle sharp \
fields differ — do NOT flag consensus vs Pinnacle discrepancies as errors.

CHECKLIST:
1. Mathematical consistency — vig %, implied probs, fair odds match data
2. Ranking consistency — sportsbook rankings match get_book_rankings / get_vig_analysis
3. Comparison validity — "X better than Y" claims match odds data
4. Contradiction detection — recommendations vs risk flags are consistent
5. Confidence alignment — confidence ratings match actual edge sizes
6. Staleness-confidence alignment — HIGH confidence ratings must not apply to bets \
   using lines identified as stale (>30 min old). Flag any HIGH confidence + stale \
   line combination as a logical contradiction.

Each issue MUST cite specific data evidence. Do NOT evaluate betting strategy \
(another agent handles that). Do NOT flag issues based on suspicion alone.

Return ONLY valid JSON:
{
  "verdict": "pass" | "warn" | "fail",
  "confidence": 0.0-1.0,
  "checks_total": <number verified including passes>,
  "checks_failed": <warnings + errors only>,
  "issues": [{"severity": "info"|"warning"|"error", "claim": "...", "finding": "..."}],
  "summary": "1-2 sentence summary"
}
Verdict: "pass" = all consistent, "warn" = minor issues, "fail" = clear logical errors.
Your final message MUST be the JSON verdict — no markdown fences or extra text.
"""


FACTUAL_SYSTEM_PROMPT = """\
You are a factual accuracy verification agent for sports betting analysis.

TASK: Fact-check ALL specific claims (odds, EV %, vig %, arb profit, rankings, \
stale lines) against actual data. Use the PRE-FETCHED REFERENCE DATA in the \
user prompt FIRST — only call MCP tools for data not already provided.

ARITHMETIC: Use MCP arithmetic tools (arithmetic_add, arithmetic_subtract, \
arithmetic_multiply, arithmetic_divide) for ALL math — even simple operations. \
NEVER compute numbers yourself. For multi-step calculations, chain multiple \
arithmetic tool calls.

FAIR ODDS: get_fair_odds() = consensus baseline. Do NOT flag consensus vs \
Pinnacle discrepancies unless text explicitly says "Pinnacle fair odds".

PRIORITY ORDER (most error-prone first):
1. +EV claims — verify against find_expected_value_bets reference data. Flag \
   any +EV opportunity NOT in the data as "error: Fabricated +EV opportunity".
2. Arbitrage profit % — verify against find_arbitrage_opportunities data.
3. Kelly sizing — verify against get_kelly_sizing / get_best_bets_today data.
4. All remaining claims (odds, vig, rankings, staleness, etc.).

TOOL CALL POLICY: If a claim references data from an MCP tool NOT in the \
reference data (e.g., get_gamlss_analysis, detect_knn_anomalies, \
get_poisson_score_predictions, get_shin_fair_odds, get_information_flow), \
you MUST call that tool to verify. NEVER mark a claim as "cannot verify" \
without attempting the tool call first. For large results, focus on the \
"summary" section which contains the aggregate numbers.

ACCURACY THRESHOLDS:
- Odds: >3 pts = error, 1-3 = warning
- EV/Vig %: >0.5% = error, 0.1-0.5% = warning
- Arb profit: >1% = error, 0.1-1% = warning
- Wrong sportsbook or direction = always error

LINE SHOPPING GAPS: The brief uses "gap_pct" (implied probability edge) from \
pre-computed line_shopping_pairs. Do NOT verify gaps by subtracting American odds \
(e.g., +165 minus -121 = 286 is INVALID — American odds are non-linear). Instead, \
verify gap_pct = (1 - combined_implied_prob) * 100. If the brief reports a "%  \
implied edge" value, check it against the pre-computed line_shopping_pairs data.

Each issue MUST cite specific data evidence. Do NOT evaluate strategy or logic \
(other agents handle those).

Return ONLY valid JSON:
{
  "verdict": "pass" | "warn" | "fail",
  "confidence": 0.0-1.0,
  "checks_total": <number verified including passes>,
  "checks_failed": <warnings + errors only>,
  "issues": [{"severity": "info"|"warning"|"error", "claim": "...", "finding": "..."}],
  "summary": "1-2 sentence summary"
}
Verdict: "pass" = accurate, "warn" = minor discrepancies, "fail" = material errors.
Your final message MUST be the JSON verdict — no markdown fences or extra text.
"""


BETTING_SYSTEM_PROMPT = """\
You are a betting recommendation verification agent. You review AI-generated \
betting analysis for sound betting principles.

TASK: Verify ALL recommendations against actual data. Use the PRE-FETCHED \
REFERENCE DATA in the user prompt FIRST — only call MCP tools for data not \
already provided.

ARITHMETIC: Use MCP arithmetic tools (arithmetic_add, arithmetic_subtract, \
arithmetic_multiply, arithmetic_divide) for ALL math — even simple operations. \
NEVER compute numbers yourself. For multi-step calculations, chain multiple \
arithmetic tool calls.

CHECKLIST (verify against reference data or MCP tools):
1. Bankroll management — Kelly sizing proportional to edges; flag if sizing \
   exceeds Kelly optimal or if any bet lacks quarter-Kelly guidance.
2. Risk assessment — verify EV edges. The analysis uses a relative staleness \
   threshold (see stale_threshold_minutes in the data — typically 30 min behind \
   the newest update for the same game). When verifying stale line counts, use \
   detect_stale_lines with the SAME threshold. ONLY flag staleness as an issue \
   if the analysis omits any mention of staleness for genuinely stale lines.
3. Diversification — flag correlated bets on same game unless explained as arb.
4. Value basis — every bet must have measurable edge above fair odds.
5. Realistic expectations — "guaranteed profit" claims must be actual arbitrage.
6. Responsible gambling — high-risk bets need caution language.

Do NOT verify specific number accuracy or logical consistency (other agents do that).
Each issue MUST cite specific data evidence.

Return ONLY valid JSON:
{
  "verdict": "pass" | "warn" | "fail",
  "confidence": 0.0-1.0,
  "checks_total": <number verified including passes>,
  "checks_failed": <warnings + errors only>,
  "issues": [{"severity": "info"|"warning"|"error", "claim": "...", "finding": "..."}],
  "summary": "1-2 sentence summary"
}
Verdict: "pass" = sound principles, "warn" = minor concerns, "fail" = dangerous advice.
Your final message MUST be the JSON verdict — no markdown fences or extra text.
"""


# ---------------------------------------------------------------------------
# Claim extraction prompt — used to split text into individual claims
# ---------------------------------------------------------------------------

CLAIM_EXTRACTION_PROMPT = """\
You are a claim extractor. Given a betting analysis text and an agent type, \
extract every verifiable claim into a structured JSON array.

AGENT TYPE: {agent_type}

Agent-specific focus:
- reasoning: Extract logical assertions, ranking claims, cross-references, \
  and consistency claims (e.g., "Book X has lowest vig", "this contradicts \
  the recommendation above").
- factual: Extract specific numbers — odds values, EV percentages, vig \
  percentages, arbitrage profit %, sportsbook names, line values, implied \
  probabilities, Kelly sizing numbers, power rankings, stale line times.
- betting: Extract recommendations — bet suggestions, sizing advice, risk \
  assessments, guaranteed-profit claims, bankroll management guidance, \
  responsible gambling statements.

Return ONLY a valid JSON array (no markdown fences, no extra text):
[
  {{
    "claim_text": "verbatim or near-verbatim claim from the text",
    "claim_type": "short category label (e.g. ev_claim, odds_value, vig_ranking, \
kelly_sizing, arb_profit, recommendation, risk_flag)",
    "required_tools": ["list", "of", "MCP", "tool", "names", "to", "verify"],
    "context": "1-2 sentences of surrounding context needed to understand the claim"
  }},
  ...
]

IMPORTANT:
- Extract EVERY verifiable claim — do not skip any.
- Each claim should be a single, atomic, verifiable statement.
- Do NOT combine multiple facts into one claim.
- Include 1-3 relevant MCP tool names in required_tools.
- If the text has very few verifiable claims (e.g. just 1), still return them.
"""


# ---------------------------------------------------------------------------
# Sub-agent prompt prefix — scopes a sub-agent to specific claims
# ---------------------------------------------------------------------------

SUB_AGENT_PROMPT_PREFIX = """\
IMPORTANT SCOPE RESTRICTION:
You are a focused sub-agent. Verify ONLY the specific claim(s) listed below. \
Do NOT verify anything else in the analysis text. Your entire job is to check \
these specific claims and return your verdict for them only.

CLAIMS TO VERIFY:
{claims_block}

FULL ANALYSIS TEXT (for context only — verify ONLY the claims above):
"""


# ---------------------------------------------------------------------------
# Sub-agent helpers — extract, verify, aggregate
# ---------------------------------------------------------------------------

async def _extract_claims(
    text: str,
    agent_type: str,
    run_logger: Optional[logging.Logger] = None,
) -> list[dict]:
    """Use a lightweight LLM call (no MCP tools) to parse claims from text.

    Returns a list of claim dicts, or an empty list on failure (triggering
    fallback to the serial single-call approach).
    """
    prompt = CLAIM_EXTRACTION_PROMPT.format(agent_type=agent_type)
    if run_logger:
        run_logger.info("VERIFICATION [%s] extracting claims via LLM ...", agent_type)

    try:
        result = await call_ai(
            system_prompt=prompt,
            user_prompt=text,
            run_logger=run_logger,
            max_tokens=4096,
        )
        raw = result["text"].strip()

        # Parse JSON — handle markdown fences
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
        if fence_match:
            raw = fence_match.group(1).strip()

        claims = json.loads(raw)
        if not isinstance(claims, list):
            raise ValueError("Expected JSON array of claims")

        if run_logger:
            run_logger.info(
                "VERIFICATION [%s] extracted %d claims", agent_type, len(claims)
            )
        return claims

    except Exception as exc:
        logger.warning("Claim extraction failed for %s: %s", agent_type, exc)
        if run_logger:
            run_logger.warning(
                "VERIFICATION [%s] claim extraction failed: %s — falling back to serial",
                agent_type, exc,
            )
        return []


async def _run_claim_sub_agent(
    claim: dict,
    sub_index: int,
    agent_type: str,
    parent_system_prompt: str,
    text_to_verify: str,
    semaphore: asyncio.Semaphore,
    on_tool_event=None,
    run_logger: Optional[logging.Logger] = None,
) -> dict:
    """Verify a single claim (or small batch) via a scoped LLM call with MCP tools."""
    claim_text = claim.get("claim_text", str(claim))
    claim_type = claim.get("claim_type", "unknown")
    required_tools = claim.get("required_tools", [])
    context = claim.get("context", "")

    claims_block = (
        f"Claim: {claim_text}\n"
        f"Type: {claim_type}\n"
        f"Suggested MCP tools: {', '.join(required_tools) if required_tools else 'use your judgment'}\n"
        f"Context: {context}"
    )

    scoped_system = parent_system_prompt + "\n\n" + SUB_AGENT_PROMPT_PREFIX.format(
        claims_block=claims_block
    )

    messages = [
        {
            "role": "user",
            "content": (
                f"Verify the following specific claim from a betting analysis. "
                f"Use MCP tools to check it. Return your verdict as JSON.\n\n"
                f"=== CLAIM TO VERIFY ===\n{claim_text}\n\n"
                f"=== FULL ANALYSIS (for context) ===\n{text_to_verify}"
            ),
        }
    ]

    async with semaphore:
        if run_logger:
            run_logger.info(
                "VERIFICATION [%s] sub-agent #%d starting — claim_type=%s",
                agent_type, sub_index, claim_type,
            )

        sub_start = time.time()
        try:
            result = await call_ai_chat_stream(
                messages=messages,
                system_prompt=scoped_system,
                on_tool_event=on_tool_event,
                provider_id=None,
                run_logger=run_logger,
            )
        except Exception as exc:
            elapsed = round(time.time() - sub_start, 2)
            logger.error(
                "Sub-agent #%d [%s] failed: %s", sub_index, agent_type, exc
            )
            return {
                "verdict": "error",
                "confidence": 0.0,
                "checks_total": 1,
                "checks_failed": 1,
                "issues": [
                    {
                        "severity": "error",
                        "claim": claim_text[:200],
                        "finding": f"Sub-agent failed: {exc}",
                    }
                ],
                "summary": f"Sub-agent error: {exc}",
                "ai_meta": {"elapsed_seconds": elapsed},
                "tool_calls": [],
                "assistant_response": "",
            }

        elapsed = round(time.time() - sub_start, 2)
        if run_logger:
            run_logger.info(
                "VERIFICATION [%s] sub-agent #%d complete in %.2fs",
                agent_type, sub_index, elapsed,
            )

        parsed = _parse_agent_response(result["text"], f"{agent_type}_sub{sub_index}")
        issues = parsed.get("issues", [])
        return {
            "verdict": parsed.get("verdict", "warn"),
            "confidence": parsed.get("confidence", 0.5),
            "checks_total": parsed.get("checks_total") or max(len(issues), 1),
            "checks_failed": parsed.get("checks_failed") or sum(
                1 for i in issues if i.get("severity") in ("warning", "error")
            ),
            "issues": issues,
            "summary": parsed.get("summary", ""),
            "ai_meta": {
                "provider": result.get("provider_name", ""),
                "model": result.get("model", ""),
                "elapsed_seconds": elapsed,
                "usage": result.get("usage", {}),
            },
            "tool_calls": _truncate_tool_results(result.get("tool_calls", [])),
            "assistant_response": result.get("text", ""),
        }


def _aggregate_sub_results(
    sub_results: list[dict],
    agent_name: str,
    parent_system_prompt: str,
    original_user_prompt: str,
    wall_elapsed: float,
    claims: list[dict] | None = None,
) -> dict:
    """Merge N sub-agent results into a single agent result dict.

    Preserves the exact same schema as the original serial agent runner so
    that callers (app.py, fix phase, cache, UI) see no difference.

    When *claims* is provided the individual sub-agent results are preserved
    in a ``sub_agents`` list so the UI can drill down per-claim.
    """
    verdict_priority = {"pass": 0, "warn": 1, "fail": 2, "error": 3}

    # Worst verdict wins
    verdicts = [r["verdict"] for r in sub_results]
    worst = max(verdicts, key=lambda v: verdict_priority.get(v, 3))

    # Sum checks
    total_checks = sum(r.get("checks_total", 0) for r in sub_results)
    failed_checks = sum(r.get("checks_failed", 0) for r in sub_results)

    # Concatenate issues
    all_issues = []
    for r in sub_results:
        all_issues.extend(r.get("issues", []))

    # Weighted-average confidence
    weighted_conf = 0.0
    total_weight = 0
    for r in sub_results:
        w = max(r.get("checks_total", 1), 1)
        weighted_conf += r.get("confidence", 0.5) * w
        total_weight += w
    avg_confidence = round(weighted_conf / total_weight, 3) if total_weight else 0.5

    # Combine summaries
    summaries = [r.get("summary", "") for r in sub_results if r.get("summary")]
    combined_summary = " | ".join(summaries) if summaries else ""
    # Truncate if very long
    if len(combined_summary) > 500:
        combined_summary = combined_summary[:497] + "..."

    # Aggregate token usage
    total_input = sum(r.get("ai_meta", {}).get("usage", {}).get("input_tokens", 0) for r in sub_results)
    total_output = sum(r.get("ai_meta", {}).get("usage", {}).get("output_tokens", 0) for r in sub_results)
    provider = sub_results[0].get("ai_meta", {}).get("provider", "") if sub_results else ""
    model = sub_results[0].get("ai_meta", {}).get("model", "") if sub_results else ""

    # Concatenate tool calls from all sub-agents
    all_tool_calls = []
    for r in sub_results:
        all_tool_calls.extend(r.get("tool_calls", []))

    # Combine assistant responses for conversation record
    all_responses = [r.get("assistant_response", "") for r in sub_results if r.get("assistant_response")]
    combined_response = "\n\n---\n\n".join(all_responses)

    return {
        "agent": agent_name,
        "verdict": worst,
        "confidence": avg_confidence,
        "checks_total": total_checks,
        "checks_failed": failed_checks,
        "issues": all_issues,
        "summary": combined_summary,
        "ai_meta": {
            "provider": provider,
            "model": model,
            "elapsed_seconds": wall_elapsed,
            "usage": {
                "input_tokens": total_input,
                "output_tokens": total_output,
            },
            "sub_agent_count": len(sub_results),
        },
        "conversation": {
            "system_prompt": parent_system_prompt,
            "user_prompt": original_user_prompt,
            "assistant_response": combined_response,
            "tool_calls": all_tool_calls,
        },
        # Per-claim sub-agent results for UI drill-down
        "sub_agents": [
            {
                "claim_text": (claims[i].get("claim_text", "") if claims and i < len(claims) else ""),
                "claim_type": (claims[i].get("claim_type", "") if claims and i < len(claims) else ""),
                "verdict": r["verdict"],
                "confidence": r.get("confidence", 0.5),
                "checks_total": r.get("checks_total", 0),
                "checks_failed": r.get("checks_failed", 0),
                "issues": r.get("issues", []),
                "summary": r.get("summary", ""),
                "ai_meta": r.get("ai_meta", {}),
                "tool_calls_count": len(r.get("tool_calls", [])),
            }
            for i, r in enumerate(sub_results)
        ],
    }


# ---------------------------------------------------------------------------
# Individual agent runners (parallel sub-agent pattern with serial fallback)
# ---------------------------------------------------------------------------

async def _run_agent_serial(
    agent_name: str,
    text_to_verify: str,
    system_prompt: str,
    user_prompt_text: str,
    run_logger: Optional[logging.Logger] = None,
    on_tool_event=None,
) -> dict:
    """Original serial single-LLM-call path — used as fallback when claim
    extraction fails or yields fewer than 2 claims."""
    agent_start = time.time()

    messages = [{"role": "user", "content": user_prompt_text}]

    if run_logger:
        run_logger.info("VERIFICATION [%s] starting SERIAL (MCP-enabled)", agent_name)

    result = await call_ai_chat_stream(
        messages=messages,
        system_prompt=system_prompt,
        on_tool_event=on_tool_event,
        provider_id=None,
        run_logger=run_logger,
    )

    elapsed = round(time.time() - agent_start, 2)
    if run_logger:
        run_logger.info("VERIFICATION [%s] serial complete in %.2fs", agent_name, elapsed)

    parsed = _parse_agent_response(result["text"], agent_name)
    issues = parsed.get("issues", [])
    return {
        "agent": agent_name,
        "verdict": parsed.get("verdict", "warn"),
        "confidence": parsed.get("confidence", 0.5),
        "checks_total": parsed.get("checks_total") or len(issues) or 0,
        "checks_failed": parsed.get("checks_failed") or sum(1 for i in issues if i.get("severity") in ("warning", "error")),
        "issues": issues,
        "summary": parsed.get("summary", ""),
        "ai_meta": {
            "provider": result["provider_name"],
            "model": result["model"],
            "elapsed_seconds": elapsed,
            "usage": result.get("usage", {}),
        },
        "conversation": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt_text,
            "assistant_response": result["text"],
            "tool_calls": _truncate_tool_results(result.get("tool_calls", [])),
        },
    }


async def _run_agent_parallel(
    agent_name: str,
    text_to_verify: str,
    system_prompt: str,
    user_prompt_text: str,
    run_logger: Optional[logging.Logger] = None,
    on_tool_event=None,
) -> dict:
    """Extract claims, fan out parallel sub-agents, aggregate results.

    Falls back to _run_agent_serial if claim extraction fails or returns < 2 claims.
    """
    agent_start = time.time()

    # Step 1 — extract claims
    claims = await _extract_claims(text_to_verify, agent_name, run_logger)

    if len(claims) < 2:
        if run_logger:
            run_logger.info(
                "VERIFICATION [%s] only %d claim(s) extracted — using serial fallback",
                agent_name, len(claims),
            )
        return await _run_agent_serial(
            agent_name, text_to_verify, system_prompt,
            user_prompt_text, run_logger, on_tool_event,
        )

    # Step 2 — fan out sub-agents with semaphore
    if run_logger:
        run_logger.info(
            "VERIFICATION [%s] launching %d parallel sub-agents (max_concurrent=%d)",
            agent_name, len(claims), MAX_CONCURRENT_SUB_AGENTS,
        )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SUB_AGENTS)

    sub_coros = [
        _run_claim_sub_agent(
            claim=claim,
            sub_index=i,
            agent_type=agent_name,
            parent_system_prompt=system_prompt,
            text_to_verify=text_to_verify,
            semaphore=semaphore,
            on_tool_event=on_tool_event,
            run_logger=run_logger,
        )
        for i, claim in enumerate(claims)
    ]

    sub_results = await asyncio.gather(*sub_coros, return_exceptions=True)

    # Convert exceptions to error results
    clean_results = []
    for i, r in enumerate(sub_results):
        if isinstance(r, BaseException):
            logger.error("Sub-agent #%d [%s] raised: %s", i, agent_name, r)
            clean_results.append({
                "verdict": "error",
                "confidence": 0.0,
                "checks_total": 1,
                "checks_failed": 1,
                "issues": [{"severity": "error", "claim": claims[i].get("claim_text", "")[:200], "finding": f"Sub-agent exception: {r}"}],
                "summary": f"Sub-agent error: {r}",
                "ai_meta": {"elapsed_seconds": 0},
                "tool_calls": [],
                "assistant_response": "",
            })
        else:
            clean_results.append(r)

    # Step 3 — aggregate
    wall_elapsed = round(time.time() - agent_start, 2)
    aggregated = _aggregate_sub_results(
        clean_results, agent_name, system_prompt, user_prompt_text, wall_elapsed,
        claims=claims,
    )

    if run_logger:
        run_logger.info(
            "VERIFICATION [%s] parallel complete — %d sub-agents, verdict=%s, elapsed=%.2fs",
            agent_name, len(clean_results), aggregated["verdict"], wall_elapsed,
        )

    return aggregated


async def _run_reasoning_agent(
    text_to_verify: str,
    source_data: str,
    run_logger: Optional[logging.Logger] = None,
    on_tool_event=None,
    reference_data: str = "",
) -> dict:
    """Verify logical consistency — parallel sub-agents with serial fallback."""
    ref_block = f"\n\n{reference_data}" if reference_data else ""
    user_prompt = (
        "Verify the logical consistency of this betting analysis. "
        "Use the reference data below FIRST — only call MCP tools for data "
        "not included in the reference.\n\n"
        "=== AI-GENERATED ANALYSIS TO VERIFY ===\n"
        f"{text_to_verify}"
        f"{ref_block}"
    )
    return await _run_agent_parallel(
        "reasoning", text_to_verify, REASONING_SYSTEM_PROMPT,
        user_prompt, run_logger, on_tool_event,
    )


async def _run_factual_agent(
    text_to_verify: str,
    run_logger: Optional[logging.Logger] = None,
    on_tool_event=None,
    reference_data: str = "",
) -> dict:
    """Fact-check claims via parallel sub-agents with serial fallback."""
    ref_block = f"\n\n{reference_data}" if reference_data else ""
    user_prompt = (
        "Fact-check the following betting analysis. Use the reference data "
        "below FIRST — only call MCP tools for data not included.\n\n"
        "PRIORITY: Verify ALL +EV claims and arb profits FIRST. Flag any "
        "+EV opportunity NOT in the reference data as fabrication.\n\n"
        "=== AI-GENERATED ANALYSIS TO FACT-CHECK ===\n"
        f"{text_to_verify}"
        f"{ref_block}"
    )
    return await _run_agent_parallel(
        "factual", text_to_verify, FACTUAL_SYSTEM_PROMPT,
        user_prompt, run_logger, on_tool_event,
    )


async def _run_betting_agent(
    text_to_verify: str,
    run_logger: Optional[logging.Logger] = None,
    on_tool_event=None,
    reference_data: str = "",
) -> dict:
    """Verify betting recommendation soundness — parallel sub-agents with serial fallback."""
    ref_block = f"\n\n{reference_data}" if reference_data else ""
    user_prompt = (
        "Review the following betting analysis for sound betting principles. "
        "Use the reference data below FIRST — only call MCP tools for data "
        "not included.\n\n"
        "=== AI-GENERATED ANALYSIS TO REVIEW ===\n"
        f"{text_to_verify}"
        f"{ref_block}"
    )
    return await _run_agent_parallel(
        "betting", text_to_verify, BETTING_SYSTEM_PROMPT,
        user_prompt, run_logger, on_tool_event,
    )


# ---------------------------------------------------------------------------
# Public orchestrator — runs all 3 agents concurrently
# ---------------------------------------------------------------------------

async def run_verification(
    text_to_verify: str,
    source_data: str = "",
    run_logger: Optional[logging.Logger] = None,
    on_agent_complete=None,
    on_tool_event=None,
    reference_data: str = "",
) -> dict:
    """Run all three verification agents in parallel.

    Parameters
    ----------
    text_to_verify : str
        The AI-generated text (brief or chat response) to verify.
    source_data : str, optional
        The underlying data that produced the text (detection + analysis JSON).
        Used by the reasoning agent for cross-referencing.
    run_logger : logging.Logger, optional
        Per-run logger for detailed logging.
    on_agent_complete : async callable, optional
        Called with (agent_name, agent_result) each time an individual agent
        finishes — enables real-time streaming of audit progress to the frontend.
    on_tool_event : async callable, optional
        Called with (agent_name, event_type, data) each time an agent makes or
        receives a tool call — enables real-time streaming of MCP tool activity.
    reference_data : str, optional
        Pre-fetched MCP tool results (built via ``build_reference_data``).
        When provided, agents use this data first and only call MCP tools
        for data not already included — dramatically reducing duplicate calls.

    Returns
    -------
    dict
        VerificationReport with overall verdict and per-agent results.
    """
    # ------------------------------------------------------------------
    # Check audit cache — skip all 3 agents if we already verified this text
    # ------------------------------------------------------------------
    cached = _get_cached_audit(text_to_verify)
    if cached is not None:
        if run_logger:
            run_logger.info("=" * 60)
            run_logger.info("VERIFICATION PIPELINE — cache HIT (skipping 3 agents)")
            run_logger.info("  cached verdict=%s  original_elapsed=%.2fs",
                            cached["overall_verdict"], cached["elapsed_seconds"])
            run_logger.info("=" * 60)

        # Still fire on_agent_complete callbacks so the UI gets its updates
        if on_agent_complete:
            for agent_name in ["reasoning", "factual", "betting"]:
                agent_result = cached.get("agents", {}).get(agent_name)
                if agent_result:
                    try:
                        await on_agent_complete(agent_name, agent_result)
                    except Exception as cb_err:
                        logger.warning("on_agent_complete callback error for %s: %s", agent_name, cb_err)

        # Return a shallow copy with cached flag so callers can distinguish
        return {**cached, "from_cache": True}

    if run_logger:
        run_logger.info("=" * 60)
        run_logger.info("VERIFICATION PIPELINE — cache MISS, starting 3 agents in parallel")
        run_logger.info("=" * 60)

    start = time.time()

    # Create per-agent tool event callbacks that tag events with the agent name
    def _make_agent_tool_cb(agent_name):
        if not on_tool_event:
            return None
        async def _cb(event_type, data):
            await on_tool_event(agent_name, event_type, data)
        return _cb

    # Wrap each agent so we can emit its result as soon as it finishes
    agent_names = ["reasoning", "factual", "betting"]
    agents = {}

    async def _run_and_notify(name, coro):
        try:
            result = await coro
        except BaseException as exc:
            logger.error("Verification agent '%s' failed: %s", name, exc)
            if run_logger:
                run_logger.error("VERIFICATION [%s] FAILED: %s", name, exc)
            result = _error_result(name, str(exc))
        agents[name] = result
        # Emit per-agent result immediately for real-time UI updates
        if on_agent_complete:
            try:
                await on_agent_complete(name, result)
            except Exception as cb_err:
                logger.warning("on_agent_complete callback error for %s: %s", name, cb_err)
        return result

    # Launch all 3 agents concurrently, each wrapped with notification.
    # reference_data (pre-fetched tool results) is passed to all agents so
    # they can verify claims without making redundant MCP tool calls.
    await asyncio.gather(
        _run_and_notify("reasoning", _run_reasoning_agent(text_to_verify, source_data, run_logger, on_tool_event=_make_agent_tool_cb("reasoning"), reference_data=reference_data)),
        _run_and_notify("factual", _run_factual_agent(text_to_verify, run_logger, on_tool_event=_make_agent_tool_cb("factual"), reference_data=reference_data)),
        _run_and_notify("betting", _run_betting_agent(text_to_verify, run_logger, on_tool_event=_make_agent_tool_cb("betting"), reference_data=reference_data)),
    )

    # Compute overall verdict — worst of (pass < warn < fail < error)
    verdict_priority = {"pass": 0, "warn": 1, "fail": 2, "error": 3}
    verdicts = [agents[n]["verdict"] for n in agent_names]
    worst = max(verdicts, key=lambda v: verdict_priority.get(v, 3))

    elapsed = round(time.time() - start, 2)

    if run_logger:
        run_logger.info("=" * 60)
        run_logger.info(
            "VERIFICATION PIPELINE complete — overall=%s elapsed=%.2fs",
            worst, elapsed,
        )
        for name in agent_names:
            a = agents[name]
            run_logger.info(
                "  [%s] verdict=%s confidence=%.2f issues=%d",
                name, a["verdict"], a["confidence"], len(a.get("issues", [])),
            )
            # Log the agent's summary / verdict reasoning
            summary = a.get("summary", "")
            if summary:
                run_logger.info("  [%s] summary: %s", name, summary)
            # Log each issue found by the agent
            for issue in a.get("issues", []):
                run_logger.info(
                    "  [%s]   • (%s) %s — %s",
                    name,
                    issue.get("severity", "?"),
                    issue.get("claim", ""),
                    issue.get("finding", ""),
                )
        run_logger.info("=" * 60)

    result = {
        "verified": worst == "pass",
        "overall_verdict": worst,
        "elapsed_seconds": elapsed,
        "agents": agents,
    }

    # Store in cache for future lookups on the same text
    _store_audit_cache(text_to_verify, result)

    return result
