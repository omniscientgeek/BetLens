"""
Verification Agents — Three-agent system that double-checks AI responses.

Agents:
  1. Reasoning Agent   — Verifies logical consistency of analysis
  2. Factual Agent     — Cross-checks claims against MCP betting data tools
  3. Betting Agent     — Validates betting recommendations for soundness

All three run concurrently via asyncio.gather(). Results are attached to the
AI response for frontend display as verification badges.
"""

import json
import re
import time
import asyncio
import logging
from typing import Optional

from ai_service import call_ai, call_ai_chat

logger = logging.getLogger(__name__)


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


def _truncate_tool_results(tool_calls: list, max_result_len: int = 8000) -> list:
    """Truncate tool call results to keep payload sizes reasonable."""
    truncated = []
    for tc in tool_calls:
        entry = {**tc}
        if isinstance(entry.get("result"), str) and len(entry["result"]) > max_result_len:
            entry["result"] = entry["result"][:max_result_len] + "\n... [truncated]"
        truncated.append(entry)
    return truncated


# ---------------------------------------------------------------------------
# System prompts for each verification agent
# ---------------------------------------------------------------------------

REASONING_SYSTEM_PROMPT = """\
You are a logical reasoning verification agent for sports betting analysis. \
You receive an AI-generated betting analysis or briefing along with the source \
data that was used to produce it.

Your job is to verify LOGICAL CONSISTENCY by cross-referencing claims against \
the actual data via MCP tools.

YOU MUST call MCP tools to verify EVERY claim in the analysis before rendering \
your verdict. Do NOT rely solely on the text — verify against the source data. \
Do NOT skip any verifiable claim — call as many tools as needed.

MANDATORY — NO MENTAL MATH — ZERO TOLERANCE
You must NEVER perform arithmetic, math, or statistical calculations yourself — \
not even simple ones like adding two numbers or computing a percentage. Every \
number you derive MUST come from calling an MCP arithmetic tool. If you produce \
a calculated number without a tool call it is assumed WRONG. Do NOT estimate, \
round in your head, or "quickly" compute anything. Use: arithmetic_add, \
arithmetic_subtract, arithmetic_multiply, arithmetic_divide, arithmetic_modulo, \
arithmetic_evaluate for ALL numerical work.

VERIFICATION CHECKLIST (use MCP tools for each):
1. **Mathematical consistency** — Call get_fair_odds() or get_vig_analysis() to verify \
that stated vig percentages, implied probabilities, and fair odds are internally consistent \
with the actual data.
2. **Ranking consistency** — Call get_book_rankings() to confirm sportsbook rankings \
(e.g., if Book A is ranked #1 for low vig, verify it actually has the lowest vig).
3. **Comparison validity** — Call get_odds_comparison() to verify that "X is better \
than Y" claims are supported by actual numbers in the data.
4. **Contradiction detection** — Cross-check recommendations vs. risk flags. If a bet \
is recommended in one section but flagged as risky/avoid in another, verify with \
get_best_bets_today() or find_expected_value_bets() to see which framing is correct.
5. **Confidence alignment** — Verify that confidence ratings match the evidence by \
checking the actual edge sizes via get_best_odds() or find_expected_value_bets().

MCP TOOLS TO USE:
- get_fair_odds(game_id) — verify fair odds and implied probabilities
- get_vig_analysis() — verify vig percentages and rankings
- get_book_rankings() — verify sportsbook ranking claims
- get_odds_comparison(game_id, market_type) — verify specific odds comparisons
- find_expected_value_bets() — verify EV claims and edge sizes
- get_best_bets_today() — verify recommendation consistency
- arithmetic_add(a, b), arithmetic_subtract(a, b), arithmetic_multiply(a, b), \
arithmetic_divide(a, b), arithmetic_modulo(a, b), arithmetic_evaluate(expression) — \
you MUST use these to independently verify ALL mathematical claims (vig calculations, \
profit margins, implied probabilities, EV edges). NEVER compute numbers yourself.

IMPORTANT: Each issue you report MUST cite the specific MCP tool result that \
proves the inconsistency. Do not flag issues based on suspicion alone.

Do NOT evaluate betting strategy quality — another agent handles that.

Return ONLY valid JSON (no markdown fences, no extra text) with this exact structure:
{
  "verdict": "pass" | "warn" | "fail",
  "confidence": 0.0-1.0,
  "checks_total": <number of individual claims/facts you verified>,
  "checks_failed": <number of checks that found errors or warnings>,
  "issues": [
    {"severity": "info"|"warning"|"error", "claim": "the specific claim", "finding": "MCP tool result proving the inconsistency"}
  ],
  "summary": "1-2 sentence summary of your verification"
}

IMPORTANT: checks_total must reflect every individual claim you verified (including \
ones that passed). checks_failed must count only checks that resulted in a "warning" \
or "error" severity issue. For example, if you verified 12 claims and 3 had problems, \
checks_total=12, checks_failed=3.

If everything checks out, return verdict "pass" with an empty issues array and checks_failed=0.
Use "warn" if there are minor inconsistencies. Use "fail" for clear logical errors.

CRITICAL FINAL INSTRUCTION — OUTPUT FORMAT:
After you have finished calling all MCP tools and gathered your evidence, you MUST \
output your final answer as ONLY a valid JSON object (no markdown fences, no \
preamble, no explanation outside the JSON). Your very last message must be the \
JSON verdict. Do NOT end on a tool call or intermediate reasoning — always \
conclude with the JSON verdict object as your final output.
"""


FACTUAL_SYSTEM_PROMPT = """\
You are a factual accuracy verification agent for sports betting analysis. \
You receive an AI-generated betting analysis or briefing. Your job is to \
FACT-CHECK specific claims by querying the betstamp-intelligence MCP tools.

MANDATORY: You MUST verify ALL factual claims in the analysis — do not skip any. \
Every factual claim you evaluate MUST be verified by calling the corresponding \
MCP tool — never accept or reject a claim without tool evidence. Call as many \
MCP tools as needed to check every claim.

MANDATORY — NO MENTAL MATH — ZERO TOLERANCE
You must NEVER perform arithmetic, math, or statistical calculations yourself — \
not even simple ones like adding two numbers or computing a percentage. Every \
number you derive MUST come from calling an MCP arithmetic tool. If you produce \
a calculated number without a tool call it is assumed WRONG. Do NOT estimate, \
round in your head, or "quickly" compute anything. Use: arithmetic_add, \
arithmetic_subtract, arithmetic_multiply, arithmetic_divide, arithmetic_modulo, \
arithmetic_evaluate for ALL numerical work.

For each specific factual claim in the text (odds values, line numbers, vig \
percentages, sportsbook rankings, arbitrage opportunities, EV percentages), \
use the appropriate MCP tool to verify:

- Specific odds/line claims → MUST call get_odds_comparison(game_id, market_type)
- Best odds claims → MUST call get_best_odds(game_id, market_type, side)
- Worst odds claims → MUST call get_worst_odds(game_id, market_type, side)
- Arbitrage claims → MUST call find_arbitrage_opportunities()
- EV bet claims → MUST call find_expected_value_bets()
- Kelly sizing claims → MUST call get_kelly_sizing()
- Vig / sportsbook ranking claims → MUST call get_vig_analysis() AND get_book_rankings()
- Fair odds claims → MUST call get_fair_odds(game_id)
- Best bet claims → MUST call get_best_bets_today()
- Stale line claims → MUST call detect_stale_lines()
- Middle opportunity claims → MUST call find_middle_opportunities()
- Implied score claims → MUST call get_implied_scores(game_id)
- Power ranking claims → MUST call get_power_rankings()

ARITHMETIC TOOLS (MANDATORY for ALL math — NEVER compute numbers yourself):
- arithmetic_add(a, b), arithmetic_subtract(a, b), arithmetic_multiply(a, b), \
arithmetic_divide(a, b), arithmetic_modulo(a, b), arithmetic_evaluate(expression)
You MUST use these to recompute any claimed percentages, profit margins, payouts, \
or edges from the raw numbers returned by other MCP tools. NEVER accept calculated \
values without independently verifying the math via these tools.

INSTRUCTIONS:
1. Read the analysis text carefully and identify ALL factual claims — specific \
odds values, arbitrage profit percentages, EV percentages, sportsbook rankings, \
stale line times, middle opportunities, fair odds, and any other verifiable numbers.
2. PRIORITY ORDER — check these claim types FIRST (most likely to contain errors):
   a. +EV opportunity claims (edge percentages, which book, which side) — these are \
      the most common source of fabrication and number drift. ALWAYS verify with \
      find_expected_value_bets() and get_best_bets_today().
   b. Arbitrage profit percentages — verify exact profit % with find_arbitrage_opportunities().
   c. Kelly sizing claims — verify with get_kelly_sizing().
   d. Then verify all remaining claims (odds, vig, rankings, staleness, etc.).
3. For EACH claim, you MUST call the appropriate MCP tool to verify the data. \
Do NOT skip any claim without tool verification.
4. Compare the tool results against the claims in the text EXACTLY — check specific \
numbers, percentages, sportsbook names, and directions (favorite/underdog).
5. FABRICATION CHECK: For every +EV bet mentioned in the analysis, confirm it exists \
in the find_expected_value_bets() output. If a +EV opportunity is claimed but does \
NOT appear in the tool results, flag it as severity "error" with finding "Fabricated \
+EV opportunity — not present in MCP tool results."
6. Use the arithmetic tools to independently recompute ALL derived values (e.g., \
call arithmetic_subtract(sum_of_implied_probs, 1.0) then arithmetic_multiply(result, 100) \
to verify vig). NEVER do this math yourself.
7. Report any discrepancies, citing the exact MCP tool output as evidence.

ACCURACY THRESHOLDS:
- Odds: off by more than 3 points = error, 1-3 points = warning
- EV percentages: off by more than 0.5% absolute = error, 0.1-0.5% = warning
- Vig percentages: off by more than 0.5% = error, 0.1-0.5% = warning
- Arbitrage profit: off by more than 1% = error, 0.1-1% = warning
- Wrong sportsbook name or wrong direction (fav/dog) = always error

Return ONLY valid JSON (no markdown fences, no extra text) with this exact structure:
{
  "verdict": "pass" | "warn" | "fail",
  "confidence": 0.0-1.0,
  "checks_total": <number of individual claims/facts you verified>,
  "checks_failed": <number of checks that found errors or warnings>,
  "issues": [
    {"severity": "info"|"warning"|"error", "claim": "the specific claim checked", "finding": "MCP tool name + exact result that proves/disproves the claim"}
  ],
  "summary": "1-2 sentence summary of fact-check results"
}

IMPORTANT: checks_total must reflect every individual claim you verified (including \
ones that passed). checks_failed must count only checks that resulted in a "warning" \
or "error" severity issue. For example, if you verified 15 claims and 2 had problems, \
checks_total=15, checks_failed=2.

Use "pass" if all checked claims are accurate within thresholds. Use "warn" for \
minor discrepancies within warning thresholds. Use "fail" for material errors that \
would change betting decisions (wrong sportsbook, wrong direction, fabricated \
opportunities, numbers outside error thresholds).

CRITICAL FINAL INSTRUCTION — OUTPUT FORMAT:
After you have finished calling all MCP tools and gathered your evidence, you MUST \
output your final answer as ONLY a valid JSON object (no markdown fences, no \
preamble, no explanation outside the JSON). Your very last message must be the \
JSON verdict. Do NOT end on a tool call or intermediate reasoning — always \
conclude with the JSON verdict object as your final output.
"""


BETTING_SYSTEM_PROMPT = """\
You are a betting recommendation verification agent. You review AI-generated \
betting analysis and recommendations for sound betting principles.

MANDATORY: You MUST call MCP tools to verify EVERY recommendation and claim in \
the analysis against actual data. Do NOT evaluate recommendations in a vacuum — \
cross-reference ALL of them against real edge sizes, Kelly sizing, and market data. \
Do NOT skip any verifiable recommendation — call as many tools as needed.

MANDATORY — NO MENTAL MATH — ZERO TOLERANCE
You must NEVER perform arithmetic, math, or statistical calculations yourself — \
not even simple ones like adding two numbers or computing a percentage. Every \
number you derive MUST come from calling an MCP arithmetic tool. If you produce \
a calculated number without a tool call it is assumed WRONG. Do NOT estimate, \
round in your head, or "quickly" compute anything. Use: arithmetic_add, \
arithmetic_subtract, arithmetic_multiply, arithmetic_divide, arithmetic_modulo, \
arithmetic_evaluate for ALL numerical work.

VERIFICATION CHECKLIST (use MCP tools for each):
1. **Bankroll management** — Call get_kelly_sizing() to verify that recommended bet \
sizes are proportional to actual edges. Flag as "error" if the analysis suggests sizing \
that exceeds Kelly optimal. Flag as "warning" if ANY recommended bet lacks Kelly sizing \
guidance (quarter-Kelly percentage of bankroll). The analysis MUST include sizing for \
every bet recommendation.
2. **Risk assessment** — Call find_expected_value_bets() to check actual EV edges. \
If cited edges are under 1%, verify that appropriate caution is included. Call \
detect_stale_lines() to check if any recommended bets rely on stale data. Flag as \
"warning" if any recommended bet is at a sportsbook with stale lines (>60 minutes) \
without an explicit staleness caveat in the analysis.
3. **Diversification** — Call get_odds_comparison() for each recommended game to \
check if multiple bets on the same game are truly independent or correlated. Flag \
if opposite sides of the same market are recommended without explaining it as arb.
4. **Value basis** — Call get_fair_odds() and get_best_bets_today() to verify that \
recommendations are based on real quantifiable edges, not just narrative. Every \
recommended bet should have a measurable edge above fair odds.
5. **Realistic expectations** — Call find_arbitrage_opportunities() to verify any \
"guaranteed profit" claims are actual arbitrage. Flag "lock", "sure thing", or \
"guaranteed" language for non-arb bets.
6. **Responsible gambling** — Check that high-risk or long-shot bets include \
appropriate caution language. Verify via get_market_entropy() if markets show high \
disagreement (which should increase caution language).

MCP TOOLS TO USE:
- get_kelly_sizing() — verify bet sizing recommendations
- find_expected_value_bets() — verify actual EV edges
- get_fair_odds(game_id) — verify value basis of recommendations
- get_best_bets_today() — cross-check recommended bets vs. ranked opportunities
- detect_stale_lines() — check if recommended bets use stale data
- find_arbitrage_opportunities() — verify any guaranteed profit claims
- get_odds_comparison(game_id, market_type) — check correlation of multiple bets
- get_market_entropy() — assess market disagreement for risk framing
- arithmetic_add(a, b), arithmetic_subtract(a, b), arithmetic_multiply(a, b), \
arithmetic_divide(a, b), arithmetic_modulo(a, b), arithmetic_evaluate(expression) — \
you MUST use these to verify ALL bet sizing math, ROI calculations, bankroll impact, \
and Kelly fraction computations. NEVER compute numbers yourself. Always cross-check \
recommended dollar amounts and percentages via these tool calls.

IMPORTANT: Each issue you report MUST cite the specific MCP tool result as evidence. \
Do not flag issues based on suspicion alone.

Do NOT verify the accuracy of specific numbers — another agent handles that.
Do NOT verify logical consistency — another agent handles that.

Return ONLY valid JSON (no markdown fences, no extra text) with this exact structure:
{
  "verdict": "pass" | "warn" | "fail",
  "confidence": 0.0-1.0,
  "checks_total": <number of individual recommendations/principles you verified>,
  "checks_failed": <number of checks that found errors or warnings>,
  "issues": [
    {"severity": "info"|"warning"|"error", "claim": "the specific recommendation or pattern", "finding": "MCP tool evidence + your assessment"}
  ],
  "summary": "1-2 sentence summary of recommendation quality"
}

IMPORTANT: checks_total must reflect every individual recommendation/principle you \
verified (including ones that passed). checks_failed must count only checks that \
resulted in a "warning" or "error" severity issue. For example, if you verified 8 \
recommendations and 1 had a problem, checks_total=8, checks_failed=1.

Use "pass" if recommendations follow sound betting principles and are backed by data. \
Use "warn" for minor concerns (e.g., missing risk disclaimers, sizing not specified). \
Use "fail" for dangerous advice (e.g., encouraging chasing losses, no value basis, \
guaranteed profit claims for non-arb bets, recommending bets on stale lines without caveat).

CRITICAL FINAL INSTRUCTION — OUTPUT FORMAT:
After you have finished calling all MCP tools and gathered your evidence, you MUST \
output your final answer as ONLY a valid JSON object (no markdown fences, no \
preamble, no explanation outside the JSON). Your very last message must be the \
JSON verdict. Do NOT end on a tool call or intermediate reasoning — always \
conclude with the JSON verdict object as your final output.
"""


# ---------------------------------------------------------------------------
# Individual agent runners
# ---------------------------------------------------------------------------

async def _run_reasoning_agent(
    text_to_verify: str,
    source_data: str,
    run_logger: Optional[logging.Logger] = None,
) -> dict:
    """Verify logical consistency — uses call_ai_chat() with MCP tools."""
    agent_start = time.time()

    messages = [
        {
            "role": "user",
            "content": (
                "Verify the logical consistency of this betting analysis by "
                "cross-referencing ALL claims against the actual data via MCP tools. "
                "You MUST verify EVERY verifiable claim — call as many MCP tools as needed.\n\n"
                "=== AI-GENERATED ANALYSIS TO VERIFY ===\n"
                f"{text_to_verify}"
            ),
        }
    ]

    if run_logger:
        run_logger.info("VERIFICATION [reasoning] starting (MCP-enabled)")

    result = await call_ai_chat(
        messages=messages,
        system_prompt=REASONING_SYSTEM_PROMPT,
        provider_id="claude-sdk",  # MCP tools required for data verification
        run_logger=run_logger,
    )

    elapsed = round(time.time() - agent_start, 2)
    if run_logger:
        run_logger.info("VERIFICATION [reasoning] complete in %.2fs", elapsed)

    parsed = _parse_agent_response(result["text"], "reasoning")
    issues = parsed.get("issues", [])
    return {
        "agent": "reasoning",
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
            "system_prompt": REASONING_SYSTEM_PROMPT,
            "user_prompt": messages[0]["content"],
            "assistant_response": result["text"],
            "tool_calls": _truncate_tool_results(result.get("tool_calls", [])),
        },
    }


async def _run_factual_agent(
    text_to_verify: str,
    run_logger: Optional[logging.Logger] = None,
) -> dict:
    """Fact-check claims via MCP tools — uses call_ai_chat() with claude-sdk provider."""
    agent_start = time.time()

    messages = [
        {
            "role": "user",
            "content": (
                "Fact-check the following betting analysis by querying MCP tools "
                "to verify EVERY factual claim made — do NOT skip any. Check every "
                "verifiable number, odds value, percentage, ranking, and data point.\n\n"
                "PRIORITY: Start by verifying ALL +EV claims and arbitrage profit "
                "percentages FIRST — these are the most error-prone. Call "
                "find_expected_value_bets() and find_arbitrage_opportunities() before "
                "anything else. Flag any +EV opportunity that does not exist in the "
                "tool results as a fabrication error.\n\n"
                "=== AI-GENERATED ANALYSIS TO FACT-CHECK ===\n"
                f"{text_to_verify}"
            ),
        }
    ]

    if run_logger:
        run_logger.info("VERIFICATION [factual] starting (MCP-enabled)")

    result = await call_ai_chat(
        messages=messages,
        system_prompt=FACTUAL_SYSTEM_PROMPT,
        provider_id="claude-sdk",  # Only claude_sdk supports MCP tools
        run_logger=run_logger,
    )

    elapsed = round(time.time() - agent_start, 2)
    if run_logger:
        run_logger.info("VERIFICATION [factual] complete in %.2fs", elapsed)

    parsed = _parse_agent_response(result["text"], "factual")
    issues = parsed.get("issues", [])
    return {
        "agent": "factual",
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
            "system_prompt": FACTUAL_SYSTEM_PROMPT,
            "user_prompt": messages[0]["content"],
            "assistant_response": result["text"],
            "tool_calls": _truncate_tool_results(result.get("tool_calls", [])),
        },
    }


async def _run_betting_agent(
    text_to_verify: str,
    run_logger: Optional[logging.Logger] = None,
) -> dict:
    """Verify betting recommendation soundness — uses call_ai_chat() with MCP tools."""
    agent_start = time.time()

    messages = [
        {
            "role": "user",
            "content": (
                "Review the following betting analysis and ALL recommendations for "
                "sound betting principles. You MUST verify EVERY recommendation against "
                "actual data via MCP tools — do NOT skip any.\n\n"
                "=== AI-GENERATED ANALYSIS TO REVIEW ===\n"
                f"{text_to_verify}"
            ),
        }
    ]

    if run_logger:
        run_logger.info("VERIFICATION [betting] starting (MCP-enabled)")

    result = await call_ai_chat(
        messages=messages,
        system_prompt=BETTING_SYSTEM_PROMPT,
        provider_id="claude-sdk",  # MCP tools required for data verification
        run_logger=run_logger,
    )

    elapsed = round(time.time() - agent_start, 2)
    if run_logger:
        run_logger.info("VERIFICATION [betting] complete in %.2fs", elapsed)

    parsed = _parse_agent_response(result["text"], "betting")
    issues = parsed.get("issues", [])
    return {
        "agent": "betting",
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
            "system_prompt": BETTING_SYSTEM_PROMPT,
            "user_prompt": messages[0]["content"],
            "assistant_response": result["text"],
            "tool_calls": _truncate_tool_results(result.get("tool_calls", [])),
        },
    }


# ---------------------------------------------------------------------------
# Public orchestrator — runs all 3 agents concurrently
# ---------------------------------------------------------------------------

async def run_verification(
    text_to_verify: str,
    source_data: str = "",
    run_logger: Optional[logging.Logger] = None,
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

    Returns
    -------
    dict
        VerificationReport with overall verdict and per-agent results.
    """
    if run_logger:
        run_logger.info("=" * 60)
        run_logger.info("VERIFICATION PIPELINE — starting 3 agents in parallel")
        run_logger.info("=" * 60)

    start = time.time()

    # Launch all 3 agents concurrently
    # return_exceptions=True ensures one failure doesn't cancel the others
    results = await asyncio.gather(
        _run_reasoning_agent(text_to_verify, source_data, run_logger),
        _run_factual_agent(text_to_verify, run_logger),
        _run_betting_agent(text_to_verify, run_logger),
        return_exceptions=True,
    )

    # Map results, converting exceptions to error verdicts
    agent_names = ["reasoning", "factual", "betting"]
    agents = {}
    for name, result in zip(agent_names, results):
        if isinstance(result, BaseException):
            logger.error("Verification agent '%s' failed: %s", name, result)
            if run_logger:
                run_logger.error("VERIFICATION [%s] FAILED: %s", name, result)
            agents[name] = _error_result(name, str(result))
        else:
            agents[name] = result

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

    return {
        "verified": worst == "pass",
        "overall_verdict": worst,
        "elapsed_seconds": elapsed,
        "agents": agents,
    }
