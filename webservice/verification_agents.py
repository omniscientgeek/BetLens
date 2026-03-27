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
    logger.warning("Could not parse JSON from %s agent response, using fallback", agent_name)
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


# ---------------------------------------------------------------------------
# System prompts for each verification agent
# ---------------------------------------------------------------------------

REASONING_SYSTEM_PROMPT = """\
You are a logical reasoning verification agent for sports betting analysis. \
You receive an AI-generated betting analysis or briefing along with the source \
data that was used to produce it.

Your job is to verify LOGICAL CONSISTENCY only:
1. Do conclusions follow from the stated premises and data?
2. Are comparisons logically valid (e.g., "X is better than Y" supported by the numbers)?
3. Are there contradictions within the text (e.g., recommending a bet in one section \
but flagging it as risky/avoid in another without explanation)?
4. Are confidence ratings consistent with the evidence cited?
5. Are mathematical relationships correct (e.g., if vig is stated as X%, does that \
align with the odds cited)? Check that implied probabilities, vig percentages, and \
fair odds are internally consistent.
6. Are ranking claims consistent (e.g., if Book A is ranked #1 for low vig, it should \
not appear with the highest vig in another section)?

Do NOT verify factual accuracy of specific numbers — another agent handles that.
Do NOT evaluate betting strategy quality — another agent handles that.

Return ONLY valid JSON (no markdown fences, no extra text) with this exact structure:
{
  "verdict": "pass" | "warn" | "fail",
  "confidence": 0.0-1.0,
  "issues": [
    {"severity": "info"|"warning"|"error", "claim": "the specific claim", "finding": "what you found"}
  ],
  "summary": "1-2 sentence summary of your verification"
}

If everything checks out, return verdict "pass" with an empty issues array.
Use "warn" if there are minor inconsistencies. Use "fail" for clear logical errors.
"""


FACTUAL_SYSTEM_PROMPT = """\
You are a factual accuracy verification agent for sports betting analysis. \
You receive an AI-generated betting analysis or briefing. Your job is to \
FACT-CHECK specific claims by querying the betstamp-intelligence MCP tools.

For each specific factual claim in the text (odds values, line numbers, vig \
percentages, sportsbook rankings, arbitrage opportunities, EV percentages), \
use the appropriate MCP tool to verify:

- Specific odds/line claims → use get_odds_comparison(game_id, market_type)
- Best odds claims → use get_best_odds(game_id, market_type, side)
- Arbitrage claims → use find_arbitrage_opportunities()
- EV bet claims → use find_expected_value_bets()
- Vig / sportsbook ranking claims → use get_vig_analysis() or get_book_rankings()
- Fair odds claims → use get_fair_odds(game_id)
- Best bet claims → use get_best_bets_today()

INSTRUCTIONS:
1. Read the analysis text carefully and identify UP TO 5 of the most important \
factual claims — prioritize claims that directly influence betting decisions \
(specific odds, arbitrage profit percentages, EV percentages, sportsbook rankings).
2. For each claim, call the appropriate MCP tool to verify the data.
3. Compare the tool results against the claims in the text.
4. Report any discrepancies.

Return ONLY valid JSON (no markdown fences, no extra text) with this exact structure:
{
  "verdict": "pass" | "warn" | "fail",
  "confidence": 0.0-1.0,
  "issues": [
    {"severity": "info"|"warning"|"error", "claim": "the specific claim checked", "finding": "MCP verification result"}
  ],
  "summary": "1-2 sentence summary of fact-check results"
}

Use "pass" if all checked claims are accurate. Use "warn" for minor discrepancies \
(e.g., odds off by 1-2 points). Use "fail" for material errors that would change \
betting decisions (wrong sportsbook, wrong direction, fabricated opportunities).
"""


BETTING_SYSTEM_PROMPT = """\
You are a betting recommendation verification agent. You review AI-generated \
betting analysis and recommendations for sound betting principles.

Evaluate the recommendations for:
1. **Bankroll management** — Does the analysis avoid suggesting reckless bet sizing? \
Are recommendations proportional to the edge cited?
2. **Risk assessment** — Are risks properly disclosed? Are long-shot bets flagged \
appropriately? Is there acknowledgment that edges can disappear?
3. **Diversification** — Does the analysis avoid over-concentrating on correlated \
bets (e.g., multiple bets on the same game from the same angle)?
4. **Value basis** — Are recommendations based on quantifiable edges (EV, vig \
advantage, fair odds comparison) rather than gut feelings or vague language?
5. **Realistic expectations** — Does the analysis avoid promising guaranteed profits \
(except for true arbitrage with cited numbers)? Are phrases like "lock", "sure thing", \
or "guaranteed" used appropriately?
6. **Responsible gambling** — Are there any red flags for encouraging problem \
gambling behavior? Is there appropriate caution language?

Do NOT verify the accuracy of specific numbers — another agent handles that.
Do NOT verify logical consistency — another agent handles that.

Return ONLY valid JSON (no markdown fences, no extra text) with this exact structure:
{
  "verdict": "pass" | "warn" | "fail",
  "confidence": 0.0-1.0,
  "issues": [
    {"severity": "info"|"warning"|"error", "claim": "the specific recommendation or pattern", "finding": "your assessment"}
  ],
  "summary": "1-2 sentence summary of recommendation quality"
}

Use "pass" if recommendations follow sound betting principles. Use "warn" for \
minor concerns (e.g., missing risk disclaimers). Use "fail" for dangerous advice \
(e.g., encouraging chasing losses, no value basis, guaranteed profit claims for non-arb bets).
"""


# ---------------------------------------------------------------------------
# Individual agent runners
# ---------------------------------------------------------------------------

async def _run_reasoning_agent(
    text_to_verify: str,
    source_data: str,
    run_logger: Optional[logging.Logger] = None,
) -> dict:
    """Verify logical consistency — uses call_ai() (no MCP needed)."""
    agent_start = time.time()

    # Truncate source data to keep token costs manageable
    truncated_source = source_data[:10000] if source_data else "(no source data provided)"

    user_prompt = (
        "Verify the logical consistency of this betting analysis.\n\n"
        "=== AI-GENERATED ANALYSIS TO VERIFY ===\n"
        f"{text_to_verify}\n\n"
        "=== SOURCE DATA USED TO PRODUCE THE ANALYSIS ===\n"
        f"{truncated_source}"
    )

    if run_logger:
        run_logger.info("VERIFICATION [reasoning] starting")

    result = await call_ai(
        system_prompt=REASONING_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        run_logger=run_logger,
    )

    elapsed = round(time.time() - agent_start, 2)
    if run_logger:
        run_logger.info("VERIFICATION [reasoning] complete in %.2fs", elapsed)

    parsed = _parse_agent_response(result["text"], "reasoning")
    return {
        "agent": "reasoning",
        "verdict": parsed.get("verdict", "warn"),
        "confidence": parsed.get("confidence", 0.5),
        "issues": parsed.get("issues", []),
        "summary": parsed.get("summary", ""),
        "ai_meta": {
            "provider": result["provider_name"],
            "model": result["model"],
            "elapsed_seconds": elapsed,
            "usage": result.get("usage", {}),
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
                "to verify the specific claims made. Check up to 5 of the most "
                "important factual claims.\n\n"
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
    return {
        "agent": "factual",
        "verdict": parsed.get("verdict", "warn"),
        "confidence": parsed.get("confidence", 0.5),
        "issues": parsed.get("issues", []),
        "summary": parsed.get("summary", ""),
        "ai_meta": {
            "provider": result["provider_name"],
            "model": result["model"],
            "elapsed_seconds": elapsed,
            "usage": result.get("usage", {}),
        },
    }


async def _run_betting_agent(
    text_to_verify: str,
    run_logger: Optional[logging.Logger] = None,
) -> dict:
    """Verify betting recommendation soundness — uses call_ai() (no MCP needed)."""
    agent_start = time.time()

    user_prompt = (
        "Review the following betting analysis and recommendations for sound "
        "betting principles.\n\n"
        "=== AI-GENERATED ANALYSIS TO REVIEW ===\n"
        f"{text_to_verify}"
    )

    if run_logger:
        run_logger.info("VERIFICATION [betting] starting")

    result = await call_ai(
        system_prompt=BETTING_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        run_logger=run_logger,
    )

    elapsed = round(time.time() - agent_start, 2)
    if run_logger:
        run_logger.info("VERIFICATION [betting] complete in %.2fs", elapsed)

    parsed = _parse_agent_response(result["text"], "betting")
    return {
        "agent": "betting",
        "verdict": parsed.get("verdict", "warn"),
        "confidence": parsed.get("confidence", 0.5),
        "issues": parsed.get("issues", []),
        "summary": parsed.get("summary", ""),
        "ai_meta": {
            "provider": result["provider_name"],
            "model": result["model"],
            "elapsed_seconds": elapsed,
            "usage": result.get("usage", {}),
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
        run_logger.info("=" * 60)

    return {
        "verified": worst == "pass",
        "overall_verdict": worst,
        "elapsed_seconds": elapsed,
        "agents": agents,
    }
