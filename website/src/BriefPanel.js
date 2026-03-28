import React, { useEffect, useRef, useState } from "react";
import AnalyzeConversation from "./AnalyzeConversation";

/* ------------------------------------------------------------------ */
/*  Section heading icons based on content                             */
/* ------------------------------------------------------------------ */

const SECTION_ICONS = {
  "market snapshot": "\u{1F4CA}",
  "top value bets": "\u{1F4B0}",
  "value bets": "\u{1F4B0}",
  "best line shopping": "\u{1F6D2}",
  "line shopping": "\u{1F6D2}",
  "arbitrage opportunities": "\u{1F504}",
  "arbitrage": "\u{1F504}",
  "middle opportunities": "\u{1F3AF}",
  "middles": "\u{1F3AF}",
  "stale & suspect lines": "\u26A0\uFE0F",
  "stale lines": "\u26A0\uFE0F",
  "suspect lines": "\u26A0\uFE0F",
  "fair odds & expected value": "\u{1F4B9}",
  "fair odds": "\u{1F4B9}",
  "expected value": "\u{1F4B9}",
  "sportsbook rankings": "\u{1F3C6}",
  "market movements": "\u{1F4C8}",
  "analyst notes": "\u{1F4DD}",
  "summary": "\u{1F4CB}",
  "best lines": "\u2B50",
  "outlier lines": "\u{1F50D}",
  "line outliers": "\u{1F50D}",
};

function getSectionIcon(headingText) {
  const lower = headingText.toLowerCase().trim();
  for (const [key, icon] of Object.entries(SECTION_ICONS)) {
    if (lower.includes(key)) return icon;
  }
  return "\u25B8";
}

/* ------------------------------------------------------------------ */
/*  Inline markdown processor (bold, confidence badges, odds)          */
/* ------------------------------------------------------------------ */

function processInline(text) {
  const parts = [];
  let remaining = text;
  let idx = 0;

  while (remaining.length > 0) {
    // Confidence badges: HIGH, MEDIUM, LOW CONFIDENCE
    const confMatch = remaining.match(
      /\b(HIGH|MEDIUM|LOW)\s*(CONFIDENCE)?\b/
    );
    const boldStart = remaining.indexOf("**");

    // Determine which comes first
    const confPos = confMatch ? remaining.indexOf(confMatch[0]) : -1;

    if (boldStart !== -1 && (confPos === -1 || boldStart < confPos)) {
      const boldEnd = remaining.indexOf("**", boldStart + 2);
      if (boldEnd === -1) {
        parts.push(remaining);
        break;
      }
      if (boldStart > 0) {
        parts.push(remaining.substring(0, boldStart));
      }
      parts.push(
        <strong key={`b-${idx}`}>
          {remaining.substring(boldStart + 2, boldEnd)}
        </strong>
      );
      remaining = remaining.substring(boldEnd + 2);
      idx++;
    } else if (confPos !== -1) {
      if (confPos > 0) {
        parts.push(remaining.substring(0, confPos));
      }
      const level = confMatch[1].toLowerCase();
      parts.push(
        <span key={`conf-${idx}`} className={`bp-confidence bp-confidence--${level}`}>
          {confMatch[0]}
        </span>
      );
      remaining = remaining.substring(confPos + confMatch[0].length);
      idx++;
    } else {
      parts.push(remaining);
      break;
    }
  }
  return parts.length === 1 ? parts[0] : parts;
}

/* ------------------------------------------------------------------ */
/*  Enhanced markdown-to-JSX renderer                                  */
/* ------------------------------------------------------------------ */

function renderBriefMarkdown(text) {
  if (!text) return null;
  const lines = text.split("\n");
  const elements = [];
  let listItems = [];
  let inSection = false;

  const flushList = () => {
    if (listItems.length > 0) {
      elements.push(
        <ul key={`ul-${elements.length}`} className="bp-list">
          {listItems.map((li, i) => (
            <li key={i}>{processInline(li)}</li>
          ))}
        </ul>
      );
      listItems = [];
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    // === CALLOUT: lines starting with CAUTION, CONFIRMED, NOTE, WARNING ===
    if (/^(CAUTION|WARNING|NOTE)[:\s]/i.test(trimmed)) {
      flushList();
      elements.push(
        <div key={`callout-${i}`} className="bp-callout bp-callout--warning">
          <span className="bp-callout-icon">{"\u26A0\uFE0F"}</span>
          <span>{processInline(trimmed)}</span>
        </div>
      );
      continue;
    }
    if (/^CONFIRMED[:\s]/i.test(trimmed)) {
      flushList();
      elements.push(
        <div key={`callout-${i}`} className="bp-callout bp-callout--success">
          <span className="bp-callout-icon">{"\u2705"}</span>
          <span>{processInline(trimmed)}</span>
        </div>
      );
      continue;
    }

    // === HEADINGS ===
    if (trimmed.startsWith("## ")) {
      flushList();
      const headingText = trimmed.substring(3);
      const icon = getSectionIcon(headingText);
      if (inSection) {
        elements.push(
          <div key={`sec-end-${i}`} className="bp-section-divider" />
        );
      }
      inSection = true;
      elements.push(
        <h3 key={`h-${i}`} className="bp-heading">
          <span className="bp-heading-icon">{icon}</span>
          {headingText}
        </h3>
      );
    } else if (trimmed.startsWith("### ")) {
      flushList();
      elements.push(
        <h4 key={`h-${i}`} className="bp-subheading">
          {trimmed.substring(4)}
        </h4>
      );
    } else if (trimmed.startsWith("# ")) {
      flushList();
      elements.push(
        <h2 key={`h-${i}`} className="bp-title-heading">
          {trimmed.substring(2)}
        </h2>
      );

    // === LIST ITEMS ===
    } else if (/^[-*] /.test(trimmed)) {
      listItems.push(trimmed.substring(2));
    } else if (/^\d+\.\s/.test(trimmed)) {
      listItems.push(trimmed.replace(/^\d+\.\s/, ""));

    // === BLANK LINES / PARAGRAPHS ===
    } else {
      flushList();
      if (trimmed === "") {
        if (
          elements.length > 0 &&
          elements[elements.length - 1]?.type !== "br"
        ) {
          elements.push(<div key={`sp-${i}`} className="bp-spacer" />);
        }
      } else {
        elements.push(
          <p key={`p-${i}`} className="bp-paragraph">
            {processInline(trimmed)}
          </p>
        );
      }
    }
  }
  flushList();
  return elements;
}

/* ------------------------------------------------------------------ */
/*  Audit Badge — shows audit agent results                              */
/* ------------------------------------------------------------------ */

const VERDICT_CONFIG = {
  pass:  { icon: "\u2705", label: "Verified",   className: "vb--pass"  },
  warn:  { icon: "\u26A0\uFE0F", label: "Warnings",  className: "vb--warn"  },
  fail:  { icon: "\u274C", label: "Issues",     className: "vb--fail"  },
  error: { icon: "\u2753", label: "Error",      className: "vb--error" },
};

const AGENT_LABELS = {
  reasoning: "Reasoning",
  factual:   "Fact Check",
  betting:   "Bet Quality",
};

const ALL_AGENT_NAMES = ["reasoning", "factual", "betting"];

function VerificationBadge({ verification, streaming = false }) {
  const [expanded, setExpanded] = useState(false);

  if (!verification) return null;

  const { overall_verdict, agents, elapsed_seconds } = verification;
  const isPending = streaming || verification._pending;
  const cfg = isPending
    ? { icon: "\u23F3", label: "Auditing\u2026", className: "vb--pending" }
    : (VERDICT_CONFIG[overall_verdict] || VERDICT_CONFIG.error);

  // Count how many agents have completed
  const completedAgents = agents ? Object.keys(agents) : [];
  const completedCount = completedAgents.length;
  const totalAgents = ALL_AGENT_NAMES.length;

  // Compute overall failure percentage across completed agents
  let overallTotal = 0;
  let overallFailed = 0;
  if (agents) {
    Object.values(agents).forEach((a) => {
      if (a.checks_total > 0) {
        overallTotal += a.checks_total;
        overallFailed += a.checks_failed || 0;
      }
    });
  }
  const overallFailPct = overallTotal > 0 ? ((overallFailed / overallTotal) * 100).toFixed(0) : null;

  // Auto-expand when streaming so users see agents arriving in real time
  const effectiveExpanded = expanded || (isPending && completedCount > 0);

  return (
    <div className={`vb-container ${cfg.className}`}>
      <button
        className="vb-summary-btn"
        onClick={() => setExpanded(!expanded)}
        title="Click to expand audit details"
      >
        <span className="vb-icon">{cfg.icon}</span>
        <span className="vb-label">{cfg.label}</span>
        {isPending && (
          <span className="vb-progress">
            {completedCount}/{totalAgents} agents
          </span>
        )}
        {!isPending && overallFailPct !== null && (
          <span className={`vb-fail-pct ${overallFailed === 0 ? "vb-fail-pct--zero" : "vb-fail-pct--nonzero"}`}>
            {overallFailed === 0 ? "0% failed" : `${overallFailPct}% failed`}
          </span>
        )}
        <span className="vb-agents-mini">
          {ALL_AGENT_NAMES.map((name) => {
            const a = agents && agents[name];
            if (a) {
              const ac = VERDICT_CONFIG[a.verdict] || VERDICT_CONFIG.error;
              const agentFailPct = a.checks_total > 0
                ? ((a.checks_failed || 0) / a.checks_total * 100).toFixed(0)
                : null;
              return (
                <span
                  key={name}
                  className="vb-agent-dot vb-agent-dot--done"
                  title={`${AGENT_LABELS[name]}: ${a.verdict}${agentFailPct !== null ? ` (${agentFailPct}% failed)` : ""}`}
                >
                  {ac.icon}
                </span>
              );
            }
            // Pending agent — show spinner dot
            return (
              <span
                key={name}
                className="vb-agent-dot vb-agent-dot--pending"
                title={`${AGENT_LABELS[name]}: running…`}
              >
                <span className="vb-agent-spinner" />
              </span>
            );
          })}
        </span>
        {elapsed_seconds != null && (
          <span className="vb-elapsed">{elapsed_seconds.toFixed(1)}s</span>
        )}
        <span className={`vb-chevron ${effectiveExpanded ? "vb-chevron--open" : ""}`}>{"\u25BC"}</span>
      </button>

      {effectiveExpanded && (
        <div className="vb-details">
          {ALL_AGENT_NAMES.map((name) => {
            const agent = agents && agents[name];

            // Pending agent placeholder — show live tool call activity
            if (!agent) {
              const toolEvents = verification?._toolEvents?.[name] || [];
              const toolCalls = toolEvents.filter(e => e.type === "tool_call");
              const toolResults = toolEvents.filter(e => e.type === "tool_result");
              const latestCall = toolCalls[toolCalls.length - 1];
              const toolName = latestCall?.name || latestCall?.tool_name || null;
              return (
                <div key={name} className="vb-agent vb--pending">
                  <div className="vb-agent-header">
                    <span className="vb-agent-icon"><span className="vb-agent-spinner" /></span>
                    <span className="vb-agent-name">{AGENT_LABELS[name] || name}</span>
                    <span className="vb-agent-verdict vb-agent-verdict--pending">RUNNING</span>
                    {toolCalls.length > 0 && (
                      <span className="vb-tool-count">{toolResults.length}/{toolCalls.length} tools</span>
                    )}
                  </div>
                  {toolName ? (
                    <p className="vb-agent-summary vb-agent-summary--pending vb-tool-live">
                      {toolResults.length < toolCalls.length ? "Calling" : "Called"}{" "}
                      <code>{toolName}</code>
                      {toolCalls.length > 1 && (
                        <span className="vb-tool-history">
                          {" "}— previous: {toolCalls.slice(0, -1).map(tc => tc.name || tc.tool_name).filter(Boolean).join(", ")}
                        </span>
                      )}
                    </p>
                  ) : (
                    <p className="vb-agent-summary vb-agent-summary--pending">
                      Verifying with MCP tools…
                    </p>
                  )}
                </div>
              );
            }

            const ac = VERDICT_CONFIG[agent.verdict] || VERDICT_CONFIG.error;
            const agentTotal = agent.checks_total || 0;
            const agentFailed = agent.checks_failed || 0;
            const agentFailPct = agentTotal > 0 ? ((agentFailed / agentTotal) * 100).toFixed(0) : null;
            return (
              <div key={name} className={`vb-agent ${ac.className} vb-agent--animate-in`}>
                <div className="vb-agent-header">
                  <span className="vb-agent-icon">{ac.icon}</span>
                  <span className="vb-agent-name">{AGENT_LABELS[name] || name}</span>
                  <span className="vb-agent-verdict">{(agent.verdict || "unknown").toUpperCase()}</span>
                  {agentFailPct !== null && (
                    <span className={`vb-agent-fail-pct ${agentFailed === 0 ? "vb-agent-fail-pct--zero" : "vb-agent-fail-pct--nonzero"}`}>
                      {agentFailed}/{agentTotal} failed ({agentFailPct}%)
                    </span>
                  )}
                  {agent.confidence != null && (
                    <span className="vb-agent-confidence">
                      {(agent.confidence * 100).toFixed(0)}% confidence
                    </span>
                  )}
                </div>
                {agent.summary && (
                  <p className="vb-agent-summary">{agent.summary}</p>
                )}
                {agent.issues && agent.issues.length > 0 && (
                  <ul className="vb-issues">
                    {agent.issues.map((issue, idx) => (
                      <li key={idx} className={`vb-issue vb-issue--${issue.severity}`}>
                        <span className="vb-issue-severity">
                          {issue.severity === "error" ? "\u274C" : issue.severity === "warning" ? "\u26A0\uFE0F" : "\u2139\uFE0F"}
                        </span>
                        <div>
                          <strong className="vb-issue-claim">{issue.claim}</strong>
                          <span className="vb-issue-finding">{issue.finding}</span>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
                {agent.ai_meta && (
                  <div className="vb-agent-meta">
                    {agent.ai_meta.provider} &middot; {agent.ai_meta.model} &middot; {agent.ai_meta.elapsed_seconds?.toFixed(1)}s
                  </div>
                )}
                {agent.conversation && (
                  <AnalyzeConversation
                    analyzeResult={{ conversation: agent.conversation, ai_meta: agent.ai_meta }}
                    title={`${AGENT_LABELS[name] || name} Agent Conversation`}
                  />
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  BriefPanel component                                               */
/* ------------------------------------------------------------------ */

export default function BriefPanel({ briefResult, isInterim = false }) {
  const bodyRef = useRef(null);

  const brief_text = briefResult?.brief_text || null;
  const generated_at = briefResult?.generated_at;
  const ai_meta = briefResult?.ai_meta;

  // Auto-scroll to bottom as streaming content arrives
  useEffect(() => {
    if (isInterim && bodyRef.current) {
      const el = bodyRef.current;
      // Only auto-scroll if user is near the bottom
      const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 150;
      if (isNearBottom) {
        el.scrollTop = el.scrollHeight;
      }
    }
  }, [brief_text, isInterim]);

  if (!brief_text) return null;

  const fmtDate = generated_at
    ? new Date(generated_at).toLocaleString()
    : "Unknown";

  return (
    <div className={`bp-container${isInterim ? " bp-container--interim" : ""}`}>
      {/* Header */}
      <div className="bp-header">
        <div className="bp-header-left">
          <div className="bp-header-title-row">
            <span className="bp-header-icon">{isInterim ? "\u{1F4E1}" : "\u{1F4CA}"}</span>
            <h2>{isInterim ? "Live Briefing" : "Daily Market Briefing"}</h2>
          </div>
          <span className="bp-timestamp">
            {isInterim ? "Streaming live..." : `Generated ${fmtDate}`}
          </span>
          {isInterim && (
            <div className="bp-interim-badge">
              <span className="bp-pulse-dot" />
              Receiving data from AI analyst
            </div>
          )}
        </div>
        {ai_meta && (
          <div className="bp-meta">
            <span className="bp-meta-provider">{ai_meta.provider}</span>
            <span className="bp-meta-model">{ai_meta.model}</span>
            {ai_meta.elapsed_seconds != null && (
              <span className="bp-meta-time">
                {ai_meta.elapsed_seconds.toFixed(1)}s
              </span>
            )}
          </div>
        )}
      </div>

      {/* Brief text body */}
      <div className="bp-body" ref={bodyRef}>
        <div className="bp-content">
          {renderBriefMarkdown(brief_text)}
          {isInterim && <span className="bp-cursor" />}
        </div>
      </div>
    </div>
  );
}

export { VerificationBadge };
