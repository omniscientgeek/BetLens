import React, { useEffect, useRef } from "react";

/* ------------------------------------------------------------------ */
/*  Section heading icons based on content                             */
/* ------------------------------------------------------------------ */

const SECTION_ICONS = {
  "market snapshot": "\u{1F4CA}",
  "top value bets": "\u{1F4B0}",
  "value bets": "\u{1F4B0}",
  "arbitrage opportunities": "\u{1F504}",
  "arbitrage": "\u{1F504}",
  "stale & suspect lines": "\u26A0\uFE0F",
  "stale lines": "\u26A0\uFE0F",
  "suspect lines": "\u26A0\uFE0F",
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
/*  BriefPanel component                                               */
/* ------------------------------------------------------------------ */

export default function BriefPanel({ briefResult, isInterim = false }) {
  if (!briefResult || !briefResult.brief_text) return null;

  const { brief_text, generated_at, ai_meta } = briefResult;
  const bodyRef = useRef(null);

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
