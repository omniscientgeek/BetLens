import React from "react";

/* ------------------------------------------------------------------ */
/*  Simple markdown-to-JSX renderer for the briefing text              */
/* ------------------------------------------------------------------ */

function processInline(text) {
  const parts = [];
  let remaining = text;
  let idx = 0;
  while (remaining.length > 0) {
    const boldStart = remaining.indexOf("**");
    if (boldStart === -1) {
      parts.push(remaining);
      break;
    }
    const boldEnd = remaining.indexOf("**", boldStart + 2);
    if (boldEnd === -1) {
      parts.push(remaining);
      break;
    }
    if (boldStart > 0) {
      parts.push(remaining.substring(0, boldStart));
    }
    parts.push(
      <strong key={`b-${idx}`}>{remaining.substring(boldStart + 2, boldEnd)}</strong>
    );
    remaining = remaining.substring(boldEnd + 2);
    idx++;
  }
  return parts.length === 1 ? parts[0] : parts;
}

function renderBriefMarkdown(text) {
  if (!text) return null;
  const lines = text.split("\n");
  const elements = [];
  let listItems = [];

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

    // Headings
    if (trimmed.startsWith("## ")) {
      flushList();
      elements.push(
        <h3 key={`h-${i}`} className="bp-heading">
          {trimmed.substring(3)}
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
    } else if (/^[-*] /.test(trimmed)) {
      // List item
      listItems.push(trimmed.substring(2));
    } else if (/^\d+\.\s/.test(trimmed)) {
      // Numbered list — treat as bullet for simplicity
      listItems.push(trimmed.replace(/^\d+\.\s/, ""));
    } else {
      flushList();
      if (trimmed === "") {
        // Skip consecutive blank lines
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

export default function BriefPanel({ briefResult }) {
  if (!briefResult || !briefResult.brief_text) return null;

  const { brief_text, generated_at, ai_meta } = briefResult;

  const fmtDate = generated_at
    ? new Date(generated_at).toLocaleString()
    : "Unknown";

  return (
    <div className="bp-container">
      {/* Header */}
      <div className="bp-header">
        <div>
          <h2>Daily Market Briefing</h2>
          <span className="bp-timestamp">Generated {fmtDate}</span>
        </div>
        {ai_meta && (
          <div className="bp-meta">
            <span>{ai_meta.provider}</span>
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
      <div className="bp-body">
        <div className="bp-content">{renderBriefMarkdown(brief_text)}</div>
      </div>
    </div>
  );
}
