import React, { useState } from "react";

/* ------------------------------------------------------------------ */
/*  Token collection — deep aggregation across all pipeline phases     */
/* ------------------------------------------------------------------ */
export function collectTokens(pr) {
  const rows = [];
  let grandIn = 0;
  let grandOut = 0;

  const addRow = (label, meta, toolCalls) => {
    if (!meta?.usage) return;
    const inT = meta.usage.input_tokens || 0;
    const outT = meta.usage.output_tokens || 0;
    grandIn += inT;
    grandOut += outT;
    rows.push({
      label,
      provider: meta.provider,
      model: meta.model,
      elapsed: meta.elapsed_seconds,
      inTokens: inT,
      outTokens: outT,
      toolCalls: toolCalls || 0,
    });
  };

  // Analyze phase
  if (pr.analyze) {
    const tc = pr.analyze.conversation?.tool_calls?.length || 0;
    addRow("Analyze", pr.analyze.ai_meta, tc);
  }

  // Audit Analyze agents
  const auditAnalyze = pr.analyze?.verification || pr.audit_analyze;
  if (auditAnalyze?.agents) {
    for (const [name, agent] of Object.entries(auditAnalyze.agents)) {
      const tc = agent.conversation?.tool_calls?.length || 0;
      addRow(`Audit Analyze \u203A ${name}`, agent.ai_meta, tc);
    }
  }

  // Analyze fix history
  const analyzeHistory = (pr.analyze?.verification || pr.audit_analyze)?.fix_history || [];
  analyzeHistory.forEach((h) => {
    if (h.type === "fix" && h.ai_meta) {
      const tc = h.conversation?.tool_calls?.length || 0;
      addRow(`Analyze Fix #${h.attempt}`, h.ai_meta, tc);
    }
    if (h.type !== "fix" && h.verification?.agents) {
      for (const [name, agent] of Object.entries(h.verification.agents)) {
        const tc = agent.conversation?.tool_calls?.length || 0;
        addRow(`Re-audit #${h.attempt} \u203A ${name}`, agent.ai_meta, tc);
      }
    }
  });

  // Brief phase
  if (pr.brief) {
    addRow("Brief", pr.brief.ai_meta, 0);
  }

  // Audit Brief agents
  const auditBrief = pr.brief?.verification || pr.audit_brief;
  if (auditBrief?.agents) {
    for (const [name, agent] of Object.entries(auditBrief.agents)) {
      const tc = agent.conversation?.tool_calls?.length || 0;
      addRow(`Audit Brief \u203A ${name}`, agent.ai_meta, tc);
    }
  }

  // Brief fix history
  const briefHistory = (pr.brief?.verification || pr.audit_brief)?.fix_history || [];
  briefHistory.forEach((h) => {
    if (h.type === "fix" && h.ai_meta) {
      const tc = h.conversation?.tool_calls?.length || 0;
      addRow(`Brief Fix #${h.attempt}`, h.ai_meta, tc);
    }
    if (h.type !== "fix" && h.verification?.agents) {
      for (const [name, agent] of Object.entries(h.verification.agents)) {
        const tc = agent.conversation?.tool_calls?.length || 0;
        addRow(`Re-audit #${h.attempt} \u203A ${name}`, agent.ai_meta, tc);
      }
    }
  });

  return { rows, grandIn, grandOut, grandTotal: grandIn + grandOut };
}

/* ------------------------------------------------------------------ */
/*  TokenUsageSummary component                                        */
/* ------------------------------------------------------------------ */
export default function TokenUsageSummary({ pipelineResults }) {
  const [expanded, setExpanded] = useState(false);
  if (!pipelineResults) return null;

  const { rows, grandIn, grandOut, grandTotal } = collectTokens(pipelineResults);
  if (rows.length === 0) return null;

  const totalToolCalls = rows.reduce((s, r) => s + r.toolCalls, 0);

  return (
    <div className="pr-token-summary">
      <button className="pr-token-summary-header" onClick={() => setExpanded(!expanded)}>
        <span className="pr-token-summary-icon">{"\u{1F4CA}"}</span>
        <span className="pr-token-summary-title">Token Usage</span>
        <span className="pr-token-summary-totals">
          <span className="pr-token-pill pr-token-pill--in">{grandIn.toLocaleString()} in</span>
          <span className="pr-token-pill pr-token-pill--out">{grandOut.toLocaleString()} out</span>
          <span className="pr-token-pill pr-token-pill--total">{grandTotal.toLocaleString()} total</span>
          {totalToolCalls > 0 && (
            <span className="pr-token-pill pr-token-pill--tools">{totalToolCalls} tool call{totalToolCalls !== 1 ? "s" : ""}</span>
          )}
        </span>
        <span className={`pr-token-chevron ${expanded ? "pr-token-chevron--open" : ""}`}>{"\u25BC"}</span>
      </button>
      {expanded && (
        <div className="pr-token-table-wrap">
          <table className="pr-token-table">
            <thead>
              <tr>
                <th>Phase</th>
                <th>Provider / Model</th>
                <th>Time</th>
                <th>Input Tokens</th>
                <th>Output Tokens</th>
                <th>Total</th>
                <th>Tool Calls</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  <td className="pr-token-phase">{r.label}</td>
                  <td className="pr-token-model">{r.provider ? `${r.provider} / ` : ""}{r.model || "-"}</td>
                  <td>{r.elapsed != null ? `${r.elapsed.toFixed(1)}s` : "-"}</td>
                  <td className="pr-token-in">{r.inTokens.toLocaleString()}</td>
                  <td className="pr-token-out">{r.outTokens.toLocaleString()}</td>
                  <td className="pr-token-total">{(r.inTokens + r.outTokens).toLocaleString()}</td>
                  <td>{r.toolCalls > 0 ? r.toolCalls : "-"}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="pr-token-footer">
                <td colSpan="3"><strong>Grand Total</strong></td>
                <td className="pr-token-in"><strong>{grandIn.toLocaleString()}</strong></td>
                <td className="pr-token-out"><strong>{grandOut.toLocaleString()}</strong></td>
                <td className="pr-token-total"><strong>{grandTotal.toLocaleString()}</strong></td>
                <td><strong>{totalToolCalls > 0 ? totalToolCalls : "-"}</strong></td>
              </tr>
            </tfoot>
          </table>
        </div>
      )}
    </div>
  );
}
