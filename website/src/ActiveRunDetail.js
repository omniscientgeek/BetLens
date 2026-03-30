import React, { useState, useEffect, useCallback, useRef } from "react";
import BriefPanel, { VerificationBadge } from "./BriefPanel";
import AnalyzeConversation from "./AnalyzeConversation";
import { API_BASE, fetchWithRetry } from "./api";
import TokenUsageSummary from "./TokenUsageSummary";

const PHASE_LABELS = {
  detect: "Detect",
  analyze: "Analyze",
  audit_analyze: "Audit Analyze",
  brief: "Brief",
  audit_brief: "Audit Brief",
  done: "Complete",
};

function formatElapsed(totalSec) {
  const hrs = Math.floor(totalSec / 3600);
  const mins = Math.floor((totalSec % 3600) / 60);
  const secs = totalSec % 60;
  if (hrs > 0) return `${hrs}:${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

/* ------------------------------------------------------------------ */
/*  Active run detail — polls for live progress                        */
/* ------------------------------------------------------------------ */
export default function ActiveRunDetail({ filename, onBack, backLabel }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);
  const dataRef = useRef(null);

  const fetchDetail = useCallback(async () => {
    try {
      const res = await fetchWithRetry(
        `${API_BASE}/active-runs/${encodeURIComponent(filename)}`
      );
      if (res.status === 404) {
        // Run no longer active — it completed and was evicted from cache
        setError("Run is no longer active. It may have completed — check the History list.");
        setLoading(false);
        clearInterval(pollRef.current);
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
      dataRef.current = json;
      setLoading(false);

      // Stop polling once the run is done
      if (json.status === "complete" || json.status === "error") {
        clearInterval(pollRef.current);
      }
    } catch (err) {
      // Don't overwrite existing data on transient errors
      if (!dataRef.current) {
        setError(err.message);
        setLoading(false);
      }
    }
  }, [filename]);

  useEffect(() => {
    fetchDetail();
    pollRef.current = setInterval(fetchDetail, 2000);
    return () => clearInterval(pollRef.current);
  }, [fetchDetail]);

  if (loading) {
    return (
      <div className="pr-loading">
        <span className="pr-loading-spinner" />
        Loading active run...
      </div>
    );
  }

  if (error && !data) {
    return (
      <div>
        <button className="pr-back-btn" onClick={onBack}>← {backLabel || "Back"}</button>
        <div className="pr-error">{error}</div>
      </div>
    );
  }

  if (!data) return null;

  const pr = data.pipeline_results || {};
  const phases = ["detect", "analyze", "audit_analyze", "brief", "audit_brief"];

  return (
    <div className="pr-detail">
      {/* Header */}
      <div className="pr-detail-header">
        <button className="pr-back-btn" onClick={onBack}>← {backLabel || "Back"}</button>
        <div className="pr-detail-meta">
          <span className="pr-detail-source">{data.filename}</span>
          <span className="pr-detail-date">
            {data.status === "running" ? (
              <>
                <span className="pr-active-pulse" style={{ display: "inline-block", width: 8, height: 8, marginRight: 6, verticalAlign: "middle" }} />
                {data.current_phase_label} ({data.phase_index + 1}/{data.total_phases})
              </>
            ) : data.status === "complete" ? (
              "Completed"
            ) : (
              `Error: ${data.error || "Unknown"}`
            )}
            {" · "}{formatElapsed(data.elapsed_seconds)}
          </span>
        </div>
      </div>

      {/* Full-width progress stepper */}
      <div className="pr-live-stepper">
        {phases.map((phaseName, i) => {
          let stepCls = "pr-active-step--pending";
          if (i < data.phase_index) stepCls = "pr-active-step--complete";
          else if (i === data.phase_index && data.status === "running") stepCls = "pr-active-step--active";
          else if (data.status === "complete") stepCls = "pr-active-step--complete";
          return (
            <div key={phaseName} className="pr-live-step">
              <div className={`pr-active-step ${stepCls}`} />
              <span className="pr-live-step-label">{PHASE_LABELS[phaseName]}</span>
            </div>
          );
        })}
      </div>

      {/* Token Usage Summary */}
      <TokenUsageSummary pipelineResults={pr} />

      {/* Completed phase results — same rendering as RunDetail */}
      {pr.analyze && (
        <AnalyzeConversation
          analyzeResult={pr.analyze}
          streaming={false}
          defaultExpanded={false}
          title="Analysis Conversation"
        />
      )}

      {/* Audit Analyze */}
      {(() => {
        const analyzeV = pr.analyze?.verification;
        const analyzeHistory = analyzeV?.fix_history || [];
        const auditEntries = analyzeHistory.filter((h) => h.type !== "fix");
        const fixEntries = analyzeHistory.filter((h) => h.type === "fix");
        const hasTimeline = auditEntries.length > 0;

        if (hasTimeline) {
          const timelineItems = analyzeHistory.map((h, idx) => {
            if (h.type === "fix") {
              return (
                <div key={`fix-${idx}`} className="fix-block">
                  <div className="fix-block-header">
                    <span className="fix-block-indicator">{"\u{1F527}"}</span>
                    <span className="fix-block-title">Fix #{h.attempt}</span>
                    {h.ai_meta && (
                      <span className="fix-block-meta">
                        {h.ai_meta.provider} / {h.ai_meta.model}
                        {h.ai_meta.elapsed_seconds != null && ` \u00B7 ${h.ai_meta.elapsed_seconds.toFixed(1)}s`}
                      </span>
                    )}
                  </div>
                  {h.conversation && (
                    <AnalyzeConversation
                      analyzeResult={{ conversation: h.conversation, ai_meta: h.ai_meta }}
                      streaming={false}
                      defaultExpanded={false}
                      title={`Fix #${h.attempt} Conversation`}
                    />
                  )}
                </div>
              );
            }
            const verdict = h.verdict || "unknown";
            const isLatest = idx === analyzeHistory.length - 1 ||
              analyzeHistory.slice(idx + 1).every((e) => e.type === "fix");
            const isPassed = verdict === "pass";
            const issueCount = Object.values(h.verification?.agents || {}).reduce(
              (sum, a) => sum + (a.issues?.length || 0), 0
            );
            return (
              <div key={`audit-${idx}`} className={`audit-block audit-block--${verdict} ${isLatest ? "audit-block--latest" : "audit-block--historical"}`}>
                <div className="audit-block-header">
                  <span className={`audit-block-indicator audit-block-indicator--${verdict}`}>
                    {isPassed ? "\u2705" : verdict === "warn" ? "\u26A0\uFE0F" : "\u274C"}
                  </span>
                  <span className="audit-block-title">
                    {h.attempt === 0 ? "Initial Audit" : `Re-audit after Fix #${h.attempt}`}
                  </span>
                  <span className={`audit-block-verdict verdict--${verdict}`}>
                    {verdict.toUpperCase()}
                  </span>
                  {issueCount > 0 && (
                    <span className="audit-block-issues">
                      {issueCount} issue{issueCount !== 1 ? "s" : ""}
                    </span>
                  )}
                  {isLatest && <span className="audit-block-latest-tag">Latest</span>}
                </div>
                <VerificationBadge verification={h.verification} />
              </div>
            );
          });

          return (
            <div className="audit-timeline">
              <div className="audit-timeline-header">
                <span className="audit-timeline-icon">{"\u{1F6E1}"}</span>
                <h3>Audit Analyze Timeline</h3>
                <span className="audit-timeline-count">
                  {auditEntries.length} audit{auditEntries.length !== 1 ? "s" : ""}
                  {fixEntries.length > 0 && `, ${fixEntries.length} fix${fixEntries.length !== 1 ? "es" : ""}`}
                </span>
              </div>
              {timelineItems}
            </div>
          );
        }

        if (analyzeV) {
          return (
            <div className="verification-card">
              <div className="verification-card-header">
                <span className="verification-card-icon">{"\u{1F6E1}"}</span>
                <h3>Audit Analyze</h3>
              </div>
              <VerificationBadge verification={analyzeV} streaming={false} />
            </div>
          );
        }

        return null;
      })()}

      {/* Brief Panel */}
      {pr.brief && !pr.brief.error && (
        <BriefPanel briefResult={pr.brief} isInterim={false} />
      )}
      {pr.brief?.error && (
        <div className="error">Brief generation failed: {pr.brief.error}</div>
      )}

      {/* Audit Brief */}
      {(() => {
        const briefV = pr.brief?.verification;
        const briefHistory = briefV?.fix_history || [];
        const auditEntries = briefHistory.filter((h) => h.type !== "fix");
        const fixEntries = briefHistory.filter((h) => h.type === "fix");
        const hasTimeline = auditEntries.length > 0;

        if (hasTimeline) {
          const timelineItems = briefHistory.map((h, idx) => {
            if (h.type === "fix") {
              return (
                <div key={`fix-${idx}`} className="fix-block">
                  <div className="fix-block-header">
                    <span className="fix-block-indicator">{"\u{1F527}"}</span>
                    <span className="fix-block-title">Fix #{h.attempt}</span>
                    {h.ai_meta && (
                      <span className="fix-block-meta">
                        {h.ai_meta.provider} / {h.ai_meta.model}
                        {h.ai_meta.elapsed_seconds != null && ` \u00B7 ${h.ai_meta.elapsed_seconds.toFixed(1)}s`}
                      </span>
                    )}
                  </div>
                  {h.conversation && (
                    <AnalyzeConversation
                      analyzeResult={{ conversation: h.conversation, ai_meta: h.ai_meta }}
                      streaming={false}
                      defaultExpanded={false}
                      title={`Fix #${h.attempt} Conversation`}
                    />
                  )}
                </div>
              );
            }
            const verdict = h.verdict || "unknown";
            const isLatest = idx === briefHistory.length - 1 ||
              briefHistory.slice(idx + 1).every((e) => e.type === "fix");
            const isPassed = verdict === "pass";
            const issueCount = Object.values(h.verification?.agents || {}).reduce(
              (sum, a) => sum + (a.issues?.length || 0), 0
            );
            return (
              <div key={`audit-${idx}`} className={`audit-block audit-block--${verdict} ${isLatest ? "audit-block--latest" : "audit-block--historical"}`}>
                <div className="audit-block-header">
                  <span className={`audit-block-indicator audit-block-indicator--${verdict}`}>
                    {isPassed ? "\u2705" : verdict === "warn" ? "\u26A0\uFE0F" : "\u274C"}
                  </span>
                  <span className="audit-block-title">
                    {h.attempt === 0 ? "Initial Audit" : `Re-audit after Fix #${h.attempt}`}
                  </span>
                  <span className={`audit-block-verdict verdict--${verdict}`}>
                    {verdict.toUpperCase()}
                  </span>
                  {issueCount > 0 && (
                    <span className="audit-block-issues">
                      {issueCount} issue{issueCount !== 1 ? "s" : ""}
                    </span>
                  )}
                  {isLatest && <span className="audit-block-latest-tag">Latest</span>}
                </div>
                <VerificationBadge verification={h.verification} />
              </div>
            );
          });

          return (
            <div className="audit-timeline">
              <div className="audit-timeline-header">
                <span className="audit-timeline-icon">{"\u{1F6E1}"}</span>
                <h3>Audit Brief Timeline</h3>
                <span className="audit-timeline-count">
                  {auditEntries.length} audit{auditEntries.length !== 1 ? "s" : ""}
                  {fixEntries.length > 0 && `, ${fixEntries.length} fix${fixEntries.length !== 1 ? "es" : ""}`}
                </span>
              </div>
              {timelineItems}
            </div>
          );
        }

        if (briefV) {
          return (
            <div className="verification-card">
              <div className="verification-card-header">
                <span className="verification-card-icon">{"\u{1F6E1}"}</span>
                <h3>Audit Brief</h3>
              </div>
              <VerificationBadge verification={briefV} streaming={false} />
            </div>
          );
        }

        return null;
      })()}

      {/* Currently running phase indicator */}
      {data.status === "running" && (
        <div className="pr-live-running-phase">
          <span className="pr-active-spinner" />
          <span>{data.current_phase_label}...</span>
        </div>
      )}

      {/* Completed banner */}
      {data.status === "complete" && (
        <div className="pr-live-complete-banner">
          ✓ Run completed. Results have been saved and will appear in the History list.
        </div>
      )}

      {/* Error banner */}
      {data.status === "error" && (
        <div className="pr-error" style={{ marginTop: "var(--space-md)" }}>
          Pipeline error: {data.error || "Unknown error"}
        </div>
      )}
    </div>
  );
}
