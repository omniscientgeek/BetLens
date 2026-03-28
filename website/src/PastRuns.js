import React, { useState, useEffect, useCallback, useRef } from "react";
import { useLocation } from "react-router-dom";
import BriefPanel, { VerificationBadge } from "./BriefPanel";
import AnalyzeConversation from "./AnalyzeConversation";
import ActiveRunDetail from "./ActiveRunDetail";
import { API_BASE, fetchWithRetry } from "./api";

/* ------------------------------------------------------------------ */
/*  Verdict badge helper                                               */
/* ------------------------------------------------------------------ */
const VERDICT_CONFIG = {
  pass: { icon: "\u2705", label: "Pass", cls: "verdict--pass" },
  warn: { icon: "\u26A0\uFE0F", label: "Warn", cls: "verdict--warn" },
  fail: { icon: "\u274C", label: "Fail", cls: "verdict--fail" },
  error: { icon: "\u2753", label: "Error", cls: "verdict--error" },
};

function VerdictPill({ verdict }) {
  const cfg = VERDICT_CONFIG[verdict] || { icon: "\u2022", label: verdict || "N/A", cls: "" };
  return (
    <span className={`pr-verdict-pill ${cfg.cls}`}>
      <span className="pr-verdict-icon">{cfg.icon}</span>
      {cfg.label}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/*  Phase progress labels                                              */
/* ------------------------------------------------------------------ */
const PHASE_LABELS = {
  detect: "Detect",
  analyze: "Analyze",
  audit_analyze: "Audit Analyze",
  brief: "Brief",
  audit_brief: "Audit Brief",
  done: "Complete",
};

/* ------------------------------------------------------------------ */
/*  Parse metadata from filename + optional saved_at                   */
/* ------------------------------------------------------------------ */
function parseRunMeta(filename) {
  // betlens_results_2026-03-28_013024.json  or  sample_2026-03-28_013024.json
  // Timestamps in filenames are server local time (generated via datetime.now())
  const match = filename.match(/(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})(\d{2})\.json$/);
  if (match) {
    const [, date, hh, mm, ss] = match;
    const dateTime = new Date(`${date}T${hh}:${mm}:${ss}`);
    return {
      date: dateTime.toLocaleDateString(),
      time: dateTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
      dateTime,
    };
  }
  return { date: null, time: null, dateTime: null };
}

/* ------------------------------------------------------------------ */
/*  Format elapsed seconds as m:ss or h:mm:ss                         */
/* ------------------------------------------------------------------ */
function formatElapsed(totalSec) {
  const hrs = Math.floor(totalSec / 3600);
  const mins = Math.floor((totalSec % 3600) / 60);
  const secs = totalSec % 60;
  if (hrs > 0) return `${hrs}:${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

/* ------------------------------------------------------------------ */
/*  Active runs section                                                */
/* ------------------------------------------------------------------ */
function ActiveRuns({ activeRuns, onSelect }) {
  if (!activeRuns || activeRuns.length === 0) return null;

  return (
    <div className="pr-active-section">
      <div className="pr-active-header">
        <span className="pr-active-header-icon">
          <span className="pr-active-pulse" />
        </span>
        <h3>Running Now</h3>
        <span className="pr-active-count">{activeRuns.length}</span>
      </div>
      <div className="pr-list">
        {activeRuns.map((run) => (
          <button
            key={run.run_id}
            className={`pr-list-item pr-list-item--active pr-list-item--${run.status}`}
            onClick={() => onSelect(run.filename, { isActive: true })}
          >
            <div className="pr-list-item-main">
              <span className="pr-list-item-icon">
                {run.status === "running" ? (
                  <span className="pr-active-spinner" />
                ) : run.status === "complete" ? (
                  "\u2705"
                ) : (
                  "\u274C"
                )}
              </span>
              <div className="pr-list-item-info">
                <span className="pr-list-item-source">{run.filename}</span>
                <span className="pr-list-item-date">
                  {run.status === "running"
                    ? `${PHASE_LABELS[run.current_phase] || run.current_phase} (${run.phase_index + 1}/${run.total_phases})`
                    : run.status === "complete"
                    ? "Completed — saving results..."
                    : `Error: ${run.error || "Unknown"}`}
                </span>
              </div>
            </div>
            <div className="pr-list-item-verdicts">
              {/* Mini pipeline stepper */}
              <div className="pr-active-stepper">
                {Array.from({ length: run.total_phases }, (_, i) => {
                  let stepCls = "pr-active-step--pending";
                  if (i < run.phase_index) stepCls = "pr-active-step--complete";
                  else if (i === run.phase_index && run.status === "running") stepCls = "pr-active-step--active";
                  return <div key={i} className={`pr-active-step ${stepCls}`} />;
                })}
              </div>
              <span className="pr-active-timer">{formatElapsed(run.elapsed_seconds)}</span>
            </div>
            <span className="pr-list-item-arrow">›</span>
          </button>
        ))}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  List view                                                          */
/* ------------------------------------------------------------------ */
function RunList({ runs, activeRuns, onSelect, loading, error }) {
  if (loading) {
    return (
      <div className="pr-loading">
        <span className="pr-loading-spinner" />
        Loading past runs...
      </div>
    );
  }

  if (error) {
    return <div className="pr-error">{error}</div>;
  }

  const hasContent = runs.length > 0 || (activeRuns && activeRuns.length > 0);

  if (!hasContent) {
    return (
      <div className="pr-empty">
        <span className="pr-empty-icon">📭</span>
        <p>No saved runs yet.</p>
        <p className="pr-empty-hint">
          Run an analysis on the BetLens page — results are saved automatically when complete.
        </p>
      </div>
    );
  }

  return (
    <>
      <ActiveRuns activeRuns={activeRuns} onSelect={onSelect} />
      {runs.length > 0 && (
        <div className="pr-list">
          {runs.map((run) => {
            const meta = parseRunMeta(run.filename);
            return (
              <button
                key={run.filename}
                className="pr-list-item"
                onClick={() => onSelect(run.filename)}
              >
                <div className="pr-list-item-main">
                  <span className="pr-list-item-icon">📊</span>
                  <div className="pr-list-item-info">
                    <span className="pr-list-item-source">
                      {run.source_file || run.filename}
                    </span>
                    <span className="pr-list-item-date">
                      {meta.date ? `${meta.date} at ${meta.time}` : run.filename}
                    </span>
                    {run.total_runtime_seconds != null && (
                      <span className="pr-list-item-runtime">
                        ⏱ {formatElapsed(Math.round(run.total_runtime_seconds))}
                      </span>
                    )}
                  </div>
                </div>
                <div className="pr-list-item-verdicts">
                  {run.analyze_verdict && (
                    <div className="pr-list-item-verdict">
                      <span className="pr-verdict-label">Analyze</span>
                      <VerdictPill verdict={run.analyze_verdict} />
                    </div>
                  )}
                  {run.brief_verdict && (
                    <div className="pr-list-item-verdict">
                      <span className="pr-verdict-label">Brief</span>
                      <VerdictPill verdict={run.brief_verdict} />
                    </div>
                  )}
                </div>
                <span className="pr-list-item-arrow">›</span>
              </button>
            );
          })}
        </div>
      )}
    </>
  );
}

/* ------------------------------------------------------------------ */
/*  Detail view — reuses BriefPanel, AnalyzeConversation, etc.         */
/* ------------------------------------------------------------------ */
function RunDetail({ filename, onBack }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchWithRetry(`${API_BASE}/saved-results/${encodeURIComponent(filename)}`)
      .then(async (res) => {
        const json = await res.json().catch(() => null);
        if (!res.ok) {
          throw new Error(json?.error || `HTTP ${res.status}`);
        }
        return json;
      })
      .then((json) => {
        if (!cancelled) setData(json);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [filename]);

  if (loading) {
    return (
      <div className="pr-loading">
        <span className="pr-loading-spinner" />
        Loading run details...
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <button className="pr-back-btn" onClick={onBack}>← Back to History</button>
        <div className="pr-error">Failed to load run: {error}</div>
      </div>
    );
  }

  if (!data) return null;

  const pr = data.pipeline_results || {};
  const meta = parseRunMeta(filename);
  const savedAt = data.saved_at
    ? new Date(data.saved_at).toLocaleString()
    : meta.dateTime
    ? meta.dateTime.toLocaleString()
    : filename;

  // Compute total pipeline runtime from phase + verification elapsed times
  let totalRuntime = 0;
  for (const phaseKey of ["analyze", "brief"]) {
    const phase = pr[phaseKey] || {};
    totalRuntime += (phase.ai_meta?.elapsed_seconds || 0);
    totalRuntime += (phase.verification?.elapsed_seconds || 0);
  }

  return (
    <div className="pr-detail">
      {/* Header bar */}
      <div className="pr-detail-header">
        <button className="pr-back-btn" onClick={onBack}>← Back to History</button>
        <div className="pr-detail-meta">
          <span className="pr-detail-source">{data.source_file || filename}</span>
          <span className="pr-detail-date">
            Saved {savedAt}
            {totalRuntime > 0 && (
              <span className="pr-detail-runtime"> · ⏱ {formatElapsed(Math.round(totalRuntime))}</span>
            )}
          </span>
        </div>
      </div>

      {/* Analyze Conversation */}
      {pr.analyze && (() => {
        const analyzeV = pr.analyze?.verification;
        const analyzeIsDraft = analyzeV && analyzeV.overall_verdict !== "pass";
        return (
          <div className={`draft-wrapper ${analyzeIsDraft ? "draft-wrapper--draft" : analyzeV ? "draft-wrapper--verified" : ""}`}>
            {analyzeIsDraft && (
              <div className="draft-badge">
                <span className="draft-badge-icon">{"\u26A0\uFE0F"}</span>
                <span>DRAFT — Analysis did not pass all audits</span>
              </div>
            )}
            {!analyzeIsDraft && analyzeV && (
              <div className="verified-badge">
                <span className="verified-badge-icon">{"\u2705"}</span>
                <span>VERIFIED — Analysis passed all audits</span>
              </div>
            )}
            <AnalyzeConversation
              analyzeResult={pr.analyze}
              streaming={false}
              defaultExpanded={false}
              title="Analysis Conversation"
            />
          </div>
        );
      })()}

      {/* Audit Analyze */}
      {(() => {
        const analyzeV = pr.analyze?.verification;
        const analyzeHistory = analyzeV?.fix_history || [];
        // Support both legacy (no type field) and new format
        const auditEntries = analyzeHistory.filter((h) => h.type !== "fix");
        const fixEntries = analyzeHistory.filter((h) => h.type === "fix");
        const hasTimeline = auditEntries.length > 0;

        if (hasTimeline) {
          // Build interleaved timeline: audit blocks and fix blocks in order
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
            // audit entry (type === "audit" or legacy entries without type)
            const isLatest = idx === analyzeHistory.length - 1 ||
              analyzeHistory.slice(idx + 1).every((e) => e.type === "fix");
            const isPassed = h.verdict === "pass";
            const issueCount = Object.values(h.verification?.agents || {}).reduce(
              (sum, a) => sum + (a.issues?.length || 0), 0
            );
            return (
              <div key={`audit-${idx}`} className={`audit-block audit-block--${h.verdict} ${isLatest ? "audit-block--latest" : "audit-block--historical"}`}>
                <div className="audit-block-header">
                  <span className={`audit-block-indicator audit-block-indicator--${h.verdict}`}>
                    {isPassed ? "\u2705" : h.verdict === "warn" ? "\u26A0\uFE0F" : "\u274C"}
                  </span>
                  <span className="audit-block-title">
                    {h.attempt === 0 ? "Initial Audit" : `Re-audit after Fix #${h.attempt}`}
                  </span>
                  <span className={`audit-block-verdict verdict--${h.verdict}`}>
                    {h.verdict.toUpperCase()}
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
      {pr.brief && !pr.brief.error && (() => {
        const briefV = pr.brief?.verification;
        const briefIsDraft = briefV && briefV.overall_verdict !== "pass";
        return (
          <div className={`draft-wrapper ${briefIsDraft ? "draft-wrapper--draft" : briefV ? "draft-wrapper--verified" : ""}`}>
            {briefIsDraft && (
              <div className="draft-badge">
                <span className="draft-badge-icon">{"\u26A0\uFE0F"}</span>
                <span>DRAFT — Brief did not pass all audits</span>
              </div>
            )}
            {!briefIsDraft && briefV && (
              <div className="verified-badge">
                <span className="verified-badge-icon">{"\u2705"}</span>
                <span>VERIFIED — Brief passed all audits</span>
              </div>
            )}
            <BriefPanel briefResult={pr.brief} isInterim={false} />
          </div>
        );
      })()}
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
            const isLatest = idx === briefHistory.length - 1 ||
              briefHistory.slice(idx + 1).every((e) => e.type === "fix");
            const isPassed = h.verdict === "pass";
            const issueCount = Object.values(h.verification?.agents || {}).reduce(
              (sum, a) => sum + (a.issues?.length || 0), 0
            );
            return (
              <div key={`audit-${idx}`} className={`audit-block audit-block--${h.verdict} ${isLatest ? "audit-block--latest" : "audit-block--historical"}`}>
                <div className="audit-block-header">
                  <span className={`audit-block-indicator audit-block-indicator--${h.verdict}`}>
                    {isPassed ? "\u2705" : h.verdict === "warn" ? "\u26A0\uFE0F" : "\u274C"}
                  </span>
                  <span className="audit-block-title">
                    {h.attempt === 0 ? "Initial Audit" : `Re-audit after Fix #${h.attempt}`}
                  </span>
                  <span className={`audit-block-verdict verdict--${h.verdict}`}>
                    {h.verdict.toUpperCase()}
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
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main PastRuns page                                                 */
/* ------------------------------------------------------------------ */
export default function PastRuns() {
  const location = useLocation();
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedView, setSelectedView] = useState(
    location.state?.openFile ? { filename: location.state.openFile, isActive: false } : null
  );
  const [activeRuns, setActiveRuns] = useState([]);
  const pollRef = useRef(null);
  const activeRunsRef = useRef(activeRuns);
  activeRunsRef.current = activeRuns;

  const loadRuns = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchWithRetry(`${API_BASE}/saved-results`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      // Use pre-built metadata from the API if available, fallback to filenames.
      // Sort by date (newest first) using saved_at or the date parsed from filename.
      const rawRuns = data.runs?.length
        ? data.runs
        : (data.files || []).map((f) => ({ filename: f }));

      const sorted = [...rawRuns].sort((a, b) => {
        const dateA = a.saved_at || parseRunMeta(a.filename).dateTime?.toISOString() || "";
        const dateB = b.saved_at || parseRunMeta(b.filename).dateTime?.toISOString() || "";
        return dateB.localeCompare(dateA); // newest first
      });
      setRuns(sorted);
    } catch (err) {
      setError("Failed to load saved runs: " + err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const pollActiveRuns = useCallback(async () => {
    try {
      const res = await fetchWithRetry(`${API_BASE}/active-runs`);
      if (!res.ok) return;
      const data = await res.json();
      const prev = activeRunsRef.current;
      const current = data.runs || [];
      setActiveRuns(current);

      // If a run just completed, refresh the saved runs list
      const prevRunning = prev.filter((r) => r.status === "running");
      const currentRunning = current.filter((r) => r.status === "running");
      if (prevRunning.length > 0 && currentRunning.length < prevRunning.length) {
        // A run just finished — reload saved runs after a brief delay for the auto-save
        setTimeout(() => loadRuns(), 1500);
      }
    } catch {
      // Silently ignore polling errors
    }
  }, [loadRuns]);

  useEffect(() => {
    loadRuns();
    pollActiveRuns();
  }, [loadRuns, pollActiveRuns]);

  // Poll active runs every 3 seconds
  useEffect(() => {
    pollRef.current = setInterval(pollActiveRuns, 3000);
    return () => clearInterval(pollRef.current);
  }, [pollActiveRuns]);

  const handleSelect = useCallback((filename, opts) => {
    setSelectedView({ filename, isActive: opts?.isActive || false });
  }, []);

  if (selectedView) {
    if (selectedView.isActive) {
      return (
        <ActiveRunDetail
          filename={selectedView.filename}
          onBack={() => setSelectedView(null)}
          backLabel="Back to History"
        />
      );
    }
    return (
      <RunDetail
        filename={selectedView.filename}
        onBack={() => setSelectedView(null)}
      />
    );
  }

  return (
    <div className="pr-page">
      <div className="pr-page-header">
        <h2>Past Runs</h2>
        <p className="pr-page-subtitle">
          Browse and review previous BetLens analysis results.
          {activeRuns.some((r) => r.status === "running") && (
            <span className="pr-page-subtitle-live"> Active runs are shown in real-time. Click to view details.</span>
          )}
        </p>
      </div>
      <RunList
        runs={runs}
        activeRuns={activeRuns.filter((r) => r.status === "running")}
        onSelect={handleSelect}
        loading={loading}
        error={error}
      />
    </div>
  );
}
