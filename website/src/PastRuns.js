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
/*  Pagination constant                                                */
/* ------------------------------------------------------------------ */
const PAGE_SIZE = 20;

/* ------------------------------------------------------------------ */
/*  Collapsible section wrapper                                        */
/* ------------------------------------------------------------------ */
function Collapsible({ title, icon, defaultOpen = false, badge, children }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={`pr-collapsible ${open ? "pr-collapsible--open" : ""}`}>
      <button className="pr-collapsible-header" onClick={() => setOpen(!open)}>
        <span className="pr-collapsible-arrow">{open ? "\u25BE" : "\u25B8"}</span>
        {icon && <span className="pr-collapsible-icon">{icon}</span>}
        <span className="pr-collapsible-title">{title}</span>
        {badge && <span className="pr-collapsible-badge">{badge}</span>}
      </button>
      {open && <div className="pr-collapsible-body">{children}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  DetectSummary — Issue #1                                           */
/* ------------------------------------------------------------------ */
function DetectSummary({ detect }) {
  if (!detect || typeof detect !== "object") return null;

  const {
    ev_summary,
    vig_summary,
    stale_summary,
    arb_profit_curves,
    consensus,
    synthetic_perfect_book,
  } = detect;

  const hasSomething =
    ev_summary || vig_summary || stale_summary || arb_profit_curves || consensus || synthetic_perfect_book;
  if (!hasSomething) return null;

  return (
    <Collapsible title="Detection Results" icon={"\u{1F50D}"} defaultOpen={false}>
      <div className="pr-detect-grid">
        {/* EV Opportunities */}
        {ev_summary && (
          <Collapsible title="EV Opportunities" icon={"\u{1F4B0}"} defaultOpen={false}
            badge={Array.isArray(ev_summary) ? `${ev_summary.length}` : null}>
            <div className="pr-detect-table-wrap">
              {Array.isArray(ev_summary) && ev_summary.length > 0 ? (
                <table className="pr-detect-table">
                  <thead>
                    <tr>
                      <th>Game</th>
                      <th>Side</th>
                      <th>Book</th>
                      <th>Edge%</th>
                      <th>EV$</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ev_summary.slice(0, 20).map((row, i) => (
                      <tr key={i}>
                        <td>{row.game || row.event || row.matchup || "-"}</td>
                        <td>{row.side || row.bet || "-"}</td>
                        <td>{row.book || row.sportsbook || "-"}</td>
                        <td>{row.edge != null ? `${(row.edge * 100).toFixed(2)}%` : row.edge_pct || "-"}</td>
                        <td>{row.ev_dollar != null ? `$${row.ev_dollar.toFixed(2)}` : row.ev || "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <pre className="pr-detect-json">{JSON.stringify(ev_summary, null, 2)}</pre>
              )}
            </div>
          </Collapsible>
        )}

        {/* Vig Rankings */}
        {vig_summary && (
          <Collapsible title="Vig Rankings" icon={"\u{1F4CA}"} defaultOpen={false}>
            <div className="pr-detect-table-wrap">
              {Array.isArray(vig_summary) && vig_summary.length > 0 ? (
                <table className="pr-detect-table">
                  <thead>
                    <tr>
                      <th>Book</th>
                      <th>Avg Vig</th>
                      <th>Markets</th>
                    </tr>
                  </thead>
                  <tbody>
                    {vig_summary.map((row, i) => (
                      <tr key={i}>
                        <td>{row.book || row.sportsbook || "-"}</td>
                        <td>{row.avg_vig != null ? `${(row.avg_vig * 100).toFixed(2)}%` : row.vig || "-"}</td>
                        <td>{row.market_count || row.markets || "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <pre className="pr-detect-json">{JSON.stringify(vig_summary, null, 2)}</pre>
              )}
            </div>
          </Collapsible>
        )}

        {/* Stale Lines */}
        {stale_summary && (
          <Collapsible title="Stale Lines" icon={"\u26A0\uFE0F"} defaultOpen={false}
            badge={Array.isArray(stale_summary) ? `${stale_summary.length}` : null}>
            <div className="pr-detect-table-wrap">
              {Array.isArray(stale_summary) && stale_summary.length > 0 ? (
                <table className="pr-detect-table">
                  <thead>
                    <tr>
                      <th>Game</th>
                      <th>Book</th>
                      <th>Market</th>
                      <th>Minutes Behind</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stale_summary.slice(0, 20).map((row, i) => (
                      <tr key={i}>
                        <td>{row.game || row.event || "-"}</td>
                        <td>{row.book || row.sportsbook || "-"}</td>
                        <td>{row.market || "-"}</td>
                        <td>{row.minutes_behind != null ? row.minutes_behind.toFixed(1) : row.staleness || "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <pre className="pr-detect-json">{JSON.stringify(stale_summary, null, 2)}</pre>
              )}
            </div>
          </Collapsible>
        )}

        {/* Arbitrage */}
        {arb_profit_curves && (
          <Collapsible title="Arbitrage" icon={"\u{1F504}"} defaultOpen={false}>
            <div className="pr-detect-table-wrap">
              {Array.isArray(arb_profit_curves) && arb_profit_curves.length > 0 ? (
                <table className="pr-detect-table">
                  <thead>
                    <tr>
                      <th>Game</th>
                      <th>Market</th>
                      <th>Profit%</th>
                      <th>Books</th>
                    </tr>
                  </thead>
                  <tbody>
                    {arb_profit_curves.map((row, i) => (
                      <tr key={i}>
                        <td>{row.game || row.event || "-"}</td>
                        <td>{row.market || "-"}</td>
                        <td>{row.profit_pct != null ? `${row.profit_pct.toFixed(2)}%` : row.profit || "-"}</td>
                        <td>{row.books ? (Array.isArray(row.books) ? row.books.join(", ") : row.books) : "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <pre className="pr-detect-json">{JSON.stringify(arb_profit_curves, null, 2)}</pre>
              )}
            </div>
          </Collapsible>
        )}

        {/* Consensus Odds */}
        {consensus && (
          <Collapsible title="Consensus Odds" icon={"\u{1F4B9}"} defaultOpen={false}>
            <div className="pr-detect-table-wrap">
              {Array.isArray(consensus) && consensus.length > 0 ? (
                <table className="pr-detect-table">
                  <thead>
                    <tr>
                      <th>Game</th>
                      <th>Market</th>
                      <th>Side</th>
                      <th>Fair Odds</th>
                      <th>Fair Prob</th>
                    </tr>
                  </thead>
                  <tbody>
                    {consensus.slice(0, 30).map((row, i) => (
                      <tr key={i}>
                        <td>{row.game || row.event || "-"}</td>
                        <td>{row.market || "-"}</td>
                        <td>{row.side || row.bet || "-"}</td>
                        <td>{row.fair_odds || row.odds || "-"}</td>
                        <td>{row.fair_prob != null ? `${(row.fair_prob * 100).toFixed(1)}%` : row.probability || "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : typeof consensus === "object" ? (
                <pre className="pr-detect-json">{JSON.stringify(consensus, null, 2)}</pre>
              ) : (
                <span>{String(consensus)}</span>
              )}
            </div>
          </Collapsible>
        )}

        {/* Synthetic Perfect Book */}
        {synthetic_perfect_book && (
          <Collapsible title="Synthetic Perfect Book" icon={"\u2B50"} defaultOpen={false}>
            <div className="pr-detect-table-wrap">
              {Array.isArray(synthetic_perfect_book) && synthetic_perfect_book.length > 0 ? (
                <table className="pr-detect-table">
                  <thead>
                    <tr>
                      <th>Game</th>
                      <th>Market</th>
                      <th>Side</th>
                      <th>Best Line</th>
                      <th>Best Book</th>
                    </tr>
                  </thead>
                  <tbody>
                    {synthetic_perfect_book.slice(0, 30).map((row, i) => (
                      <tr key={i}>
                        <td>{row.game || row.event || "-"}</td>
                        <td>{row.market || "-"}</td>
                        <td>{row.side || row.bet || "-"}</td>
                        <td>{row.best_odds || row.odds || "-"}</td>
                        <td>{row.best_book || row.book || "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : typeof synthetic_perfect_book === "object" ? (
                <pre className="pr-detect-json">{JSON.stringify(synthetic_perfect_book, null, 2)}</pre>
              ) : (
                <span>{String(synthetic_perfect_book)}</span>
              )}
            </div>
          </Collapsible>
        )}
      </div>
    </Collapsible>
  );
}

/* ------------------------------------------------------------------ */
/*  AI Meta display — Issue #5                                         */
/* ------------------------------------------------------------------ */
function AiMetaRow({ label, meta }) {
  if (!meta) return null;
  const tokens = (meta.usage?.input_tokens || 0) + (meta.usage?.output_tokens || 0);
  return (
    <span className="pr-ai-meta-item">
      <span className="pr-ai-meta-label">{label}:</span>{" "}
      {meta.provider && <span>{meta.provider}/</span>}
      {meta.model && <span>{meta.model}</span>}
      {meta.elapsed_seconds != null && <span> ({meta.elapsed_seconds.toFixed(1)}s)</span>}
      {tokens > 0 && <span> {tokens.toLocaleString()} tok</span>}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/*  Helper: resolve audit data with fallback — Issue #8                */
/* ------------------------------------------------------------------ */
function resolveAudit(pr, phaseKey) {
  // Try nested verification first, then top-level audit key
  const nested = pr[phaseKey]?.verification;
  if (nested) return nested;
  const topKey = phaseKey === "analyze" ? "audit_analyze" : "audit_brief";
  return pr[topKey] || null;
}

/* ------------------------------------------------------------------ */
/*  Filter toolbar — Issue #3                                          */
/* ------------------------------------------------------------------ */
function FilterToolbar({ search, onSearchChange, verdictFilter, onVerdictChange, sort, onSortChange }) {
  return (
    <div className="pr-toolbar">
      <input
        className="pr-toolbar-search"
        type="text"
        placeholder="Search by filename or source..."
        value={search}
        onChange={(e) => onSearchChange(e.target.value)}
      />
      <select
        className="pr-toolbar-select"
        value={verdictFilter}
        onChange={(e) => onVerdictChange(e.target.value)}
      >
        <option value="all">All verdicts</option>
        <option value="pass">Pass</option>
        <option value="warn">Warn</option>
        <option value="fail">Fail</option>
      </select>
      <select
        className="pr-toolbar-select"
        value={sort}
        onChange={(e) => onSortChange(e.target.value)}
      >
        <option value="newest">Newest first</option>
        <option value="oldest">Oldest first</option>
        <option value="fastest">Fastest runtime</option>
        <option value="slowest">Slowest runtime</option>
      </select>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Active runs section — Issue #10: show completed & errored too      */
/* ------------------------------------------------------------------ */
function ActiveRuns({ activeRuns, onSelect }) {
  if (!activeRuns || activeRuns.length === 0) return null;

  return (
    <div className="pr-active-section">
      <div className="pr-active-header">
        <span className="pr-active-header-icon">
          <span className="pr-active-pulse" />
        </span>
        <h3>Active Runs</h3>
        <span className="pr-active-count">{activeRuns.length}</span>
      </div>
      <div className="pr-list">
        {activeRuns.map((run) => {
          const isRunning = run.status === "running";
          const isComplete = run.status === "complete";
          const isError = run.status === "error";
          return (
            <button
              key={run.run_id}
              className={`pr-list-item pr-list-item--active pr-list-item--${run.status}`}
              onClick={() => onSelect(run.filename, { isActive: true })}
            >
              <div className="pr-list-item-main">
                <span className="pr-list-item-icon">
                  {isRunning ? (
                    <span className="pr-active-spinner" />
                  ) : isComplete ? (
                    "\u2705"
                  ) : (
                    "\u274C"
                  )}
                </span>
                <div className="pr-list-item-info">
                  <span className="pr-list-item-source">{run.filename}</span>
                  <span className="pr-list-item-date">
                    {isRunning
                      ? `${PHASE_LABELS[run.current_phase] || run.current_phase} (${run.phase_index + 1}/${run.total_phases})`
                      : isComplete
                      ? "Completed \u2014 saving results..."
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
                    else if (i === run.phase_index && isRunning) stepCls = "pr-active-step--active";
                    else if (isComplete) stepCls = "pr-active-step--complete";
                    else if (isError && i <= run.phase_index) stepCls = "pr-active-step--error";
                    return <div key={i} className={`pr-active-step ${stepCls}`} />;
                  })}
                </div>
                <span className="pr-active-timer">{formatElapsed(run.elapsed_seconds)}</span>
              </div>
              <span className="pr-list-item-arrow">{"\u203A"}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  List view — Issues #3, #9, #2                                      */
/* ------------------------------------------------------------------ */
function RunList({ runs, activeRuns, onSelect, onDelete, loading, error, search, onSearchChange, verdictFilter, onVerdictChange, sort, onSortChange }) {
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);

  // Reset pagination when filters change
  useEffect(() => {
    setVisibleCount(PAGE_SIZE);
  }, [search, verdictFilter, sort]);

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

  if (!hasContent && !search && verdictFilter === "all") {
    return (
      <div className="pr-empty">
        <span className="pr-empty-icon">{"\u{1F4ED}"}</span>
        <p>No saved runs yet.</p>
        <p className="pr-empty-hint">
          Run an analysis on the BetLens page — results are saved automatically when complete.
        </p>
      </div>
    );
  }

  // Filter — Issue #3
  const searchLower = search.toLowerCase();
  let filtered = runs.filter((run) => {
    // Text search
    if (searchLower) {
      const haystack = `${run.filename || ""} ${run.source_file || ""}`.toLowerCase();
      if (!haystack.includes(searchLower)) return false;
    }
    // Verdict filter
    if (verdictFilter !== "all") {
      const v = run.analyze_verdict || run.brief_verdict;
      if (v !== verdictFilter) return false;
    }
    return true;
  });

  // Sort — Issue #3
  filtered = [...filtered].sort((a, b) => {
    if (sort === "oldest") {
      const da = a.saved_at || parseRunMeta(a.filename).dateTime?.toISOString() || "";
      const db = b.saved_at || parseRunMeta(b.filename).dateTime?.toISOString() || "";
      return da.localeCompare(db);
    }
    if (sort === "fastest") {
      return (a.total_runtime_seconds || Infinity) - (b.total_runtime_seconds || Infinity);
    }
    if (sort === "slowest") {
      return (b.total_runtime_seconds || 0) - (a.total_runtime_seconds || 0);
    }
    // newest (default)
    const da = a.saved_at || parseRunMeta(a.filename).dateTime?.toISOString() || "";
    const db = b.saved_at || parseRunMeta(b.filename).dateTime?.toISOString() || "";
    return db.localeCompare(da);
  });

  const visible = filtered.slice(0, visibleCount);
  const hasMore = visibleCount < filtered.length;

  return (
    <>
      <FilterToolbar
        search={search}
        onSearchChange={onSearchChange}
        verdictFilter={verdictFilter}
        onVerdictChange={onVerdictChange}
        sort={sort}
        onSortChange={onSortChange}
      />
      <ActiveRuns activeRuns={activeRuns} onSelect={onSelect} />
      {visible.length > 0 && (
        <div className="pr-list">
          {visible.map((run) => {
            const meta = parseRunMeta(run.filename);
            return (
              <div key={run.filename} className="pr-list-item-row">
                <button
                  className="pr-list-item"
                  onClick={() => onSelect(run.filename)}
                >
                  <div className="pr-list-item-main">
                    <span className="pr-list-item-icon">{"\u{1F4CA}"}</span>
                    <div className="pr-list-item-info">
                      <span className="pr-list-item-source">
                        {run.source_file || run.filename}
                      </span>
                      <span className="pr-list-item-date">
                        {meta.date ? `${meta.date} at ${meta.time}` : run.filename}
                      </span>
                      {run.total_runtime_seconds != null && (
                        <span className="pr-list-item-runtime">
                          {"\u23F1"} {formatElapsed(Math.round(run.total_runtime_seconds))}
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
                  <span className="pr-list-item-arrow">{"\u203A"}</span>
                </button>
                {/* Delete button — Issue #2 */}
                <button
                  className="pr-list-item-delete"
                  title="Delete run"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(run.filename);
                  }}
                >
                  {"\u2715"}
                </button>
              </div>
            );
          })}
        </div>
      )}
      {filtered.length === 0 && (search || verdictFilter !== "all") && (
        <div className="pr-empty">
          <p>No runs match the current filters.</p>
        </div>
      )}
      {/* Pagination — Issue #9 */}
      {hasMore && (
        <button
          className="pr-show-more"
          onClick={() => setVisibleCount((c) => c + PAGE_SIZE)}
        >
          Show more ({filtered.length - visibleCount} remaining)
        </button>
      )}
    </>
  );
}

/* ------------------------------------------------------------------ */
/*  Detail view — reuses BriefPanel, AnalyzeConversation, etc.         */
/*  Issues: #1, #2, #4, #5, #6, #8                                    */
/* ------------------------------------------------------------------ */
function RunDetail({ filename, onBack, onDelete }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sourceDataOpen, setSourceDataOpen] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);

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

  // Download JSON — Issue #4
  const handleDownload = useCallback(() => {
    if (!data) return;
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [data, filename]);

  // Copy brief — Issue #4
  const handleCopyBrief = useCallback(() => {
    const pr = data?.pipeline_results || {};
    const briefText = pr.brief?.brief_text;
    if (!briefText) return;
    navigator.clipboard.writeText(briefText).then(() => {
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    }).catch(() => {});
  }, [data]);

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
        <button className="pr-back-btn" onClick={onBack}>{"\u2190"} Back to History</button>
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
    // Check both nested verification and top-level audit keys
    const audit = resolveAudit(pr, phaseKey);
    totalRuntime += (audit?.elapsed_seconds || 0);
  }

  const analyzeMeta = pr.analyze?.ai_meta;
  const briefMeta = pr.brief?.ai_meta;

  return (
    <div className="pr-detail">
      {/* Header bar — Issues #2, #4, #5 */}
      <div className="pr-detail-header">
        <div className="pr-detail-header-top">
          <button className="pr-back-btn" onClick={onBack}>{"\u2190"} Back to History</button>
          <div className="pr-detail-header-actions">
            {pr.brief?.brief_text && (
              <button className="pr-action-btn" onClick={handleCopyBrief}>
                {copySuccess ? "\u2713 Copied" : "Copy Brief"}
              </button>
            )}
            <button className="pr-action-btn" onClick={handleDownload}>
              Download JSON
            </button>
            <button
              className="pr-action-btn pr-action-btn--danger"
              onClick={() => onDelete(filename)}
            >
              Delete
            </button>
          </div>
        </div>
        <div className="pr-detail-meta">
          <span className="pr-detail-source">{data.source_file || filename}</span>
          <span className="pr-detail-date">
            Saved {savedAt}
            {totalRuntime > 0 && (
              <span className="pr-detail-runtime"> {"\u00B7"} {"\u23F1"} {formatElapsed(Math.round(totalRuntime))}</span>
            )}
          </span>
          {/* AI Metadata — Issue #5 */}
          {(analyzeMeta || briefMeta) && (
            <div className="pr-ai-meta-row">
              <AiMetaRow label="Analyze" meta={analyzeMeta} />
              <AiMetaRow label="Brief" meta={briefMeta} />
            </div>
          )}
        </div>
      </div>

      {/* Detect Summary — Issue #1 */}
      <DetectSummary detect={pr.detect} />

      {/* Analyze Conversation */}
      {pr.analyze && (() => {
        const analyzeV = resolveAudit(pr, "analyze");
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
        const analyzeV = resolveAudit(pr, "analyze");
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
        const briefV = resolveAudit(pr, "brief");
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
        const briefV = resolveAudit(pr, "brief");
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

      {/* Source Data — Issue #6 */}
      {data.file_data && (
        <Collapsible title="Source Data" icon={"\u{1F4C1}"} defaultOpen={false}>
          <pre className="pr-source-data">
            {JSON.stringify(data.file_data, null, 2)}
          </pre>
        </Collapsible>
      )}
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

  // Filter/sort state — Issue #3
  const [search, setSearch] = useState("");
  const [verdictFilter, setVerdictFilter] = useState("all");
  const [sort, setSort] = useState("newest");

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

  // Delete handler — Issue #2
  const handleDelete = useCallback(async (filenameToDelete) => {
    if (!window.confirm(`Delete "${filenameToDelete}"? This cannot be undone.`)) return;
    try {
      const res = await fetchWithRetry(
        `${API_BASE}/saved-results/${encodeURIComponent(filenameToDelete)}`,
        { method: "DELETE" }
      );
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.error || `HTTP ${res.status}`);
      }
      // If we were viewing this run, go back to list
      if (selectedView?.filename === filenameToDelete) {
        setSelectedView(null);
      }
      // Refresh list
      loadRuns();
    } catch (err) {
      alert("Failed to delete: " + err.message);
    }
  }, [loadRuns, selectedView]);

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
        onDelete={handleDelete}
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
        activeRuns={activeRuns}
        onSelect={handleSelect}
        onDelete={handleDelete}
        loading={loading}
        error={error}
        search={search}
        onSearchChange={setSearch}
        verdictFilter={verdictFilter}
        onVerdictChange={setVerdictFilter}
        sort={sort}
        onSortChange={setSort}
      />
    </div>
  );
}
