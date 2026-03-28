import React, { useState, useCallback, useRef, useEffect } from "react";
import { API_BASE, fetchWithRetry } from "./api";

/* ───────────────────────────────────────────────────────────────────
   TestRunner — Run MCP unit tests from the browser & display results
   ─────────────────────────────────────────────────────────────────── */

const STATUS_ICONS = {
  passed: "\u2705",   // green check
  failed: "\u274C",   // red X
  error: "\u26A0\uFE0F",    // warning
  skipped: "\u23ED\uFE0F",  // skip
};

const STATUS_LABELS = {
  passed: "PASS",
  failed: "FAIL",
  error: "ERROR",
  skipped: "SKIP",
};

function formatDuration(seconds) {
  if (seconds == null) return "--";
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  return `${seconds.toFixed(2)}s`;
}

function ProgressRing({ passed, failed, errors, total }) {
  const size = 120;
  const stroke = 10;
  const r = (size - stroke) / 2;
  const circ = 2 * Math.PI * r;

  const passRatio = total > 0 ? passed / total : 0;
  const failRatio = total > 0 ? failed / total : 0;
  const errRatio = total > 0 ? errors / total : 0;

  const passLen = circ * passRatio;
  const failLen = circ * failRatio;
  const errLen = circ * errRatio;

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ transform: "rotate(-90deg)" }}>
      {/* Background track */}
      <circle cx={size/2} cy={size/2} r={r} fill="none"
        stroke="var(--border-subtle)" strokeWidth={stroke} />
      {/* Pass arc */}
      <circle cx={size/2} cy={size/2} r={r} fill="none"
        stroke="var(--accent-green)" strokeWidth={stroke}
        strokeDasharray={`${passLen} ${circ}`}
        strokeDashoffset={0}
        strokeLinecap="round"
        style={{ transition: "stroke-dasharray 0.6s var(--ease-out)" }} />
      {/* Fail arc */}
      <circle cx={size/2} cy={size/2} r={r} fill="none"
        stroke="var(--accent-coral)" strokeWidth={stroke}
        strokeDasharray={`${failLen} ${circ}`}
        strokeDashoffset={-passLen}
        strokeLinecap="round"
        style={{ transition: "stroke-dasharray 0.6s var(--ease-out)" }} />
      {/* Error arc */}
      <circle cx={size/2} cy={size/2} r={r} fill="none"
        stroke="var(--accent-amber)" strokeWidth={stroke}
        strokeDasharray={`${errLen} ${circ}`}
        strokeDashoffset={-(passLen + failLen)}
        strokeLinecap="round"
        style={{ transition: "stroke-dasharray 0.6s var(--ease-out)" }} />
    </svg>
  );
}

function SummaryCard({ label, value, accent }) {
  return (
    <div className="test-summary-card" style={{ borderColor: `var(--${accent})` }}>
      <span className="test-summary-value" style={{ color: `var(--${accent})` }}>{value}</span>
      <span className="test-summary-label">{label}</span>
    </div>
  );
}

function TestRow({ test, index }) {
  const [expanded, setExpanded] = useState(false);
  const status = test.outcome || "passed";
  const hasFail = status === "failed" || status === "error";

  return (
    <div className={`test-row test-row--${status}`}>
      <div className="test-row-header" onClick={() => hasFail && setExpanded(!expanded)}>
        <span className="test-row-index">{index + 1}</span>
        <span className="test-row-status">{STATUS_ICONS[status]} {STATUS_LABELS[status]}</span>
        <span className="test-row-name" title={test.nodeid}>{test.nodeid}</span>
        <span className="test-row-duration">{formatDuration(test.duration)}</span>
        {hasFail && (
          <span className="test-row-expand">{expanded ? "\u25B2" : "\u25BC"}</span>
        )}
      </div>
      {expanded && test.longrepr && (
        <pre className="test-row-output">{test.longrepr}</pre>
      )}
    </div>
  );
}

function TestGroup({ name, tests }) {
  const [collapsed, setCollapsed] = useState(false);
  const passed = tests.filter(t => t.outcome === "passed").length;
  const failed = tests.filter(t => t.outcome === "failed" || t.outcome === "error").length;
  const allPass = failed === 0;

  return (
    <div className="test-group">
      <div className="test-group-header" onClick={() => setCollapsed(!collapsed)}>
        <span className="test-group-chevron">{collapsed ? "\u25B6" : "\u25BC"}</span>
        <span className={`test-group-icon ${allPass ? "test-group-icon--pass" : "test-group-icon--fail"}`}>
          {allPass ? "\u2705" : "\u274C"}
        </span>
        <span className="test-group-name">{name}</span>
        <span className="test-group-count">
          <span className="test-count-pass">{passed}</span>
          {failed > 0 && <span className="test-count-fail"> / {failed} failed</span>}
          <span className="test-count-total"> of {tests.length}</span>
        </span>
      </div>
      {!collapsed && (
        <div className="test-group-body">
          {tests.map((t, i) => <TestRow key={t.nodeid} test={t} index={i} />)}
        </div>
      )}
    </div>
  );
}

export default function TestRunner() {
  const [results, setResults] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState("all"); // all | passed | failed
  const [lastRun, setLastRun] = useState(null);
  const timerRef = useRef(null);
  const [elapsed, setElapsed] = useState(0);

  // Timer during test run
  useEffect(() => {
    if (running) {
      const start = Date.now();
      timerRef.current = setInterval(() => setElapsed((Date.now() - start) / 1000), 100);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [running]);

  const runTests = useCallback(async () => {
    setRunning(true);
    setError(null);
    setResults(null);
    setElapsed(0);
    try {
      const res = await fetchWithRetry(`${API_BASE}/tests/run`, { method: "POST" });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResults(data);
      setLastRun(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(false);
    }
  }, []);

  // Derive stats
  const allTests = results?.tests || [];
  const totalPassed = allTests.filter(t => t.outcome === "passed").length;
  const totalFailed = allTests.filter(t => t.outcome === "failed").length;
  const totalErrors = allTests.filter(t => t.outcome === "error").length;
  const totalSkipped = allTests.filter(t => t.outcome === "skipped").length;
  const totalCount = allTests.length;
  const allGreen = totalCount > 0 && totalFailed === 0 && totalErrors === 0;

  // Group by class
  const grouped = {};
  const filteredTests = allTests.filter(t => {
    if (filter === "passed") return t.outcome === "passed";
    if (filter === "failed") return t.outcome === "failed" || t.outcome === "error";
    return true;
  });

  filteredTests.forEach(t => {
    // Extract class from nodeid:  tests/test_odds_math.py::TestFoo::test_bar → TestFoo
    const parts = t.nodeid.split("::");
    const className = parts.length >= 2 ? parts[parts.length - 2] : "Ungrouped";
    if (!grouped[className]) grouped[className] = [];
    grouped[className].push(t);
  });

  return (
    <div className="test-runner">
      {/* ── Header ── */}
      <div className="test-runner-header">
        <div className="test-runner-title-row">
          <h2 className="test-runner-title">
            <span className="test-runner-icon">{"\uD83E\uDDEA"}</span>
            Test Runner
          </h2>
          <span className="test-runner-subtitle">MCP Probability Engine &amp; Service Tests</span>
        </div>

        <div className="test-runner-actions">
          {lastRun && <span className="test-runner-lastrun">Last run: {lastRun}</span>}
          <button
            className={`test-run-btn ${running ? "test-run-btn--running" : ""}`}
            onClick={runTests}
            disabled={running}
          >
            {running ? (
              <>
                <span className="test-run-spinner" />
                Running... {formatDuration(elapsed)}
              </>
            ) : (
              <>{"\u25B6"} Run All Tests</>
            )}
          </button>
        </div>
      </div>

      {/* ── Error Banner ── */}
      {error && (
        <div className="test-error-banner">
          <span>{"\u26A0\uFE0F"} {error}</span>
          <button onClick={() => setError(null)}>{"\u2715"}</button>
        </div>
      )}

      {/* ── Results ── */}
      {results && (
        <>
          {/* Summary dashboard */}
          <div className={`test-summary ${allGreen ? "test-summary--pass" : "test-summary--fail"}`}>
            <div className="test-summary-ring">
              <ProgressRing passed={totalPassed} failed={totalFailed} errors={totalErrors} total={totalCount} />
              <div className="test-summary-ring-label">
                <span className="test-summary-ring-pct">
                  {totalCount > 0 ? Math.round((totalPassed / totalCount) * 100) : 0}%
                </span>
                <span className="test-summary-ring-sub">passing</span>
              </div>
            </div>
            <div className="test-summary-cards">
              <SummaryCard label="Total" value={totalCount} accent="accent-cyan" />
              <SummaryCard label="Passed" value={totalPassed} accent="accent-green" />
              <SummaryCard label="Failed" value={totalFailed} accent="accent-coral" />
              <SummaryCard label="Errors" value={totalErrors} accent="accent-amber" />
              {totalSkipped > 0 && <SummaryCard label="Skipped" value={totalSkipped} accent="accent-purple" />}
              <SummaryCard label="Duration" value={formatDuration(results.duration)} accent="text-secondary" />
            </div>
            <div className={`test-verdict ${allGreen ? "test-verdict--pass" : "test-verdict--fail"}`}>
              {allGreen ? "\u2705 ALL TESTS PASSING" : `\u274C ${totalFailed + totalErrors} TEST${totalFailed + totalErrors > 1 ? "S" : ""} FAILING`}
            </div>
          </div>

          {/* Filter tabs */}
          <div className="test-filter-bar">
            {[
              { key: "all", label: `All (${totalCount})` },
              { key: "passed", label: `Passed (${totalPassed})` },
              { key: "failed", label: `Failed (${totalFailed + totalErrors})` },
            ].map(f => (
              <button
                key={f.key}
                className={`test-filter-btn ${filter === f.key ? "test-filter-btn--active" : ""}`}
                onClick={() => setFilter(f.key)}
              >
                {f.label}
              </button>
            ))}
          </div>

          {/* Test groups */}
          <div className="test-groups">
            {Object.entries(grouped).map(([name, tests]) => (
              <TestGroup key={name} name={name} tests={tests} />
            ))}
          </div>

          {/* Raw output toggle */}
          {results.stdout && <RawOutput output={results.stdout} />}
        </>
      )}

      {/* ── Empty state ── */}
      {!results && !running && !error && (
        <div className="test-empty">
          <div className="test-empty-icon">{"\uD83E\uDDEA"}</div>
          <h3>Unit Test Suite</h3>
          <p>
            Run the full test suite for the MCP probability engine, odds calculations,
            anomaly detection, and statistical models.
          </p>
          <p className="test-empty-detail">
            <strong>236 tests</strong> across 2 test files covering implied probability,
            vig calculation, Shin model, Kelly Criterion, Bayesian updating, GAMLSS,
            Poisson score prediction, KNN anomaly detection, Isolation Forest, Shannon entropy,
            and cross-formula consistency checks.
          </p>
          <button className="test-run-btn" onClick={runTests}>
            {"\u25B6"} Run All Tests
          </button>
        </div>
      )}

      {/* Running overlay */}
      {running && !results && (
        <div className="test-running-overlay">
          <div className="test-running-pulse" />
          <span>Executing 236 tests...</span>
          <span className="test-running-timer">{formatDuration(elapsed)}</span>
        </div>
      )}
    </div>
  );
}

function RawOutput({ output }) {
  const [show, setShow] = useState(false);
  return (
    <div className="test-raw-output">
      <button className="test-raw-toggle" onClick={() => setShow(!show)}>
        {show ? "\u25BC" : "\u25B6"} Raw pytest output
      </button>
      {show && <pre className="test-raw-pre">{output}</pre>}
    </div>
  );
}
