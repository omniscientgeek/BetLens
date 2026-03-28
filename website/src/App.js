import React, { useState, useEffect, useRef } from "react";
import { BrowserRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import { io } from "socket.io-client";
import DevNotes from "./DevNotes";
import DataExplorer from "./DataExplorer";
import AISettings from "./AISettings";
import ChatPanel from "./ChatPanel";
import BriefPanel from "./BriefPanel";
import "./App.css";

const API_BASE = "";
const SOCKET_URL = "";

const PHASE_LABELS = { detect: "Detect", analyze: "Analyze", brief: "Brief" };
const PHASE_NAMES = ["detect", "analyze", "brief"];

// Session persistence helpers — survive refresh, cleared on tab close
const SESSION_PREFIX = "betstamp_";
function sessionGet(key, fallback = null) {
  try {
    const raw = sessionStorage.getItem(SESSION_PREFIX + key);
    return raw !== null ? JSON.parse(raw) : fallback;
  } catch { return fallback; }
}
function sessionSet(key, value) {
  try { sessionStorage.setItem(SESSION_PREFIX + key, JSON.stringify(value)); } catch {}
}
function sessionRemove(key) {
  try { sessionStorage.removeItem(SESSION_PREFIX + key); } catch {}
}

function Navigation() {
  const location = useLocation();

  return (
    <nav className="App-nav">
      <Link
        to="/"
        className={`nav-btn ${location.pathname === "/" ? "active" : ""}`}
      >
        Bet Lens
      </Link>
      <Link
        to="/devnotes"
        className={`nav-btn ${location.pathname === "/devnotes" ? "active" : ""}`}
      >
        Dev Notes
      </Link>
      <Link
        to="/ai-settings"
        className={`nav-btn ${location.pathname === "/ai-settings" ? "active" : ""}`}
      >
        AI Settings
      </Link>
    </nav>
  );
}

function BetLens() {
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(() => sessionGet("selectedFile", ""));
  const [fileData, setFileData] = useState(() => sessionGet("fileData", null));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Pipeline state
  const [pipeline, setPipeline] = useState(null);
  const [pipelineComplete, setPipelineComplete] = useState(() => sessionGet("pipelineComplete", false));
  const [pipelineError, setPipelineError] = useState(null);
  const socketRef = useRef(null);

  // Pipeline results (used by BriefPanel and ChatPanel)
  const [pipelineResults, setPipelineResults] = useState(() => sessionGet("pipelineResults", null));

  // Incremental phase results (captured as each phase completes)
  const [phaseResults, setPhaseResults] = useState({});

  // Streaming brief text (accumulated from brief_chunk events)
  const [streamingBrief, setStreamingBrief] = useState("");

  // Debug mode
  const [debugMode, setDebugMode] = useState(
    () => localStorage.getItem("betstamp_debug") === "true"
  );
  const [runId, setRunId] = useState(null);

  const toggleDebug = () => {
    const next = !debugMode;
    setDebugMode(next);
    localStorage.setItem("betstamp_debug", String(next));
  };

  // Persist key state to sessionStorage so data survives page refresh
  useEffect(() => { sessionSet("selectedFile", selectedFile); }, [selectedFile]);
  useEffect(() => { sessionSet("fileData", fileData); }, [fileData]);
  useEffect(() => { sessionSet("pipelineComplete", pipelineComplete); }, [pipelineComplete]);
  useEffect(() => { sessionSet("pipelineResults", pipelineResults); }, [pipelineResults]);

  // Fetch available JSON files on mount
  useEffect(() => {
    fetch(`${API_BASE}/files`)
      .then((res) => res.json())
      .then((data) => setFiles(data.files || []))
      .catch((err) => setError("Failed to load file list: " + err.message));
  }, []);

  // Cleanup socket on unmount
  useEffect(() => {
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  // Load selected file and kick off processing pipeline
  const handleFileSelect = async (filename) => {
    setSelectedFile(filename);
    setFileData(null);
    setError(null);
    setPipeline(null);
    setPipelineComplete(false);
    setPipelineError(null);
    setPhaseResults({});
    setStreamingBrief("");
    setRunId(null);

    if (!filename) return;

    // Disconnect any previous socket
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/files/${filename}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setFileData(data);
    } catch (err) {
      setError("Failed to load file: " + err.message);
      setLoading(false);
      return;
    }
    setLoading(false);

    // Start WebSocket connection for processing pipeline
    const socket = io(SOCKET_URL, { transports: ["websocket", "polling"] });
    socketRef.current = socket;

    socket.on("connect", () => {
      socket.emit("start_processing", { filename });
    });

    socket.on("phase_update", (data) => {
      if (data.run_id) setRunId(data.run_id);
      setPipeline({
        phase: data.phase,
        status: data.status,
        phaseIndex: data.phaseIndex,
        totalPhases: data.totalPhases,
        label: data.label,
      });
      // Capture phase results as they complete
      if (data.status === "complete" && data.result) {
        setPhaseResults((prev) => ({ ...prev, [data.phase]: data.result }));
      }
    });

    // Accumulate streaming brief text chunks as they arrive
    socket.on("brief_chunk", (data) => {
      if (data.text) {
        setStreamingBrief((prev) => prev + data.text);
      }
    });

    // Receive verification results (arrives after brief phase, before processing_complete)
    socket.on("verification_update", (data) => {
      if (data.verification) {
        setPhaseResults((prev) => ({
          ...prev,
          brief: { ...prev.brief, verification: data.verification },
        }));
      }
    });

    socket.on("processing_complete", (data) => {
      if (data.run_id) setRunId(data.run_id);
      setPipelineComplete(true);
      setPipelineResults(data.results || null);
      setStreamingBrief(""); // Clear streaming text — final brief is in results
      socket.disconnect();
      socketRef.current = null;
    });

    socket.on("processing_error", (data) => {
      setPipelineError(data.error);
      socket.disconnect();
      socketRef.current = null;
    });
  };

  // Convert analyze-phase result into a BriefPanel-compatible shape
  const buildInterimBrief = (analyzeResult) => {
    if (!analyzeResult) return null;
    const { analysis, ai_meta } = analyzeResult;
    let briefText;
    if (typeof analysis === "string") {
      briefText = analysis;
    } else if (analysis && typeof analysis === "object") {
      const parts = [];

      // AI-generated summary takes priority if available
      if (analysis.ai_summary) {
        parts.push(`## AI Analysis Summary\n${analysis.ai_summary}`);
      } else if (analysis.summary) {
        parts.push(`## Summary\n${analysis.summary}`);
      }

      // Market assessment from AI
      if (analysis.market_assessment && Object.keys(analysis.market_assessment).length > 0) {
        const ma = analysis.market_assessment;
        const maLines = [];
        if (ma.overall_health) maLines.push(`**Market Health:** ${ma.overall_health}`);
        if (ma.efficiency_score !== undefined) maLines.push(`**Efficiency Score:** ${ma.efficiency_score}/100`);
        if (ma.key_themes?.length) maLines.push(`**Key Themes:** ${ma.key_themes.join(", ")}`);
        if (ma.risk_flags?.length) maLines.push(`**Risk Flags:** ${ma.risk_flags.join(", ")}`);
        if (maLines.length > 0) parts.push(`## Market Assessment\n${maLines.join("\n")}`);
      }

      // AI insights (prioritized actions & findings)
      if (analysis.ai_insights?.length) {
        const severityOrder = { critical: 0, high: 1, medium: 2, low: 3, info: 4 };
        const sorted = [...analysis.ai_insights].sort(
          (a, b) => (severityOrder[a.severity] ?? 5) - (severityOrder[b.severity] ?? 5)
        );
        const insightLines = sorted.map((ins) => {
          const badge = ins.severity === "critical" ? "🔴" : ins.severity === "high" ? "🟠" : ins.severity === "medium" ? "🟡" : "🟢";
          return `- ${badge} **${ins.title}** (${ins.confidence} confidence)\n  ${ins.description}`;
        });
        parts.push(`## AI Insights\n${insightLines.join("\n")}`);
      }

      // Top actions from AI
      if (analysis.top_actions?.length) {
        const actionLines = analysis.top_actions.map(
          (a) => `${a.priority}. **[${(a.urgency || "").toUpperCase()}]** ${a.action}\n   _${a.reasoning}_`
        );
        parts.push(`## Recommended Actions\n${actionLines.join("\n")}`);
      }

      // Book grades from AI
      if (analysis.book_grades && Object.keys(analysis.book_grades).length > 0) {
        const gradeLines = Object.entries(analysis.book_grades).map(([book, info]) => {
          const g = typeof info === "object" ? info : { grade: info };
          return `- **${book}**: Grade ${g.grade || "N/A"}${g.avg_vig !== undefined ? ` (avg vig: ${g.avg_vig}%)` : ""}`;
        });
        parts.push(`## Sportsbook Grades\n${gradeLines.join("\n")}`);
      }

      // Original data sections (kept for completeness)
      if (analysis.best_lines?.length) {
        parts.push(
          `## Best Line Shopping\n${analysis.best_lines.map((l) => `- ${typeof l === "string" ? l : JSON.stringify(l)}`).join("\n")}`
        );
      }
      if (analysis.arbitrage?.length) {
        parts.push(
          `## Arbitrage Opportunities\n${analysis.arbitrage.map((a) => `- ${typeof a === "string" ? a : JSON.stringify(a)}`).join("\n")}`
        );
      }
      if (analysis.middles?.length) {
        parts.push(
          `## Middle Opportunities\n${analysis.middles.map((m) => `- ${typeof m === "string" ? m : JSON.stringify(m)}`).join("\n")}`
        );
      }
      if (analysis.outliers?.length) {
        parts.push(
          `## Outlier Lines\n${analysis.outliers.map((o) => `- ${typeof o === "string" ? o : JSON.stringify(o)}`).join("\n")}`
        );
      }
      if (analysis.fair_odds_summary?.length) {
        parts.push(
          `## Fair Odds & Expected Value\n${analysis.fair_odds_summary.map((f) => `- ${typeof f === "string" ? f : JSON.stringify(f)}`).join("\n")}`
        );
      }

      // AI verification notes
      if (analysis.ai_verification_notes) {
        parts.push(`## Verification Notes\n_${analysis.ai_verification_notes}_`);
      }

      briefText = parts.length > 0 ? parts.join("\n\n") : JSON.stringify(analysis, null, 2);
    } else {
      briefText = "Analysis complete. Generating full briefing...";
    }
    return {
      brief_text: briefText,
      generated_at: new Date().toISOString(),
      ai_meta: ai_meta,
    };
  };

  // Determine step status for the pipeline stepper
  const getStepStatus = (idx) => {
    if (!pipeline) return "pending";
    if (idx < pipeline.phaseIndex) return "complete";
    if (idx === pipeline.phaseIndex) return pipeline.status;
    return "pending";
  };

  return (
    <>
      <div className="file-selector">
        <label htmlFor="file-select">Select a data file:</label>
        <select
          id="file-select"
          value={selectedFile}
          onChange={(e) => handleFileSelect(e.target.value)}
        >
          <option value="">-- Choose a file --</option>
          {files.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </select>
        <button
          className={`debug-toggle ${debugMode ? "debug-toggle--on" : ""}`}
          onClick={toggleDebug}
          title="Toggle debug mode — shows links to per-run log files"
        >
          {debugMode ? "Debug: ON" : "Debug: OFF"}
        </button>
        {debugMode && runId && (
          <a
            href={`${API_BASE}/logs/${runId}`}
            target="_blank"
            rel="noopener noreferrer"
            className="debug-log-link"
          >
            View Log ({runId})
          </a>
        )}
      </div>

      {error && <div className="error">{error}</div>}
      {loading && <div className="loading">Loading...</div>}

      {/* Pipeline Progress Stepper */}
      {selectedFile && pipeline && !pipelineComplete && !pipelineError && (
        <div className="pipeline-progress">
          {PHASE_NAMES.map((phaseName, idx) => {
            const status = getStepStatus(idx);
            return (
              <React.Fragment key={phaseName}>
                {idx > 0 && (
                  <div
                    className={`pipeline-connector ${
                      status !== "pending" ? "pipeline-connector--active" : ""
                    }`}
                  />
                )}
                <div className={`pipeline-step pipeline-step--${status}`}>
                  <div className="pipeline-step-indicator">
                    {status === "complete" ? "\u2713" : status === "error" ? "\u2717" : idx + 1}
                  </div>
                  <span className="pipeline-step-label">
                    {PHASE_LABELS[phaseName]}
                  </span>
                  {status === "in_progress" && (
                    <span className="pipeline-step-spinner" />
                  )}
                </div>
              </React.Fragment>
            );
          })}
        </div>
      )}

      {/* Show streaming brief text as it arrives during the brief phase */}
      {!pipelineComplete && !pipelineError && streamingBrief && (
        <BriefPanel
          briefResult={{
            brief_text: streamingBrief,
            generated_at: new Date().toISOString(),
            ai_meta: null,
          }}
          isInterim={true}
        />
      )}

      {pipelineComplete && (
        <>
          <div className="pipeline-done">Processing complete</div>
          {pipelineResults?.brief?.error && (
            <div className="error">Brief generation failed: {pipelineResults.brief.error}</div>
          )}
          {pipelineResults?.analyze?.error && (
            <div className="error">Analysis failed: {pipelineResults.analyze.error}</div>
          )}
          {pipelineResults?.brief && !pipelineResults.brief.error && (
            <BriefPanel briefResult={pipelineResults.brief} />
          )}
        </>
      )}

      {pipelineError && (
        <div className="error">Pipeline error: {pipelineError}</div>
      )}

      {pipelineComplete && pipelineResults && (
        <ChatPanel key={selectedFile} pipelineResults={pipelineResults} debugMode={debugMode} />
      )}

      {fileData && (
        <div className="bet-lens">
          <h2>{selectedFile}</h2>
          <DataExplorer data={fileData} />
        </div>
      )}
    </>
  );
}

const PAGE_TITLES = {
  "/": "BetLens",
  "/devnotes": "Dev Notes",
  "/ai-settings": "AI Settings",
};

function AppContent() {
  const location = useLocation();

  useEffect(() => {
    const pageName = PAGE_TITLES[location.pathname] || "Bet Stamp";
    document.title = `${pageName} — Bet Stamp`;
  }, [location.pathname]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Bet Stamp</h1>
        <Navigation />
      </header>

      <main className="App-main">
        <Routes>
          <Route path="/" element={<BetLens />} />
          <Route path="/devnotes" element={<DevNotes />} />
          <Route path="/ai-settings" element={<AISettings />} />
        </Routes>
      </main>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}

export default App;
