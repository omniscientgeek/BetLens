import React, { useState, useEffect, useRef, useCallback } from "react";
import { BrowserRouter, Routes, Route, Link, useLocation, useNavigate } from "react-router-dom";
import { io } from "socket.io-client";
import DevNotes from "./DevNotes";
import AISettings from "./AISettings";
import PastRuns from "./PastRuns";
import ActiveRunDetail from "./ActiveRunDetail";
import ChatPanel from "./ChatPanel";
import BriefPanel, { VerificationBadge } from "./BriefPanel";
import AnalyzeConversation from "./AnalyzeConversation";
import { API_BASE, SOCKET_URL, fetchWithRetry } from "./api";
import "./App.css"; // updated

const PHASE_LABELS = {
  detect: "Detect",
  analyze: "Analyze",
  audit_analyze: "Audit Analyze",
  brief: "Brief",
  audit_brief: "Audit Brief",
};
const PHASE_NAMES = ["detect", "analyze", "audit_analyze", "brief", "audit_brief"];

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

  const NAV_ITEMS = [
    { path: "/", label: "BetLens", icon: "◎" },
    { path: "/past-runs", label: "History", icon: "▤" },
    { path: "/devnotes", label: "Notes", icon: "◆" },
    { path: "/ai-settings", label: "Settings", icon: "⚙" },
  ];

  return (
    <nav className="App-nav">
      {NAV_ITEMS.map(({ path, label, icon }) => (
        <Link
          key={path}
          to={path}
          className={`nav-btn ${location.pathname === path ? "active" : ""}`}
        >
          <span className="nav-btn-icon">{icon}</span>
          <span className="nav-btn-label">{label}</span>
        </Link>
      ))}
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

  // Reconnection tracking
  const [reconnecting, setReconnecting] = useState(false);
  const reconnectAttemptsRef = useRef(0);
  const MAX_RECONNECT_RESTARTS = 3; // max times we'll auto-restart the pipeline

  // Pipeline results (used by BriefPanel and ChatPanel)
  const [pipelineResults, setPipelineResults] = useState(() => sessionGet("pipelineResults", null));

  // Incremental phase results (captured as each phase completes)
  const [phaseResults, setPhaseResults] = useState({});

  // Streaming brief text (accumulated from brief_chunk events)
  const [streamingBrief, setStreamingBrief] = useState("");

  // Streaming analyze conversation (built up from analyze_conversation events)
  const [analyzeConversation, setAnalyzeConversation] = useState(null);

  // Pipeline runtime timer
  const [pipelineStartTime, setPipelineStartTime] = useState(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  // Self-healing fix tracking
  const [fixStatus, setFixStatus] = useState({}); // { analyze: {attempt, max, verdict}, brief: {...} }

  // Save state
  const [saving, setSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null); // { ok, message }

  // Debug mode
  const [debugMode, setDebugMode] = useState(
    () => localStorage.getItem("betstamp_debug") === "true"
  );
  const [runId, setRunId] = useState(null);

  // Active runs polling — shows all currently running pipelines (including from other tabs)
  const [activeRuns, setActiveRuns] = useState([]);
  const activeRunsPollRef = useRef(null);

  // Viewing a specific active run inline
  const [viewingActiveRun, setViewingActiveRun] = useState(null);

  // Past runs — displayed inline when no file is selected
  const [pastRuns, setPastRuns] = useState([]);
  const [pastRunsLoading, setPastRunsLoading] = useState(false);
  const navigate = useNavigate();

  const pollActiveRuns = useCallback(async () => {
    try {
      const res = await fetchWithRetry(`${API_BASE}/active-runs`);
      if (!res.ok) return;
      const data = await res.json();
      setActiveRuns((data.runs || []).filter((r) => r.status === "running"));
    } catch {
      // Silently ignore polling errors
    }
  }, []);

  useEffect(() => {
    pollActiveRuns();
    activeRunsPollRef.current = setInterval(pollActiveRuns, 3000);
    return () => clearInterval(activeRunsPollRef.current);
  }, [pollActiveRuns]);

  const toggleDebug = () => {
    const next = !debugMode;
    setDebugMode(next);
    localStorage.setItem("betstamp_debug", String(next));
  };

  // Upload a new data file
  const fileInputRef = useRef(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null); // { ok, message }

  const handleUploadClick = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    // Reset the input so re-selecting the same file triggers onChange
    e.target.value = "";

    if (!file.name.endsWith(".json")) {
      setUploadStatus({ ok: false, message: "Only .json files are supported" });
      setTimeout(() => setUploadStatus(null), 5000);
      return;
    }

    setUploading(true);
    setUploadStatus(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetchWithRetry(`${API_BASE}/files/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: "Upload failed" }));
        throw new Error(err.error || `HTTP ${res.status}`);
      }

      const result = await res.json();

      // Refresh file list and auto-select the uploaded file
      const listRes = await fetchWithRetry(`${API_BASE}/files`);
      const listData = await listRes.json();
      setFiles(listData.files || []);

      setUploadStatus({ ok: true, message: `Uploaded ${result.filename}` });
      handleFileSelect(result.filename);
    } catch (err) {
      setUploadStatus({ ok: false, message: err.message });
    } finally {
      setUploading(false);
      setTimeout(() => setUploadStatus(null), 5000);
    }
  };

  // Save pipeline results to server + trigger JSON download
  const handleSaveResults = async () => {
    if (!pipelineResults || !selectedFile) return;
    setSaving(true);
    setSaveStatus(null);

    try {
      // 1. Save to server
      const res = await fetchWithRetry(`${API_BASE}/save-results`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: selectedFile,
          pipelineResults,
          fileData: fileData,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: "Save failed" }));
        throw new Error(err.error || `HTTP ${res.status}`);
      }

      const result = await res.json();

      // 2. Also trigger a browser download of the JSON
      // Restructure: promote sub-agent verification results to top-level
      // audit_analyze / audit_brief to mirror the 5-phase pipeline structure.
      const downloadResults = { ...pipelineResults };
      if (downloadResults.analyze?.verification) {
        const { verification, ...analyzeRest } = downloadResults.analyze;
        downloadResults.analyze = analyzeRest;
        downloadResults.audit_analyze = verification;
      }
      if (downloadResults.brief?.verification) {
        const { verification, ...briefRest } = downloadResults.brief;
        downloadResults.brief = briefRest;
        downloadResults.audit_brief = verification;
      }
      const blob = new Blob(
        [JSON.stringify({
          source_file: selectedFile,
          saved_at: new Date().toISOString(),
          pipeline_results: downloadResults,
        }, null, 2)],
        { type: "application/json" }
      );
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = result.filename || `betlens_results_${new Date().toISOString().slice(0, 19).replace(/[T:]/g, "_")}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setSaveStatus({ ok: true, message: `Saved as ${result.filename}` });
    } catch (err) {
      setSaveStatus({ ok: false, message: err.message });
    } finally {
      setSaving(false);
      // Clear status after a few seconds
      setTimeout(() => setSaveStatus(null), 5000);
    }
  };

  // Tick the elapsed timer every second while the pipeline is running
  useEffect(() => {
    if (!pipelineStartTime || pipelineComplete || pipelineError) return;
    const interval = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - pipelineStartTime) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [pipelineStartTime, pipelineComplete, pipelineError]);

  // Persist key state to sessionStorage so data survives page refresh
  useEffect(() => { sessionSet("selectedFile", selectedFile); }, [selectedFile]);
  useEffect(() => { sessionSet("fileData", fileData); }, [fileData]);
  useEffect(() => { sessionSet("pipelineComplete", pipelineComplete); }, [pipelineComplete]);
  useEffect(() => { sessionSet("pipelineResults", pipelineResults); }, [pipelineResults]);

  // Fetch available JSON files on mount (with retry)
  useEffect(() => {
    fetchWithRetry(`${API_BASE}/files`)
      .then((res) => res.json())
      .then((data) => setFiles(data.files || []))
      .catch((err) => setError("Failed to load file list: " + err.message));
  }, []);

  // Fetch past runs when no file is selected (landing state)
  useEffect(() => {
    if (selectedFile) return;
    setPastRunsLoading(true);
    fetchWithRetry(`${API_BASE}/saved-results`)
      .then((res) => res.json())
      .then((data) => {
        if (data.runs?.length) {
          setPastRuns(data.runs);
        } else {
          setPastRuns((data.files || []).map((f) => ({ filename: f })));
        }
      })
      .catch(() => setPastRuns([]))
      .finally(() => setPastRunsLoading(false));
  }, [selectedFile]);

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
    setAnalyzeConversation(null);
    setRunId(null);
    setPipelineStartTime(null);
    setElapsedSeconds(0);
    setSaving(false);
    setSaveStatus(null);
    setFixStatus({});

    if (!filename) {
      sessionRemove("selectedFile");
      sessionRemove("fileData");
      sessionRemove("pipelineComplete");
      sessionRemove("pipelineResults");
      return;
    }

    // Disconnect any previous socket
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }

    setLoading(true);
    try {
      const res = await fetchWithRetry(`${API_BASE}/files/${filename}`);
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
    // Auto-reconnect is enabled — on reconnect we resume the pipeline
    // from where it left off (completed phases are cached server-side).
    reconnectAttemptsRef.current = 0;
    const socket = io(SOCKET_URL, {
      transports: ["websocket", "polling"],
      reconnection: true,
      reconnectionAttempts: 15,
      reconnectionDelay: 2000,
      reconnectionDelayMax: 30000,
    });
    socketRef.current = socket;

    socket.on("connect", () => {
      const isReconnect = reconnectAttemptsRef.current > 0;
      setReconnecting(false);

      if (isReconnect) {
        // Reconnected — clear only streaming state; keep completed phase results.
        // Server will replay completed phases and resume live streaming.
        setStreamingBrief("");
        setPipelineError(null);
        socket.emit("start_processing", { filename, resume: true });
      } else {
        setPipelineStartTime(Date.now());
        socket.emit("start_processing", { filename });
      }
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

    // Stream analyze conversation events as they arrive
    socket.on("analyze_conversation", (data) => {
      const { event, data: eventData } = data;
      if (event === "prompts") {
        setAnalyzeConversation({
          system_prompt: eventData.system_prompt,
          user_prompt: eventData.user_prompt,
          assistant_response: "",
          thinking: null,
          tool_calls: [],
          streaming: true,
        });
      } else if (event === "chunk") {
        setAnalyzeConversation((prev) => {
          if (!prev) return prev;
          return { ...prev, assistant_response: (prev.assistant_response || "") + (eventData.text || "") };
        });
      } else if (event === "tool_call") {
        // Live tool call event — append to tool_calls array during streaming
        setAnalyzeConversation((prev) => {
          if (!prev) return prev;
          const existing = prev.tool_calls || [];
          return { ...prev, tool_calls: [...existing, eventData] };
        });
      } else if (event === "tool_result") {
        // Live tool result event — attach result to matching tool call
        setAnalyzeConversation((prev) => {
          if (!prev) return prev;
          const updated = (prev.tool_calls || []).map((tc) =>
            tc.id === eventData.tool_use_id
              ? { ...tc, result: eventData.result, is_error: eventData.is_error || false }
              : tc
          );
          return { ...prev, tool_calls: updated };
        });
      } else if (event === "complete") {
        setAnalyzeConversation((prev) => ({
          ...(prev || {}),
          ...eventData.conversation,
          ai_meta: eventData.ai_meta,
          streaming: false,
        }));
      }
    });

    // Accumulate streaming brief text chunks as they arrive
    socket.on("brief_chunk", (data) => {
      if (data.text) {
        setStreamingBrief((prev) => prev + data.text);
      }
    });

    // Receive per-agent verification results in real time as each audit agent finishes
    socket.on("verification_agent_update", (data) => {
      if (data.agent_name && data.agent_result) {
        const phase = data.phase || "brief";
        setPhaseResults((prev) => {
          const existing = prev[phase] || {};
          const existingVerification = existing.verification || {
            overall_verdict: "pending",
            elapsed_seconds: null,
            agents: {},
            _pending: true,
          };
          const updatedAgents = {
            ...existingVerification.agents,
            [data.agent_name]: data.agent_result,
          };
          return {
            ...prev,
            [phase]: {
              ...existing,
              verification: {
                ...existingVerification,
                agents: updatedAgents,
              },
            },
          };
        });
      }
    });

    // Receive final verification results (arrives after all agents finish, before processing_complete)
    socket.on("verification_update", (data) => {
      if (data.verification) {
        const phase = data.phase || "brief";
        setPhaseResults((prev) => ({
          ...prev,
          [phase]: { ...prev[phase], verification: data.verification },
        }));
      }
    });

    // Self-healing: fix started — AI is correcting audit failures
    socket.on("fix_started", (data) => {
      const phase = data.phase || "brief";
      setFixStatus((prev) => ({
        ...prev,
        [phase]: {
          attempt: data.fix_attempt,
          max: data.max_attempts,
          previousVerdict: data.previous_verdict,
          status: "fixing",
        },
      }));
      // Clear previous verification agents for this phase so the UI shows fresh results
      setPhaseResults((prev) => {
        const existing = prev[phase] || {};
        return {
          ...prev,
          [phase]: { ...existing, verification: undefined },
        };
      });
      // For brief fixes, clear streaming text so new fixed text can stream in
      if (phase === "brief") {
        setStreamingBrief("");
      }
    });

    // Self-healing: fix complete — re-auditing now
    socket.on("fix_complete", (data) => {
      const phase = data.phase || "brief";
      setFixStatus((prev) => ({
        ...prev,
        [phase]: {
          ...prev[phase],
          status: "re-auditing",
        },
      }));
    });

    // Heartbeat from server — reset any "stale" timers / keep UI alive
    socket.on("heartbeat", (data) => {
      // Update pipeline status so user sees the phase is still active
      if (data.phase) {
        setPipeline((prev) => prev ? { ...prev, phase: data.phase, status: "in_progress" } : prev);
      }
    });

    socket.on("processing_complete", (data) => {
      if (data.run_id) setRunId(data.run_id);
      setPipelineComplete(true);
      // Merge streaming phaseResults verification into server results as fallback
      // so audit data captured during streaming isn't lost
      const serverResults = data.results || null;
      if (serverResults) {
        setPhaseResults((prev) => {
          const merged = { ...serverResults };
          // If server results lack verification but streaming captured it, fill in
          if (!merged.analyze?.verification && prev.analyze?.verification) {
            merged.analyze = { ...merged.analyze, verification: prev.analyze.verification };
          }
          if (!merged.brief?.verification && prev.brief?.verification) {
            merged.brief = { ...merged.brief, verification: prev.brief.verification };
          }
          setPipelineResults(merged);
          return prev;
        });
      } else {
        setPipelineResults(null);
      }
      setStreamingBrief(""); // Clear streaming text — final brief is in results
      setSaveStatus({ ok: true, message: "Results saved automatically" });
      setTimeout(() => setSaveStatus(null), 5000);
      socket.disconnect();
      socketRef.current = null;
    });

    socket.on("processing_error", (data) => {
      setPipelineError(data.error);
      socket.disconnect();
      socketRef.current = null;
    });

    // Handle unexpected disconnects — attempt auto-reconnect + pipeline restart
    socket.on("disconnect", (reason) => {
      // Ignore intentional disconnects (we call socket.disconnect() on complete/error)
      if (reason === "io client disconnect") return;

      reconnectAttemptsRef.current += 1;

      if (reconnectAttemptsRef.current <= MAX_RECONNECT_RESTARTS) {
        // Socket.IO will auto-reconnect; show "reconnecting" state instead of error
        setReconnecting(true);
        setPipelineError(null);
      } else {
        // Exhausted auto-restart attempts — give up and show error
        setPipelineError(
          `Connection lost (${reason}) after ${reconnectAttemptsRef.current} reconnection attempts. Please try again.`
        );
        socket.disconnect();
        socketRef.current = null;
      }
    });

    // If Socket.IO itself gives up reconnecting (all attempts exhausted)
    socket.io.on("reconnect_failed", () => {
      setReconnecting(false);
      setPipelineError(
        "Unable to reconnect to the server. Please check your connection and try again."
      );
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

      // AI audit notes
      if (analysis.ai_verification_notes) {
        parts.push(`## Audit Notes\n_${analysis.ai_verification_notes}_`);
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

  // Format elapsed seconds as m:ss or h:mm:ss
  const formatElapsed = (totalSec) => {
    const hrs = Math.floor(totalSec / 3600);
    const mins = Math.floor((totalSec % 3600) / 60);
    const secs = totalSec % 60;
    if (hrs > 0) return `${hrs}:${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
    return `${mins}:${String(secs).padStart(2, "0")}`;
  };

  // Determine step status for the 5-step pipeline
  // Server sends 5 phases: 0=Detect, 1=Analyze, 2=Audit Analyze, 3=Brief, 4=Audit Brief
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
        <input
          type="file"
          accept=".json"
          ref={fileInputRef}
          onChange={handleFileUpload}
          style={{ display: "none" }}
        />
        <button
          className="upload-btn"
          onClick={handleUploadClick}
          disabled={uploading}
          title="Upload a new JSON data file"
        >
          {uploading ? "Uploading…" : "Upload File"}
        </button>
        {uploadStatus && (
          <span className={`upload-status ${uploadStatus.ok ? "upload-status--ok" : "upload-status--err"}`}>
            {uploadStatus.message}
          </span>
        )}
        {selectedFile && (pipeline || pipelineComplete || pipelineError) && (
          <button
            className="regenerate-btn"
            onClick={() => handleFileSelect(selectedFile)}
            title="Re-run the analysis pipeline for this file"
          >
            Regenerate
          </button>
        )}
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

      {/* Active Runs — shows pipelines currently running (including from other tabs/sessions) */}
      {activeRuns.length > 0 && !viewingActiveRun && (
        <div className="bl-active-runs">
          <div className="bl-active-runs-header">
            <span className="bl-active-pulse" />
            <span className="bl-active-runs-title">
              {activeRuns.length} pipeline{activeRuns.length !== 1 ? "s" : ""} running
            </span>
          </div>
          <div className="bl-active-runs-list">
            {activeRuns.map((run) => (
              <button
                key={run.run_id}
                className="bl-active-run"
                onClick={() => setViewingActiveRun(run.filename)}
              >
                <span className="bl-active-run-spinner" />
                <span className="bl-active-run-file">{run.filename}</span>
                <span className="bl-active-run-phase">
                  {run.current_phase_label || run.current_phase}
                  {" "}({run.phase_index + 1}/{run.total_phases})
                </span>
                <span className="bl-active-run-timer">
                  {(() => {
                    const t = run.elapsed_seconds;
                    const m = Math.floor(t / 60);
                    const s = t % 60;
                    return `${m}:${String(s).padStart(2, "0")}`;
                  })()}
                </span>
                <span className="bl-active-run-arrow">›</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Active Run Detail — inline view when a run is selected */}
      {viewingActiveRun && (
        <ActiveRunDetail
          filename={viewingActiveRun}
          onBack={() => setViewingActiveRun(null)}
          backLabel="Close"
        />
      )}

      {/* Past Runs — shown inline when no file/run is selected */}
      {!selectedFile && !pipeline && !pipelineComplete && (
        <div className="bl-past-runs">
          <div className="bl-past-runs-header">
            <h3>Recent Runs</h3>
            <Link to="/past-runs" className="bl-past-runs-link">
              View all in History <span aria-hidden>›</span>
            </Link>
          </div>
          {pastRunsLoading ? (
            <div className="bl-past-runs-loading">
              <span className="pr-loading-spinner" />
              Loading past runs...
            </div>
          ) : pastRuns.length === 0 ? (
            <div className="bl-past-runs-empty">
              <span className="bl-past-runs-empty-icon">📭</span>
              <p>No saved runs yet. Select a data file above to start your first analysis.</p>
            </div>
          ) : (
            <div className="pr-list">
              {pastRuns.slice(0, 10).map((run) => {
                const match = run.filename.match(/(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})(\d{2})\.json$/);
                const dateStr = match ? `${match[1]} at ${match[2]}:${match[3]}:${match[4]}` : run.filename;
                return (
                  <button
                    key={run.filename}
                    className="pr-list-item"
                    onClick={() => navigate(`/past-runs`, { state: { openFile: run.filename } })}
                  >
                    <div className="pr-list-item-main">
                      <span className="pr-list-item-icon">📊</span>
                      <div className="pr-list-item-info">
                        <span className="pr-list-item-source">
                          {run.source_file || run.filename}
                        </span>
                        <span className="pr-list-item-date">{dateStr}</span>
                      </div>
                    </div>
                    <div className="pr-list-item-verdicts">
                      {run.analyze_verdict && (
                        <div className="pr-list-item-verdict">
                          <span className="pr-verdict-label">Analyze</span>
                          <span className={`pr-verdict-pill verdict--${run.analyze_verdict}`}>
                            {run.analyze_verdict}
                          </span>
                        </div>
                      )}
                      {run.brief_verdict && (
                        <div className="pr-list-item-verdict">
                          <span className="pr-verdict-label">Brief</span>
                          <span className={`pr-verdict-pill verdict--${run.brief_verdict}`}>
                            {run.brief_verdict}
                          </span>
                        </div>
                      )}
                    </div>
                    <span className="pr-list-item-arrow">›</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      )}

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
          <div className="pipeline-timer">
            {formatElapsed(elapsedSeconds)}
          </div>
        </div>
      )}

      {/* Reconnecting banner */}
      {reconnecting && (
        <div className="reconnecting-banner">
          <span className="reconnecting-spinner" />
          Connection lost — reconnecting and resuming analysis
          <span className="reconnecting-attempt">
            (attempt {reconnectAttemptsRef.current} of {MAX_RECONNECT_RESTARTS})
          </span>
        </div>
      )}

      {/* Show streaming analyze conversation as it arrives during the analyze phase */}
      {!pipelineComplete && !pipelineError && analyzeConversation && (
        <AnalyzeConversation
          analyzeResult={{ conversation: analyzeConversation, ai_meta: analyzeConversation.ai_meta }}
          streaming={analyzeConversation.streaming}
          defaultExpanded={true}
        />
      )}

      {/* Audit Analyze — show during processing with real-time agent updates + self-healing */}
      {!pipelineComplete && !pipelineError && pipeline && pipeline.phaseIndex >= 2 && (() => {
        const auditV = phaseResults.analyze?.verification;
        const agents = auditV?.agents || {};
        const completedCount = Object.keys(agents).length;
        const totalAgents = 3;
        const progressPct = (completedCount / totalAgents) * 100;
        const isAuditing = !auditV || auditV._pending;

        return (
          <div className="verification-card">
            <div className="verification-card-header">
              <span className="verification-card-icon">{"\u{1F6E1}"}</span>
              <h3>Audit Analyze</h3>
              {isAuditing && (
                <span className="audit-phase-badge">
                  <span className="pipeline-step-spinner" />
                  {completedCount}/{totalAgents} agents
                </span>
              )}
              {fixStatus.analyze && (
                <span className="fix-badge">
                  {fixStatus.analyze.status === "fixing"
                    ? `Fixing (${fixStatus.analyze.attempt}/${fixStatus.analyze.max})`
                    : fixStatus.analyze.status === "re-auditing"
                    ? `Re-auditing (${fixStatus.analyze.attempt}/${fixStatus.analyze.max})`
                    : null}
                </span>
              )}
            </div>

            {/* Segmented progress bar */}
            {isAuditing && (
              <div className="audit-agent-progress">
                <div className="audit-agent-progress-bar">
                  {["reasoning", "factual", "betting"].map((name) => {
                    const a = agents[name];
                    const verdict = a?.verdict || "pending";
                    return (
                      <div
                        key={name}
                        className={`audit-agent-segment audit-agent-segment--${verdict}`}
                        title={`${name}: ${a ? verdict : "running…"}`}
                      />
                    );
                  })}
                </div>
                <span className="audit-agent-progress-label">{Math.round(progressPct)}%</span>
              </div>
            )}

            {/* Per-agent status chips during streaming */}
            {isAuditing && completedCount > 0 && (
              <div className="audit-agent-chips">
                {["reasoning", "factual", "betting"].map((name) => {
                  const a = agents[name];
                  const label = name === "reasoning" ? "Reasoning" : name === "factual" ? "Fact Check" : "Bet Quality";
                  return (
                    <div key={name} className={`audit-agent-chip audit-agent-chip--${a ? a.verdict : "pending"}`}>
                      <span className="audit-agent-chip-icon">
                        {a ? (a.verdict === "pass" ? "\u2705" : a.verdict === "warn" ? "\u26A0\uFE0F" : "\u274C")
                          : <span className="pipeline-step-spinner" />}
                      </span>
                      <span className="audit-agent-chip-name">{label}</span>
                      {a && <span className="audit-agent-chip-verdict">{a.verdict}</span>}
                      {a?.summary && <span className="audit-agent-chip-summary">{a.summary}</span>}
                    </div>
                  );
                })}
              </div>
            )}

            {fixStatus.analyze?.status === "fixing" && (
              <div className="fix-status-banner">
                <span className="pipeline-step-spinner" />
                <span>
                  Audit failed with <strong>{fixStatus.analyze.previousVerdict}</strong> verdict
                  — AI is fixing issues (attempt {fixStatus.analyze.attempt} of {fixStatus.analyze.max})…
                </span>
              </div>
            )}

            {auditV ? (
              <VerificationBadge
                verification={auditV}
                streaming={!!auditV._pending}
              />
            ) : (
              <div className="verification-card-pending">
                <span className="pipeline-step-spinner" />
                <span>
                  {fixStatus.analyze?.status === "re-auditing"
                    ? `Re-auditing after fix ${fixStatus.analyze.attempt}…`
                    : "Auditing analysis…"}
                </span>
              </div>
            )}
          </div>
        );
      })()}

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

      {/* Audit Brief — show during processing with real-time agent updates + self-healing */}
      {!pipelineComplete && !pipelineError && pipeline && pipeline.phaseIndex >= 4 && (() => {
        const auditV = phaseResults.brief?.verification;
        const agents = auditV?.agents || {};
        const completedCount = Object.keys(agents).length;
        const totalAgents = 3;
        const progressPct = (completedCount / totalAgents) * 100;
        const isAuditing = !auditV || auditV._pending;

        return (
          <div className="verification-card">
            <div className="verification-card-header">
              <span className="verification-card-icon">{"\u{1F6E1}"}</span>
              <h3>Audit Brief</h3>
              {isAuditing && (
                <span className="audit-phase-badge">
                  <span className="pipeline-step-spinner" />
                  {completedCount}/{totalAgents} agents
                </span>
              )}
              {fixStatus.brief && (
                <span className="fix-badge">
                  {fixStatus.brief.status === "fixing"
                    ? `Fixing (${fixStatus.brief.attempt}/${fixStatus.brief.max})`
                    : fixStatus.brief.status === "re-auditing"
                    ? `Re-auditing (${fixStatus.brief.attempt}/${fixStatus.brief.max})`
                    : null}
                </span>
              )}
            </div>

            {/* Segmented progress bar */}
            {isAuditing && (
              <div className="audit-agent-progress">
                <div className="audit-agent-progress-bar">
                  {["reasoning", "factual", "betting"].map((name) => {
                    const a = agents[name];
                    const verdict = a?.verdict || "pending";
                    return (
                      <div
                        key={name}
                        className={`audit-agent-segment audit-agent-segment--${verdict}`}
                        title={`${name}: ${a ? verdict : "running…"}`}
                      />
                    );
                  })}
                </div>
                <span className="audit-agent-progress-label">{Math.round(progressPct)}%</span>
              </div>
            )}

            {/* Per-agent status chips during streaming */}
            {isAuditing && completedCount > 0 && (
              <div className="audit-agent-chips">
                {["reasoning", "factual", "betting"].map((name) => {
                  const a = agents[name];
                  const label = name === "reasoning" ? "Reasoning" : name === "factual" ? "Fact Check" : "Bet Quality";
                  return (
                    <div key={name} className={`audit-agent-chip audit-agent-chip--${a ? a.verdict : "pending"}`}>
                      <span className="audit-agent-chip-icon">
                        {a ? (a.verdict === "pass" ? "\u2705" : a.verdict === "warn" ? "\u26A0\uFE0F" : "\u274C")
                          : <span className="pipeline-step-spinner" />}
                      </span>
                      <span className="audit-agent-chip-name">{label}</span>
                      {a && <span className="audit-agent-chip-verdict">{a.verdict}</span>}
                      {a?.summary && <span className="audit-agent-chip-summary">{a.summary}</span>}
                    </div>
                  );
                })}
              </div>
            )}

            {fixStatus.brief?.status === "fixing" && (
              <div className="fix-status-banner">
                <span className="pipeline-step-spinner" />
                <span>
                  Audit failed with <strong>{fixStatus.brief.previousVerdict}</strong> verdict
                  — AI is fixing issues (attempt {fixStatus.brief.attempt} of {fixStatus.brief.max})…
                </span>
              </div>
            )}

            {auditV ? (
              <VerificationBadge
                verification={auditV}
                streaming={!!auditV._pending}
              />
            ) : (
              <div className="verification-card-pending">
                <span className="pipeline-step-spinner" />
                <span>
                  {fixStatus.brief?.status === "re-auditing"
                    ? `Re-auditing after fix ${fixStatus.brief.attempt}…`
                    : "Auditing brief…"}
                </span>
              </div>
            )}
          </div>
        );
      })()}

      {pipelineComplete && (
        <>
          <div className="pipeline-done">
            Processing complete in {formatElapsed(elapsedSeconds)}
            {saveStatus && (
              <span className={`save-status ${saveStatus.ok ? "save-status--ok" : "save-status--err"}`}>
                {saveStatus.message}
              </span>
            )}
            <button
              className="save-results-btn"
              onClick={handleSaveResults}
              disabled={saving || !pipelineResults}
              title="Save results to server and download as JSON"
            >
              {saving ? "Saving..." : "Download Results"}
            </button>
          </div>
          {pipelineResults?.brief?.error && (
            <div className="error">Brief generation failed: {pipelineResults.brief.error}</div>
          )}
          {pipelineResults?.analyze?.error && (
            <div className="error">Analysis failed: {pipelineResults.analyze.error}</div>
          )}

          {/* === ANALYSIS SECTION === */}
          {(() => {
            const analyzeV = pipelineResults?.analyze?.verification;
            const analyzeHistory = analyzeV?.fix_history || [];
            const analyzeIsDraft = analyzeV && analyzeV.overall_verdict !== "pass";

            return (
              <>
                {/* Analysis — with draft/verified badge */}
                {pipelineResults?.analyze && (
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
                    <AnalyzeConversation analyzeResult={pipelineResults.analyze} />
                  </div>
                )}

                {/* Audit Analyze — each attempt as a separate block */}
                {analyzeHistory.length > 0 ? (
                  <div className="audit-timeline">
                    <div className="audit-timeline-header">
                      <span className="audit-timeline-icon">{"\u{1F6E1}"}</span>
                      <h3>Audit Analyze Timeline</h3>
                      <span className="audit-timeline-count">
                        {analyzeHistory.length} audit{analyzeHistory.length !== 1 ? "s" : ""}
                      </span>
                    </div>
                    {analyzeHistory.map((h, idx) => {
                      const isLatest = idx === analyzeHistory.length - 1;
                      const isPassed = h.verdict === "pass";
                      const issueCount = Object.values(h.verification?.agents || {}).reduce(
                        (sum, a) => sum + (a.issues?.length || 0), 0
                      );
                      return (
                        <div key={idx} className={`audit-block audit-block--${h.verdict} ${isLatest ? "audit-block--latest" : "audit-block--historical"}`}>
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
                    })}
                  </div>
                ) : analyzeV && (
                  <div className="verification-card">
                    <div className="verification-card-header">
                      <span className="verification-card-icon">{"\u{1F6E1}"}</span>
                      <h3>Audit Analyze</h3>
                    </div>
                    <VerificationBadge verification={analyzeV} />
                  </div>
                )}
              </>
            );
          })()}

          {/* === BRIEF SECTION === */}
          {(() => {
            const briefV = pipelineResults?.brief?.verification;
            const briefHistory = briefV?.fix_history || [];
            const briefIsDraft = briefV && briefV.overall_verdict !== "pass";
            const hasRevisions = briefHistory.length > 1;

            return (
              <>
                {/* Brief revisions — each superseded version as a separate block */}
                {hasRevisions && (
                  <div className="brief-revisions">
                    <div className="brief-revisions-header">
                      <span className="brief-revisions-icon">{"\u{1F4DD}"}</span>
                      <h3>Brief Revisions</h3>
                      <span className="brief-revisions-count">
                        {briefHistory.length} version{briefHistory.length !== 1 ? "s" : ""}
                      </span>
                    </div>
                    {briefHistory.slice(0, -1).map((h, idx) => (
                      <div key={idx} className="brief-revision-block brief-revision-block--superseded">
                        <div className="brief-revision-header">
                          <span className="brief-revision-indicator">{"\u{1F4C4}"}</span>
                          <span className="brief-revision-title">
                            {h.attempt === 0 ? "Original Brief" : `Revision #${h.attempt}`}
                          </span>
                          <span className={`brief-revision-verdict verdict--${h.verdict}`}>
                            Audit: {h.verdict.toUpperCase()}
                          </span>
                          <span className="brief-revision-superseded-tag">Superseded</span>
                        </div>
                        {h.text && (
                          <BriefPanel
                            briefResult={{
                              brief_text: h.text,
                              generated_at: pipelineResults.brief.generated_at,
                              ai_meta: null,
                            }}
                          />
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* Current/final brief — with draft or verified badge */}
                {pipelineResults?.brief && !pipelineResults.brief.error && (
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
                    {hasRevisions && (
                      <div className="brief-revision-current-tag">
                        Current Version (Revision #{briefHistory[briefHistory.length - 1]?.attempt || 0})
                      </div>
                    )}
                    <BriefPanel briefResult={pipelineResults.brief} />
                  </div>
                )}

                {/* Audit Brief — each attempt as a separate block */}
                {briefHistory.length > 0 ? (
                  <div className="audit-timeline">
                    <div className="audit-timeline-header">
                      <span className="audit-timeline-icon">{"\u{1F6E1}"}</span>
                      <h3>Audit Brief Timeline</h3>
                      <span className="audit-timeline-count">
                        {briefHistory.length} audit{briefHistory.length !== 1 ? "s" : ""}
                      </span>
                    </div>
                    {briefHistory.map((h, idx) => {
                      const isLatest = idx === briefHistory.length - 1;
                      const isPassed = h.verdict === "pass";
                      const issueCount = Object.values(h.verification?.agents || {}).reduce(
                        (sum, a) => sum + (a.issues?.length || 0), 0
                      );
                      return (
                        <div key={idx} className={`audit-block audit-block--${h.verdict} ${isLatest ? "audit-block--latest" : "audit-block--historical"}`}>
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
                    })}
                  </div>
                ) : briefV && (
                  <div className="verification-card">
                    <div className="verification-card-header">
                      <span className="verification-card-icon">{"\u{1F6E1}"}</span>
                      <h3>Audit Brief</h3>
                    </div>
                    <VerificationBadge verification={briefV} />
                  </div>
                )}
              </>
            );
          })()}
        </>
      )}

      {pipelineError && (
        <div className="error">Pipeline error: {pipelineError}</div>
      )}

      {pipelineComplete && pipelineResults && (
        <ChatPanel key={selectedFile} pipelineResults={pipelineResults} debugMode={debugMode} />
      )}

    </>
  );
}

const PAGE_TITLES = {
  "/": "BetLens",
  "/past-runs": "History",
  "/devnotes": "Notes",
  "/ai-settings": "Settings",
};

function AppContent() {
  const location = useLocation();

  useEffect(() => {
    const pageName = PAGE_TITLES[location.pathname] || "BetStamp";
    document.title = `${pageName} — BetStamp`;
  }, [location.pathname]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>BetStamp</h1>
        <Navigation />
      </header>

      <main className="App-main">
        <Routes>
          <Route path="/" element={<BetLens />} />
          <Route path="/past-runs" element={<PastRuns />} />
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
