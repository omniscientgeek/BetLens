import React, { useState, useEffect, useRef } from "react";
import { BrowserRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import { io } from "socket.io-client";
import DevNotes from "./DevNotes";
import DataExplorer from "./DataExplorer";
import AISettings from "./AISettings";
import ChatPanel from "./ChatPanel";
import "./App.css";

const API_BASE = "";
const SOCKET_URL = "";

const PHASE_LABELS = { detect: "Detect", analyze: "Analyze", brief: "Brief" };
const PHASE_NAMES = ["detect", "analyze", "brief"];

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
  const [selectedFile, setSelectedFile] = useState("");
  const [fileData, setFileData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Pipeline state
  const [pipeline, setPipeline] = useState(null);
  const [pipelineComplete, setPipelineComplete] = useState(false);
  const [pipelineError, setPipelineError] = useState(null);
  const socketRef = useRef(null);

  // Chat state
  const [chatOpen, setChatOpen] = useState(false);
  const [pipelineResults, setPipelineResults] = useState(null);

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
      setPipeline({
        phase: data.phase,
        status: data.status,
        phaseIndex: data.phaseIndex,
        totalPhases: data.totalPhases,
        label: data.label,
      });
    });

    socket.on("processing_complete", (data) => {
      setPipelineComplete(true);
      setPipelineResults(data.results || null);
      socket.disconnect();
      socketRef.current = null;
    });

    socket.on("processing_error", (data) => {
      setPipelineError(data.error);
      socket.disconnect();
      socketRef.current = null;
    });
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
                    {status === "complete" ? "\u2713" : idx + 1}
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

      {pipelineComplete && (
        <div className="pipeline-done">Processing complete</div>
      )}

      {pipelineError && (
        <div className="error">Pipeline error: {pipelineError}</div>
      )}

      {fileData && (
        <div className="bet-lens">
          <h2>{selectedFile}</h2>
          <DataExplorer data={fileData} />
        </div>
      )}

      <button
        className="chat-fab"
        onClick={() => setChatOpen(!chatOpen)}
        title="Chat with AI Assistant"
      >
        {chatOpen ? '\u00d7' : '\ud83d\udcac'}
      </button>
      <ChatPanel
        pipelineResults={pipelineResults}
        isOpen={chatOpen}
        onToggle={() => setChatOpen(!chatOpen)}
      />
    </>
  );
}

function AppContent() {
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
