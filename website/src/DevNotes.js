import React, { useState, useEffect } from "react";
import { API_BASE } from "./api";
const PROJECT_ID = 10;

const isRailway = window.location.hostname.endsWith(".railway.app");

function DevNotes() {
  const [conversations, setConversations] = useState([]);
  const [notes, setNotes] = useState([]);
  const [gitStats, setGitStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [generated, setGenerated] = useState(false);
  const [mdReady, setMdReady] = useState(false);
  const [expandedId, setExpandedId] = useState(null);
  const [sortOrder, setSortOrder] = useState("newest");
  const [filter, setFilter] = useState("both"); // "both", "conversations", "notes"

  // Auto-load saved devnotes on mount
  useEffect(() => {
    const loadSaved = async () => {
      try {
        const res = await fetch(`${API_BASE}/devnotes/${PROJECT_ID}`);
        if (!res.ok) return;
        const data = await res.json();
        setConversations(data.conversations || []);
        setGenerated(true);
      } catch {
        // Silent fail on auto-load
      }
    };
    loadSaved();
  }, []);

  // Auto-load notes from NOTES.md on mount
  useEffect(() => {
    const loadNotes = async () => {
      try {
        const res = await fetch(`${API_BASE}/notes`);
        if (!res.ok) return;
        const data = await res.json();
        setNotes(data.notes || []);
      } catch {
        // Silent fail on auto-load
      }
    };
    loadNotes();
  }, []);

  // Auto-load saved git stats on mount
  useEffect(() => {
    const loadGitStats = async () => {
      try {
        const res = await fetch(`${API_BASE}/git-stats/saved`);
        if (!res.ok) return;
        const data = await res.json();
        setGitStats(data);
      } catch {
        // Silent fail on auto-load
      }
    };
    loadGitStats();
  }, []);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    setConversations([]);
    setNotes([]);
    setGenerated(false);
    setMdReady(false);

    try {
      // Fetch conversations, git stats, devnotes, and generate combined MD in parallel
      const [convRes, gitRes, notesRes, mdRes] = await Promise.all([
        fetch(`${API_BASE}/conversations?project_id=${PROJECT_ID}`),
        fetch(`${API_BASE}/git-stats`),
        fetch(`${API_BASE}/notes`),
        fetch(`${API_BASE}/generate-devnotes-md?project_id=${PROJECT_ID}`, { method: "POST" }),
      ]);

      if (!convRes.ok) {
        const data = await convRes.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${convRes.status}`);
      }
      const convData = await convRes.json();
      setConversations(convData.conversations || []);

      if (gitRes.ok) {
        const gitData = await gitRes.json();
        setGitStats(gitData);
      }

      if (notesRes.ok) {
        const notesData = await notesRes.json();
        setNotes(notesData.notes || []);
      }

      if (mdRes.ok) {
        setMdReady(true);
      } else {
        console.warn("Failed to generate combined MD file");
      }

      setGenerated(true);
    } catch (err) {
      setError("Failed to load conversations: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const toggleExpand = (id) => {
    setExpandedId(expandedId === id ? null : id);
  };

  // Build combined display items
  const conversationItems = conversations.map((c) => ({
    ...c,
    itemType: "conversation",
    itemId: c.filename,
    sortDate: (c.messages && c.messages.length > 0 ? c.messages[0].timestamp : c.created) || "",
  }));

  const noteItems = notes.map((n) => ({
    ...n,
    itemType: "note",
    itemId: `note-${n.id}`,
    sortDate: (n.created || "").replace("T", " "),
  }));

  let displayItems = [];
  if (filter === "conversations" || filter === "both") {
    displayItems = displayItems.concat(conversationItems);
  }
  if (filter === "notes" || filter === "both") {
    displayItems = displayItems.concat(noteItems);
  }

  displayItems.sort((a, b) => {
    return sortOrder === "newest"
      ? b.sortDate.localeCompare(a.sortDate)
      : a.sortDate.localeCompare(b.sortDate);
  });

  const filterLabel =
    filter === "both"
      ? "item"
      : filter === "conversations"
      ? "conversation"
      : "note";

  return (
    <div className="dev-notes">
      <div className="dev-notes-controls">
        <button
          className="generate-btn"
          onClick={handleGenerate}
          disabled={loading || isRailway}
          title={isRailway ? "Not available in production" : undefined}
        >
          {loading ? "Generating..." : isRailway ? "Unavailable" : "Generate"}
        </button>

        {mdReady && (
          <a
            className="generate-btn"
            href={`${API_BASE}/download-devnotes-md`}
            download="devnotes_combined.MD"
            style={{ textDecoration: "none" }}
          >
            ⬇ Download MD
          </a>
        )}

        <div className="filter-group">
          {[
            { key: "both", label: "Both" },
            { key: "conversations", label: "AI Conversations" },
            { key: "notes", label: "DevNotes" },
          ].map((f) => (
            <button
              key={f.key}
              className={`filter-btn ${filter === f.key ? "active" : ""}`}
              onClick={() => setFilter(f.key)}
            >
              {f.label}
            </button>
          ))}
        </div>

        {displayItems.length > 0 && (
          <button
            className="sort-btn"
            onClick={() =>
              setSortOrder(sortOrder === "newest" ? "oldest" : "newest")
            }
          >
            {sortOrder === "newest" ? "↓ Newest First" : "↑ Oldest First"}
          </button>
        )}

        {(generated || notes.length > 0) && (
          <span className="conversation-count">
            {displayItems.length} {filterLabel}
            {displayItems.length !== 1 ? "s" : ""} found
          </span>
        )}
      </div>

      {gitStats && gitStats.summary && (
        <div className="git-stats-card">
          <h3>Git Contribution Stats</h3>
          <div className="git-stats-bars">
            <div className="git-stat-row">
              <span className="git-stat-label">Claude</span>
              <div className="git-stat-bar-track">
                <div
                  className="git-stat-bar-fill git-stat-bar-claude"
                  style={{ width: `${gitStats.summary.claude_percentage}%` }}
                />
              </div>
              <span className="git-stat-value">
                {gitStats.summary.claude_percentage}%
              </span>
              <span className="git-stat-count">
                ({gitStats.summary.claude_commits} commits)
              </span>
            </div>
            {gitStats.summary.autosync_commits > 0 && (
              <div className="git-stat-row">
                <span className="git-stat-label">AutoSync</span>
                <div className="git-stat-bar-track">
                  <div
                    className="git-stat-bar-fill git-stat-bar-autosync"
                    style={{ width: `${gitStats.summary.autosync_percentage}%` }}
                  />
                </div>
                <span className="git-stat-value">
                  {gitStats.summary.autosync_percentage}%
                </span>
                <span className="git-stat-count">
                  ({gitStats.summary.autosync_commits} commits)
                </span>
              </div>
            )}
            <div className="git-stat-row">
              <span className="git-stat-label">User</span>
              <div className="git-stat-bar-track">
                <div
                  className="git-stat-bar-fill git-stat-bar-user"
                  style={{ width: `${gitStats.summary.user_percentage}%` }}
                />
              </div>
              <span className="git-stat-value">
                {gitStats.summary.user_percentage}%
              </span>
              <span className="git-stat-count">
                ({gitStats.summary.user_commits} commits)
              </span>
            </div>
          </div>
          <div className="git-stats-footer">
            {gitStats.summary.total_commits} total commits &middot; Generated{" "}
            {new Date(gitStats.summary.generated_at).toLocaleString()}
          </div>
        </div>
      )}

      {error && <div className="error">{error}</div>}
      {loading && (
        <div className="loading">Reading conversation files...</div>
      )}

      {generated &&
        displayItems.length === 0 &&
        !loading && (
          <div className="empty-state">
            No {filter === "both" ? "items" : filterLabel + "s"} found.
          </div>
        )}

      <div className="conversation-list">
        {displayItems.map((item) =>
          item.itemType === "conversation" ? (
            <div key={item.itemId} className="conversation-card">
              <div
                className="conversation-header"
                onClick={() => toggleExpand(item.itemId)}
              >
                <div className="conversation-title-row">
                  <h3>{item.title || "Untitled"}</h3>
                  <span className={`status-badge status-${item.status}`}>
                    {item.status}
                  </span>
                </div>
                {item.description && (
                  <p className="conversation-desc">{item.description}</p>
                )}
                <div className="conversation-meta">
                  <span>{item.created}</span>
                  <span>
                    {item.messages.length} message
                    {item.messages.length !== 1 ? "s" : ""}
                  </span>
                  <span className="expand-indicator">
                    {expandedId === item.itemId ? "\u25B2" : "\u25BC"}
                  </span>
                </div>
              </div>

              {expandedId === item.itemId && (
                <div className="conversation-messages">
                  {item.messages.map((msg, idx) => (
                    <div
                      key={idx}
                      className={`message message-${msg.role}`}
                    >
                      <div className="message-header">
                        <span className="message-role">{msg.role}</span>
                        <span className="message-time">{msg.timestamp}</span>
                      </div>
                      <div className="message-content">
                        <pre>{msg.content}</pre>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div key={item.itemId} className="conversation-card note-card">
              <div
                className="conversation-header"
                onClick={() => toggleExpand(item.itemId)}
              >
                <div className="conversation-title-row">
                  <h3>{item.title}</h3>
                  <span className="status-badge status-note">Note</span>
                </div>
                <div className="conversation-meta">
                  <span>{item.timestamp}</span>
                  <span className="expand-indicator">
                    {expandedId === item.itemId ? "\u25B2" : "\u25BC"}
                  </span>
                </div>
              </div>

              {expandedId === item.itemId && (
                <div className="conversation-messages">
                  <div className="message message-note">
                    <div className="message-content">
                      <pre>{item.content}</pre>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )
        )}
      </div>
    </div>
  );
}

export default DevNotes;
