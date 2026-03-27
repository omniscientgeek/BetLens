import React, { useState, useEffect } from "react";

const API_BASE = "";

const TYPE_LABELS = {
  anthropic: "Anthropic",
  openai: "OpenAI",
  openai_compatible: "Custom / Ollama",
};

function AISettings() {
  const [config, setConfig] = useState(null);
  const [originalConfig, setOriginalConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [saveMsg, setSaveMsg] = useState(null);
  const [testResults, setTestResults] = useState({});

  useEffect(() => {
    fetch(`${API_BASE}/ai/config`)
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load config (${r.status})`);
        return r.json();
      })
      .then((data) => {
        setConfig(data);
        setOriginalConfig(JSON.parse(JSON.stringify(data)));
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const isDirty = () =>
    JSON.stringify(config) !== JSON.stringify(originalConfig);

  const updateProvider = (index, field, value) => {
    setConfig((prev) => {
      const next = JSON.parse(JSON.stringify(prev));
      next.providers[index][field] = value;
      return next;
    });
  };

  const updateGlobal = (field, value) => {
    setConfig((prev) => ({ ...prev, [field]: value }));
  };

  const handleSave = async () => {
    setSaving(true);
    setSaveMsg(null);
    try {
      const res = await fetch(`${API_BASE}/ai/config`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || `Save failed (${res.status})`);
      }
      setOriginalConfig(JSON.parse(JSON.stringify(config)));
      setSaveMsg("Configuration saved successfully.");
      setTimeout(() => setSaveMsg(null), 3000);
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    setConfig(JSON.parse(JSON.stringify(originalConfig)));
    setTestResults({});
    setSaveMsg(null);
  };

  const handleTest = async (providerId) => {
    setTestResults((prev) => ({
      ...prev,
      [providerId]: { status: "testing" },
    }));
    try {
      const res = await fetch(`${API_BASE}/ai/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider_id: providerId }),
      });
      const data = await res.json();
      if (data.status === "ok") {
        setTestResults((prev) => ({
          ...prev,
          [providerId]: {
            status: "success",
            message: data.response?.content || "Provider responded successfully.",
          },
        }));
      } else {
        setTestResults((prev) => ({
          ...prev,
          [providerId]: { status: "error", message: data.error || "Test failed" },
        }));
      }
    } catch (e) {
      setTestResults((prev) => ({
        ...prev,
        [providerId]: { status: "error", message: e.message },
      }));
    }
  };

  if (loading) return <p className="loading">Loading AI configuration...</p>;
  if (error && !config)
    return <p className="error">Error: {error}</p>;

  return (
    <div className="ai-settings">
      <div className="ai-header">
        <h2>AI Provider Settings</h2>
        <p className="ai-subtitle">
          Configure AI providers for analysis and briefing. Providers are tried
          in priority order with automatic failover.
        </p>
      </div>

      {error && <p className="error">{error}</p>}
      {saveMsg && <p className="ai-success-msg">{saveMsg}</p>}

      {/* ---- Global Settings ---- */}
      <div className="ai-global-card">
        <h3>Global Settings</h3>
        <div className="ai-global-fields">
          <label className="ai-toggle-label">
            <span className="ai-toggle-track" data-on={config.failover_enabled}>
              <span className="ai-toggle-thumb" />
            </span>
            <input
              type="checkbox"
              className="ai-toggle-input"
              checked={config.failover_enabled}
              onChange={(e) => updateGlobal("failover_enabled", e.target.checked)}
            />
            Automatic failover
          </label>
          <div className="ai-field">
            <label>Timeout (seconds)</label>
            <input
              type="number"
              min={5}
              max={300}
              value={config.timeout_seconds}
              onChange={(e) =>
                updateGlobal("timeout_seconds", Number(e.target.value))
              }
            />
          </div>
          <div className="ai-field">
            <label>Retry attempts</label>
            <input
              type="number"
              min={0}
              max={10}
              value={config.retry_attempts}
              onChange={(e) =>
                updateGlobal("retry_attempts", Number(e.target.value))
              }
            />
          </div>
        </div>
      </div>

      {/* ---- Provider Cards ---- */}
      <div className="ai-provider-list">
        {config.providers
          .sort((a, b) => a.priority - b.priority)
          .map((p, idx) => {
            const test = testResults[p.id];
            return (
              <div
                key={p.id}
                className={`ai-provider-card ${p.enabled ? "" : "ai-provider-card--disabled"}`}
              >
                <div className="ai-provider-card-header">
                  <div className="ai-provider-title">
                    <span className="ai-provider-name">{p.name}</span>
                    <span className={`ai-type-badge ai-type-badge--${p.type}`}>
                      {TYPE_LABELS[p.type] || p.type}
                    </span>
                  </div>
                  <label className="ai-toggle-label">
                    <span className="ai-toggle-track" data-on={p.enabled}>
                      <span className="ai-toggle-thumb" />
                    </span>
                    <input
                      type="checkbox"
                      className="ai-toggle-input"
                      checked={p.enabled}
                      onChange={(e) =>
                        updateProvider(idx, "enabled", e.target.checked)
                      }
                    />
                  </label>
                </div>

                <div className="ai-provider-card-body">
                  <div className="ai-field">
                    <label>Model</label>
                    <input
                      type="text"
                      value={p.model}
                      onChange={(e) => updateProvider(idx, "model", e.target.value)}
                    />
                  </div>
                  <div className="ai-field">
                    <label>Priority</label>
                    <input
                      type="number"
                      min={1}
                      max={99}
                      value={p.priority}
                      onChange={(e) =>
                        updateProvider(idx, "priority", Number(e.target.value))
                      }
                    />
                  </div>
                  <div className="ai-field">
                    <label>Max Tokens</label>
                    <input
                      type="number"
                      min={1}
                      max={200000}
                      value={p.max_tokens}
                      onChange={(e) =>
                        updateProvider(idx, "max_tokens", Number(e.target.value))
                      }
                    />
                  </div>
                  <div className="ai-field">
                    <label>Temperature</label>
                    <input
                      type="number"
                      min={0}
                      max={2}
                      step={0.1}
                      value={p.temperature}
                      onChange={(e) =>
                        updateProvider(idx, "temperature", Number(e.target.value))
                      }
                    />
                  </div>
                  {p.type === "openai_compatible" && (
                    <div className="ai-field ai-field--wide">
                      <label>Base URL</label>
                      <input
                        type="text"
                        value={p.base_url || ""}
                        placeholder="http://localhost:11434/v1"
                        onChange={(e) =>
                          updateProvider(idx, "base_url", e.target.value)
                        }
                      />
                    </div>
                  )}
                </div>

                <div className="ai-provider-card-footer">
                  <div className="ai-key-status-row">
                    <span className="ai-key-label">API Key ({p.api_key_env}):</span>
                    {p.api_key_set ? (
                      <span className="ai-key-status ai-key-status--set">Set</span>
                    ) : (
                      <span className="ai-key-status ai-key-status--unset">
                        Not Set
                      </span>
                    )}
                  </div>
                  <button
                    className="ai-test-btn"
                    disabled={test?.status === "testing"}
                    onClick={() => handleTest(p.id)}
                  >
                    {test?.status === "testing" ? (
                      <>
                        <span className="ai-test-spinner" /> Testing...
                      </>
                    ) : (
                      "Test Connection"
                    )}
                  </button>
                  {test && test.status !== "testing" && (
                    <span
                      className={`ai-test-result ai-test-result--${test.status}`}
                    >
                      {test.status === "success" ? "Connected" : test.message}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
      </div>

      {/* ---- Action Buttons ---- */}
      <div className="ai-actions">
        <button
          className="ai-save-btn"
          disabled={!isDirty() || saving}
          onClick={handleSave}
        >
          {saving ? "Saving..." : "Save Configuration"}
        </button>
        <button
          className="ai-reset-btn"
          disabled={!isDirty()}
          onClick={handleReset}
        >
          Reset
        </button>
      </div>
    </div>
  );
}

export default AISettings;
