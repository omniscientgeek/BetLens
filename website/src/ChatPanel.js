import React, { useState, useEffect, useRef } from "react";
import { VerificationBadge } from "./BriefPanel";

const API_BASE = "";

function formatTime(date) {
  return new Date(date).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function renderMarkdown(text) {
  if (!text) return text;
  // Split into lines, process each
  const lines = text.split("\n");
  const elements = [];
  let inList = false;
  let listItems = [];

  const flushList = () => {
    if (listItems.length > 0) {
      elements.push(
        <ul key={`ul-${elements.length}`}>
          {listItems.map((li, i) => (
            <li key={i}>{processInline(li)}</li>
          ))}
        </ul>
      );
      listItems = [];
      inList = false;
    }
  };

  const processInline = (line) => {
    // Bold: **text** and Italic: *text* (single asterisk, not double)
    const parts = [];
    let remaining = line;
    let idx = 0;
    while (remaining.length > 0) {
      // Find the first * (could be ** for bold or * for italic)
      const starPos = remaining.indexOf("*");
      if (starPos === -1) {
        parts.push(remaining);
        break;
      }
      const isBold = remaining[starPos + 1] === "*";
      const marker = isBold ? "**" : "*";
      const markerLen = marker.length;
      const endPos = remaining.indexOf(marker, starPos + markerLen);
      if (endPos === -1) {
        parts.push(remaining);
        break;
      }
      if (starPos > 0) {
        parts.push(remaining.substring(0, starPos));
      }
      const inner = remaining.substring(starPos + markerLen, endPos);
      if (isBold) {
        parts.push(<strong key={`b-${idx}`}>{inner}</strong>);
      } else {
        parts.push(<em key={`i-${idx}`}>{inner}</em>);
      }
      remaining = remaining.substring(endPos + markerLen);
      idx++;
    }
    return parts.length === 1 ? parts[0] : parts;
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      inList = true;
      listItems.push(trimmed.substring(2));
    } else {
      flushList();
      if (trimmed === "") {
        elements.push(<br key={`br-${i}`} />);
      } else {
        elements.push(
          <span key={`line-${i}`}>
            {processInline(trimmed)}
            {i < lines.length - 1 ? <br /> : null}
          </span>
        );
      }
    }
  }
  flushList();

  return elements;
}

function WelcomeMessage({ onSuggestionClick }) {
  const suggestions = [
    "What are the best value bets right now?",
    "Are there any arbitrage opportunities?",
    "Which sportsbooks have the highest vig?",
    "Explain the outlier lines detected",
  ];

  return (
    <div className="chat-welcome">
      <h4>AI Assistant</h4>
      <p>
        Ask questions about your betting data, odds analysis, or pipeline
        results. I can help identify value bets, arbitrage opportunities, and
        more.
      </p>
      <div className="chat-suggestions">
        {suggestions.map((s, i) => (
          <button
            key={i}
            className="chat-suggestion-btn"
            onClick={() => onSuggestionClick(s)}
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="chat-typing">
      <div className="chat-typing-dot" />
      <div className="chat-typing-dot" />
      <div className="chat-typing-dot" />
    </div>
  );
}

function ChatPanel({ pipelineResults, debugMode }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [conversationId, setConversationId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const prevPipelineRef = useRef(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollTop = messagesEndRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  // Show notice when pipeline results change
  useEffect(() => {
    if (
      pipelineResults &&
      prevPipelineRef.current !== null &&
      pipelineResults !== prevPipelineRef.current &&
      messages.length > 0
    ) {
      setMessages((prev) => [
        ...prev,
        {
          role: "notice",
          content: "New data has been processed. You can ask about the updated results.",
          timestamp: new Date(),
        },
      ]);
    }
    prevPipelineRef.current = pipelineResults;
  }, [pipelineResults]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleNewChat = () => {
    setMessages([]);
    setConversationId(null);
    setError(null);
    setInput("");
  };

  const handleSend = async (messageText) => {
    const text = (messageText || input).trim();
    if (!text || isLoading) return;

    const userMsg = { role: "user", content: text, timestamp: new Date() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setError(null);
    setIsLoading(true);

    // Resize textarea back
    if (textareaRef.current) {
      textareaRef.current.style.height = "40px";
    }

    try {
      const body = {
        message: text,
        conversation_id: conversationId,
        pipeline_context: pipelineResults,
      };

      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const data = await res.json();

      if (data.conversation_id && !conversationId) {
        setConversationId(data.conversation_id);
      }

      const responseText =
        data.response?.text || data.response || data.message || "No response received.";
      const assistantMsg = {
        role: "assistant",
        content: typeof responseText === "string" ? responseText : JSON.stringify(responseText),
        timestamp: new Date(),
        run_id: data.run_id || null,
        verification: data.response?.verification || null,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleTextareaChange = (e) => {
    setInput(e.target.value);
    // Auto-resize
    e.target.style.height = "40px";
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
  };

  const handleSuggestionClick = (suggestion) => {
    handleSend(suggestion);
  };

  const messageCount = messages.filter(
    (m) => m.role === "user" || m.role === "assistant"
  ).length;

  return (
    <div className="chat-panel">
      <div className="chat-panel-header">
        <h3>
          AI Assistant
          {messageCount > 0 && (
            <span className="chat-msg-count">({messageCount})</span>
          )}
        </h3>
        <div className="chat-panel-header-actions">
          {conversationId && (
            <button onClick={handleNewChat}>New Chat</button>
          )}
        </div>
      </div>

      <div className="chat-messages" ref={messagesEndRef}>
        {messages.length === 0 && (
          <WelcomeMessage onSuggestionClick={handleSuggestionClick} />
        )}
        {messages.map((msg, i) => {
          if (msg.role === "notice") {
            return (
              <div key={i} className="chat-context-notice">
                {msg.content}
              </div>
            );
          }
          return (
            <div
              key={i}
              className={`chat-msg chat-msg--${msg.role}`}
            >
              <div className="chat-msg-content">
                {msg.role === "assistant"
                  ? renderMarkdown(msg.content)
                  : msg.content}
              </div>
              {msg.role === "assistant" && msg.verification && (
                <VerificationBadge verification={msg.verification} />
              )}
              <div className="chat-msg-meta">
                <span className="chat-msg-time">{formatTime(msg.timestamp)}</span>
                {debugMode && msg.run_id && (
                  <a
                    href={`${API_BASE}/logs/${msg.run_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="debug-log-link debug-log-link--chat"
                  >
                    log:{msg.run_id}
                  </a>
                )}
              </div>
            </div>
          );
        })}
        {isLoading && <TypingIndicator />}
        {error && <div className="chat-error">Error: {error}</div>}
      </div>

      <div className="chat-input-area">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleTextareaChange}
          onKeyDown={handleKeyDown}
          placeholder="Ask about your betting data..."
          rows={1}
        />
        <button
          className="chat-send-btn"
          onClick={() => handleSend()}
          disabled={!input.trim() || isLoading}
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default ChatPanel;
