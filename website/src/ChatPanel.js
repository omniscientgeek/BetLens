import React, { useState, useEffect, useRef } from "react";
import { VerificationBadge } from "./BriefPanel";
import { API_BASE } from "./api";

// Session persistence helpers
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

const ASSISTANT_SUGGESTIONS = [
  "What are the best value bets right now?",
  "Are there any arbitrage opportunities?",
  "Which sportsbooks have the highest vig?",
  "Explain the outlier lines detected",
];

const AGENT_SUGGESTIONS = [
  "Give me today's daily digest",
  "Which sportsbooks are sharpest right now?",
  "Find the best value bets across all sports",
  "Are there any arbitrage opportunities today?",
  "What games have the most market disagreement?",
  "Show me the power rankings for today's games",
];

function WelcomeMessage({ onSuggestionClick, agentMode }) {
  const suggestions = agentMode ? AGENT_SUGGESTIONS : ASSISTANT_SUGGESTIONS;

  return (
    <div className="chat-welcome">
      <h4>{agentMode ? "BetStamp Agent" : "AI Assistant"}</h4>
      <p>
        {agentMode
          ? "I can fetch live betting data and analyze it for you. I'll always show my reasoning and tell you honestly when I'm unsure or don't have enough data."
          : "Ask questions about your betting data, odds analysis, or pipeline results. I can help identify value bets, arbitrage opportunities, and more."}
      </p>
      {agentMode && (
        <div className="agent-honesty-badge">
          <span className="agent-honesty-icon">◈</span>
          <span>This agent will clearly state when it doesn't know something</span>
        </div>
      )}
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

/**
 * Parse raw streamed text to separate <thinking>...</thinking> content from the visible response.
 * Handles partial/incomplete tags during streaming.
 * Returns { thinking, response, isThinking }
 */
function parseThinkingFromText(raw) {
  if (!raw) return { thinking: "", response: "", isThinking: false };

  const openTag = "<thinking>";
  const closeTag = "</thinking>";
  let thinking = "";
  let response = "";
  let isThinking = false;

  let remaining = raw;

  while (remaining.length > 0) {
    if (!isThinking) {
      const openIdx = remaining.indexOf(openTag);
      if (openIdx === -1) {
        // Check if we're partially into a <thinking> tag at the end
        // e.g. the stream ended with "<thin" — don't show that as response
        let partialTagLen = 0;
        for (let len = Math.min(openTag.length - 1, remaining.length); len > 0; len--) {
          if (openTag.startsWith(remaining.slice(remaining.length - len))) {
            partialTagLen = len;
            break;
          }
        }
        response += remaining.slice(0, remaining.length - partialTagLen);
        break;
      }
      response += remaining.slice(0, openIdx);
      remaining = remaining.slice(openIdx + openTag.length);
      isThinking = true;
    } else {
      const closeIdx = remaining.indexOf(closeTag);
      if (closeIdx === -1) {
        // Still inside thinking — no closing tag yet
        thinking += remaining;
        break;
      }
      thinking += remaining.slice(0, closeIdx);
      remaining = remaining.slice(closeIdx + closeTag.length);
      isThinking = false;
    }
  }

  return { thinking: thinking.trim(), response: response.trim(), isThinking };
}

/**
 * Collapsible thinking block shown inside chat messages.
 */
function ThinkingBlock({ text, isStreaming, defaultOpen }) {
  const [open, setOpen] = useState(defaultOpen !== undefined ? defaultOpen : true);
  const contentRef = useRef(null);

  // Auto-scroll thinking content during streaming
  useEffect(() => {
    if (open && isStreaming && contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [text, open, isStreaming]);

  if (!text) return null;

  return (
    <div className={`chat-thinking ${isStreaming ? "chat-thinking--streaming" : ""}`}>
      <button className="chat-thinking-header" onClick={() => setOpen(!open)}>
        <span className="chat-thinking-arrow">{open ? "\u25BC" : "\u25B6"}</span>
        <span className="chat-thinking-label">Thinking</span>
        {isStreaming && <span className="chat-thinking-badge">reasoning...</span>}
      </button>
      {open && (
        <div className="chat-thinking-content" ref={contentRef}>
          {text}
          {isStreaming && <span className="chat-streaming-cursor" />}
        </div>
      )}
    </div>
  );
}

function ChatPanel({ pipelineResults, debugMode, agentMode }) {
  // Use separate session keys for agent vs assistant to keep histories independent
  const keyPrefix = agentMode ? "agentChatMessages" : "chatMessages";
  const convKeyPrefix = agentMode ? "agentChatConversationId" : "chatConversationId";
  const [messages, setMessages] = useState(() => sessionGet(keyPrefix, []));
  const [input, setInput] = useState("");
  const [conversationId, setConversationId] = useState(() => sessionGet(convKeyPrefix, null));
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isAuditing, setIsAuditing] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [streamingParsed, setStreamingParsed] = useState({ thinking: "", response: "", isThinking: false });
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const prevPipelineRef = useRef(null);
  const abortControllerRef = useRef(null);
  const streamingTextRef = useRef("");

  // Persist chat state to sessionStorage (skip during active streaming to avoid perf issues)
  useEffect(() => {
    if (!isStreaming) { sessionSet(keyPrefix, messages); }
  }, [messages, isStreaming, keyPrefix]);
  useEffect(() => { sessionSet(convKeyPrefix, conversationId); }, [conversationId, convKeyPrefix]);

  // Auto-scroll to bottom on new messages or streaming updates
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollTop = messagesEndRef.current.scrollHeight;
    }
  }, [messages, isLoading, streamingParsed]);

  // Cleanup abort controller on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

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
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setMessages([]);
    setConversationId(null);
    setError(null);
    setInput("");
    setIsLoading(false);
    setIsStreaming(false);
    setIsAuditing(false);
    setStreamingText("");
    setStreamingParsed({ thinking: "", response: "", isThinking: false });
    streamingTextRef.current = "";
    sessionSet(keyPrefix, []);
    sessionSet(convKeyPrefix, null);
  };

  /**
   * Parse SSE events from a text buffer.
   * Returns { events: [{event, data}], remaining: string }
   */
  const parseSSE = (buffer) => {
    const events = [];
    const parts = buffer.split("\n\n");
    // Last part may be incomplete — keep it as remainder
    const remaining = parts.pop() || "";

    for (const part of parts) {
      if (!part.trim()) continue;
      let eventType = "message";
      let dataStr = "";
      for (const line of part.split("\n")) {
        if (line.startsWith("event: ")) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith("data: ")) {
          dataStr += line.slice(6);
        }
      }
      if (dataStr) {
        try {
          events.push({ event: eventType, data: JSON.parse(dataStr) });
        } catch {
          events.push({ event: eventType, data: { raw: dataStr } });
        }
      }
    }
    return { events, remaining };
  };

  const handleSend = async (messageText) => {
    const text = (messageText || input).trim();
    if (!text || isLoading) return;

    const userMsg = { role: "user", content: text, timestamp: new Date() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setError(null);
    setIsLoading(true);
    setIsStreaming(false);
    setStreamingText("");
    setStreamingParsed({ thinking: "", response: "", isThinking: false });
    streamingTextRef.current = "";

    // Resize textarea back
    if (textareaRef.current) {
      textareaRef.current.style.height = "40px";
    }

    // Abort any existing stream
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    let runId = null;
    let verification = null;

    try {
      const body = {
        message: text,
        conversation_id: conversationId,
        pipeline_context: pipelineResults || undefined,
        agent_mode: agentMode || false,
      };

      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      setIsStreaming(true);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const { events, remaining } = parseSSE(buffer);
        buffer = remaining;

        for (const evt of events) {
          switch (evt.event) {
            case "chunk":
              streamingTextRef.current += evt.data.text;
              setStreamingText(streamingTextRef.current);
              setStreamingParsed(parseThinkingFromText(streamingTextRef.current));
              break;
            case "metadata":
              if (evt.data.conversation_id && !conversationId) {
                setConversationId(evt.data.conversation_id);
              }
              runId = evt.data.run_id || null;
              setIsAuditing(true);
              break;
            case "verification":
              verification = evt.data.verification || null;
              setIsAuditing(false);
              break;
            case "error":
              throw new Error(evt.data.error || "Stream error");
            case "done":
              break;
            default:
              break;
          }
        }
      }

      // Stream finished — finalize the assistant message
      const finalRaw = streamingTextRef.current || "No response received.";
      const finalParsed = parseThinkingFromText(finalRaw);
      const assistantMsg = {
        role: "assistant",
        content: finalParsed.response || finalRaw,
        thinking: finalParsed.thinking || null,
        timestamp: new Date(),
        run_id: runId,
        verification: verification,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      if (err.name !== "AbortError") {
        // If we accumulated streaming text before the error, preserve it as a message
        // so the user doesn't lose the response they were reading
        if (streamingTextRef.current) {
          const partialRaw = streamingTextRef.current;
          const partialParsed = parseThinkingFromText(partialRaw);
          const partialMsg = {
            role: "assistant",
            content: partialParsed.response || partialRaw,
            thinking: partialParsed.thinking || null,
            timestamp: new Date(),
            run_id: runId,
            verification: verification,
          };
          setMessages((prev) => [...prev, partialMsg]);
        }
        setError(err.message);
      }
    } finally {
      setIsLoading(false);
      setIsStreaming(false);
      setIsAuditing(false);
      setStreamingText("");
      setStreamingParsed({ thinking: "", response: "", isThinking: false });
      streamingTextRef.current = "";
      abortControllerRef.current = null;
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
          {agentMode ? "BetStamp Agent" : "AI Assistant"}
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
          <WelcomeMessage onSuggestionClick={handleSuggestionClick} agentMode={agentMode} />
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
              {msg.role === "assistant" && msg.thinking && (
                <ThinkingBlock text={msg.thinking} isStreaming={false} defaultOpen={false} />
              )}
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
        {isStreaming && streamingText && (
          <div className="chat-msg chat-msg--assistant">
            {streamingParsed.thinking && (
              <ThinkingBlock
                text={streamingParsed.thinking}
                isStreaming={streamingParsed.isThinking}
                defaultOpen={true}
              />
            )}
            {streamingParsed.response ? (
              <div className="chat-msg-content">
                {renderMarkdown(streamingParsed.response)}
                {!streamingParsed.isThinking && !isAuditing && <span className="chat-streaming-cursor" />}
              </div>
            ) : streamingParsed.isThinking ? null : (
              <div className="chat-msg-content">
                <span className="chat-streaming-cursor" />
              </div>
            )}
            {isAuditing && (
              <div className="chat-audit-indicator">
                <span className="chat-audit-spinner" />
                <span className="chat-audit-label">Verifying response...</span>
              </div>
            )}
          </div>
        )}
        {isLoading && (!isStreaming || !streamingText) && !isAuditing && <TypingIndicator />}
        {error && <div className="chat-error">Error: {error}</div>}
      </div>

      <div className="chat-input-area">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleTextareaChange}
          onKeyDown={handleKeyDown}
          placeholder={agentMode ? "Ask the agent about live betting data..." : "Ask about your betting data..."}
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
