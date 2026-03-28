import React, { useState, useEffect, useRef } from "react";
import { VerificationBadge } from "./BriefPanel";

/**
 * Displays the AI conversation from the Analyze pipeline phase.
 * Supports real-time streaming during the analyze phase and
 * static display after completion.
 */

function CollapsibleSection({ title, badge, defaultOpen, children }) {
  const [open, setOpen] = useState(defaultOpen || false);
  return (
    <div className={`ac-section ${open ? "ac-section--open" : ""}`}>
      <button className="ac-section-header" onClick={() => setOpen(!open)}>
        <span className="ac-section-arrow">{open ? "\u25BC" : "\u25B6"}</span>
        <span className="ac-section-title">{title}</span>
        {badge && <span className="ac-section-badge">{badge}</span>}
      </button>
      {open && <div className="ac-section-body">{children}</div>}
    </div>
  );
}

function ToolCallCard({ call, index }) {
  const [showInput, setShowInput] = useState(false);
  const [showResult, setShowResult] = useState(false);

  return (
    <div className={`ac-tool-call ${call.is_error ? "ac-tool-call--error" : ""}`}>
      <div className="ac-tool-call-header">
        <span className="ac-tool-call-index">#{index + 1}</span>
        <span className="ac-tool-call-name">{call.name}</span>
        {call.is_error && <span className="ac-tool-call-error-badge">ERROR</span>}
      </div>
      <div className="ac-tool-call-toggles">
        <button
          className="ac-tool-toggle"
          onClick={() => setShowInput(!showInput)}
        >
          {showInput ? "Hide" : "Show"} Input
        </button>
        {call.result !== undefined && (
          <button
            className="ac-tool-toggle"
            onClick={() => setShowResult(!showResult)}
          >
            {showResult ? "Hide" : "Show"} Result
          </button>
        )}
      </div>
      {showInput && (
        <pre className="ac-code-block ac-code-block--input">
          {typeof call.input === "string"
            ? call.input
            : JSON.stringify(call.input, null, 2)}
        </pre>
      )}
      {showResult && call.result !== undefined && (
        <pre className={`ac-code-block ${call.is_error ? "ac-code-block--error" : "ac-code-block--result"}`}>
          {typeof call.result === "string"
            ? call.result
            : JSON.stringify(call.result, null, 2)}
        </pre>
      )}
    </div>
  );
}

function truncateText(text, maxLen) {
  if (!text || text.length <= maxLen) return text;
  return text.slice(0, maxLen) + "\n... [truncated]";
}

/** Auto-scrolling code block that follows new content during streaming */
function StreamingCodeBlock({ className, children }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [children]);
  return (
    <pre className={className} ref={ref}>
      {children}
    </pre>
  );
}

function AnalyzeConversation({ analyzeResult, streaming, defaultExpanded, title }) {
  const [expanded, setExpanded] = useState(defaultExpanded || false);

  // Auto-expand when streaming starts
  useEffect(() => {
    if (streaming && defaultExpanded) {
      setExpanded(true);
    }
  }, [streaming, defaultExpanded]);

  if (!analyzeResult) return null;

  const conversation = analyzeResult.conversation;
  const aiMeta = analyzeResult.ai_meta;
  const isStreaming = streaming || false;

  // If no conversation data available, show a minimal notice
  if (!conversation) {
    return (
      <div className="ac-container">
        <button
          className="ac-expand-btn"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? "\u25BC" : "\u25B6"} {title || "Analyze AI Conversation"}
          {aiMeta && (
            <span className="ac-meta-inline">
              {aiMeta.provider} / {aiMeta.model}
            </span>
          )}
        </button>
        {expanded && (
          <div className="ac-body">
            <p className="ac-no-data">
              Conversation data not available for this run.
            </p>
          </div>
        )}
      </div>
    );
  }

  const toolCalls = conversation.tool_calls || [];
  const responseText = conversation.assistant_response || "";
  const responseLen = responseText.length;

  return (
    <div className={`ac-container ${isStreaming ? "ac-container--streaming" : ""}`}>
      <button
        className="ac-expand-btn"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="ac-expand-arrow">{expanded ? "\u25BC" : "\u25B6"}</span>
        <span>{title || "Analyze AI Conversation"}</span>
        {isStreaming && <span className="ac-streaming-badge">STREAMING</span>}
        {aiMeta && (
          <span className="ac-meta-inline">
            {aiMeta.provider} / {aiMeta.model}
            {aiMeta.elapsed_seconds && ` \u00B7 ${aiMeta.elapsed_seconds}s`}
            {aiMeta.usage?.input_tokens && ` \u00B7 ${aiMeta.usage.input_tokens} in / ${aiMeta.usage.output_tokens} out`}
          </span>
        )}
        {toolCalls.length > 0 && (
          <span className="ac-tool-count">{toolCalls.length} tool call{toolCalls.length !== 1 ? "s" : ""}</span>
        )}
        {isStreaming && responseLen > 0 && (
          <span className="ac-meta-inline">{(responseLen / 1024).toFixed(1)} KB received</span>
        )}
      </button>

      {expanded && (
        <div className="ac-body">
          {/* Conversation error */}
          {conversation.error && (
            <div className="ac-error">
              AI call failed: {conversation.error}
            </div>
          )}

          {/* System Prompt */}
          <CollapsibleSection title="System Prompt" badge="system">
            <pre className="ac-code-block ac-code-block--system">
              {conversation.system_prompt || "(none)"}
            </pre>
          </CollapsibleSection>

          {/* User Prompt (data payload) */}
          <CollapsibleSection
            title="User Prompt (Data Payload)"
            badge={conversation.user_prompt ? `${(conversation.user_prompt.length / 1024).toFixed(1)} KB` : null}
          >
            <pre className="ac-code-block ac-code-block--user">
              {truncateText(conversation.user_prompt, 20000) || "(none)"}
            </pre>
          </CollapsibleSection>

          {/* Streaming Response — shown during streaming as the live feed */}
          {isStreaming && (
            <CollapsibleSection
              title="AI Response (Live)"
              badge={`${(responseLen / 1024).toFixed(1)} KB`}
              defaultOpen={true}
            >
              <StreamingCodeBlock className="ac-code-block ac-code-block--assistant ac-code-block--streaming">
                {responseText || "(waiting for response...)"}
                {responseLen > 0 && <span className="ac-cursor" />}
              </StreamingCodeBlock>
            </CollapsibleSection>
          )}

          {/* Thinking (Chain of Thought) — shown after streaming completes */}
          {!isStreaming && conversation.thinking && (
            <CollapsibleSection
              title="AI Thinking (Chain of Thought)"
              badge="reasoning"
              defaultOpen={true}
            >
              <pre className="ac-code-block ac-code-block--thinking">
                {conversation.thinking}
              </pre>
            </CollapsibleSection>
          )}

          {/* Tool Calls — shown during streaming and after completion */}
          <CollapsibleSection
            title={isStreaming && toolCalls.length > 0 ? "Tool Calls (Live)" : "Tool Calls"}
            badge={toolCalls.length > 0 ? `${toolCalls.length}` : (isStreaming ? "waiting..." : "none")}
            defaultOpen={toolCalls.length > 0 || isStreaming}
          >
            {toolCalls.length === 0 ? (
              <p className="ac-no-data">
                {isStreaming
                  ? "Waiting for tool calls..."
                  : "No tool calls were made during this analysis."}
              </p>
            ) : (
              <div className="ac-tool-calls-list">
                {toolCalls.map((call, i) => (
                  <ToolCallCard key={call.id || i} call={call} index={i} />
                ))}
              </div>
            )}
          </CollapsibleSection>

          {/* Full Assistant Response — shown after streaming completes */}
          {!isStreaming && (
            <CollapsibleSection
              title="Full AI Response"
              badge={responseText ? `${(responseLen / 1024).toFixed(1)} KB` : null}
            >
              <pre className="ac-code-block ac-code-block--assistant">
                {truncateText(responseText, 30000) || "(no response)"}
              </pre>
            </CollapsibleSection>
          )}

          {/* Verification Badge — shown after verification agents complete */}
          {!isStreaming && analyzeResult.verification && (
            <VerificationBadge verification={analyzeResult.verification} />
          )}
        </div>
      )}
    </div>
  );
}

export default AnalyzeConversation;
