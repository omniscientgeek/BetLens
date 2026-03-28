#!/usr/bin/env node
/**
 * Claude Agent SDK Wrapper for BetStamp
 *
 * Simplified request-response bridge between Python and the Claude Agent SDK.
 * Reads a single JSON command from stdin, runs the query, outputs the result
 * as JSON to stdout, then exits.
 *
 * Migrated from: SeniorAIDeveloper/desktop-service/claude-sdk-wrapper.mjs
 *
 * Input (JSON on stdin):
 *   { "system_prompt": "...", "user_prompt": "...", "model": "...", "max_turns": 1, "cwd": "...",
 *     "use_mcp": false, "mcp_config_path": "..." }
 *
 * Output (JSON on stdout):
 *   { "type": "result", "text": "...", "session_id": "...", "usage": {...} }
 *   or
 *   { "type": "error", "message": "..." }
 */
import { query } from '@anthropic-ai/claude-agent-sdk';
import { readFileSync } from 'fs';
// readline import removed — raw stdin chunks used instead to avoid 64KB line limit

// Ensure stdout writes UTF-8 on all platforms (prevents emoji mojibake on Windows)
if (typeof process.stdout.setDefaultEncoding === 'function') {
  process.stdout.setDefaultEncoding('utf-8');
}

// Global handler for unhandled promise rejections — prevents Node.js from crashing
// when the SDK's ProcessTransport throws after the CLI process exits unexpectedly.
process.on('unhandledRejection', (reason, promise) => {
  const msg = reason?.message || String(reason);
  process.stderr.write(`[Unhandled Rejection] ${msg}\n`);
  sendResult({ type: 'error', message: `Unhandled rejection: ${msg}` });
  process.exit(1);
});

function sendResult(obj) {
  // Write result as a single JSON line to stdout
  process.stdout.write(JSON.stringify(obj) + '\n');
}

// Read all of stdin as raw chunks (avoids readline's 64KB line-length limit
// which causes "Separator is not found, and chunk exceed the limit" when the
// JSON payload is large, e.g. during analyze phases with big prompts).
const chunks = [];

process.stdin.on('data', (chunk) => chunks.push(chunk));

process.stdin.on('end', async () => {
  let command;
  try {
    command = JSON.parse(Buffer.concat(chunks).toString('utf-8'));
  } catch (err) {
    sendResult({ type: 'error', message: `Invalid JSON input: ${err.message}` });
    process.exit(1);
  }

  try {
    await handleQuery(command);
  } catch (err) {
    sendResult({ type: 'error', message: err.message, stack: err.stack });
    process.exit(1);
  }

  process.exit(0);
});

async function handleQuery(command) {
  const { system_prompt, user_prompt, model, max_turns, cwd, timeout_seconds, use_mcp, mcp_config_path } = command;

  if (!user_prompt) {
    sendResult({ type: 'error', message: 'user_prompt is required' });
    return;
  }

  const resolvedCwd = cwd || process.cwd();
  const mcpEnabled = use_mcp === true;
  process.stderr.write(`[claude-sdk] Starting query: model=${model || 'default'}, cwd=${resolvedCwd}, mcp=${mcpEnabled}\n`);

  const options = {
    cwd: resolvedCwd,
    settingSources: [],          // Don't load user/project settings — we inject MCP config directly
    ...(max_turns ? { maxTurns: max_turns } : mcpEnabled ? {} : { maxTurns: 1 }),
    permissionMode: mcpEnabled ? 'bypassPermissions' : 'default',
  };

  // When MCP is enabled, load the betstamp-intelligence server config
  if (mcpEnabled) {
    const defaultMcpPath = mcp_config_path || new URL('../mcp-server/.mcp.json', import.meta.url).pathname.replace(/^\/([A-Z]:)/, '$1');
    try {
      const raw = readFileSync(defaultMcpPath, 'utf-8');
      const parsed = JSON.parse(raw);
      if (parsed.mcpServers) {
        options.mcpServers = parsed.mcpServers;
        process.stderr.write(`[claude-sdk] Loaded MCP servers: ${Object.keys(parsed.mcpServers).join(', ')}\n`);
      }
    } catch (err) {
      process.stderr.write(`[claude-sdk] Warning: could not load MCP config from ${defaultMcpPath}: ${err.message}\n`);
    }
  }

  if (model) {
    options.model = model;
  }

  if (system_prompt) {
    options.systemPrompt = system_prompt;
  }

  // Set up timeout — generous default for large CoT analyze calls
  const timeoutMs = (timeout_seconds || 300) * 1000;
  const timeoutPromise = new Promise((_, reject) =>
    setTimeout(() => reject(new Error(`Claude SDK query timed out after ${timeout_seconds || 300}s`)), timeoutMs)
  );

  const queryPromise = (async () => {
    const session = query({
      prompt: user_prompt,
      options: options
    });

    let resultText = '';
    let sessionId = null;
    let usage = null;
    const toolCalls = [];

    for await (const message of session) {
      process.stderr.write(`[claude-sdk] Message type: ${message.type}\n`);

      // Capture session ID
      if (message.session_id) {
        sessionId = message.session_id;
      }

      // Extract text and tool_use blocks from assistant messages (streaming)
      if (message.type === 'assistant' && message.message?.content) {
        for (const item of message.message.content) {
          if (item.type === 'text' && item.text) {
            // Each assistant turn has its own text block — emit the full text
            // as a chunk (resultText is overwritten per turn, not accumulated)
            const text = item.text;
            if (text) {
              process.stdout.write(JSON.stringify({ type: 'chunk', text }) + '\n');
            }
            resultText = text;
          }
          // Capture tool_use blocks (MCP tool calls) and emit live event
          if (item.type === 'tool_use') {
            const toolCall = {
              id: item.id,
              name: item.name,
              input: item.input,
            };
            toolCalls.push(toolCall);
            // Emit tool_call event so the UI can show it in real-time
            process.stdout.write(JSON.stringify({ type: 'tool_call', tool_call: toolCall }) + '\n');
          }
        }
      }

      // Capture tool results and emit live event
      if (message.type === 'tool_result' || (message.type === 'user' && message.message?.content)) {
        const content = message.message?.content || [];
        for (const item of (Array.isArray(content) ? content : [])) {
          if (item.type === 'tool_result') {
            const matching = toolCalls.find(tc => tc.id === item.tool_use_id);
            if (matching) {
              matching.result = typeof item.content === 'string' ? item.content : JSON.stringify(item.content);
              matching.is_error = item.is_error || false;
              // Emit tool_result event so the UI can show results in real-time
              process.stdout.write(JSON.stringify({
                type: 'tool_result',
                tool_use_id: item.tool_use_id,
                result: matching.result,
                is_error: matching.is_error,
              }) + '\n');
            }
          }
        }
      }

      // Check for result message (end of session)
      if (message.type === 'result') {
        if (message.subtype === 'success' && message.result) {
          resultText = message.result;
        }
        // Capture usage from result
        if (message.usage) {
          usage = {
            input_tokens: message.usage.input_tokens || 0,
            output_tokens: message.usage.output_tokens || 0,
            cache_read_input_tokens: message.usage.cache_read_input_tokens || 0,
            cache_creation_input_tokens: message.usage.cache_creation_input_tokens || 0,
          };
        }
        break;
      }
    }

    return { resultText, sessionId, usage, toolCalls };
  })();

  const { resultText, sessionId, usage, toolCalls } = await Promise.race([queryPromise, timeoutPromise]);

  if (!resultText) {
    sendResult({ type: 'error', message: 'Empty response from Claude SDK' });
    return;
  }

  process.stderr.write(`[claude-sdk] Query complete: ${resultText.length} chars\n`);

  sendResult({
    type: 'result',
    text: resultText,
    session_id: sessionId,
    usage: usage || {},
    tool_calls: toolCalls,
  });
}
