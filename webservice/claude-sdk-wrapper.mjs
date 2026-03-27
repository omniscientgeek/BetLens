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
 *   { "system_prompt": "...", "user_prompt": "...", "model": "...", "max_turns": 1, "cwd": "..." }
 *
 * Output (JSON on stdout):
 *   { "type": "result", "text": "...", "session_id": "...", "usage": {...} }
 *   or
 *   { "type": "error", "message": "..." }
 */
import { query } from '@anthropic-ai/claude-agent-sdk';
import readline from 'readline';

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

// Read all of stdin as a single JSON payload
const rl = readline.createInterface({ input: process.stdin, terminal: false });
const lines = [];

rl.on('line', (line) => lines.push(line));

rl.on('close', async () => {
  let command;
  try {
    command = JSON.parse(lines.join('\n'));
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
  const { system_prompt, user_prompt, model, max_turns, cwd, timeout_seconds } = command;

  if (!user_prompt) {
    sendResult({ type: 'error', message: 'user_prompt is required' });
    return;
  }

  const resolvedCwd = cwd || process.cwd();
  process.stderr.write(`[claude-sdk] Starting query: model=${model || 'default'}, cwd=${resolvedCwd}\n`);

  const options = {
    cwd: resolvedCwd,
    settingSources: [],          // Don't load user/project settings for pipeline calls
    maxTurns: max_turns || 1,    // Single turn for analysis/brief phases
    permissionMode: 'plan',      // Read-only, no tool execution needed
  };

  if (model) {
    options.model = model;
  }

  if (system_prompt) {
    options.systemPrompt = system_prompt;
  }

  // Set up timeout
  const timeoutMs = (timeout_seconds || 120) * 1000;
  const timeoutPromise = new Promise((_, reject) =>
    setTimeout(() => reject(new Error(`Claude SDK query timed out after ${timeout_seconds || 120}s`)), timeoutMs)
  );

  const queryPromise = (async () => {
    const session = query({
      prompt: user_prompt,
      options: options
    });

    let resultText = '';
    let sessionId = null;
    let usage = null;

    for await (const message of session) {
      process.stderr.write(`[claude-sdk] Message type: ${message.type}\n`);

      // Capture session ID
      if (message.session_id) {
        sessionId = message.session_id;
      }

      // Extract text from assistant messages (streaming)
      if (message.type === 'assistant' && message.message?.content) {
        for (const item of message.message.content) {
          if (item.type === 'text' && item.text) {
            resultText = item.text;
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

    return { resultText, sessionId, usage };
  })();

  const { resultText, sessionId, usage } = await Promise.race([queryPromise, timeoutPromise]);

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
  });
}
