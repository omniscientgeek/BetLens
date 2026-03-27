import os
import re
import json
import uuid
import asyncio
import subprocess
from datetime import datetime

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import socketio

from detect import run_detection
from ai_service import (
    load_config as load_ai_config,
    save_config as save_ai_config,
    get_enabled_providers,
    run_analyze_phase,
    run_brief_phase,
    call_ai_chat,
    CHAT_SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# FastAPI + Socket.IO (async) setup
# ---------------------------------------------------------------------------

app = FastAPI()


# Ensure all JSON responses include charset=utf-8 to prevent emoji/multi-byte
# character corruption (mojibake) across proxies and browsers.
class Utf8JsonMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        ct = response.headers.get("content-type", "")
        if ct.startswith("application/json") and "charset" not in ct:
            response.headers["content-type"] = "application/json; charset=utf-8"
        return response


app.add_middleware(Utf8JsonMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    ping_timeout=300,
    ping_interval=60,
)
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# ---------------------------------------------------------------------------
# Three-phase processing pipeline (Detect → Analyze → Brief)
# ---------------------------------------------------------------------------
PHASES = [
    {"name": "detect", "label": "Detecting patterns"},
    {"name": "analyze", "label": "Analyzing structure"},
    {"name": "brief", "label": "Generating brief"},
]


async def _emit_phase(sid, filename, phase, i, status, result=None):
    """Helper to emit a phase_update event."""
    payload = {
        "filename": filename,
        "phase": phase["name"],
        "label": phase["label"],
        "status": status,
        "phaseIndex": i,
        "totalPhases": len(PHASES),
    }
    if result is not None:
        payload["result"] = result
    await sio.emit("phase_update", payload, to=sid)


async def run_processing_pipeline(filename, sid):
    """Execute the 3-phase pipeline, emitting status updates via WebSocket."""
    try:
        pipeline_results = {}

        for i, phase in enumerate(PHASES):
            await _emit_phase(sid, filename, phase, i, "in_progress")

            if phase["name"] == "detect":
                # --- Real detection: compute probabilities, vig, fair odds ---
                detection = await run_detection(filename)
                pipeline_results["detect"] = detection
                await _emit_phase(sid, filename, phase, i, "complete", result=detection)
            elif phase["name"] == "analyze":
                # --- AI-powered cross-sportsbook analysis ---
                try:
                    analysis = await run_analyze_phase(pipeline_results.get("detect", {}))
                    pipeline_results["analyze"] = analysis
                    await _emit_phase(sid, filename, phase, i, "complete", result=analysis)
                except Exception as exc:
                    # Graceful degradation: report the error but continue pipeline
                    err_result = {"error": str(exc), "ai_meta": None}
                    pipeline_results["analyze"] = err_result
                    await _emit_phase(sid, filename, phase, i, "complete", result=err_result)

            elif phase["name"] == "brief":
                # --- AI-powered actionable briefing (streamed to client) ---
                try:
                    async def on_brief_chunk(text_delta):
                        await sio.emit("brief_chunk", {"text": text_delta}, to=sid)

                    brief = await run_brief_phase(
                        pipeline_results.get("detect", {}),
                        pipeline_results.get("analyze", {}),
                        on_chunk=on_brief_chunk,
                    )
                    pipeline_results["brief"] = brief
                    await _emit_phase(sid, filename, phase, i, "complete", result=brief)
                except Exception as exc:
                    err_result = {"error": str(exc), "ai_meta": None}
                    pipeline_results["brief"] = err_result
                    await _emit_phase(sid, filename, phase, i, "complete", result=err_result)

            else:
                # Unknown phase — placeholder
                await asyncio.sleep(2)
                await _emit_phase(sid, filename, phase, i, "complete")

        await sio.emit("processing_complete", {
            "filename": filename,
            "results": pipeline_results,
        }, to=sid)
    except Exception as e:
        await sio.emit("processing_error", {
            "filename": filename,
            "error": str(e),
        }, to=sid)


@sio.on("start_processing")
async def handle_start_processing(sid, data):
    filename = data.get("filename", "")
    if not filename.endswith(".json"):
        await sio.emit("processing_error", {"filename": filename, "error": "Only .json files supported"}, to=sid)
        return
    asyncio.create_task(run_processing_pipeline(filename, sid))


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DEV_NOTES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "devNotesData"))
CONVERSATIONS_DIR = r"C:\ProgramData\DesktopDevService\Conversations"


@app.get("/api/files")
async def list_files():
    """Return a list of all JSON files in the data directory."""
    if not os.path.isdir(DATA_DIR):
        return {"files": []}

    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".json"))
    return {"files": files}


@app.get("/api/files/{filename}")
async def get_file(filename: str):
    """Return the contents of a specific JSON file."""
    if not filename.endswith(".json"):
        return JSONResponse({"error": "Only .json files are supported"}, status_code=400)

    filepath = os.path.abspath(os.path.join(DATA_DIR, filename))

    # Prevent directory traversal
    if not filepath.startswith(DATA_DIR):
        return JSONResponse({"error": "Invalid file path"}, status_code=403)

    if not os.path.isfile(filepath):
        return JSONResponse({"error": "File not found"}, status_code=404)

    import aiofiles
    async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
        content = await f.read()
    data = json.loads(content)

    return data


def _parse_conversation_file(filepath):
    """Parse a conversation .txt file and return its header and messages."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return None

    # Split header from body at the closing separator line
    parts = content.split("================================================\n")
    if len(parts) < 3:
        return None

    header_text = parts[1].strip()
    body_text = "================================================\n".join(parts[2:]).strip()

    # Parse header key-value pairs
    header = {}
    for line in header_text.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            header[key.strip()] = value.strip()

    # Parse messages from the body
    messages = []
    # Pattern: [YYYY-MM-DD HH:MM:SS] ACTOR:\n--------------------------------------------------\n<content>
    msg_pattern = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+(USER|ASSISTANT):\s*\n-{50}\n(.*?)(?=\n\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s+(?:USER|ASSISTANT):|\Z)",
        re.DOTALL,
    )
    for match in msg_pattern.finditer(body_text):
        messages.append({
            "timestamp": match.group(1),
            "role": match.group(2).lower(),
            "content": match.group(3).strip(),
        })

    return {
        "filename": os.path.basename(filepath),
        "title": header.get("Conversation", ""),
        "description": header.get("Description", ""),
        "project": header.get("Project", ""),
        "created": header.get("Created", ""),
        "status": header.get("Status", ""),
        "messages": messages,
    }


# Mapping of numeric project IDs to project GUIDs used in conversation files
PROJECT_GUID_MAP = {
    10: "ebfd2c6d-663f-400d-b0f9-f4b5499d28d9",
}


@app.get("/api/conversations")
async def list_conversations(project_id: int = Query(...)):
    """Return conversations filtered by project ID."""
    project_guid = PROJECT_GUID_MAP.get(project_id)
    if not project_guid:
        return JSONResponse({"error": f"Unknown project_id: {project_id}"}, status_code=404)

    if not os.path.isdir(CONVERSATIONS_DIR):
        return JSONResponse({"error": "Conversations directory not found"}, status_code=500)

    # Run file parsing in a thread to avoid blocking the event loop
    def _parse_all():
        conversations = []
        for fname in os.listdir(CONVERSATIONS_DIR):
            if not fname.endswith(".txt"):
                continue
            filepath = os.path.join(CONVERSATIONS_DIR, fname)
            parsed = _parse_conversation_file(filepath)
            if parsed and parsed["project"] == project_guid:
                conversations.append(parsed)
        return conversations

    conversations = await asyncio.to_thread(_parse_all)

    # Sort by first message timestamp descending (newest first)
    conversations.sort(
        key=lambda c: c["messages"][0]["timestamp"] if c.get("messages") else c.get("created", ""),
        reverse=True,
    )

    result = {"conversations": conversations, "count": len(conversations)}

    # Persist to a JSON file so it can be auto-loaded later
    try:
        os.makedirs(DEV_NOTES_DIR, exist_ok=True)
        save_path = os.path.join(DEV_NOTES_DIR, f"devnotes_project_{project_id}.json")
        import aiofiles
        async with aiofiles.open(save_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(result, indent=2))
    except Exception:
        pass  # Don't fail the response if saving fails

    return result


@app.get("/api/devnotes/{project_id}")
async def get_devnotes(project_id: int):
    """Return previously saved devnotes for a project, or 404 if none exist."""
    filename = f"devnotes_project_{project_id}.json"
    filepath = os.path.join(DEV_NOTES_DIR, filename)
    if not os.path.isfile(filepath):
        return JSONResponse({"error": "No saved devnotes found"}, status_code=404)

    import aiofiles
    async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
        content = await f.read()
    data = json.loads(content)
    return data


def _parse_notes_file():
    """Parse NOTES.md and return individual notes split by ## headings."""
    notes_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DEVLOG.md"))
    if not os.path.isfile(notes_path):
        return []

    try:
        with open(notes_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return []

    # Split on ## headings, keeping the heading with its block
    chunks = re.split(r'\n(?=## )', content)
    notes = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk.startswith("## "):
            continue

        lines = chunk.split("\n")
        title = lines[0].lstrip("# ").strip()

        # Look for italic timestamp line: _Some timestamp_
        timestamp_raw = ""
        created = ""
        body_start = 1
        for j, line in enumerate(lines[1:], start=1):
            stripped = line.strip()
            ts_match = re.match(r'^_(.+?)_$', stripped)
            if ts_match:
                timestamp_raw = ts_match.group(1)
                body_start = j + 1
                break
            elif stripped:
                # Non-empty, non-timestamp line — stop looking
                body_start = j
                break

        # Parse timestamp to ISO for sorting
        if timestamp_raw:
            try:
                # Normalize unicode spaces (e.g. narrow no-break space \u202f) to regular spaces
                normalized = re.sub(r'[\u00a0\u202f\u2009\u2007]', ' ', timestamp_raw)
                dt = datetime.strptime(normalized, "%a, %b %d, %Y at %I:%M %p")
                created = dt.isoformat()
            except ValueError:
                created = timestamp_raw

        body = "\n".join(lines[body_start:]).strip()

        notes.append({
            "id": i,
            "title": title,
            "timestamp": timestamp_raw,
            "created": created,
            "content": body,
            "type": "note",
        })

    return notes


@app.get("/api/notes")
async def get_notes():
    """Return parsed notes from NOTES.md."""
    notes = await asyncio.to_thread(_parse_notes_file)
    return {"notes": notes, "count": len(notes)}


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GIT_STATS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "devNotesData"))


async def _analyze_git_history():
    """Analyze git commit history and classify commits as Claude or User authored."""
    proc = await asyncio.create_subprocess_exec(
        "git", "log", "--pretty=format:%H||%an||%ae||%ai||%s||%b%x00",
        cwd=REPO_DIR,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"git log failed: {stderr_bytes.decode('utf-8', errors='replace')}")

    raw = stdout_bytes.decode("utf-8", errors="replace").strip()
    if not raw:
        return {"commits": [], "summary": {}}

    # Split on null byte delimiter
    entries = [e.strip() for e in raw.split("\x00") if e.strip()]

    commits = []
    claude_patterns = [
        re.compile(r"Co-Authored-By:.*Claude", re.IGNORECASE),
        re.compile(r"co-authored-by:.*anthropic", re.IGNORECASE),
        re.compile(r"Generated.*Claude", re.IGNORECASE),
    ]

    for entry in entries:
        parts = entry.split("||", 5)
        if len(parts) < 5:
            continue

        commit_hash = parts[0].strip()
        author_name = parts[1].strip()
        author_email = parts[2].strip()
        date = parts[3].strip()
        subject = parts[4].strip()
        body = parts[5].strip() if len(parts) > 5 else ""

        full_message = f"{subject}\n{body}"

        # Determine if Claude authored/co-authored this commit
        is_claude = any(p.search(full_message) for p in claude_patterns)

        commits.append({
            "hash": commit_hash[:8],
            "author": author_name,
            "email": author_email,
            "date": date,
            "subject": subject,
            "is_claude": is_claude,
        })

    total = len(commits)
    claude_count = sum(1 for c in commits if c["is_claude"])
    user_count = total - claude_count

    summary = {
        "total_commits": total,
        "claude_commits": claude_count,
        "user_commits": user_count,
        "claude_percentage": round((claude_count / total) * 100, 1) if total > 0 else 0,
        "user_percentage": round((user_count / total) * 100, 1) if total > 0 else 0,
        "generated_at": datetime.now().isoformat(),
    }

    return {"commits": commits, "summary": summary}


@app.get("/api/git-stats")
async def get_git_stats():
    """Analyze git history, calculate Claude vs User contribution percentages, and save to JSON."""
    try:
        stats = await _analyze_git_history()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    # Persist to JSON file
    try:
        os.makedirs(GIT_STATS_DIR, exist_ok=True)
        save_path = os.path.join(GIT_STATS_DIR, "git_stats.json")
        import aiofiles
        async with aiofiles.open(save_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(stats, indent=2))
    except Exception:
        pass

    return stats


@app.get("/api/git-stats/saved")
async def get_saved_git_stats():
    """Return previously saved git stats, or 404 if none exist."""
    filepath = os.path.join(GIT_STATS_DIR, "git_stats.json")
    if not os.path.isfile(filepath):
        return JSONResponse({"error": "No saved git stats found"}, status_code=404)

    import aiofiles
    async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
        content = await f.read()
    data = json.loads(content)
    return data


# ---------------------------------------------------------------------------
# AI Configuration Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/ai/config")
async def get_ai_config():
    """Return the current AI provider configuration (keys redacted)."""
    try:
        config = load_ai_config()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    # Redact API keys — only show whether they are set
    for p in config.get("providers", []):
        env_var = p.get("api_key_env", "")
        p["api_key_set"] = bool(os.environ.get(env_var)) if env_var else False
        p.pop("api_key", None)  # Never expose direct keys

    return config


@app.put("/api/ai/config")
async def update_ai_config(request: Request):
    """Update AI provider configuration."""
    try:
        new_config = await request.json()
        if not new_config or "providers" not in new_config:
            return JSONResponse({"error": "Invalid config: 'providers' array required"}, status_code=400)

        # Validate providers
        for p in new_config["providers"]:
            if not p.get("id") or not p.get("type"):
                return JSONResponse({"error": "Each provider needs 'id' and 'type'"}, status_code=400)

        save_ai_config(new_config)
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/ai/providers")
async def list_ai_providers():
    """Return a summary of enabled AI providers."""
    try:
        config = load_ai_config()
        providers = []
        for p in config.get("providers", []):
            env_var = p.get("api_key_env", "")
            providers.append({
                "id": p["id"],
                "name": p.get("name", p["id"]),
                "type": p.get("type"),
                "model": p.get("model"),
                "enabled": p.get("enabled", False),
                "priority": p.get("priority", 999),
                "api_key_set": bool(os.environ.get(env_var)) if env_var else False,
            })
        providers.sort(key=lambda x: x["priority"])
        return {"providers": providers, "failover_enabled": config.get("failover_enabled", True)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/ai/test")
async def test_ai_provider(request: Request):
    """Send a quick test prompt to a specific provider to verify it works."""
    data = await request.json()
    provider_id = data.get("provider_id")

    if not provider_id:
        return JSONResponse({"error": "provider_id is required"}, status_code=400)

    try:
        from ai_service import call_ai
        result = await call_ai(
            system_prompt="You are a helpful assistant. Respond in one short sentence.",
            user_prompt="Say hello and confirm you are working.",
            provider_id=provider_id,
        )
        return {"status": "ok", "response": result}
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# In-memory Chat Conversation Store
# ---------------------------------------------------------------------------

# Keyed by "{session_id}:{conversation_id}" for per-user isolation.
# Each browser gets a unique session_id cookie so conversations never leak
# across users even though we use a single in-memory dict.
_chat_sessions = {}


def _get_session_id(request: Request) -> str:
    """Return the session_id from the cookie, or generate a new one."""
    return request.cookies.get("betstamp_session") or uuid.uuid4().hex


def _make_key(session_id: str, conversation_id: str) -> str:
    return f"{session_id}:{conversation_id}"


@app.get("/api/chat/sessions")
async def list_chat_sessions(request: Request):
    """Return a list of active chat sessions for the current user."""
    session_id = _get_session_id(request)
    prefix = f"{session_id}:"
    sessions = []
    for key, session in _chat_sessions.items():
        if key.startswith(prefix):
            cid = key[len(prefix):]
            sessions.append({
                "id": cid,
                "message_count": len(session["messages"]),
                "created": session["created"],
                "has_pipeline_context": bool(session.get("pipeline_context")),
            })
    return {"sessions": sessions}


@app.get("/api/chat/{conversation_id}")
async def get_chat_conversation(conversation_id: str, request: Request):
    """Return the full conversation history for a given ID (scoped to user)."""
    session_id = _get_session_id(request)
    key = _make_key(session_id, conversation_id)
    session = _chat_sessions.get(key)
    if not session:
        return JSONResponse({"error": "Conversation not found"}, status_code=404)
    return {
        "conversation_id": conversation_id,
        "messages": session["messages"],
        "created": session["created"],
        "has_pipeline_context": bool(session.get("pipeline_context")),
        "message_count": len(session["messages"]),
    }


@app.delete("/api/chat/{conversation_id}")
async def delete_chat_conversation(conversation_id: str, request: Request):
    """Delete a conversation session (scoped to user)."""
    session_id = _get_session_id(request)
    key = _make_key(session_id, conversation_id)
    _chat_sessions.pop(key, None)
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: Request):
    """Send a message to the AI chat and get a response with conversation memory.

    Conversations are scoped to a browser session via a cookie so multiple
    users never share state.
    """
    data = await request.json()
    message = (data.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)

    session_id = _get_session_id(request)
    conversation_id = data.get("conversation_id") or uuid.uuid4().hex[:12]
    key = _make_key(session_id, conversation_id)
    pipeline_context = data.get("pipeline_context")

    # Get or create session (scoped to user)
    if key in _chat_sessions:
        session = _chat_sessions[key]
    else:
        session = {
            "messages": [],
            "pipeline_context": None,
            "created": datetime.now().isoformat(),
        }
        _chat_sessions[key] = session

    # Update pipeline context if provided
    if pipeline_context:
        session["pipeline_context"] = pipeline_context

    # Build system prompt with optional pipeline context
    system_prompt = CHAT_SYSTEM_PROMPT
    if session.get("pipeline_context"):
        context_str = json.dumps(session["pipeline_context"], indent=2)
        # Truncate to 20KB
        if len(context_str) > 20480:
            context_str = context_str[:20480] + "\n... (truncated)"
        system_prompt += (
            "\n\n=== PIPELINE CONTEXT ===\n"
            + context_str
        )

    # Append user message
    session["messages"].append({"role": "user", "content": message})

    try:
        result = await call_ai_chat(
            messages=session["messages"],
            system_prompt=system_prompt,
        )

        # Append assistant response
        session["messages"].append({"role": "assistant", "content": result["text"]})

        # Set session cookie so subsequent requests are tied to this user
        response = JSONResponse({
            "conversation_id": conversation_id,
            "response": {
                "text": result["text"],
                "ai_meta": {
                    "provider": result["provider_name"],
                    "model": result["model"],
                    "usage": result["usage"],
                    "elapsed_seconds": result["elapsed_seconds"],
                },
            },
            "message_count": len(session["messages"]),
        })
        response.set_cookie(
            key="betstamp_session",
            value=session_id,
            httponly=True,
            samesite="lax",
            max_age=60 * 60 * 24,  # 24 hours
        )
        return response
    except Exception as e:
        # Remove the user message we just added since the call failed
        session["messages"].pop()
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    print(f"Serving JSON files from: {DATA_DIR}")
    uvicorn.run(socket_app, host="0.0.0.0", port=8191, log_level="info")
