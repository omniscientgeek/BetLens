import eventlet
eventlet.monkey_patch()

import os
import re
import json
import subprocess
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ---------------------------------------------------------------------------
# Three-phase processing pipeline (Detect → Analyze → Brief)
# ---------------------------------------------------------------------------
PHASES = [
    {"name": "detect", "label": "Detecting patterns"},
    {"name": "analyze", "label": "Analyzing structure"},
    {"name": "brief", "label": "Generating brief"},
]


def run_processing_pipeline(filename, sid):
    """Execute the 3-phase pipeline, emitting status updates via WebSocket."""
    try:
        for i, phase in enumerate(PHASES):
            socketio.emit("phase_update", {
                "filename": filename,
                "phase": phase["name"],
                "label": phase["label"],
                "status": "in_progress",
                "phaseIndex": i,
                "totalPhases": len(PHASES),
            }, to=sid)

            # Stub: each phase takes 10 seconds (replace with real logic later)
            socketio.sleep(10)

            socketio.emit("phase_update", {
                "filename": filename,
                "phase": phase["name"],
                "label": phase["label"],
                "status": "complete",
                "phaseIndex": i,
                "totalPhases": len(PHASES),
            }, to=sid)

        socketio.emit("processing_complete", {"filename": filename}, to=sid)
    except Exception as e:
        socketio.emit("processing_error", {
            "filename": filename,
            "error": str(e),
        }, to=sid)


@socketio.on("start_processing")
def handle_start_processing(data):
    filename = data.get("filename", "")
    if not filename.endswith(".json"):
        emit("processing_error", {"filename": filename, "error": "Only .json files supported"})
        return
    sid = request.sid
    socketio.start_background_task(run_processing_pipeline, filename, sid)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DEV_NOTES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "devNotesData"))
CONVERSATIONS_DIR = r"C:\ProgramData\DesktopDevService\Conversations"


@app.route("/api/files", methods=["GET"])
def list_files():
    """Return a list of all JSON files in the data directory."""
    if not os.path.isdir(DATA_DIR):
        return jsonify({"files": []})

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    files.sort()
    return jsonify({"files": files})


@app.route("/api/files/<filename>", methods=["GET"])
def get_file(filename):
    """Return the contents of a specific JSON file."""
    if not filename.endswith(".json"):
        return jsonify({"error": "Only .json files are supported"}), 400

    filepath = os.path.join(DATA_DIR, filename)
    filepath = os.path.abspath(filepath)

    # Prevent directory traversal
    if not filepath.startswith(DATA_DIR):
        return jsonify({"error": "Invalid file path"}), 403

    if not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 404

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return jsonify(data)


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


@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    """Return conversations filtered by project ID."""
    project_id = request.args.get("project_id", type=int)
    if project_id is None:
        return jsonify({"error": "project_id query parameter is required"}), 400

    project_guid = PROJECT_GUID_MAP.get(project_id)
    if not project_guid:
        return jsonify({"error": f"Unknown project_id: {project_id}"}), 404

    if not os.path.isdir(CONVERSATIONS_DIR):
        return jsonify({"error": "Conversations directory not found"}), 500

    conversations = []
    for fname in os.listdir(CONVERSATIONS_DIR):
        if not fname.endswith(".txt"):
            continue
        filepath = os.path.join(CONVERSATIONS_DIR, fname)
        parsed = _parse_conversation_file(filepath)
        if parsed and parsed["project"] == project_guid:
            conversations.append(parsed)

    # Sort by created date descending (newest first)
    conversations.sort(key=lambda c: c.get("created", ""), reverse=True)

    result = {"conversations": conversations, "count": len(conversations)}

    # Persist to a JSON file so it can be auto-loaded later
    try:
        os.makedirs(DEV_NOTES_DIR, exist_ok=True)
        save_path = os.path.join(DEV_NOTES_DIR, f"devnotes_project_{project_id}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass  # Don't fail the response if saving fails

    return jsonify(result)


@app.route("/api/devnotes/<int:project_id>", methods=["GET"])
def get_devnotes(project_id):
    """Return previously saved devnotes for a project, or 404 if none exist."""
    filename = f"devnotes_project_{project_id}.json"
    filepath = os.path.join(DEV_NOTES_DIR, filename)
    if not os.path.isfile(filepath):
        return jsonify({"error": "No saved devnotes found"}), 404
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


def _parse_notes_file():
    """Parse NOTES.md and return individual notes split by ## headings."""
    notes_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "NOTES.md"))
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


@app.route("/api/notes", methods=["GET"])
def get_notes():
    """Return parsed notes from NOTES.md."""
    notes = _parse_notes_file()
    return jsonify({"notes": notes, "count": len(notes)})


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GIT_STATS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "devNotesData"))


def _analyze_git_history():
    """Analyze git commit history and classify commits as Claude or User authored."""
    # Get git log with commit hash, author, date, subject, and body
    result = subprocess.run(
        ["git", "log", "--pretty=format:%H||%an||%ae||%ai||%s||%b%x00"],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"git log failed: {result.stderr}")

    raw = result.stdout.strip()
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


@app.route("/api/git-stats", methods=["GET"])
def get_git_stats():
    """Analyze git history, calculate Claude vs User contribution percentages, and save to JSON."""
    try:
        stats = _analyze_git_history()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Persist to JSON file
    try:
        os.makedirs(GIT_STATS_DIR, exist_ok=True)
        save_path = os.path.join(GIT_STATS_DIR, "git_stats.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    except Exception:
        pass

    return jsonify(stats)


@app.route("/api/git-stats/saved", methods=["GET"])
def get_saved_git_stats():
    """Return previously saved git stats, or 404 if none exist."""
    filepath = os.path.join(GIT_STATS_DIR, "git_stats.json")
    if not os.path.isfile(filepath):
        return jsonify({"error": "No saved git stats found"}), 404
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


if __name__ == "__main__":
    print(f"Serving JSON files from: {DATA_DIR}")
    socketio.run(app, host="0.0.0.0", port=8191, debug=True, use_reloader=False)
