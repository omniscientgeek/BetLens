import os
import re
import json
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
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
        os.makedirs(DATA_DIR, exist_ok=True)
        save_path = os.path.join(DATA_DIR, f"devnotes_project_{project_id}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass  # Don't fail the response if saving fails

    return jsonify(result)


@app.route("/api/devnotes/<int:project_id>", methods=["GET"])
def get_devnotes(project_id):
    """Return previously saved devnotes for a project, or 404 if none exist."""
    filename = f"devnotes_project_{project_id}.json"
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(filepath):
        return jsonify({"error": "No saved devnotes found"}), 404
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


if __name__ == "__main__":
    print(f"Serving JSON files from: {DATA_DIR}")
    app.run(host="0.0.0.0", port=8191, debug=True)
