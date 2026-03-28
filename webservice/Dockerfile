# BetStamp API — Python 3.13 + Node.js 20 (for Claude Agent SDK wrapper)
FROM python:3.13-slim AS base

# Install Node.js 20.x (required for claude-sdk-wrapper.mjs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory to project root (preserves ../data, ../mcp-server layout)
WORKDIR /app

# ---- Python dependencies ----
COPY webservice/requirements.txt /app/webservice/requirements.txt
RUN pip install --no-cache-dir -r /app/webservice/requirements.txt

# ---- Node dependencies (Claude Agent SDK) ----
COPY webservice/package.json webservice/package-lock.json* /app/webservice/
RUN cd /app/webservice && npm install --omit=dev

# ---- Copy application code ----
COPY webservice/ /app/webservice/
COPY mcp-server/ /app/mcp-server/
COPY data/ /app/data/
COPY devNotesData/ /app/devNotesData/
COPY DEVLOG.md /app/DEVLOG.md

# Create saved_results and logs directories
RUN mkdir -p /app/saved_results /app/webservice/logs/runs

# MCP config for Docker — uses container-absolute path
RUN echo '{"mcpServers":{"betstamp-intelligence":{"command":"python","args":["/app/mcp-server/mcp_server.py"]}}}' \
    > /app/mcp-server/.mcp.json

WORKDIR /app/webservice

# Railway injects $PORT; uvicorn binds to 0.0.0.0:$PORT
ENV PORT=8191
EXPOSE ${PORT}

CMD ["sh", "-c", "python app.py"]
