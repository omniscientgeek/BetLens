import os
import re
import json
import uuid
import time
import asyncio
import logging
import subprocess
import dataclasses
from datetime import datetime

from logging_config import setup_logging, create_run_logger, close_run_logger, RUNS_LOG_DIR

setup_logging()
logger = logging.getLogger(__name__)

from fastapi import FastAPI, Query, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse
import socketio

from detect import run_detection
from ai_service import (
    load_config as load_ai_config,
    save_config as save_ai_config,
    get_enabled_providers,
    run_analyze_phase,
    run_brief_phase,
    run_fix_phase,
    call_ai_chat,
    call_ai_chat_stream,
    CHAT_SYSTEM_PROMPT,
    _build_brief_payload,
)
from verification_agents import run_verification

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
    ping_timeout=600,
    ping_interval=30,
)
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# ---------------------------------------------------------------------------
# Three-phase processing pipeline (Detect → Analyze → Brief)
# ---------------------------------------------------------------------------
PHASES = [
    {"name": "detect", "label": "Detecting patterns"},
    {"name": "analyze", "label": "Analyzing structure"},
    {"name": "audit_analyze", "label": "Auditing analysis"},
    {"name": "brief", "label": "Generating brief"},
    {"name": "audit_brief", "label": "Auditing brief"},
]

# Self-healing: max fix+re-audit attempts before accepting a non-pass verdict
MAX_FIX_ATTEMPTS = 3


@dataclasses.dataclass
class PipelineState:
    """Tracks a pipeline's lifecycle independently of any WebSocket session.

    Keyed by filename in ``_pipeline_cache``.  The ``attached_sid`` field
    points to the *currently connected* socket — when a client disconnects
    it becomes ``None``, but the pipeline keeps running.  A reconnecting
    client re-attaches by setting ``attached_sid`` to its new sid.
    """
    run_id: str
    filename: str
    current_phase: int        # index into PHASES; len(PHASES) = done
    status: str               # "running" | "complete" | "error"
    results: dict             # {"detect": {…}, "analyze": {…}, "brief": {…}}
    replay_events: list       # completion-level events for instant replay on reconnect
    task: asyncio.Task | None
    attached_sid: str | None  # currently connected socket (None = disconnected)
    created_at: float
    error: str | None = None


# Pipeline cache — keyed by filename so reconnecting clients resume the same run
_pipeline_cache: dict[str, PipelineState] = {}
PIPELINE_CACHE_TTL = 1800  # 30 minutes


def _is_cache_expired(state: PipelineState) -> bool:
    return time.time() - state.created_at > PIPELINE_CACHE_TTL


async def _replay_completed_phases(sid: str, state: PipelineState):
    """Send completion-level events for all already-finished phases to *sid*.

    This gives a reconnecting client instant catch-up without replaying
    hundreds of streaming chunks.
    """
    for event in state.replay_events:
        await sio.emit(event["type"], event["payload"], to=sid)


async def _replay_completed_pipeline(sid: str, state: PipelineState):
    """Pipeline already done — replay every phase result + processing_complete."""
    await _replay_completed_phases(sid, state)
    await sio.emit("processing_complete", {
        "filename": state.filename,
        "results": state.results,
        "run_id": state.run_id,
    }, to=sid)


async def _safe_emit(state: PipelineState, event: str, payload: dict):
    """Emit to the attached client if one exists; silently skip otherwise."""
    sid = state.attached_sid
    if sid:
        try:
            await sio.emit(event, payload, to=sid)
        except Exception:
            pass  # Client may have just disconnected — non-fatal


async def _emit_phase(state, filename, phase, i, status, result=None, run_id=None):
    """Helper to emit a phase_update event and record completions for replay."""
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
    if run_id is not None:
        payload["run_id"] = run_id
    # Record completion events so reconnecting clients get instant catch-up
    if status in ("complete", "error"):
        state.replay_events.append({"type": "phase_update", "payload": payload})
    await _safe_emit(state, "phase_update", payload)


async def _auto_save_results(filename: str, pipeline_results: dict, run_logger):
    """Automatically save pipeline results after completion (no browser download)."""
    import aiofiles

    os.makedirs(SAVED_RESULTS_DIR, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    save_filename = f"betlens_results_{ts}.json"
    save_path = os.path.join(SAVED_RESULTS_DIR, save_filename)

    # Load the original file data from disk
    file_data = None
    data_path = os.path.join(DATA_DIR, filename)
    if os.path.isfile(data_path):
        async with aiofiles.open(data_path, "r", encoding="utf-8") as f:
            content = await f.read()
        file_data = json.loads(content)

    # Restructure verification to top-level audit_analyze / audit_brief
    structured_results = dict(pipeline_results)
    analyze = structured_results.get("analyze")
    if isinstance(analyze, dict) and "verification" in analyze:
        structured_results["audit_analyze"] = analyze.pop("verification")
        structured_results["analyze"] = analyze
    brief = structured_results.get("brief")
    if isinstance(brief, dict) and "verification" in brief:
        structured_results["audit_brief"] = brief.pop("verification")
        structured_results["brief"] = brief

    payload = {
        "source_file": filename,
        "saved_at": datetime.now().isoformat(),
        "pipeline_results": structured_results,
        "file_data": file_data,
    }

    async with aiofiles.open(save_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(payload, indent=2, ensure_ascii=False))

    run_logger.info("Auto-saved pipeline results to %s", save_path)
    logger.info("PIPELINE Auto-saved results to %s", save_filename)


async def run_processing_pipeline(filename: str, state: PipelineState):
    """Execute the 5-phase pipeline, emitting status updates via WebSocket.

    The pipeline runs to completion even if the client disconnects mid-way.
    Completed phase results are cached in *state* so a reconnecting client
    can resume instantly instead of re-running expensive AI calls.
    """
    run_id = state.run_id
    run_logger, run_handler = create_run_logger(run_id)
    run_logger.info("PIPELINE run_id=%s file=%s sid=%s", run_id, filename, state.attached_sid)
    logger.info("PIPELINE Starting pipeline for %s (run_id=%s)", filename, run_id)
    try:
        pipeline_results = state.results  # may already have completed phases on resume

        for i, phase in enumerate(PHASES):
            # Skip phases that already completed (e.g. after a restart/resume)
            if phase["name"] in pipeline_results:
                run_logger.info("Phase %d: %s -> already cached, skipping", i, phase["name"])
                continue

            state.current_phase = i
            run_logger.info("Phase %d: %s -> in_progress", i, phase["name"])
            logger.info("PIPELINE Phase %d: %s -> in_progress", i, phase["name"])
            phase_start = time.time()
            await _emit_phase(state, filename, phase, i, "in_progress", run_id=run_id)

            if phase["name"] == "detect":
                # --- Real detection: compute probabilities, vig, fair odds ---
                detection = await run_detection(filename)
                pipeline_results["detect"] = detection
                elapsed = time.time() - phase_start
                run_logger.info("Phase %d: %s completed in %.2fs", i, phase["name"], elapsed)
                logger.info("PIPELINE Phase %d: %s completed in %.2fs", i, phase["name"], elapsed)
                await _emit_phase(state, filename, phase, i, "complete", result=detection, run_id=run_id)
            elif phase["name"] == "analyze":
                # --- AI-powered cross-sportsbook analysis (streamed) ---
                try:
                    # Heartbeat task keeps the WebSocket alive during long AI calls
                    heartbeat_active = True

                    async def _heartbeat():
                        while heartbeat_active:
                            await asyncio.sleep(10)
                            if heartbeat_active:
                                try:
                                    elapsed = int(time.time() - phase_start)
                                    await _safe_emit(state, "heartbeat", {
                                        "phase": "analyze",
                                        "elapsed": elapsed,
                                        "run_id": run_id,
                                    })
                                except Exception:
                                    pass

                    heartbeat_task = asyncio.create_task(_heartbeat())

                    async def on_analyze_conversation(event_type, data):
                        """Emit analyze conversation events to the frontend in real-time."""
                        payload = {
                            "event": event_type,
                            "data": data,
                            "filename": filename,
                            "run_id": run_id,
                        }
                        # Save the "complete" event for replay on reconnect
                        if event_type == "complete":
                            state.replay_events.append({"type": "analyze_conversation", "payload": payload})
                        await _safe_emit(state, "analyze_conversation", payload)

                    try:
                        analysis = await asyncio.wait_for(
                            run_analyze_phase(
                                pipeline_results.get("detect", {}),
                                run_logger=run_logger,
                                on_conversation_event=on_analyze_conversation,
                                filename=filename,
                            ),
                            timeout=600,  # 10-minute hard ceiling
                        )
                    finally:
                        heartbeat_active = False
                        heartbeat_task.cancel()
                    pipeline_results["analyze"] = analysis
                    elapsed = time.time() - phase_start
                    run_logger.info("Phase %d: %s completed in %.2fs", i, phase["name"], elapsed)
                    logger.info("PIPELINE Phase %d: %s completed in %.2fs", i, phase["name"], elapsed)
                    await _emit_phase(state, filename, phase, i, "complete", result=analysis, run_id=run_id)

                except Exception as exc:
                    # Graceful degradation: report the error but continue pipeline
                    run_logger.error("Analyze phase FAILED: %s", exc)
                    logger.error("PIPELINE Analyze phase FAILED: %s", exc)
                    err_result = {"error": str(exc), "ai_meta": None}
                    pipeline_results["analyze"] = err_result
                    await _emit_phase(state, filename, phase, i, "error", result=err_result, run_id=run_id)

            elif phase["name"] == "audit_analyze":
                # --- Audit: run 3 verification agents to double-check the analysis ---
                # Self-healing loop: if audit fails, AI fixes the text and re-audits
                analysis = pipeline_results.get("analyze", {})
                analyze_text = (analysis.get("conversation") or {}).get("assistant_response", "")
                if analyze_text and not analysis.get("error"):
                    try:
                        run_logger.info("Starting audit agents for analysis...")
                        source_data = json.dumps(
                            pipeline_results.get("detect", {}),
                            separators=(",", ":"),
                            default=str,
                        )

                        current_text = analyze_text
                        fix_attempt = 0
                        fix_history = []  # track all attempts for the frontend

                        while True:
                            # Real-time callback: emit each agent's result as it completes
                            async def _on_analyze_agent(agent_name, agent_result, _attempt=fix_attempt):
                                agent_payload = {
                                    "filename": filename,
                                    "phase": "analyze",
                                    "agent_name": agent_name,
                                    "agent_result": agent_result,
                                    "run_id": run_id,
                                    "fix_attempt": _attempt,
                                }
                                state.replay_events.append({"type": "verification_agent_update", "payload": agent_payload})
                                await _safe_emit(state, "verification_agent_update", agent_payload)

                            verification = await run_verification(
                                text_to_verify=current_text,
                                source_data=source_data,
                                run_logger=run_logger,
                                on_agent_complete=_on_analyze_agent,
                            )

                            verification_payload = {
                                "filename": filename,
                                "phase": "analyze",
                                "verification": verification,
                                "run_id": run_id,
                                "fix_attempt": fix_attempt,
                            }
                            state.replay_events.append({"type": "verification_update", "payload": verification_payload})
                            await _safe_emit(state, "verification_update", verification_payload)

                            overall = verification.get("overall_verdict", "error")
                            run_logger.info(
                                "Analysis audit (attempt %d): overall=%s elapsed=%.2fs",
                                fix_attempt, overall, verification["elapsed_seconds"],
                            )

                            fix_history.append({
                                "attempt": fix_attempt,
                                "verdict": overall,
                                "verification": verification,
                                "text": current_text,
                            })

                            # If pass, or we've hit max retries, stop the loop
                            if overall == "pass" or fix_attempt >= MAX_FIX_ATTEMPTS:
                                if overall != "pass" and fix_attempt >= MAX_FIX_ATTEMPTS:
                                    run_logger.warning(
                                        "Analysis audit still %s after %d fix attempts — accepting result",
                                        overall, fix_attempt,
                                    )
                                break

                            # --- Self-healing: AI fixes the text based on audit issues ---
                            fix_attempt += 1
                            run_logger.info(
                                "Analysis audit FAILED (%s) — starting fix attempt %d/%d",
                                overall, fix_attempt, MAX_FIX_ATTEMPTS,
                            )

                            # Emit fix_started event so the UI shows the fix in progress
                            fix_payload = {
                                "filename": filename,
                                "phase": "analyze",
                                "fix_attempt": fix_attempt,
                                "max_attempts": MAX_FIX_ATTEMPTS,
                                "previous_verdict": overall,
                                "run_id": run_id,
                            }
                            state.replay_events.append({"type": "fix_started", "payload": fix_payload})
                            await _safe_emit(state, "fix_started", fix_payload)

                            try:
                                fix_result = await asyncio.wait_for(
                                    run_fix_phase(
                                        original_text=current_text,
                                        audit_result=verification,
                                        phase_type="analyze",
                                        run_logger=run_logger,
                                    ),
                                    timeout=600,
                                )
                                current_text = fix_result["fixed_text"]

                                # Update the analysis conversation with the fixed text
                                analysis["conversation"]["assistant_response"] = current_text
                                pipeline_results["analyze"] = analysis

                                fix_complete_payload = {
                                    "filename": filename,
                                    "phase": "analyze",
                                    "fix_attempt": fix_attempt,
                                    "max_attempts": MAX_FIX_ATTEMPTS,
                                    "run_id": run_id,
                                    "fix_ai_meta": fix_result.get("ai_meta"),
                                }
                                state.replay_events.append({"type": "fix_complete", "payload": fix_complete_payload})
                                await _safe_emit(state, "fix_complete", fix_complete_payload)

                                run_logger.info(
                                    "Analysis fix attempt %d complete — re-auditing...",
                                    fix_attempt,
                                )
                            except Exception as fix_exc:
                                run_logger.error(
                                    "Analysis fix attempt %d FAILED: %s — stopping self-heal loop",
                                    fix_attempt, fix_exc,
                                )
                                break

                        # Store final verification + fix history
                        verification["fix_history"] = fix_history
                        verification["fix_attempts"] = fix_attempt
                        analysis["verification"] = verification
                        pipeline_results["analyze"] = analysis

                        await _emit_phase(state, filename, phase, i, "complete", run_id=run_id)
                    except Exception as vex:
                        run_logger.error("Analysis audit FAILED (non-blocking): %s", vex)
                        logger.error("PIPELINE Analysis audit FAILED: %s", vex)
                        await _emit_phase(state, filename, phase, i, "error", run_id=run_id)
                else:
                    # No analysis text to audit — skip
                    run_logger.info("Skipping audit_analyze — no analysis text available")
                    await _emit_phase(state, filename, phase, i, "complete", run_id=run_id)

            elif phase["name"] == "brief":
                # --- AI-powered actionable briefing (streamed to client) ---
                try:
                    # Heartbeat for the brief phase too
                    brief_heartbeat_active = True

                    async def _brief_heartbeat():
                        while brief_heartbeat_active:
                            await asyncio.sleep(10)
                            if brief_heartbeat_active:
                                try:
                                    elapsed = int(time.time() - phase_start)
                                    await _safe_emit(state, "heartbeat", {
                                        "phase": "brief",
                                        "elapsed": elapsed,
                                        "run_id": run_id,
                                    })
                                except Exception:
                                    pass

                    brief_heartbeat_task = asyncio.create_task(_brief_heartbeat())

                    async def on_brief_chunk(text_delta):
                        await _safe_emit(state, "brief_chunk", {"text": text_delta})

                    try:
                        brief = await asyncio.wait_for(
                            run_brief_phase(
                                pipeline_results.get("detect", {}),
                                pipeline_results.get("analyze", {}),
                                on_chunk=on_brief_chunk,
                                run_logger=run_logger,
                            ),
                            timeout=600,  # 10-minute hard ceiling
                        )
                    finally:
                        brief_heartbeat_active = False
                        brief_heartbeat_task.cancel()
                    pipeline_results["brief"] = brief
                    elapsed = time.time() - phase_start
                    run_logger.info("Phase %d: %s completed in %.2fs", i, phase["name"], elapsed)
                    logger.info("PIPELINE Phase %d: %s completed in %.2fs", i, phase["name"], elapsed)
                    await _emit_phase(state, filename, phase, i, "complete", result=brief, run_id=run_id)

                except Exception as exc:
                    run_logger.error("Brief phase FAILED: %s", exc)
                    logger.error("PIPELINE Brief phase FAILED: %s", exc)
                    err_result = {"error": str(exc), "ai_meta": None}
                    pipeline_results["brief"] = err_result
                    await _emit_phase(state, filename, phase, i, "error", result=err_result, run_id=run_id)

            elif phase["name"] == "audit_brief":
                # --- Audit: run 3 verification agents to double-check the brief ---
                # Self-healing loop: if audit fails, AI fixes the text and re-audits
                brief = pipeline_results.get("brief", {})
                if brief.get("brief_text") and not brief.get("error"):
                    try:
                        run_logger.info("Starting audit agents for brief...")
                        source_data = json.dumps(
                            _build_brief_payload(
                                pipeline_results.get("detect", {}),
                                pipeline_results.get("analyze", {}),
                            ),
                            separators=(",", ":"),
                        )

                        current_brief_text = brief["brief_text"]
                        fix_attempt = 0
                        fix_history = []

                        while True:
                            # Real-time callback: emit each agent's result as it completes
                            async def _on_brief_agent(agent_name, agent_result, _attempt=fix_attempt):
                                agent_payload = {
                                    "filename": filename,
                                    "phase": "brief",
                                    "agent_name": agent_name,
                                    "agent_result": agent_result,
                                    "run_id": run_id,
                                    "fix_attempt": _attempt,
                                }
                                state.replay_events.append({"type": "verification_agent_update", "payload": agent_payload})
                                await _safe_emit(state, "verification_agent_update", agent_payload)

                            verification = await run_verification(
                                text_to_verify=current_brief_text,
                                source_data=source_data,
                                run_logger=run_logger,
                                on_agent_complete=_on_brief_agent,
                            )

                            verification_payload = {
                                "filename": filename,
                                "phase": "brief",
                                "verification": verification,
                                "run_id": run_id,
                                "fix_attempt": fix_attempt,
                            }
                            state.replay_events.append({"type": "verification_update", "payload": verification_payload})
                            await _safe_emit(state, "verification_update", verification_payload)

                            overall = verification.get("overall_verdict", "error")
                            run_logger.info(
                                "Brief audit (attempt %d): overall=%s elapsed=%.2fs",
                                fix_attempt, overall, verification["elapsed_seconds"],
                            )

                            fix_history.append({
                                "attempt": fix_attempt,
                                "verdict": overall,
                                "verification": verification,
                                "text": current_brief_text,
                            })

                            # If pass, or we've hit max retries, stop the loop
                            if overall == "pass" or fix_attempt >= MAX_FIX_ATTEMPTS:
                                if overall != "pass" and fix_attempt >= MAX_FIX_ATTEMPTS:
                                    run_logger.warning(
                                        "Brief audit still %s after %d fix attempts — accepting result",
                                        overall, fix_attempt,
                                    )
                                break

                            # --- Self-healing: AI fixes the brief based on audit issues ---
                            fix_attempt += 1
                            run_logger.info(
                                "Brief audit FAILED (%s) — starting fix attempt %d/%d",
                                overall, fix_attempt, MAX_FIX_ATTEMPTS,
                            )

                            fix_payload = {
                                "filename": filename,
                                "phase": "brief",
                                "fix_attempt": fix_attempt,
                                "max_attempts": MAX_FIX_ATTEMPTS,
                                "previous_verdict": overall,
                                "run_id": run_id,
                            }
                            state.replay_events.append({"type": "fix_started", "payload": fix_payload})
                            await _safe_emit(state, "fix_started", fix_payload)

                            try:
                                # Stream fix chunks so the UI can show progress
                                async def _on_fix_brief_chunk(text_delta):
                                    await _safe_emit(state, "brief_chunk", {"text": text_delta, "fix_attempt": fix_attempt})

                                fix_result = await asyncio.wait_for(
                                    run_fix_phase(
                                        original_text=current_brief_text,
                                        audit_result=verification,
                                        phase_type="brief",
                                        run_logger=run_logger,
                                        on_chunk=_on_fix_brief_chunk,
                                    ),
                                    timeout=600,
                                )
                                current_brief_text = fix_result["fixed_text"]

                                # Update the brief with the fixed text
                                brief["brief_text"] = current_brief_text
                                pipeline_results["brief"] = brief

                                fix_complete_payload = {
                                    "filename": filename,
                                    "phase": "brief",
                                    "fix_attempt": fix_attempt,
                                    "max_attempts": MAX_FIX_ATTEMPTS,
                                    "run_id": run_id,
                                    "fix_ai_meta": fix_result.get("ai_meta"),
                                }
                                state.replay_events.append({"type": "fix_complete", "payload": fix_complete_payload})
                                await _safe_emit(state, "fix_complete", fix_complete_payload)

                                run_logger.info(
                                    "Brief fix attempt %d complete — re-auditing...",
                                    fix_attempt,
                                )
                            except Exception as fix_exc:
                                run_logger.error(
                                    "Brief fix attempt %d FAILED: %s — stopping self-heal loop",
                                    fix_attempt, fix_exc,
                                )
                                break

                        # Store final verification + fix history
                        verification["fix_history"] = fix_history
                        verification["fix_attempts"] = fix_attempt
                        brief["verification"] = verification
                        pipeline_results["brief"] = brief

                        await _emit_phase(state, filename, phase, i, "complete", run_id=run_id)
                    except Exception as vex:
                        run_logger.error("Brief audit FAILED (non-blocking): %s", vex)
                        logger.error("PIPELINE Brief audit FAILED: %s", vex)
                        await _emit_phase(state, filename, phase, i, "error", run_id=run_id)
                else:
                    # No brief text to audit — skip
                    run_logger.info("Skipping audit_brief — no brief text available")
                    await _emit_phase(state, filename, phase, i, "complete", run_id=run_id)

            else:
                # Unknown phase — placeholder
                await asyncio.sleep(2)
                await _emit_phase(state, filename, phase, i, "complete", run_id=run_id)

        state.status = "complete"
        state.current_phase = len(PHASES)
        run_logger.info("PIPELINE complete for %s", filename)
        await _safe_emit(state, "processing_complete", {
            "filename": filename,
            "results": pipeline_results,
            "run_id": run_id,
        })

        # Auto-save results to saved_results/ directory
        try:
            await _auto_save_results(filename, pipeline_results, run_logger)
        except Exception as save_exc:
            run_logger.error("Auto-save failed: %s", save_exc)
    except asyncio.CancelledError:
        state.status = "error"
        state.error = "Pipeline cancelled"
        run_logger.info("PIPELINE cancelled for %s", filename)
    except Exception as e:
        state.status = "error"
        state.error = str(e)
        run_logger.error("PIPELINE error: %s", e)
        await _safe_emit(state, "processing_error", {
            "filename": filename,
            "error": str(e),
            "run_id": run_id,
        })
    finally:
        close_run_logger(run_logger, run_handler)


# ---------------------------------------------------------------------------
# Pipeline cache cleanup — evict expired entries every 60s
# ---------------------------------------------------------------------------

async def _cache_cleanup_loop():
    """Periodically remove expired pipeline cache entries."""
    while True:
        await asyncio.sleep(60)
        now = time.time()
        expired = [k for k, v in _pipeline_cache.items() if now - v.created_at > PIPELINE_CACHE_TTL]
        for k in expired:
            state = _pipeline_cache.pop(k, None)
            if state:
                logger.info("PIPELINE Cache expired for %s (run_id=%s)", k, state.run_id)
                if state.task and not state.task.done():
                    state.task.cancel()


@app.on_event("startup")
async def startup_cache_cleanup():
    asyncio.create_task(_cache_cleanup_loop())


# ---------------------------------------------------------------------------
# WebSocket handlers — connect / disconnect / start_processing
# ---------------------------------------------------------------------------

@sio.on("connect")
async def handle_connect(sid, environ):
    logger.info("SOCKET Client connected: %s", sid)


@sio.on("disconnect")
async def handle_disconnect(sid):
    logger.info("SOCKET Client disconnected: %s", sid)
    # Detach sid from any pipeline but let the pipeline continue running.
    # Completed results are cached so the client can resume on reconnect.
    for fname, state in _pipeline_cache.items():
        if state.attached_sid == sid:
            state.attached_sid = None
            logger.info(
                "PIPELINE Detached sid=%s from pipeline file=%s (continues running, run_id=%s)",
                sid, fname, state.run_id,
            )


@sio.on("start_processing")
async def handle_start_processing(sid, data):
    logger.info("PIPELINE start_processing received from %s: %s", sid, data)
    filename = data.get("filename", "")
    resume = data.get("resume", False)

    if not filename.endswith(".json"):
        await sio.emit("processing_error", {"filename": filename, "error": "Only .json files supported"}, to=sid)
        return

    existing = _pipeline_cache.get(filename)

    # --- Resume path: reattach to a running or completed pipeline ---
    if resume and existing and not _is_cache_expired(existing):
        existing.attached_sid = sid
        logger.info(
            "PIPELINE Resuming pipeline for %s (run_id=%s, status=%s, phase=%d)",
            filename, existing.run_id, existing.status, existing.current_phase,
        )

        if existing.status == "complete":
            # Pipeline finished while client was away — instant replay
            await _replay_completed_pipeline(sid, existing)
        elif existing.status == "error":
            # Pipeline failed — send the error
            await _replay_completed_phases(sid, existing)
            await sio.emit("processing_error", {
                "filename": filename,
                "error": existing.error or "Pipeline failed",
                "run_id": existing.run_id,
            }, to=sid)
        else:
            # Pipeline still running — replay completed phases, then live-stream picks up
            await _replay_completed_phases(sid, existing)
            # Emit the current running phase as "in_progress" so the stepper shows it active
            if existing.current_phase < len(PHASES):
                current = PHASES[existing.current_phase]
                await sio.emit("phase_update", {
                    "filename": existing.filename,
                    "phase": current["name"],
                    "label": current["label"],
                    "status": "in_progress",
                    "phaseIndex": existing.current_phase,
                    "totalPhases": len(PHASES),
                    "run_id": existing.run_id,
                }, to=sid)
            # The running pipeline now emits to this sid via state.attached_sid
        return

    # --- Fresh start: cancel any stale pipeline for this file, create new one ---
    if existing and existing.task and not existing.task.done():
        logger.info("PIPELINE Cancelling previous pipeline for file=%s (fresh start)", filename)
        existing.task.cancel()

    state = PipelineState(
        run_id=uuid.uuid4().hex[:8],
        filename=filename,
        current_phase=0,
        status="running",
        results={},
        replay_events=[],
        task=None,
        attached_sid=sid,
        created_at=time.time(),
    )
    task = asyncio.create_task(run_processing_pipeline(filename, state))
    state.task = task
    _pipeline_cache[filename] = state


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


@app.post("/api/files/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a new JSON data file to the data directory."""
    if not file.filename or not file.filename.endswith(".json"):
        return JSONResponse({"error": "Only .json files are supported"}, status_code=400)

    # Sanitise filename – strip path components to prevent directory traversal
    safe_name = os.path.basename(file.filename)
    if not safe_name:
        return JSONResponse({"error": "Invalid filename"}, status_code=400)

    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.abspath(os.path.join(DATA_DIR, safe_name))

    if not dest.startswith(DATA_DIR):
        return JSONResponse({"error": "Invalid file path"}, status_code=403)

    # Read and validate JSON before writing
    content = await file.read()
    try:
        json.loads(content)
    except (json.JSONDecodeError, ValueError) as exc:
        return JSONResponse({"error": f"File is not valid JSON: {exc}"}, status_code=400)

    import aiofiles
    async with aiofiles.open(dest, "wb") as f:
        await f.write(content)

    logger.info("Uploaded data file: %s (%d bytes)", safe_name, len(content))
    return {"uploaded": True, "filename": safe_name, "size": len(content)}


# ---------------------------------------------------------------------------
# Save pipeline results
# ---------------------------------------------------------------------------

SAVED_RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_results"))


@app.post("/api/save-results")
async def save_results(request: Request):
    """Save pipeline results (analyze + brief + file metadata) to a JSON file."""
    body = await request.json()
    filename = body.get("filename")
    pipeline_results = body.get("pipelineResults")
    file_data = body.get("fileData")

    if not filename or not pipeline_results:
        return JSONResponse({"error": "filename and pipelineResults are required"}, status_code=400)

    os.makedirs(SAVED_RESULTS_DIR, exist_ok=True)

    # Build a timestamped filename: e.g. betlens_results_2026-03-27_143022.json
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    save_filename = f"betlens_results_{ts}.json"
    save_path = os.path.join(SAVED_RESULTS_DIR, save_filename)

    # Restructure: promote sub-agent verification results from nested
    # analyze.verification / brief.verification to top-level audit_analyze / audit_brief
    # so the saved file mirrors the 5-phase pipeline structure.
    structured_results = dict(pipeline_results)

    analyze = structured_results.get("analyze")
    if isinstance(analyze, dict) and "verification" in analyze:
        structured_results["audit_analyze"] = analyze.pop("verification")
        structured_results["analyze"] = analyze

    brief = structured_results.get("brief")
    if isinstance(brief, dict) and "verification" in brief:
        structured_results["audit_brief"] = brief.pop("verification")
        structured_results["brief"] = brief

    payload = {
        "source_file": filename,
        "saved_at": datetime.now().isoformat(),
        "pipeline_results": structured_results,
        "file_data": file_data,
    }

    import aiofiles
    async with aiofiles.open(save_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(payload, indent=2, ensure_ascii=False))

    logger.info("Saved pipeline results to %s", save_path)
    return {"saved": True, "filename": save_filename, "path": save_path}


@app.get("/api/saved-results")
async def list_saved_results():
    """List all previously saved result files with lightweight metadata."""
    if not os.path.isdir(SAVED_RESULTS_DIR):
        return {"files": [], "runs": []}
    filenames = sorted(
        (f for f in os.listdir(SAVED_RESULTS_DIR) if f.endswith(".json")),
        reverse=True,
    )

    # Build lightweight metadata for each run (extract verdicts without
    # loading full file_data).
    runs = []
    for fname in filenames:
        meta = {"filename": fname}
        try:
            fpath = os.path.join(SAVED_RESULTS_DIR, fname)
            import aiofiles
            async with aiofiles.open(fpath, "r", encoding="utf-8") as f:
                content = await f.read()
            data = json.loads(content)
            meta["source_file"] = data.get("source_file")
            meta["saved_at"] = data.get("saved_at")
            pr = data.get("pipeline_results", {})
            meta["analyze_verdict"] = (
                pr.get("analyze", {}).get("verification", {}).get("overall_verdict")
                or pr.get("audit_analyze", {}).get("overall_verdict")
            )
            meta["brief_verdict"] = (
                pr.get("brief", {}).get("verification", {}).get("overall_verdict")
                or pr.get("audit_brief", {}).get("overall_verdict")
            )
        except Exception:
            pass
        runs.append(meta)

    return {"files": filenames, "runs": runs}


@app.get("/api/saved-results/{filename}")
async def get_saved_result(filename: str):
    """Return the contents of a saved result file."""
    if not filename.endswith(".json"):
        return JSONResponse({"error": "Only .json files are supported"}, status_code=400)

    filepath = os.path.abspath(os.path.join(SAVED_RESULTS_DIR, filename))
    if not filepath.startswith(SAVED_RESULTS_DIR):
        return JSONResponse({"error": "Invalid file path"}, status_code=403)
    if not os.path.isfile(filepath):
        return JSONResponse({"error": "File not found"}, status_code=404)

    import aiofiles
    async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
        content = await f.read()
    return json.loads(content)


@app.get("/api/active-runs")
async def list_active_runs():
    """Return currently running (or recently completed) pipelines from the cache."""
    runs = []
    now = time.time()
    for filename, state in _pipeline_cache.items():
        if _is_cache_expired(state):
            continue
        phase_name = PHASES[state.current_phase]["name"] if state.current_phase < len(PHASES) else "done"
        phase_label = PHASES[state.current_phase]["label"] if state.current_phase < len(PHASES) else "Complete"
        runs.append({
            "filename": state.filename,
            "run_id": state.run_id,
            "status": state.status,
            "current_phase": phase_name,
            "current_phase_label": phase_label,
            "phase_index": state.current_phase,
            "total_phases": len(PHASES),
            "elapsed_seconds": int(now - state.created_at),
            "error": state.error,
        })
    return {"runs": runs}


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
        # Auto-commit the saved devnotes data
        await _auto_commit_devnotes(f"Auto-sync devnotes for project {project_id}")
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

# Auto-commit identity — shows as a separate contributor in git stats
AUTOSYNC_AUTHOR = "BetStamp AutoSync"
AUTOSYNC_EMAIL = "autosync@betstamp.app"


async def _auto_commit_devnotes(message: str = "Auto-sync devNotesData"):
    """Stage and commit only devNotesData/ files using the AutoSync identity.

    This ensures auto-generated data commits appear as a distinct contributor
    separate from both Claude and the user in git contribution stats.
    """
    try:
        # Stage only devNotesData/ files
        add_proc = await asyncio.create_subprocess_exec(
            "git", "add", "devNotesData/",
            cwd=REPO_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await add_proc.communicate()
        if add_proc.returncode != 0:
            return

        # Check if there are staged changes to commit
        diff_proc = await asyncio.create_subprocess_exec(
            "git", "diff", "--cached", "--quiet", "devNotesData/",
            cwd=REPO_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await diff_proc.communicate()
        if diff_proc.returncode == 0:
            # returncode 0 means no diff — nothing to commit
            return

        # Commit with the AutoSync identity
        commit_proc = await asyncio.create_subprocess_exec(
            "git", "commit",
            "--author", f"{AUTOSYNC_AUTHOR} <{AUTOSYNC_EMAIL}>",
            "-m", message,
            "--", "devNotesData/",
            cwd=REPO_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await commit_proc.communicate()
        if commit_proc.returncode == 0:
            logger.info("AutoSync committed devNotesData: %s", stdout.decode().strip())
        else:
            logger.warning("AutoSync commit failed: %s", stderr.decode().strip())
    except Exception as e:
        logger.warning("AutoSync commit error: %s", e)


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

        # Classify commit: autosync, claude, or user
        is_autosync = (author_name == AUTOSYNC_AUTHOR or author_email == AUTOSYNC_EMAIL)
        is_claude = (not is_autosync) and any(p.search(full_message) for p in claude_patterns)

        commits.append({
            "hash": commit_hash[:8],
            "author": author_name,
            "email": author_email,
            "date": date,
            "subject": subject,
            "is_claude": is_claude,
            "is_autosync": is_autosync,
        })

    total = len(commits)
    claude_count = sum(1 for c in commits if c["is_claude"])
    autosync_count = sum(1 for c in commits if c["is_autosync"])
    user_count = total - claude_count - autosync_count

    summary = {
        "total_commits": total,
        "claude_commits": claude_count,
        "autosync_commits": autosync_count,
        "user_commits": user_count,
        "claude_percentage": round((claude_count / total) * 100, 1) if total > 0 else 0,
        "autosync_percentage": round((autosync_count / total) * 100, 1) if total > 0 else 0,
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
        # Auto-commit the saved git stats data
        await _auto_commit_devnotes("Auto-sync git stats")
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
        context_str = json.dumps(session["pipeline_context"], separators=(",", ":"))
        # Truncate to 60KB (compact JSON keeps payload small)
        if len(context_str) > 61440:
            context_str = context_str[:61440] + "\n... (truncated)"
        system_prompt += (
            "\n\n=== PIPELINE CONTEXT ===\n"
            + context_str
        )

    # Append user message
    session["messages"].append({"role": "user", "content": message})

    # Per-conversation run log
    run_id = uuid.uuid4().hex[:8]
    run_logger, run_handler = create_run_logger(run_id)
    run_logger.info("CHAT run_id=%s conversation=%s message_count=%d", run_id, conversation_id, len(session["messages"]))
    run_logger.info("USER: %s", message[:500])

    try:
        result = await call_ai_chat(
            messages=session["messages"],
            system_prompt=system_prompt,
            run_logger=run_logger,
        )

        # Append assistant response
        session["messages"].append({"role": "assistant", "content": result["text"]})
        run_logger.info("ASSISTANT: %s", result["text"][:500])
        run_logger.info("CHAT complete: provider=%s model=%s elapsed=%.2fs",
                        result["provider_name"], result["model"], result["elapsed_seconds"])

        # --- Verification: run 3 agents in parallel to double-check the response ---
        verification = None
        try:
            source_data = ""
            if session.get("pipeline_context"):
                source_data = json.dumps(session["pipeline_context"], indent=2)[:10000]

            verification = await run_verification(
                text_to_verify=result["text"],
                source_data=source_data,
                run_logger=run_logger,
            )
            run_logger.info(
                "CHAT verification complete: overall=%s elapsed=%.2fs",
                verification["overall_verdict"],
                verification["elapsed_seconds"],
            )
        except Exception as vex:
            run_logger.error("CHAT verification FAILED (non-blocking): %s", vex)
            logger.error("Chat verification FAILED: %s", vex)

        # Set session cookie so subsequent requests are tied to this user
        response = JSONResponse({
            "conversation_id": conversation_id,
            "run_id": run_id,
            "response": {
                "text": result["text"],
                "ai_meta": {
                    "provider": result["provider_name"],
                    "model": result["model"],
                    "usage": result["usage"],
                    "elapsed_seconds": result["elapsed_seconds"],
                },
                "verification": verification,
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
        run_logger.error("CHAT error: %s", e)
        # Remove the user message we just added since the call failed
        session["messages"].pop()
        return JSONResponse({"error": str(e), "run_id": run_id}, status_code=500)
    finally:
        close_run_logger(run_logger, run_handler)


# ---------------------------------------------------------------------------
# Streaming Chat Endpoint (SSE)
# ---------------------------------------------------------------------------

@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    """Send a message to the AI chat and stream the response via SSE.

    SSE events emitted:
      - event: chunk      data: {"text": "..."}
      - event: metadata   data: {"conversation_id", "run_id", "ai_meta"}
      - event: verification data: {"verification": {...}}
      - event: done       data: {}
      - event: error      data: {"error": "..."}
    """
    data = await request.json()
    message = (data.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)

    session_id = _get_session_id(request)
    conversation_id = data.get("conversation_id") or uuid.uuid4().hex[:12]
    key = _make_key(session_id, conversation_id)
    pipeline_context = data.get("pipeline_context")

    # Get or create session
    if key in _chat_sessions:
        session = _chat_sessions[key]
    else:
        session = {
            "messages": [],
            "pipeline_context": None,
            "created": datetime.now().isoformat(),
        }
        _chat_sessions[key] = session

    if pipeline_context:
        session["pipeline_context"] = pipeline_context

    # Build system prompt
    system_prompt = CHAT_SYSTEM_PROMPT
    if session.get("pipeline_context"):
        context_str = json.dumps(session["pipeline_context"], separators=(",", ":"))
        if len(context_str) > 61440:
            context_str = context_str[:61440] + "\n... (truncated)"
        system_prompt += "\n\n=== PIPELINE CONTEXT ===\n" + context_str

    session["messages"].append({"role": "user", "content": message})

    run_id = uuid.uuid4().hex[:8]
    run_logger, run_handler = create_run_logger(run_id)
    run_logger.info("CHAT STREAM run_id=%s conversation=%s message_count=%d", run_id, conversation_id, len(session["messages"]))
    run_logger.info("USER: %s", message[:500])

    async def _sse_generator():
        full_text = ""

        def _sse_event(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        try:
            async def on_chunk(text_delta):
                nonlocal full_text
                full_text += text_delta

            # We need to yield chunks as they arrive.  Use an asyncio.Queue
            # so the on_chunk callback can push text deltas that the generator
            # yields immediately.
            chunk_queue = asyncio.Queue()
            SENTINEL = object()

            async def _on_chunk_queued(text_delta):
                nonlocal full_text
                full_text += text_delta
                await chunk_queue.put(text_delta)

            async def _run_ai():
                try:
                    result = await call_ai_chat_stream(
                        messages=session["messages"],
                        system_prompt=system_prompt,
                        on_chunk=_on_chunk_queued,
                        run_logger=run_logger,
                    )
                    await chunk_queue.put(SENTINEL)
                    return result
                except Exception as exc:
                    await chunk_queue.put(exc)
                    return None

            ai_task = asyncio.create_task(_run_ai())

            # Yield chunks as they arrive
            while True:
                item = await chunk_queue.get()
                if item is SENTINEL:
                    break
                if isinstance(item, Exception):
                    yield _sse_event("error", {"error": str(item)})
                    session["messages"].pop()  # remove failed user message
                    return
                yield _sse_event("chunk", {"text": item})

            result = await ai_task
            if result is None:
                yield _sse_event("error", {"error": "AI call failed"})
                session["messages"].pop()
                return

            # Append assistant response to session
            session["messages"].append({"role": "assistant", "content": full_text})
            run_logger.info("ASSISTANT (streamed): %s", full_text[:500])
            run_logger.info("CHAT STREAM complete: provider=%s model=%s elapsed=%.2fs",
                            result.get("provider_name"), result.get("model"), result.get("elapsed_seconds", 0))

            # Send metadata
            yield _sse_event("metadata", {
                "conversation_id": conversation_id,
                "run_id": run_id,
                "ai_meta": {
                    "provider": result.get("provider_name"),
                    "model": result.get("model"),
                    "usage": result.get("usage", {}),
                    "elapsed_seconds": result.get("elapsed_seconds", 0),
                },
                "message_count": len(session["messages"]),
            })

            # Run verification
            try:
                source_data = ""
                if session.get("pipeline_context"):
                    source_data = json.dumps(session["pipeline_context"], indent=2)[:10000]

                verification = await run_verification(
                    text_to_verify=full_text,
                    source_data=source_data,
                    run_logger=run_logger,
                )
                run_logger.info("CHAT STREAM verification: overall=%s elapsed=%.2fs",
                                verification["overall_verdict"], verification["elapsed_seconds"])
                yield _sse_event("verification", {"verification": verification})
            except Exception as vex:
                run_logger.error("CHAT STREAM verification FAILED (non-blocking): %s", vex)
                logger.error("Chat stream verification FAILED: %s", vex)

            yield _sse_event("done", {})

        except Exception as e:
            run_logger.error("CHAT STREAM error: %s", e)
            # Remove user message on failure
            if session["messages"] and session["messages"][-1]["role"] == "user":
                session["messages"].pop()
            yield _sse_event("error", {"error": str(e)})
        finally:
            close_run_logger(run_logger, run_handler)

    response = StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
    response.set_cookie(
        key="betstamp_session",
        value=session_id,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24,
    )
    return response


# ---------------------------------------------------------------------------
# Debug Log Endpoint
# ---------------------------------------------------------------------------

LOG_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))


@app.get("/api/logs/{run_id}")
async def get_run_log(run_id: str):
    """Serve the log file for a specific pipeline/chat run as plain text."""
    from fastapi.responses import PlainTextResponse

    # Sanitize: run_id should be alphanumeric only (hex chars)
    if not run_id.isalnum() or len(run_id) > 32:
        return JSONResponse({"error": "Invalid run_id"}, status_code=400)

    log_path = os.path.join(LOG_DIR_PATH, "runs", f"{run_id}.log")
    if not os.path.isfile(log_path):
        return JSONResponse({"error": "Log not found"}, status_code=404)

    import aiofiles
    async with aiofiles.open(log_path, "r", encoding="utf-8") as f:
        content = await f.read()
    return PlainTextResponse(content)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8191))
    logger.info("Serving JSON files from: %s", DATA_DIR)
    logger.info("Starting server on port %d", port)
    uvicorn.run(socket_app, host="0.0.0.0", port=port, log_level="info")
