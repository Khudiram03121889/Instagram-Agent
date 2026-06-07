from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse


ALLOWED_DURATIONS = (4, 6, 8, 10)
REVIEW_STATE_NAME = "review_state.json"
RUN_LOG_NAME = "dashboard_run.log"
ACTIVE_TOPIC_NAME = "dashboard_active_topic.txt"

# Google Flow Authentication Session Cache
FLOW_AUTH_CACHE = {
    "ok": False,
    "status": "unchecked",
    "message": "Session has not been verified yet.",
    "checked_at": None,
    "checking": False
}


def safe_project_name(value: str) -> str:
    safe = "".join(
        c for c in (value or "untitled_project") if c.isalnum() or c in (" ", "-", "_")
    ).strip()
    return safe[:120] or "untitled_project"


def output_dir_for_topic(topic: str) -> str:
    return os.path.join("outputs", safe_project_name(topic))


def active_topic_path() -> str:
    return os.path.join("outputs", ACTIVE_TOPIC_NAME)


def review_state_path(topic: str) -> str:
    return os.path.join(output_dir_for_topic(topic), REVIEW_STATE_NAME)


def run_log_path(topic: str) -> str:
    return os.path.join(output_dir_for_topic(topic), RUN_LOG_NAME)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip_duration(clip: Dict[str, Any]) -> int:
    try:
        duration = int(clip.get("duration_seconds", 8))
    except Exception:
        duration = 8
    return duration if duration in ALLOWED_DURATIONS else 8


def score_viral_clarity(
    blueprint: Dict[str, Any],
    script_clips: List[Dict[str, Any]],
) -> Dict[str, Any]:
    score = 100
    notes: List[str] = []
    clips = blueprint.get("clips", []) if isinstance(blueprint, dict) else []

    if not script_clips:
        return {"score": 0, "grade": "missing", "notes": ["No script clips found."]}

    first_text = str(script_clips[0].get("voice_text", "")).lower()
    if not any(token in first_text for token in ("why", "watch", "feel", "your", "happens", "starts")):
        score -= 15
        notes.append("Hook may not create a concrete first-two-second curiosity gap.")

    final_text = str(script_clips[-1].get("voice_text", "")).lower()
    if not any(token in final_text for token in ("you", "your", "next", "notice", "remember", "use", "stop")):
        score -= 12
        notes.append("Final line may not give a personal save/share reason.")

    total_duration = sum(_clip_duration(clip) for clip in clips)
    if total_duration < 24 or total_duration > 36:
        score -= 10
        notes.append("Total duration should stay close to 24-36 seconds.")

    for idx, clip in enumerate(clips, start=1):
        visual_premise = str(clip.get("visual_premise", "")).strip()
        camera_plan = str(clip.get("camera_plan", "")).strip()
        retention_reason = str(clip.get("retention_reason", "")).strip()
        if not visual_premise:
            score -= 8
            notes.append(f"Clip {idx} needs a stronger visual premise.")
        if not camera_plan:
            score -= 5
            notes.append(f"Clip {idx} needs a camera plan.")
        if not retention_reason:
            score -= 5
            notes.append(f"Clip {idx} needs a retention reason.")

    score = max(0, min(100, score))
    grade = "strong" if score >= 85 else "needs polish" if score >= 70 else "weak"
    return {"score": score, "grade": grade, "notes": notes or ["Ready for review."]}


def build_review_questions(score: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "id": "creative_direction",
            "question": "What should the agents improve before Flow?",
            "options": [
                "Approve as-is",
                "Make it more surprising",
                "Make it more relatable",
                "Make the visuals more cinematic",
            ],
        },
        {
            "id": "duration_bias",
            "question": "What pacing should the clip durations prefer?",
            "options": [
                "Use agent recommendations",
                "Shorter and punchier",
                "More room for mechanism",
            ],
        },
        {
            "id": "risk",
            "question": f"Viral clarity score is {score.get('score', 0)}/100. Continue?",
            "options": ["Approve", "Revise script", "Revise visuals"],
        },
    ]


def create_review_state(
    topic: str,
    blueprint: Dict[str, Any],
    script_text: str,
    script_clips: List[Dict[str, Any]],
    topic_profile: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    score = score_viral_clarity(blueprint, script_clips)
    
    # Preserve existing chat history if any
    existing_chat = []
    try:
        path = review_state_path(topic)
        if os.path.exists(path):
            prev = load_review_state(topic)
            existing_chat = prev.get("chat_history", [])
    except Exception:
        pass
        
    new_message = {
        "role": "assistant",
        "content": "I have successfully generated a new story blueprint and script for you! Please review the clip details below.",
        "timestamp": utc_now(),
        "type": "script_review",
        "data": {
            "blueprint": blueprint,
            "script_text": script_text,
            "script_clips": script_clips,
            "viral_clarity": score
        }
    }
    
    # Avoid duplicate final reviews if they were just generated
    should_append = True
    if existing_chat:
        last_msg = existing_chat[-1]
        if last_msg.get("type") == "script_review" and last_msg.get("data", {}).get("script_text") == script_text:
            should_append = False
            
    chat_history = existing_chat + ([new_message] if should_append else [])

    return {
        "topic": topic,
        "status": "pending_review",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "topic_profile": topic_profile or {},
        "blueprint": blueprint,
        "script_text": script_text,
        "script_clips": script_clips,
        "viral_clarity": score,
        "questions": build_review_questions(score),
        "answers": {},
        "manual_feedback": "",
        "approved": False,
        "chat_history": chat_history
    }


def save_review_state(topic: str, state: Dict[str, Any]) -> str:
    output_dir = output_dir_for_topic(topic)
    os.makedirs(output_dir, exist_ok=True)
    state["updated_at"] = utc_now()
    path = review_state_path(topic)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)
    return path


def load_review_state(topic: str) -> Dict[str, Any]:
    path = review_state_path(topic)
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def is_pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False
    except Exception:
        return False


def read_log_tail(topic: str, limit: int = 8000) -> str:
    path = run_log_path(topic)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as file:
        text = file.read()
    return text[-limit:]


def check_flow_login(timeout_seconds: int = 140) -> Dict[str, Any]:
    command = [sys.executable, "main.py", "--check-flow-login"]
    creationflags = 0x00000010 if sys.platform == "win32" else 0
    try:
        result = subprocess.run(
            command,
            cwd=os.getcwd(),
            text=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "status": "timeout",
            "message": "Google Flow session check timed out.",
            "output": str(exc),
            "checked_at": utc_now(),
        }
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    ok = result.returncode == 0 and "FLOW_LOGIN_OK" in output
    return {
        "ok": ok,
        "status": "ok" if ok else "login_required",
        "message": (
            "Google Flow session verified successfully!"
            if ok
            else "Google Flow session check failed. Import valid cookies using manual_cookies.py."
        ),
        "output": output[-4000:],
        "checked_at": utc_now(),
    }


def check_flow_login_async() -> None:
    global FLOW_AUTH_CACHE
    if FLOW_AUTH_CACHE.get("checking"):
        return
    FLOW_AUTH_CACHE["checking"] = True
    FLOW_AUTH_CACHE["status"] = "checking"
    FLOW_AUTH_CACHE["message"] = "Verifying session in background..."
    
    def run():
        try:
            res = check_flow_login()
            FLOW_AUTH_CACHE.update(res)
            FLOW_AUTH_CACHE["checking"] = False
        except Exception as e:
            FLOW_AUTH_CACHE.update({
                "ok": False,
                "status": "error",
                "message": f"Verification failed: {e}",
                "checking": False,
                "checked_at": utc_now()
            })
            
    threading.Thread(target=run, daemon=True).start()


def read_first_topic() -> str:
    if not os.path.exists("topics.txt"):
        return ""
    with open("topics.txt", "r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                if stripped.startswith("[") and "]" in stripped:
                    return stripped.split("]", 1)[1].strip()
                return stripped
    return ""


def get_all_topics() -> List[str]:
    topics = []
    if not os.path.exists("topics.txt"):
        return topics
    with open("topics.txt", "r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                topics.append(stripped)
    return topics


def set_active_topic(topic: str) -> None:
    os.makedirs("outputs", exist_ok=True)
    with open(active_topic_path(), "w", encoding="utf-8") as file:
        file.write(topic.strip())


def read_active_topic() -> str:
    path = active_topic_path()
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as file:
        return file.read().strip()


def latest_review_state() -> Dict[str, Any]:
    topic = read_active_topic() or read_first_topic()
    state = None
    if topic and os.path.exists(review_state_path(topic)):
        try:
            state = load_review_state(topic)
        except Exception:
            pass

    if state is None:
        state = {
            "topic": topic,
            "status": "idle",
            "message": "Click 'Generate Blueprint' in the sidebar or send a message below to start.",
            "chat_history": []
        }

    if not state.get("chat_history"):
        state["chat_history"] = [
            {
                "role": "assistant",
                "content": f"Hi! I'm ready to help you generate an Instagram science reel. The active topic is automatically loaded from topics.txt:\n\n💬 **'{topic}'**\n\nTo generate the visual blueprint and the script, click the **Generate Blueprint** button in the sidebar or just click the button below!",
                "timestamp": utc_now(),
                "type": "initial"
            }
        ]

    # Verify background process state
    if state.get("status") in {"review_generation_started", "approved_running_flow"}:
        process_alive = is_pid_running(state.get("process_id"))
        state["process_alive"] = process_alive
        if state.get("process_id") and not process_alive:
            state["status"] = (
                "review_generation_stopped"
                if state.get("status") == "review_generation_started"
                else "flow_run_stopped"
            )
            state["message"] = "The background generation process has finished or stopped."

    # Start async check if unchecked
    if FLOW_AUTH_CACHE.get("status") == "unchecked":
        check_flow_login_async()

    state["log_tail"] = read_log_tail(topic)
    state["topics"] = get_all_topics()
    state["flow_auth_cached"] = FLOW_AUTH_CACHE
    return state


def run_pipeline(
    args: List[str],
    topic: str,
    extra_env: Dict[str, str] | None = None,
) -> subprocess.Popen:
    command = [sys.executable, "main.py", *args]
    env = os.environ.copy()
    env["CREWAI_TELEMETRY_OPT_OUT"] = "true"
    if extra_env:
        env.update(extra_env)
    os.makedirs(output_dir_for_topic(topic), exist_ok=True)
    log_file = open(run_log_path(topic), "a", encoding="utf-8")
    log_file.write(f"\n\n[{utc_now()}] Starting: {' '.join(command)}\n")
    log_file.flush()
    creationflags = 0x00000010 if sys.platform == "win32" else 0
    return subprocess.Popen(
        command,
        cwd=os.getcwd(),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
    )


def is_proceed_command(message: str) -> bool:
    msg = message.strip().lower()
    msg = re.sub(r'[^\w\s]', '', msg)
    proceed_keywords = {
        "proceed", "move on", "looks good", "go ahead", "approved", "approve",
        "generate video", "next", "run flow", "yes", "ok", "yep", "start",
        "run", "complete", "sure", "fine", "do it", "continue", "go"
    }
    words = msg.split()
    for w in words:
        if w in proceed_keywords:
            return True
    phrases = ["looks good", "go ahead", "let's go", "lets go", "move on", "run flow", "generate video", "do it"]
    for phrase in phrases:
        if phrase in msg:
            return True
    return False


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Reel AI Generator | Command Hub</title>
  
  <!-- Premium Modern Typography -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  
  <style>
    /* Hand-Crafted Obsidian & Glass Design System */
    :root {
      --bg-obsidian: #08090C;
      --bg-slate: #10121A;
      --bg-charcoal: #171A24;
      --bg-charcoal-light: #212533;
      --border-slate: #252B3A;
      --border-glow: #3E465B;
      --text-primary: #F3F4F6;
      --text-muted: #9CA3AF;
      --accent-gradient: linear-gradient(135deg, #6366F1, #8B5CF6);
      --accent-hover: linear-gradient(135deg, #4F46E5, #7C3AED);
      --color-emerald: #10B981;
      --color-rose: #EF4444;
      --color-amber: #F59E0B;
      --color-indigo: #6366F1;
      --sidebar-width: 320px;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg-obsidian);
      color: var(--text-primary);
      font-family: 'Inter', sans-serif;
      overflow: hidden;
      height: 100vh;
      display: flex;
    }

    button, input, select, textarea {
      font-family: inherit;
      color: inherit;
    }

    /* Elegant Sidebar */
    aside {
      width: var(--sidebar-width);
      background-color: var(--bg-slate);
      border-right: 1px solid var(--border-slate);
      display: flex;
      flex-direction: column;
      height: 100%;
      z-index: 10;
      flex-shrink: 0;
    }

    .brand {
      padding: 24px;
      border-bottom: 1px solid var(--border-slate);
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-logo {
      width: 34px;
      height: 34px;
      background: var(--accent-gradient);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
    }

    .brand strong {
      font-family: 'Outfit', sans-serif;
      font-size: 20px;
      font-weight: 700;
      background: linear-gradient(to right, #F3F4F6, #D1D5DB);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .sidebar-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    .sidebar-scroll::-webkit-scrollbar {
      width: 4px;
    }
    .sidebar-scroll::-webkit-scrollbar-thumb {
      background: var(--border-slate);
      border-radius: 99px;
    }

    .section-title {
      font-family: 'Outfit', sans-serif;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--text-muted);
      margin-bottom: 8px;
    }

    /* Form & Input Elements */
    .control-group {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .select-wrapper {
      position: relative;
    }

    select {
      width: 100%;
      padding: 12px;
      background: var(--bg-charcoal);
      border: 1px solid var(--border-slate);
      border-radius: 8px;
      appearance: none;
      cursor: pointer;
      transition: all 0.2s;
    }

    select:focus {
      outline: none;
      border-color: var(--color-indigo);
      box-shadow: 0 0 8px rgba(99, 102, 241, 0.2);
    }

    .select-wrapper::after {
      content: '▼';
      font-size: 10px;
      color: var(--text-muted);
      position: absolute;
      right: 14px;
      top: 50%;
      transform: translateY(-50%);
      pointer-events: none;
    }

    /* Core Buttons */
    .btn {
      width: 100%;
      padding: 12px;
      border: 0;
      border-radius: 8px;
      font-weight: 600;
      font-family: 'Outfit', sans-serif;
      cursor: pointer;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }

    .btn-primary {
      background: var(--accent-gradient);
      box-shadow: 0 4px 12px rgba(139, 92, 246, 0.25);
    }

    .btn-primary:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 18px rgba(139, 92, 246, 0.35);
    }

    .btn-secondary {
      background: var(--bg-charcoal-light);
      border: 1px solid var(--border-slate);
    }

    .btn-secondary:hover {
      background: var(--bg-charcoal);
      border-color: var(--border-glow);
    }

    /* Login Status Widget */
    .auth-widget {
      background: var(--bg-charcoal);
      border: 1px solid var(--border-slate);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .status-badge {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 14px;
    }

    .status-badge.ok {
      background: rgba(16, 185, 129, 0.1);
      color: var(--color-emerald);
      border: 1px solid rgba(16, 185, 129, 0.2);
    }

    .status-badge.failed {
      background: rgba(239, 68, 68, 0.1);
      color: var(--color-rose);
      border: 1px solid rgba(239, 68, 68, 0.2);
    }

    .status-badge.checking {
      background: rgba(245, 158, 11, 0.1);
      color: var(--color-amber);
      border: 1px solid rgba(245, 158, 11, 0.2);
    }

    .dot-pulse {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: currentColor;
    }
    .status-badge.checking .dot-pulse {
      animation: pulse-dot 1.2s infinite;
    }
    @keyframes pulse-dot {
      0% { opacity: 0.3; }
      50% { opacity: 1; }
      100% { opacity: 0.3; }
    }

    /* Progress Stepper */
    .stepper {
      display: flex;
      flex-direction: column;
      gap: 20px;
      padding-left: 8px;
    }

    .step {
      display: flex;
      gap: 16px;
      position: relative;
    }

    .step::before {
      content: '';
      position: absolute;
      left: 10px;
      top: 24px;
      bottom: -20px;
      width: 2px;
      background: var(--border-slate);
    }

    .step:last-child::before {
      display: none;
    }

    .step-node {
      width: 22px;
      height: 22px;
      border-radius: 50%;
      background: var(--bg-charcoal-light);
      border: 2px solid var(--border-slate);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 10px;
      font-weight: 700;
      z-index: 2;
      transition: all 0.3s;
      flex-shrink: 0;
    }

    .step.active .step-node {
      background: var(--color-indigo);
      border-color: var(--color-indigo);
      box-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
    }

    .step.completed .step-node {
      background: var(--color-emerald);
      border-color: var(--color-emerald);
      box-shadow: 0 0 10px rgba(16, 185, 129, 0.4);
    }

    .step-content {
      display: flex;
      flex-direction: column;
    }

    .step-title {
      font-size: 13px;
      font-weight: 600;
    }

    .step-desc {
      font-size: 11px;
      color: var(--text-muted);
    }

    /* Main Section */
    main {
      flex: 1;
      display: flex;
      flex-direction: column;
      height: 100%;
      background-color: var(--bg-obsidian);
      position: relative;
    }

    /* Premium Top Bar */
    header {
      height: 70px;
      border-bottom: 1px solid var(--border-slate);
      padding: 0 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--bg-slate);
    }

    .header-info h1 {
      font-family: 'Outfit', sans-serif;
      font-size: 18px;
      font-weight: 700;
    }

    .header-info p {
      font-size: 12px;
      color: var(--text-muted);
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    /* Chat Hub */
    .chat-container {
      flex: 1;
      overflow-y: auto;
      padding: 30px 24px 100px 24px; /* Room for floating input */
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    .chat-container::-webkit-scrollbar {
      width: 6px;
    }
    .chat-container::-webkit-scrollbar-thumb {
      background: var(--border-slate);
      border-radius: 99px;
    }

    .message-bubble {
      max-width: 82%;
      display: flex;
      flex-direction: column;
      gap: 8px;
      animation: fade-in-slide 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    @keyframes fade-in-slide {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .message-bubble.assistant {
      align-self: flex-start;
    }

    .message-bubble.user {
      align-self: flex-end;
      max-width: 65%;
    }

    .message-meta {
      font-size: 11px;
      color: var(--text-muted);
      padding: 0 4px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .message-meta strong {
      color: var(--text-primary);
    }

    .message-content {
      padding: 16px 20px;
      border-radius: 16px;
      line-height: 1.6;
      font-size: 14.5px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }

    .message-bubble.assistant .message-content {
      background: var(--bg-charcoal);
      border: 1px solid var(--border-slate);
      border-bottom-left-radius: 4px;
    }

    .message-bubble.user .message-content {
      background: var(--accent-gradient);
      border-bottom-right-radius: 4px;
      color: white;
    }

    /* Structured Script Review Node inside Chat */
    .script-review-node {
      margin-top: 12px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .metric-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }

    .metric-card {
      background: var(--bg-slate);
      border: 1px solid var(--border-slate);
      border-radius: 10px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .metric-card .label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }

    .metric-card .value {
      font-family: 'Outfit', sans-serif;
      font-size: 26px;
      font-weight: 800;
    }

    .clip-timeline {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .clip-card {
      background: var(--bg-slate);
      border: 1px solid var(--border-slate);
      border-radius: 12px;
      padding: 16px;
      transition: border-color 0.2s;
    }

    .clip-card:hover {
      border-color: var(--border-glow);
    }

    .clip-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      padding-bottom: 8px;
    }

    .clip-title {
      font-family: 'Outfit', sans-serif;
      font-weight: 700;
      color: var(--color-indigo);
    }

    .clip-badge {
      font-size: 11px;
      padding: 3px 8px;
      background: rgba(99, 102, 241, 0.15);
      color: var(--color-indigo);
      border-radius: 99px;
      font-weight: 600;
    }

    .clip-card p {
      margin-bottom: 8px;
      font-size: 13.5px;
    }

    .clip-meta-section {
      display: grid;
      grid-template-columns: 1fr;
      gap: 6px;
      background: rgba(255, 255, 255, 0.02);
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 12px;
    }

    /* Floating Chat Prompt */
    .prompt-container {
      position: absolute;
      bottom: 24px;
      left: 24px;
      right: 24px;
      z-index: 100;
    }

    .prompt-glass {
      background: rgba(16, 18, 26, 0.85);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-slate);
      border-radius: 14px;
      padding: 8px 12px;
      display: flex;
      align-items: center;
      gap: 12px;
      box-shadow: 0 -8px 32px rgba(0, 0, 0, 0.4);
    }

    .prompt-glass input {
      flex: 1;
      background: transparent;
      border: 0;
      padding: 12px 6px;
      font-size: 14.5px;
      outline: none;
    }

    .prompt-glass input::placeholder {
      color: var(--text-muted);
    }

    .send-btn {
      width: 40px;
      height: 40px;
      border-radius: 10px;
      background: var(--accent-gradient);
      border: 0;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: transform 0.2s;
    }

    .send-btn:hover {
      transform: scale(1.05);
    }

    /* Elegant Expanding Log Console */
    .console-drawer {
      position: fixed;
      bottom: 0;
      right: 0;
      left: var(--sidebar-width);
      background: var(--bg-slate);
      border-top: 1px solid var(--border-slate);
      z-index: 150;
      transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .console-drawer.collapsed {
      transform: translateY(calc(100% - 44px));
    }

    .console-header {
      height: 44px;
      padding: 0 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      cursor: pointer;
      background: var(--bg-charcoal);
    }

    .console-header strong {
      font-family: 'Outfit', sans-serif;
      font-size: 13px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .console-terminal {
      height: 200px;
      overflow-y: auto;
      padding: 16px 24px;
      background: #060709;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      line-height: 1.5;
      color: #34D399;
      white-space: pre-wrap;
    }

    .console-terminal::-webkit-scrollbar {
      width: 6px;
    }
    .console-terminal::-webkit-scrollbar-thumb {
      background: var(--border-slate);
    }

    pre {
      white-space: pre-wrap;
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      background: #08090C;
      border: 1px solid var(--border-slate);
      padding: 12px;
      border-radius: 8px;
      color: #D1D5DB;
    }

    /* Loading Card CSS Pulse */
    .pulse-card {
      border: 1px solid var(--color-indigo);
      background: rgba(99, 102, 241, 0.05);
      border-radius: 12px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .pulse-line {
      height: 8px;
      background: var(--bg-charcoal-light);
      border-radius: 4px;
      overflow: hidden;
      position: relative;
    }
    .pulse-line::after {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.4), transparent);
      animation: loading-shimmer 1.5s infinite;
    }
    @keyframes loading-shimmer {
      0% { transform: translateX(-100%); }
      100% { transform: translateX(100%); }
    }
  </style>
</head>
<body>

  <!-- Left Sidebar Controls -->
  <aside>
    <div class="brand">
      <div class="brand-logo">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M4.5 16.5c-1.5 1.26-2.5 3.19-2.5 5.5h20c0-2.31-1-4.24-2.5-5.5"></path>
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-3.31 0-6-2.69-6-6s2.69-6 6-6 6 2.69 6 6-2.69 6-6 6z"></path>
        </svg>
      </div>
      <strong>Reels AI Hub</strong>
    </div>

    <div class="sidebar-scroll">
      
      <!-- Topic Selection Section -->
      <div class="control-group">
        <div class="section-title">Active Topic</div>
        <div class="select-wrapper">
          <select id="topicSelect" onchange="changeTopic()"></select>
        </div>
        <button class="btn btn-primary" onclick="generateReview()" style="margin-top: 6px;">
          <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"/>
          </svg>
          Generate Blueprint
        </button>
      </div>

      <!-- Flow Auth session section -->
      <div class="control-group">
        <div class="section-title">Google Flow Session</div>
        <div class="auth-widget">
          <div id="authBadge" class="status-badge checking">
            <div class="dot-pulse"></div>
            <span id="authText">Checking session...</span>
          </div>
          <button class="btn btn-secondary" onclick="checkFlowLogin()" style="padding: 8px 12px; font-size: 13px;">
            Verify Session
          </button>
        </div>
      </div>

      <!-- Live Stepper progress tracker -->
      <div class="control-group">
        <div class="section-title">Workflow Progress</div>
        <div class="stepper">
          <div id="step1" class="step">
            <div class="step-node">1</div>
            <div class="step-content">
              <span class="step-title">Blueprint Design</span>
              <span class="step-desc">Planning clips structure</span>
            </div>
          </div>
          <div id="step2" class="step">
            <div class="step-node">2</div>
            <div class="step-content">
              <span class="step-title">Script Write & Revision</span>
              <span class="step-desc">Narration fine-tuning</span>
            </div>
          </div>
          <div id="step3" class="step">
            <div class="step-node">3</div>
            <div class="step-content">
              <span class="step-title">Google Flow Video</span>
              <span class="step-desc">Playwright animation</span>
            </div>
          </div>
          <div id="step4" class="step">
            <div class="step-node">4</div>
            <div class="step-content">
              <span class="step-title">Post & Archiving</span>
              <span class="step-desc">Instagram metadata</span>
            </div>
          </div>
        </div>
      </div>

    </div>
  </aside>

  <!-- Right Main Workspace -->
  <main>
    <header>
      <div class="header-info">
        <h1 id="activeTopicHeader">Loading active topic...</h1>
        <p id="activeStatusHeader">Initializing dashboard interface</p>
      </div>
      <div class="header-actions">
        <button class="btn btn-secondary" onclick="refreshState()" style="padding: 8px 12px; font-size: 13px;">
          <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" style="margin-right: 2px;">
            <path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 1121.21 7.89M9 11l3-3m0 0l3 3m-3-3v12"/>
          </svg>
          Sync Hub
        </button>
      </div>
    </header>

    <!-- ChatGPT/Gemini conversational window -->
    <div id="chatContainer" class="chat-container"></div>

    <!-- Floating Chat prompt box -->
    <div class="prompt-container">
      <div class="prompt-glass">
        <input id="promptInput" type="text" placeholder="Type feedback to revise the script, or type 'Proceed'..." onkeydown="handleInputKey(event)">
        <button class="send-btn" onclick="sendMessage()">
          <svg width="18" height="18" fill="none" stroke="white" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M14 5l7 7m0 0l-7 7m7-7H3"/>
          </svg>
        </button>
      </div>
    </div>

    <!-- System Log Terminal Drawer -->
    <div id="consoleDrawer" class="console-drawer collapsed">
      <div class="console-header" onclick="toggleConsole()">
        <strong>
          <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"/>
          </svg>
          Live Execution Console
        </strong>
        <span id="consoleToggleArrow" style="font-size: 11px;">▲ Expand</span>
      </div>
      <div id="terminal" class="console-terminal">Waiting for active run logs...</div>
    </div>
  </main>

  <script>
    let currentTopic = "";
    let isPolling = false;
    let pollInterval = null;

    async function api(path, body) {
      const res = await fetch(path, {
        method: body ? 'POST' : 'GET',
        headers: { 'content-type': 'application/json' },
        body: body ? JSON.stringify(body) : undefined
      });
      return await res.json();
    }

    function toggleConsole() {
      const drawer = document.getElementById('consoleDrawer');
      const arrow = document.getElementById('consoleToggleArrow');
      drawer.classList.toggle('collapsed');
      if (drawer.classList.contains('collapsed')) {
        arrow.innerText = "▲ Expand";
      } else {
        arrow.innerText = "▼ Collapse";
        scrollToTerminalBottom();
      }
    }

    function scrollToTerminalBottom() {
      const t = document.getElementById('terminal');
      t.scrollTop = t.scrollHeight;
    }

    function scrollToChatBottom() {
      const c = document.getElementById('chatContainer');
      c.scrollTop = c.scrollHeight;
    }

    function handleInputKey(e) {
      if (e.key === 'Enter') {
        sendMessage();
      }
    }

    // Build timeline details for clips inside chat bubble
    function renderScriptClips(blueprint, scriptClips) {
      if (!blueprint || !blueprint.clips) return "";
      const clips = blueprint.clips;
      return `
        <div class="script-review-node">
          <div class="metric-row">
            <div class="metric-card">
              <span class="label">Viral Clarity Score</span>
              <span class="value" style="color: var(--color-indigo);">${scoreViralClarity(blueprint, scriptClips)}/100</span>
            </div>
            <div class="metric-card">
              <span class="label">Clip Count</span>
              <span class="value">${clips.length} Clips</span>
            </div>
            <div class="metric-card">
              <span class="label">Est. Duration</span>
              <span class="value">${clips.reduce((a, c) => a + (c.duration_seconds || 8), 0)}s</span>
            </div>
          </div>
          
          <div class="clip-timeline">
            ${clips.map((clip, i) => {
              const sc = scriptClips && scriptClips[i] ? scriptClips[i] : {};
              return `
                <div class="clip-card">
                  <div class="clip-header">
                    <span class="clip-title">Clip ${clip.clip_number}: ${clip.clip_role}</span>
                    <span class="clip-badge">${clip.duration_seconds || 8}s</span>
                  </div>
                  <p><strong>Narration:</strong> "${sc.voice_text || ''}"</p>
                  <div class="clip-meta-section">
                    <div><strong>Visual:</strong> ${clip.visual_premise || clip.core_idea || ''}</div>
                    <div style="margin-top: 4px;"><strong>Camera Plan:</strong> ${clip.camera_plan || ''}</div>
                    <div style="margin-top: 4px;"><strong>Retention Trigger:</strong> ${clip.retention_reason || ''}</div>
                  </div>
                </div>
              `;
            }).join('')}
          </div>
        </div>
      `;
    }

    function scoreViralClarity(blueprint, scriptClips) {
      if (!scriptClips || scriptClips.length === 0) return 0;
      let score = 100;
      const first = (scriptClips[0].voice_text || '').toLowerCase();
      if (!['why', 'watch', 'feel', 'your', 'happens', 'starts'].some(t => first.includes(t))) score -= 15;
      const final = (scriptClips[scriptClips.length - 1].voice_text || '').toLowerCase();
      if (!['you', 'your', 'next', 'notice', 'remember', 'use', 'stop'].some(t => final.includes(t))) score -= 12;
      return score;
    }

    function formatTime(isoStr) {
      if (!isoStr) return "";
      const d = new Date(isoStr);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function updateStepper(status, logTail) {
      // Clear status
      document.querySelectorAll('.step').forEach(s => s.className = 'step');
      
      const s1 = document.getElementById('step1');
      const s2 = document.getElementById('step2');
      const s3 = document.getElementById('step3');
      const s4 = document.getElementById('step4');

      if (status === 'review_generation_started') {
        s1.classList.add('active');
      } else if (status === 'pending_review' || status === 'revision_requested') {
        s1.classList.add('completed');
        s2.classList.add('active');
      } else if (status === 'approved_running_flow') {
        s1.classList.add('completed');
        s2.classList.add('completed');
        s3.classList.add('active');
      } else if (status === 'flow_run_stopped' || status === 'completed') {
        s1.classList.add('completed');
        s2.classList.add('completed');
        s3.classList.add('completed');
        s4.classList.add('completed');
      }
      
      // Fine-grained checks based on logs
      if (logTail) {
        const text = logTail.toLowerCase();
        if (text.includes('script_crew.kickoff') || text.includes('writing narration')) {
          s1.classList.add('completed');
          s2.classList.add('active');
        }
        if (text.includes('browser_operator') || text.includes('google flow')) {
          s1.classList.add('completed');
          s2.classList.add('completed');
          s3.classList.add('active');
        }
        if (text.includes('caption_writer') || text.includes('caption:') || text.includes('archiving')) {
          s1.classList.add('completed');
          s2.classList.add('completed');
          s3.classList.add('completed');
          s4.classList.add('active');
        }
      }
    }

    function renderChat(chatHistory) {
      const container = document.getElementById('chatContainer');
      
      // Calculate scroll diff
      const isPinned = container.scrollHeight - container.clientHeight <= container.scrollTop + 60;
      
      let html = "";
      chatHistory.forEach(msg => {
        const roleClass = msg.role === 'assistant' ? 'assistant' : 'user';
        const senderName = msg.role === 'assistant' ? 'AI pipeline engine' : 'You';
        
        let customContent = msg.content;
        if (msg.type === 'script_review' && msg.data) {
          customContent += renderScriptClips(msg.data.blueprint, msg.data.script_clips);
        }

        html += `
          <div class="message-bubble ${roleClass}">
            <div class="message-meta">
              <strong>${senderName}</strong> • ${formatTime(msg.timestamp)}
            </div>
            <div class="message-content">
              ${customContent.replace(/\\n/g, '<br>')}
            </div>
          </div>
        `;
      });

      container.innerHTML = html;
      
      if (isPinned) {
        scrollToChatBottom();
      }
    }

    function renderAuthBadge(auth) {
      const badge = document.getElementById('authBadge');
      const text = document.getElementById('authText');
      badge.className = "status-badge";
      
      if (auth.checking) {
        badge.classList.add('checking');
        text.innerText = "Checking session...";
      } else if (auth.ok) {
        badge.classList.add('ok');
        text.innerText = "Logged In";
      } else {
        badge.classList.add('failed');
        text.innerText = "Login Required";
      }
    }

    function populateTopicsList(topics, selected) {
      const select = document.getElementById('topicSelect');
      if (select.options.length === topics.length && currentTopic === selected) return;
      
      select.innerHTML = "";
      topics.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t;
        opt.innerText = t.substring(0, 52) + (t.length > 52 ? '...' : '');
        if (t === selected) opt.selected = true;
        select.appendChild(opt);
      });
    }

    function render(state) {
      currentTopic = state.topic || '';
      document.getElementById('activeTopicHeader').innerText = currentTopic || 'No active topic';
      
      // Subtitle status header
      let statusDesc = "Ready";
      if (state.status === 'review_generation_started') statusDesc = "💡 Designing blueprint & script...";
      if (state.status === 'pending_review') statusDesc = "✍️ Waiting for script approval / revisions";
      if (state.status === 'approved_running_flow') statusDesc = "🚀 Executing Playwright Google Flow video generation";
      if (state.status === 'flow_login_required') statusDesc = "⚠️ Cookies expired or missing - Login Required";
      document.getElementById('activeStatusHeader').innerText = statusDesc;

      populateTopicsList(state.topics || [], state.topic);
      renderAuthBadge(state.flow_auth_cached || {});
      updateStepper(state.status, state.log_tail);
      renderChat(state.chat_history || []);

      // Logs Terminal
      const terminal = document.getElementById('terminal');
      const termPinned = terminal.scrollHeight - terminal.clientHeight <= terminal.scrollTop + 30;
      terminal.innerText = state.log_tail || "Active pipeline logs will stream here during generation...";
      if (termPinned) {
        scrollToTerminalBottom();
      }

      // Manage background polling
      const backgroundRunning = ['review_generation_started', 'approved_running_flow', 'checking_flow_login'].includes(state.status);
      if (backgroundRunning && !isPolling) {
        isPolling = true;
        pollInterval = setInterval(refreshState, 1500);
      } else if (!backgroundRunning && isPolling) {
        isPolling = false;
        clearInterval(pollInterval);
      }
    }

    async function refreshState() {
      try {
        const state = await api('/api/state');
        render(state);
      } catch (err) {
        console.error("Dashboard sync error:", err);
      }
    }

    async function changeTopic() {
      const topic = document.getElementById('topicSelect').value;
      const state = await api('/api/set-active-topic', { topic });
      render(state);
    }

    async function generateReview() {
      const select = document.getElementById('topicSelect');
      const topic = select.options.length ? select.value : "";
      
      // Optimistic prompt log
      const initialChat = [
        { role: 'user', content: 'Generate blueprint and script.', timestamp: new Date().toISOString() },
        { role: 'assistant', content: 'Starting agents workflow...', timestamp: new Date().toISOString() }
      ];
      render({ topic, status: 'review_generation_started', chat_history: initialChat, topics: [topic] });

      const state = await api('/api/generate-review', { topic });
      render(state);
    }

    async function checkFlowLogin() {
      renderAuthBadge({ checking: true });
      const select = document.getElementById('topicSelect');
      const topic = select.options.length ? select.value : "";
      const state = await api('/api/auth-status', { topic });
      renderAuthBadge(state);
      refreshState();
    }

    async function sendMessage() {
      const input = document.getElementById('promptInput');
      const message = input.value.trim();
      if (!message) return;
      
      input.value = "";
      const select = document.getElementById('topicSelect');
      const topic = select.options.length ? select.value : "";
      
      const state = await api('/api/chat-message', { topic, message });
      render(state);
    }

    // Init Page state
    refreshState();
  </script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "InstagramAgentDashboard/2.0"

    def _json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or "0")
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/state":
            self._json(latest_review_state())
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        body = self._body()
        topic = (body.get("topic") or read_active_topic() or read_first_topic()).strip()
        
        if path == "/api/generate-review":
            set_active_topic(topic)
            args = ["--flow-dry-run", "--review-only"]
            if topic:
                args.extend(["--topic", topic])
            
            chat_history = [
                {
                    "role": "user",
                    "content": f"Generate blueprint and script for topic: '{topic}'",
                    "timestamp": utc_now(),
                    "type": "text"
                },
                {
                    "role": "assistant",
                    "content": f"Understood! Spawning the Story Blueprint Designer and Narration Script Writer agents to build the package for:\n\n💬 **'{topic}'**\n\nPlease wait while they research and write the script in the background... ⏳",
                    "timestamp": utc_now(),
                    "type": "text"
                }
            ]
            state = {
                "topic": topic,
                "status": "review_generation_started",
                "message": "Agents are building the blueprint, script, and visual treatment.",
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "approved": False,
                "chat_history": chat_history
            }
            save_review_state(topic, state)
            process = run_pipeline(args, topic=topic)
            state["process_id"] = process.pid
            save_review_state(topic, state)
            state["process_alive"] = True
            state["log_tail"] = read_log_tail(topic)
            self._json(state)
            return

        if path == "/api/chat-message":
            set_active_topic(topic)
            message = str(body.get("message", "")).strip()
            
            if topic and os.path.exists(review_state_path(topic)):
                state = load_review_state(topic)
            else:
                state = {
                    "topic": topic,
                    "status": "idle",
                    "approved": False,
                    "chat_history": []
                }
            
            if "chat_history" not in state:
                state["chat_history"] = []
                
            state["chat_history"].append({
                "role": "user",
                "content": message,
                "timestamp": utc_now(),
                "type": "text"
            })
            
            if is_proceed_command(message):
                state["status"] = "checking_flow_login"
                state["message"] = "Checking Google Flow session..."
                save_review_state(topic, state)
                
                flow_auth = check_flow_login()
                state["flow_auth"] = flow_auth
                
                if not flow_auth.get("ok"):
                    state["status"] = "flow_login_required"
                    state["approved"] = False
                    state["message"] = flow_auth.get("message", "Google Flow login is required.")
                    state["chat_history"].append({
                        "role": "assistant",
                        "content": f"⚠️ **Flow Login Verification Failed:**\n\n{flow_auth.get('message')}\n\nPlease check your Google Flow cookie/auth status in the sidebar and make sure you are logged in. Once resolved, type 'proceed' again to retry.",
                        "timestamp": utc_now(),
                        "type": "text"
                    })
                    save_review_state(topic, state)
                    self._json(state)
                    return
                
                state["status"] = "approved_running_flow"
                state["approved"] = True
                state["chat_history"].append({
                    "role": "assistant",
                    "content": "✅ **Flow Login Verified!** Starting Google Flow automation to generate the final videos now. 🚀\n\nYou can watch the live terminal logs and progression below.",
                    "timestamp": utc_now(),
                    "type": "text"
                })
                save_review_state(topic, state)
                
                args = ["--flow-live", "--approved-review"]
                if topic:
                    args.extend(["--topic", topic])
                process = run_pipeline(args, topic=topic)
                state["process_id"] = process.pid
                state["process_alive"] = True
                state["log_tail"] = read_log_tail(topic)
                save_review_state(topic, state)
                self._json(state)
                return
            else:
                state["manual_feedback"] = message
                state["status"] = "review_generation_started"
                state["approved"] = False
                state["chat_history"].append({
                    "role": "assistant",
                    "content": f"Understood! Appending your revision feedback:\n\n> *\"{message}\"*\n\nSpawning our agent pipeline in the background to revise the blueprint and script based on your input. Please wait... ⏳",
                    "timestamp": utc_now(),
                    "type": "text"
                })
                save_review_state(topic, state)
                
                args = ["--flow-dry-run", "--review-only"]
                if topic:
                    args.extend(["--topic", topic])
                extra_env = {"REVIEW_FEEDBACK": message}
                process = run_pipeline(args, topic=topic, extra_env=extra_env)
                state["process_id"] = process.pid
                state["process_alive"] = True
                state["log_tail"] = read_log_tail(topic)
                save_review_state(topic, state)
                self._json(state)
                return

        if path == "/api/auth-status":
            set_active_topic(topic)
            check_flow_login_async()
            self._json(FLOW_AUTH_CACHE)
            return

        if path == "/api/set-active-topic":
            set_active_topic(topic)
            self._json(latest_review_state())
            return
            
        self._json({"error": "not found"}, 404)

    def log_message(self, format: str, *args: Any) -> None:
        return


def start_dashboard(host: str = "127.0.0.1", port: int = 8765) -> str:
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://{host}:{port}"
    print(f"\n[DASHBOARD] Running premium conversational AI dashboard at {url}")
    print("[DASHBOARD] Press Ctrl+C to stop.")
    try:
        while True:
            threading.Event().wait(3600)
    except KeyboardInterrupt:
        server.shutdown()
    return url


if __name__ == "__main__":
    start_dashboard()
