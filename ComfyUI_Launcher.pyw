"""
ComfyUI Web-Based Graphical Launcher
Opens a beautiful local web interface to control ComfyUI startup
"""

import subprocess
import threading
import os
import sys
import re
import json
import webbrowser
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import time
import socket
import tempfile

# Configuration
BASE_DIR = Path(__file__).parent
COMFYUI_DIR = BASE_DIR / "ComfyUI"
PYTHON_EXE = BASE_DIR / "python_embeded" / "python.exe"
UPDATE_DIR = BASE_DIR / "update"
LAUNCHER_PORT = 8199  # Different from ComfyUI's default 8188
LOG_FILE = BASE_DIR / "comfyui_launcher.log"
UPDATE_LOG_FILE = BASE_DIR / "comfyui_update.log"

# Global state
process = None
is_running = False
log_buffer = []
log_lock = threading.Lock()
last_log_position = 0

# Update state
update_process = None
is_updating = False
update_log_buffer = []
update_log_lock = threading.Lock()
last_update_log_position = 0
settings = {
    "listen": "0.0.0.0",
    "port": 8188,
    "auto_launch_browser": True,
    "extra_args": ""
}
startup_info = {
    "phase": "idle",
    "progress": 0,
    "custom_nodes": 0,
    "status": "stopped",
    "gen_speed": "",
    "gen_progress": "",
    "is_generating": False
}

# Activity tracking for workflow steps
activity_steps = []
activity_lock = threading.Lock()
generation_progress = {"current": 0, "total": 0, "percent": 0}

def clear_activity():
    global activity_steps, generation_progress
    with activity_lock:
        activity_steps = []
        generation_progress = {"current": 0, "total": 0, "percent": 0}

def add_activity(name, status="pending"):
    """Add or update an activity step. Status: pending, active, done"""
    global activity_steps
    with activity_lock:
        # Check if activity already exists
        for step in activity_steps:
            if step["name"] == name:
                step["status"] = status
                return
        # Add new activity
        activity_steps.append({"name": name, "status": status})

def update_generation_progress(current, total, percent):
    """Update the generation progress bar data"""
    global generation_progress
    with activity_lock:
        generation_progress = {"current": current, "total": total, "percent": percent}

def complete_activity(name):
    """Mark an activity as done"""
    add_activity(name, "done")

def get_activities():
    with activity_lock:
        return list(activity_steps)

def get_generation_progress():
    with activity_lock:
        return dict(generation_progress)


def load_settings():
    global settings
    settings_file = BASE_DIR / "launcher_settings.json"
    try:
        if settings_file.exists():
            with open(settings_file, "r") as f:
                settings.update(json.load(f))
    except Exception:
        pass


def save_settings():
    settings_file = BASE_DIR / "launcher_settings.json"
    try:
        with open(settings_file, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass


def add_log(message, level="info"):
    with log_lock:
        log_buffer.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": message,
            "level": level
        })
        # Keep last 500 lines
        if len(log_buffer) > 500:
            log_buffer.pop(0)


def update_progress(line):
    """Update progress based on log content"""
    global startup_info
    line_lower = line.lower()

    # Parse tqdm progress bars for generation speed
    # Format: "  25%|██        | 2/8 [00:07<00:21, 3.55s/it]"
    tqdm_match = re.search(r'(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[[\d:]+<[\d:]+,\s*([\d.]+)(s/it|it/s)\]', line)
    if tqdm_match:
        percent = int(tqdm_match.group(1))
        current = int(tqdm_match.group(2))
        total = int(tqdm_match.group(3))
        speed_val = tqdm_match.group(4)
        speed_unit = tqdm_match.group(5)

        startup_info["is_generating"] = True
        startup_info["gen_progress"] = f"{current}/{total} ({percent}%)"
        startup_info["gen_speed"] = f"{speed_val} {speed_unit}"

        # Update activity for image generation with progress bar
        add_activity("Generating", "active")
        update_generation_progress(current, total, percent)

    # ===== WORKFLOW ACTIVITY TRACKING =====

    # New prompt started
    if "got prompt" in line_lower:
        clear_activity()
        add_activity("Processing prompt", "active")
        startup_info["is_generating"] = True

    # Model loading patterns
    if "requested to load" in line_lower:
        # Extract model name
        match = re.search(r'requested to load (\w+)', line, re.IGNORECASE)
        if match:
            model_name = match.group(1)
            # Map common model names to friendly names
            if "vae" in model_name.lower():
                add_activity("Loading VAE", "active")
            elif "clip" in model_name.lower() or "text" in model_name.lower():
                add_activity("Loading text encoder", "active")
            else:
                add_activity(f"Loading {model_name}", "active")

    # Model loaded
    if "loaded completely" in line_lower or "loaded partially" in line_lower:
        # Mark any active loading as done
        for step in get_activities():
            if step["status"] == "active" and "Loading" in step["name"]:
                complete_activity(step["name"])

    # VAE operations
    if "using pytorch attention in vae" in line_lower:
        add_activity("Preparing VAE", "active")

    # Checkpoint loading
    if "loading checkpoint" in line_lower or "loaded checkpoint" in line_lower:
        if "loading" in line_lower:
            add_activity("Loading checkpoint", "active")
        else:
            complete_activity("Loading checkpoint")

    # Sampling/Generation complete, VAE decode starting
    if "requested to load" in line_lower and "vae" in line_lower:
        # Mark generation as done if it was active
        for step in get_activities():
            if "Generating" in step["name"] and step["status"] == "active":
                complete_activity(step["name"])
        add_activity("Decoding (VAE)", "active")

    # Unloading (usually means VAE decode is happening)
    if "unloaded partially" in line_lower or "unloaded completely" in line_lower:
        for step in get_activities():
            if "Decoding" in step["name"] and step["status"] == "active":
                complete_activity(step["name"])

    # Prompt executed - everything done
    if "prompt executed" in line_lower:
        startup_info["is_generating"] = False
        startup_info["gen_progress"] = "Done"
        # Mark all remaining active steps as done
        for step in get_activities():
            if step["status"] == "active":
                complete_activity(step["name"])
        # Reset generation progress
        update_generation_progress(0, 0, 0)
        # Extract time
        time_match = re.search(r'prompt executed in ([\d.]+) seconds', line_lower)
        if time_match:
            add_activity(f"Completed in {time_match.group(1)}s", "done")

    # ===== STARTUP PROGRESS =====

    if "python version" in line_lower and startup_info["status"] != "running":
        startup_info["phase"] = "Python initialized"
        startup_info["progress"] = 10

    elif "comfyui version" in line_lower and startup_info["status"] != "running":
        startup_info["phase"] = "ComfyUI core loaded"
        startup_info["progress"] = 15

    elif "prestartup times" in line_lower:
        startup_info["phase"] = "Prestartup scripts completed"
        startup_info["progress"] = 25

    elif "total vram" in line_lower and startup_info["progress"] < 35:
        startup_info["phase"] = "GPU detected"
        startup_info["progress"] = 35

    elif "import times for custom nodes" in line_lower:
        startup_info["phase"] = "Loading custom nodes..."
        startup_info["progress"] = 40

    elif re.match(r'\s*\d+\.\d+ seconds', line) and startup_info["status"] != "running":
        startup_info["custom_nodes"] += 1
        startup_info["progress"] = min(85, 40 + startup_info["custom_nodes"] * 1.5)

    elif "starting server" in line_lower:
        startup_info["phase"] = "Starting server..."
        startup_info["progress"] = 95

    # Server is ready when we see the URL
    elif "to see the gui" in line_lower or (re.search(r'http://.*:\d+', line) and startup_info["status"] != "running"):
        startup_info["phase"] = "Server running!"
        startup_info["progress"] = 100
        startup_info["status"] = "running"


def tail_log_file():
    """Read new lines from the log file"""
    global last_log_position

    if not LOG_FILE.exists():
        return

    try:
        with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
            f.seek(last_log_position)
            new_content = f.read()
            last_log_position = f.tell()

            if new_content:
                for line in new_content.splitlines():
                    if line.strip():
                        # Determine log level - be more specific to avoid false positives
                        level = "info"
                        line_lower = line.lower()

                        # Check for actual errors (not warnings that contain "error" in path names)
                        is_real_error = False
                        if "IMPORT FAILED" in line:
                            is_real_error = True
                        elif line.strip().startswith("Error "):
                            # Skip OpenXR errors - they're harmless VR-related messages
                            if "OpenXR" not in line and "xr" not in line_lower:
                                is_real_error = True
                        elif "exception" in line_lower and "FutureWarning" not in line and "UserWarning" not in line:
                            is_real_error = True
                        elif "traceback" in line_lower:
                            is_real_error = True
                        elif "SyntaxError:" in line or "ImportError:" in line or "ModuleNotFoundError:" in line:
                            is_real_error = True

                        # Check for warnings - but not deprecation notices or Python warnings in imports
                        is_real_warning = False
                        if "DEPRECATION WARNING" in line:
                            # These are just notices about outdated extensions, not real warnings
                            is_real_warning = False
                        elif "FutureWarning:" in line or "UserWarning:" in line:
                            # Python deprecation warnings from imports - informational only
                            is_real_warning = False
                        elif "Warning:" in line and "ComfyUI" in line:
                            # Actual ComfyUI warnings (like WAS Node Suite config warnings)
                            is_real_warning = True
                        elif "attempting to free" in line_lower or "could not free port" in line_lower:
                            # Port conflict warnings from our launcher
                            is_real_warning = True

                        if is_real_error:
                            level = "error"
                        elif is_real_warning:
                            level = "warning"
                        elif "success" in line_lower or ("loaded" in line_lower and "failed" not in line_lower):
                            level = "success"

                        add_log(line, level)
                        update_progress(line)
    except Exception as e:
        pass


def monitor_process():
    """Monitor the ComfyUI process and read logs"""
    global is_running, startup_info, process

    while is_running and process:
        # Check if process is still running
        if process.poll() is not None:
            # Process ended
            is_running = False
            startup_info["status"] = "stopped"
            startup_info["phase"] = "Stopped"
            startup_info["progress"] = 0
            add_log("ComfyUI process ended", "warning")
            break

        # Read new log content
        tail_log_file()
        time.sleep(0.2)


def check_port_in_use(port):
    """Check if a port is in use and try to identify the process"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('127.0.0.1', port))
            return result == 0
    except:
        return False


def kill_process_on_port(port):
    """Try to kill whatever process is using the port (Windows)"""
    import subprocess
    try:
        # Find PID using netstat
        result = subprocess.run(
            ['netstat', '-ano'],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        for line in result.stdout.split('\n'):
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    # Kill the process
                    subprocess.run(
                        ['taskkill', '/F', '/PID', pid],
                        capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                    return True, f"Killed process {pid} on port {port}"
        return False, "No process found on port"
    except Exception as e:
        return False, str(e)


def start_comfyui():
    global process, is_running, startup_info, last_log_position

    if is_running:
        return {"success": False, "message": "Already running"}

    # Check if port is in use
    port = settings["port"]
    if check_port_in_use(port):
        add_log(f"Port {port} is in use, attempting to free it...", "warning")
        success, msg = kill_process_on_port(port)
        if success:
            add_log(msg, "success")
            # Wait a moment for port to be released
            time.sleep(1)
        else:
            add_log(f"Could not free port {port}: {msg}", "error")
            # Try to find an alternative port
            for alt_port in range(port + 1, port + 10):
                if not check_port_in_use(alt_port):
                    add_log(f"Using alternative port {alt_port}", "warning")
                    settings["port"] = alt_port
                    break
            else:
                return {"success": False, "message": f"Port {port} is in use and could not be freed"}

    # Reset state
    startup_info = {
        "phase": "Starting...",
        "progress": 5,
        "custom_nodes": 0,
        "status": "starting",
        "gen_speed": "",
        "gen_progress": "",
        "is_generating": False
    }

    # Clear log file
    try:
        with open(LOG_FILE, 'w') as f:
            f.write("")
        last_log_position = 0
    except:
        pass

    # Build command - redirect output to log file
    cmd = [
        str(PYTHON_EXE),
        "-u",  # Unbuffered output
        "-s",
        str(COMFYUI_DIR / "main.py"),
        "--windows-standalone-build",
        "--listen", settings["listen"],
        "--port", str(settings["port"])
    ]

    # Add comfy_args from checkbox/radio settings
    if settings.get("comfy_args"):
        cmd.extend(settings["comfy_args"].split())

    # Add extra_args (manual text input)
    if settings.get("extra_args"):
        cmd.extend(settings["extra_args"].split())

    if settings.get("auto_launch_browser"):
        cmd.append("--auto-launch")

    add_log("Starting ComfyUI...", "info")
    add_log(f"Command: {' '.join(cmd)}", "info")

    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0  # SW_HIDE

        # Set environment to use UTF-8 encoding for Python output
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'

        # Open log file for writing with UTF-8 encoding
        log_handle = open(LOG_FILE, 'w', encoding='utf-8', errors='replace')

        process = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            startupinfo=startupinfo,
            creationflags=subprocess.CREATE_NO_WINDOW,
            cwd=str(BASE_DIR),
            env=env
        )

        is_running = True

        # Start monitoring thread
        thread = threading.Thread(target=monitor_process, daemon=True)
        thread.start()

        return {"success": True, "message": "Started"}

    except Exception as e:
        add_log(f"Failed to start: {e}", "error")
        return {"success": False, "message": str(e)}


def kill_process_tree(pid):
    """Kill a process and all its children on Windows"""
    try:
        # Use taskkill with /T flag to kill the process tree
        subprocess.run(
            ['taskkill', '/F', '/T', '/PID', str(pid)],
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        return True
    except Exception as e:
        return False


def stop_comfyui():
    global process, is_running

    if not is_running or not process:
        return {"success": False, "message": "Not running"}

    add_log("Stopping ComfyUI...", "warning")

    pid = process.pid
    port = settings["port"]

    try:
        # First try graceful termination
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            # If it doesn't stop gracefully, kill the process tree
            add_log("Process not responding, force killing...", "warning")
            kill_process_tree(pid)
            process.wait(timeout=2)
    except Exception as e:
        add_log(f"Error during termination: {e}", "error")
        # Last resort: kill anything on the port
        kill_process_on_port(port)

    # Verify the port is actually free
    time.sleep(0.5)
    if check_port_in_use(port):
        add_log(f"Port {port} still in use, forcing cleanup...", "warning")
        success, msg = kill_process_on_port(port)
        if success:
            add_log(msg, "success")
        else:
            add_log(f"Could not free port: {msg}", "error")
        time.sleep(0.5)

    # Final check
    if check_port_in_use(port):
        add_log(f"Warning: Port {port} may still be in use", "error")
    else:
        add_log("ComfyUI stopped and port freed", "success")

    is_running = False
    process = None
    return {"success": True, "message": "Stopped"}


# Update functions
def add_update_log(message, level="info"):
    with update_log_lock:
        update_log_buffer.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": message,
            "level": level
        })
        # Keep last 500 lines
        if len(update_log_buffer) > 500:
            update_log_buffer.pop(0)


def tail_update_log_file():
    """Read new lines from the update log file"""
    global last_update_log_position

    if not UPDATE_LOG_FILE.exists():
        return

    try:
        with open(UPDATE_LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
            f.seek(last_update_log_position)
            new_content = f.read()
            last_update_log_position = f.tell()

            if new_content:
                for line in new_content.splitlines():
                    if line.strip():
                        level = "info"
                        line_lower = line.lower()
                        if "error" in line_lower or "failed" in line_lower:
                            level = "error"
                        elif "warning" in line_lower:
                            level = "warning"
                        elif "success" in line_lower or "up to date" in line_lower or "updated" in line_lower:
                            level = "success"
                        add_update_log(line, level)
    except Exception:
        pass


def monitor_update_process():
    """Monitor the update process and read logs"""
    global is_updating, update_process

    while is_updating and update_process:
        if update_process.poll() is not None:
            # Process ended
            tail_update_log_file()  # Get final output
            exit_code = update_process.returncode
            if exit_code == 0:
                add_update_log("Update completed successfully!", "success")
            else:
                add_update_log(f"Update finished with exit code {exit_code}", "warning")
            is_updating = False
            break

        tail_update_log_file()
        time.sleep(0.3)


def run_update(update_type):
    """Run an update script"""
    global update_process, is_updating, last_update_log_position

    if is_updating:
        return {"success": False, "message": "Update already in progress"}

    if is_running:
        return {"success": False, "message": "Please stop ComfyUI before updating"}

    # Map update type to script
    scripts = {
        "comfyui": "update_comfyui.bat",
        "comfyui_stable": "update_comfyui_stable.bat",
        "full": "update_comfyui_and_python_dependencies.bat"
    }

    script_name = scripts.get(update_type)
    if not script_name:
        return {"success": False, "message": f"Unknown update type: {update_type}"}

    script_path = UPDATE_DIR / script_name
    if not script_path.exists():
        return {"success": False, "message": f"Update script not found: {script_name}"}

    # Clear update log
    with update_log_lock:
        update_log_buffer.clear()

    try:
        with open(UPDATE_LOG_FILE, 'w') as f:
            f.write("")
        last_update_log_position = 0
    except:
        pass

    add_update_log(f"Starting update: {update_type}", "info")
    add_update_log(f"Running: {script_name}", "info")

    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'

        log_handle = open(UPDATE_LOG_FILE, 'w', encoding='utf-8', errors='replace')

        # Run the batch file with 'nopause' argument to prevent pause prompts
        update_process = subprocess.Popen(
            ['cmd', '/c', str(script_path), 'nopause'],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            startupinfo=startupinfo,
            creationflags=subprocess.CREATE_NO_WINDOW,
            cwd=str(UPDATE_DIR),
            env=env
        )

        is_updating = True

        # Start monitoring thread
        thread = threading.Thread(target=monitor_update_process, daemon=True)
        thread.start()

        return {"success": True, "message": f"Update started: {update_type}"}

    except Exception as e:
        add_update_log(f"Failed to start update: {e}", "error")
        return {"success": False, "message": str(e)}


def get_update_status():
    """Get current update status"""
    with update_log_lock:
        logs_copy = list(update_log_buffer)
    return {
        "is_updating": is_updating,
        "logs": logs_copy
    }


# HTML Template with embedded CSS and JavaScript
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ComfyUI Launcher</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --bg-dark: #0d1117;
            --bg-medium: #161b22;
            --bg-light: #21262d;
            --accent: #f85149;
            --accent-hover: #ff7b72;
            --text: #e6edf3;
            --text-dim: #7d8590;
            --success: #3fb950;
            --warning: #d29922;
            --error: #f85149;
            --info: #58a6ff;
            --border: #30363d;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Header */
        .header {
            background: linear-gradient(135deg, var(--bg-medium) 0%, var(--bg-light) 100%);
            border-radius: 16px;
            padding: 24px 30px;
            margin-bottom: 20px;
            border: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            font-size: 32px;
        }

        .logo h1 {
            font-size: 24px;
            font-weight: 600;
        }

        .logo h1 span {
            color: var(--accent);
        }

        .status {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--text-dim);
            animation: none;
        }

        .status-dot.running {
            background: var(--success);
            box-shadow: 0 0 10px var(--success);
        }

        .status-dot.starting {
            background: var(--warning);
            animation: pulse 1s infinite;
        }

        .status-dot.error {
            background: var(--error);
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Progress Section */
        .progress-section {
            background: var(--bg-medium);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 20px;
            border: 1px solid var(--border);
        }

        .phase-text {
            color: var(--text-dim);
            margin-bottom: 10px;
            font-size: 13px;
        }

        .progress-bar {
            height: 6px;
            background: var(--bg-light);
            border-radius: 3px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent) 0%, var(--accent-hover) 100%);
            border-radius: 3px;
            transition: width 0.3s ease;
            width: 0%;
        }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }

        .stat-card {
            background: var(--bg-medium);
            border-radius: 10px;
            padding: 16px;
            text-align: center;
            border: 1px solid var(--border);
        }

        .stat-card.errors {
            border-color: var(--error);
            background: rgba(248, 81, 73, 0.1);
        }

        .stat-card.warnings {
            border-color: var(--warning);
            background: rgba(210, 153, 34, 0.1);
        }

        .stat-label {
            color: var(--text-dim);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }

        .stat-value {
            font-size: 20px;
            font-weight: 600;
        }

        .stat-value.error-count { color: var(--error); }
        .stat-value.warning-count { color: var(--warning); }

        .stat-card.generating {
            border-color: var(--info);
            background: rgba(88, 166, 255, 0.1);
        }

        .stat-card.generating .stat-value {
            color: var(--info);
        }

        .speed-display {
            font-size: 16px;
        }

        .speed-display .speed-value {
            font-size: 22px;
            font-weight: 700;
        }

        .speed-display .speed-unit {
            font-size: 12px;
            color: var(--text-dim);
        }

        /* Split Console */
        .console-split {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }

        .console {
            background: var(--bg-medium);
            border-radius: 12px;
            border: 1px solid var(--border);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .console.errors-panel {
            border-color: rgba(248, 81, 73, 0.5);
        }

        .console-header {
            background: var(--bg-light);
            padding: 10px 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            flex-shrink: 0;
        }

        .console.errors-panel .console-header {
            background: rgba(248, 81, 73, 0.15);
        }

        .console-title {
            font-size: 12px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .console-title .icon {
            font-size: 14px;
        }

        .console-actions {
            display: flex;
            gap: 10px;
        }

        .console-action {
            color: var(--text-dim);
            cursor: pointer;
            font-size: 11px;
            padding: 4px 8px;
            border-radius: 4px;
            transition: all 0.2s;
        }

        .console-action:hover {
            color: var(--text);
            background: var(--bg-medium);
        }

        .console-body {
            height: 280px;
            overflow-y: auto;
            padding: 10px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 11px;
            line-height: 1.5;
            flex-grow: 1;
        }

        .log-entry {
            display: flex;
            gap: 8px;
            padding: 2px 0;
        }

        .log-time {
            color: var(--text-dim);
            flex-shrink: 0;
            font-size: 10px;
        }

        .log-message {
            word-break: break-word;
        }

        .log-message.info { color: var(--text); }
        .log-message.success { color: var(--success); }
        .log-message.warning { color: var(--warning); }
        .log-message.error { color: var(--error); }

        .empty-state {
            color: var(--text-dim);
            text-align: center;
            padding: 40px 20px;
            font-size: 12px;
        }

        /* Buttons */
        .button-row {
            display: flex;
            gap: 12px;
            justify-content: space-between;
        }

        .btn {
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .btn-primary {
            background: var(--accent);
            color: white;
        }

        .btn-primary:hover {
            background: var(--accent-hover);
            transform: translateY(-1px);
        }

        .btn-primary.running {
            background: var(--error);
        }

        .btn-secondary {
            background: var(--bg-light);
            color: var(--text);
            border: 1px solid var(--border);
        }

        .btn-secondary:hover {
            background: var(--bg-medium);
            border-color: var(--text-dim);
        }

        .btn-group {
            display: flex;
            gap: 8px;
        }

        /* Toast notification */
        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: var(--success);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.3s;
            z-index: 1000;
        }

        .toast.show {
            opacity: 1;
            transform: translateY(0);
        }

        /* Settings Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.7);
            z-index: 100;
            justify-content: center;
            align-items: center;
        }

        .modal-overlay.active {
            display: flex;
        }

        .modal {
            background: var(--bg-medium);
            border-radius: 16px;
            padding: 24px;
            width: 400px;
            max-width: 90%;
            border: 1px solid var(--border);
            max-height: 90vh;
            overflow-y: auto;
        }

        .modal.settings-modal {
            width: 550px;
        }

        .modal h2 {
            margin-bottom: 20px;
            font-size: 18px;
        }

        .form-group {
            margin-bottom: 16px;
        }

        .form-group label {
            display: block;
            color: var(--text-dim);
            font-size: 12px;
            margin-bottom: 6px;
        }

        .form-group input[type="text"],
        .form-group input[type="number"] {
            width: 100%;
            padding: 10px 12px;
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            font-size: 14px;
        }

        .form-group input:focus {
            outline: none;
            border-color: var(--accent);
        }

        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .checkbox-group input {
            width: 16px;
            height: 16px;
        }

        .settings-section {
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }

        .settings-section:last-of-type {
            border-bottom: none;
            margin-bottom: 10px;
        }

        .settings-section-title {
            font-size: 13px;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .settings-section-title .icon {
            font-size: 14px;
        }

        .radio-group {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .radio-option {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 8px 12px;
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 12px;
        }

        .radio-option:hover {
            border-color: var(--text-dim);
        }

        .radio-option.selected {
            border-color: var(--accent);
            background: rgba(248, 81, 73, 0.1);
        }

        .radio-option input[type="radio"] {
            display: none;
        }

        .radio-dot {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            border: 2px solid var(--text-dim);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .radio-option.selected .radio-dot {
            border-color: var(--accent);
        }

        .radio-dot::after {
            content: '';
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: transparent;
        }

        .radio-option.selected .radio-dot::after {
            background: var(--accent);
        }

        .checkbox-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }

        .checkbox-option {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 12px;
        }

        .checkbox-option:hover {
            border-color: var(--text-dim);
        }

        .checkbox-option.checked {
            border-color: var(--info);
            background: rgba(88, 166, 255, 0.1);
        }

        .checkbox-option input[type="checkbox"] {
            display: none;
        }

        .checkbox-box {
            width: 14px;
            height: 14px;
            border-radius: 3px;
            border: 2px solid var(--text-dim);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            color: transparent;
        }

        .checkbox-option.checked .checkbox-box {
            border-color: var(--info);
            background: var(--info);
            color: white;
        }

        .option-desc {
            font-size: 10px;
            color: var(--text-dim);
            margin-top: 2px;
        }

        .modal-buttons {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            margin-top: 20px;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-dark);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-dim);
        }

        /* Activity Tracker */
        .activity-tracker {
            background: var(--bg-medium);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 20px;
            border: 1px solid var(--border);
        }

        .activity-header {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .activity-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .activity-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            background: var(--bg-dark);
            border-radius: 6px;
            font-size: 13px;
        }

        .activity-icon {
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .activity-icon.pending {
            color: var(--text-dim);
        }

        .activity-icon.active {
            color: var(--info);
        }

        .activity-icon.done {
            color: var(--success);
        }

        .activity-spinner {
            width: 16px;
            height: 16px;
            border: 2px solid var(--border);
            border-top-color: var(--info);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        .activity-name {
            flex-grow: 1;
        }

        .activity-name.done {
            color: var(--text-dim);
        }

        .activity-empty {
            color: var(--text-dim);
            font-size: 12px;
            text-align: center;
            padding: 20px;
        }

        .activity-progress-container {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .activity-progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .activity-progress-text {
            font-size: 12px;
            color: var(--text-dim);
        }

        .activity-progress-bar {
            height: 8px;
            background: var(--bg-light);
            border-radius: 4px;
            overflow: hidden;
        }

        .activity-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--info) 0%, #79c0ff 100%);
            border-radius: 4px;
            transition: width 0.2s ease;
        }

        /* Update Modal */
        .modal.update-modal {
            width: 600px;
        }

        .update-options {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 20px;
        }

        .update-option {
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px 16px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .update-option:hover {
            border-color: var(--info);
            background: rgba(88, 166, 255, 0.1);
        }

        .update-option.disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .update-option-title {
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .update-option-desc {
            font-size: 12px;
            color: var(--text-dim);
        }

        .update-console {
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 8px;
            height: 200px;
            overflow-y: auto;
            padding: 10px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 11px;
            line-height: 1.5;
            display: none;
        }

        .update-console.active {
            display: block;
        }

        .update-status {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
            padding: 10px;
            background: var(--bg-dark);
            border-radius: 8px;
            display: none;
        }

        .update-status.active {
            display: flex;
        }

        .update-spinner {
            width: 20px;
            height: 20px;
            border: 2px solid var(--border);
            border-top-color: var(--info);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">
                <span class="logo-icon">⚡</span>
                <h1><span>ComfyUI</span> Launcher</h1>
            </div>
            <div class="status">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Stopped</span>
            </div>
        </div>

        <div class="progress-section">
            <div class="phase-text" id="phaseText">Ready to start</div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Status</div>
                <div class="stat-value" id="statStatus">Stopped</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Custom Nodes</div>
                <div class="stat-value" id="statNodes">0</div>
            </div>
            <div class="stat-card" id="speedCard">
                <div class="stat-label">Generation Speed</div>
                <div class="stat-value speed-display" id="statSpeed">--</div>
            </div>
            <div class="stat-card warnings">
                <div class="stat-label">Warnings</div>
                <div class="stat-value warning-count" id="statWarnings">0</div>
            </div>
            <div class="stat-card errors">
                <div class="stat-label">Errors</div>
                <div class="stat-value error-count" id="statErrors">0</div>
            </div>
        </div>

        <div class="activity-tracker" id="activityTracker" style="display: none;">
            <div class="activity-header">
                <span>⚡</span> Workflow Activity
            </div>
            <div class="activity-list" id="activityList">
                <div class="activity-empty">No active workflow</div>
            </div>
        </div>

        <div class="console-split">
            <div class="console">
                <div class="console-header">
                    <span class="console-title"><span class="icon">📋</span> Console Output</span>
                    <div class="console-actions">
                        <span class="console-action" onclick="copyLogs('all')">📋 Copy</span>
                        <span class="console-action" onclick="clearLogs()">🗑️ Clear</span>
                    </div>
                </div>
                <div class="console-body" id="consoleBody">
                    <div class="log-entry">
                        <span class="log-time">[--:--:--]</span>
                        <span class="log-message info">Launcher ready. Click Start to begin.</span>
                    </div>
                </div>
            </div>

            <div class="console errors-panel">
                <div class="console-header">
                    <span class="console-title"><span class="icon">⚠️</span> Errors & Warnings</span>
                    <div class="console-actions">
                        <span class="console-action" onclick="copyLogs('errors')">📋 Copy</span>
                        <span class="console-action" onclick="clearErrors()">🗑️ Clear</span>
                    </div>
                </div>
                <div class="console-body" id="errorsBody">
                    <div class="empty-state">No errors or warnings yet</div>
                </div>
            </div>
        </div>

        <div class="button-row">
            <div class="btn-group">
                <button class="btn btn-secondary" onclick="showSettings()">⚙️ Settings</button>
                <button class="btn btn-secondary" onclick="showUpdateModal()">🔄 Update</button>
                <button class="btn btn-secondary" onclick="openComfyUI()">🌐 Open ComfyUI</button>
                <button class="btn btn-secondary" onclick="copyLogs('all')">📋 Copy All Logs</button>
            </div>
            <button class="btn btn-primary" id="actionBtn" onclick="toggleComfyUI()">▶ Start ComfyUI</button>
        </div>
    </div>

    <!-- Toast notification -->
    <div class="toast" id="toast">Copied to clipboard!</div>

    <!-- Settings Modal -->
    <div class="modal-overlay" id="settingsModal">
        <div class="modal settings-modal">
            <h2>⚙️ Settings</h2>

            <!-- Network Settings -->
            <div class="settings-section">
                <div class="settings-section-title"><span class="icon">🌐</span> Network</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div class="form-group" style="margin-bottom: 0;">
                        <label>Listen Address</label>
                        <input type="text" id="settingListen" value="LISTEN_PLACEHOLDER">
                    </div>
                    <div class="form-group" style="margin-bottom: 0;">
                        <label>Port</label>
                        <input type="number" id="settingPort" value="PORT_PLACEHOLDER">
                    </div>
                </div>
            </div>

            <!-- VRAM Mode (mutually exclusive) -->
            <div class="settings-section">
                <div class="settings-section-title"><span class="icon">🎮</span> VRAM Mode</div>
                <div class="radio-group" id="vramModeGroup">
                    <label class="radio-option selected" data-value="">
                        <input type="radio" name="vramMode" value="" checked>
                        <span class="radio-dot"></span>
                        <span>Default</span>
                    </label>
                    <label class="radio-option" data-value="--gpu-only">
                        <input type="radio" name="vramMode" value="--gpu-only">
                        <span class="radio-dot"></span>
                        <span>GPU Only</span>
                    </label>
                    <label class="radio-option" data-value="--highvram">
                        <input type="radio" name="vramMode" value="--highvram">
                        <span class="radio-dot"></span>
                        <span>High VRAM</span>
                    </label>
                    <label class="radio-option" data-value="--normalvram">
                        <input type="radio" name="vramMode" value="--normalvram">
                        <span class="radio-dot"></span>
                        <span>Normal VRAM</span>
                    </label>
                    <label class="radio-option" data-value="--lowvram">
                        <input type="radio" name="vramMode" value="--lowvram">
                        <span class="radio-dot"></span>
                        <span>Low VRAM</span>
                    </label>
                    <label class="radio-option" data-value="--novram">
                        <input type="radio" name="vramMode" value="--novram">
                        <span class="radio-dot"></span>
                        <span>No VRAM</span>
                    </label>
                    <label class="radio-option" data-value="--cpu">
                        <input type="radio" name="vramMode" value="--cpu">
                        <span class="radio-dot"></span>
                        <span>CPU Only</span>
                    </label>
                </div>
            </div>

            <!-- Precision Options (mutually exclusive) -->
            <div class="settings-section">
                <div class="settings-section-title"><span class="icon">🔢</span> Precision</div>
                <div class="radio-group" id="precisionGroup">
                    <label class="radio-option selected" data-value="">
                        <input type="radio" name="precision" value="" checked>
                        <span class="radio-dot"></span>
                        <span>Auto</span>
                    </label>
                    <label class="radio-option" data-value="--force-fp32">
                        <input type="radio" name="precision" value="--force-fp32">
                        <span class="radio-dot"></span>
                        <span>FP32</span>
                    </label>
                    <label class="radio-option" data-value="--force-fp16">
                        <input type="radio" name="precision" value="--force-fp16">
                        <span class="radio-dot"></span>
                        <span>FP16</span>
                    </label>
                </div>
            </div>

            <!-- Preview Method (mutually exclusive) -->
            <div class="settings-section">
                <div class="settings-section-title"><span class="icon">🖼️</span> Preview Method</div>
                <div class="radio-group" id="previewGroup">
                    <label class="radio-option selected" data-value="">
                        <input type="radio" name="preview" value="" checked>
                        <span class="radio-dot"></span>
                        <span>Auto</span>
                    </label>
                    <label class="radio-option" data-value="--preview-method none">
                        <input type="radio" name="preview" value="--preview-method none">
                        <span class="radio-dot"></span>
                        <span>None</span>
                    </label>
                    <label class="radio-option" data-value="--preview-method latent2rgb">
                        <input type="radio" name="preview" value="--preview-method latent2rgb">
                        <span class="radio-dot"></span>
                        <span>Latent2RGB</span>
                    </label>
                    <label class="radio-option" data-value="--preview-method taesd">
                        <input type="radio" name="preview" value="--preview-method taesd">
                        <span class="radio-dot"></span>
                        <span>TAESD</span>
                    </label>
                </div>
            </div>

            <!-- Additional Options (checkboxes) -->
            <div class="settings-section">
                <div class="settings-section-title"><span class="icon">⚡</span> Additional Options</div>
                <div class="checkbox-grid">
                    <label class="checkbox-option" data-value="--disable-xformers">
                        <input type="checkbox" name="opt_disable_xformers">
                        <span class="checkbox-box">✓</span>
                        <span>Disable xformers</span>
                    </label>
                    <label class="checkbox-option" data-value="--use-pytorch-cross-attention">
                        <input type="checkbox" name="opt_pytorch_attention">
                        <span class="checkbox-box">✓</span>
                        <span>PyTorch Attention</span>
                    </label>
                    <label class="checkbox-option" data-value="--disable-smart-memory">
                        <input type="checkbox" name="opt_disable_smart_memory">
                        <span class="checkbox-box">✓</span>
                        <span>Disable Smart Memory</span>
                    </label>
                    <label class="checkbox-option" data-value="--dont-upcast-attention">
                        <input type="checkbox" name="opt_dont_upcast">
                        <span class="checkbox-box">✓</span>
                        <span>Don't Upcast Attention</span>
                    </label>
                    <label class="checkbox-option" data-value="--use-split-cross-attention">
                        <input type="checkbox" name="opt_split_attention">
                        <span class="checkbox-box">✓</span>
                        <span>Split Cross Attention</span>
                    </label>
                    <label class="checkbox-option" data-value="--disable-all-custom-nodes">
                        <input type="checkbox" name="opt_disable_custom_nodes">
                        <span class="checkbox-box">✓</span>
                        <span>Disable Custom Nodes</span>
                    </label>
                    <label class="checkbox-option" data-value="--fast">
                        <input type="checkbox" name="opt_fast">
                        <span class="checkbox-box">✓</span>
                        <span>Fast Mode (Experimental)</span>
                    </label>
                    <label class="checkbox-option" data-value="--cpu-vae">
                        <input type="checkbox" name="opt_cpu_vae">
                        <span class="checkbox-box">✓</span>
                        <span>VAE on CPU</span>
                    </label>
                </div>
            </div>

            <!-- Launcher Options -->
            <div class="settings-section">
                <div class="settings-section-title"><span class="icon">🚀</span> Launcher</div>
                <div class="checkbox-grid">
                    <label class="checkbox-option AUTOLAUNCH_CLASS" data-value="auto_launch">
                        <input type="checkbox" id="settingAutoLaunch" AUTOLAUNCH_PLACEHOLDER>
                        <span class="checkbox-box">✓</span>
                        <span>Auto-launch browser</span>
                    </label>
                </div>
            </div>

            <!-- Extra Arguments -->
            <div class="settings-section">
                <div class="settings-section-title"><span class="icon">📝</span> Extra Arguments</div>
                <div class="form-group" style="margin-bottom: 0;">
                    <input type="text" id="settingArgs" value="ARGS_PLACEHOLDER" placeholder="Additional arguments not covered above">
                </div>
            </div>

            <div class="modal-buttons">
                <button class="btn btn-secondary" onclick="hideSettings()">Cancel</button>
                <button class="btn btn-primary" onclick="saveSettings()">Save</button>
            </div>
        </div>
    </div>

    <!-- Update Modal -->
    <div class="modal-overlay" id="updateModal">
        <div class="modal update-modal">
            <h2>🔄 Update ComfyUI</h2>

            <div class="update-status" id="updateStatus">
                <div class="update-spinner"></div>
                <span id="updateStatusText">Updating...</span>
            </div>

            <div class="update-options" id="updateOptions">
                <div class="update-option" onclick="runUpdate('comfyui')">
                    <div class="update-option-title">
                        <span>📦</span> Update ComfyUI (Latest)
                    </div>
                    <div class="update-option-desc">
                        Update to the latest development version of ComfyUI
                    </div>
                </div>
                <div class="update-option" onclick="runUpdate('comfyui_stable')">
                    <div class="update-option-title">
                        <span>✅</span> Update ComfyUI (Stable)
                    </div>
                    <div class="update-option-desc">
                        Update to the latest stable release of ComfyUI
                    </div>
                </div>
                <div class="update-option" onclick="runUpdate('full')">
                    <div class="update-option-title">
                        <span>🔧</span> Full Update (ComfyUI + Dependencies)
                    </div>
                    <div class="update-option-desc">
                        Update ComfyUI and all Python dependencies including PyTorch. Use this if you have issues after a regular update.
                    </div>
                </div>
            </div>

            <div class="update-console" id="updateConsole"></div>

            <div class="modal-buttons">
                <button class="btn btn-secondary" id="updateCloseBtn" onclick="hideUpdateModal()">Close</button>
            </div>
        </div>
    </div>

    <script>
        let isRunning = false;
        let lastLogCount = 0;
        let lastErrorCount = 0;
        let allLogs = [];
        let errorLogs = [];

        function updateUI(data) {
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            const statStatus = document.getElementById('statStatus');
            const actionBtn = document.getElementById('actionBtn');
            const phaseText = document.getElementById('phaseText');
            const progressFill = document.getElementById('progressFill');
            const statNodes = document.getElementById('statNodes');
            const statSpeed = document.getElementById('statSpeed');
            const speedCard = document.getElementById('speedCard');

            // Update status
            statusDot.className = 'status-dot ' + data.status;
            statusText.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
            statStatus.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);

            // Update button
            isRunning = data.status !== 'stopped';
            if (isRunning) {
                actionBtn.textContent = '■ Stop ComfyUI';
                actionBtn.classList.add('running');
            } else {
                actionBtn.textContent = '▶ Start ComfyUI';
                actionBtn.classList.remove('running');
            }

            // Update progress
            phaseText.textContent = data.phase;
            progressFill.style.width = data.progress + '%';
            statNodes.textContent = data.custom_nodes;

            // Update generation speed
            if (data.is_generating) {
                speedCard.classList.add('generating');
                // Parse speed value and unit for better formatting
                if (data.gen_speed) {
                    const parts = data.gen_speed.split(' ');
                    if (parts.length === 2) {
                        statSpeed.innerHTML = '<span class="speed-value">' + parts[0] + '</span> <span class="speed-unit">' + parts[1] + '</span>';
                    } else {
                        statSpeed.textContent = data.gen_speed;
                    }
                }
            } else {
                speedCard.classList.remove('generating');
                if (data.gen_speed && data.gen_progress === 'Done') {
                    // Show last speed after completion
                    const parts = data.gen_speed.split(' ');
                    if (parts.length === 2) {
                        statSpeed.innerHTML = '<span class="speed-value">' + parts[0] + '</span> <span class="speed-unit">' + parts[1] + '</span>';
                    } else {
                        statSpeed.textContent = data.gen_speed;
                    }
                } else if (!data.gen_speed) {
                    statSpeed.textContent = '--';
                }
            }
        }

        function updateLogs(logs) {
            if (logs.length === lastLogCount) return;

            const consoleEl = document.getElementById('consoleBody');
            const errorsEl = document.getElementById('errorsBody');
            const wasAtBottom = consoleEl.scrollHeight - consoleEl.scrollTop <= consoleEl.clientHeight + 50;
            const errorsWasAtBottom = errorsEl.scrollHeight - errorsEl.scrollTop <= errorsEl.clientHeight + 50;

            // Only add new logs
            const newLogs = logs.slice(lastLogCount);

            // Clear empty state if this is first error
            if (errorLogs.length === 0 && newLogs.some(l => l.level === 'error' || l.level === 'warning')) {
                errorsEl.innerHTML = '';
            }

            let newErrorCount = 0;
            let newWarningCount = 0;

            for (const log of newLogs) {
                allLogs.push(log);

                // Add to main console
                const entry = document.createElement('div');
                entry.className = 'log-entry';
                entry.innerHTML = `
                    <span class="log-time">[${log.time}]</span>
                    <span class="log-message ${log.level}">${escapeHtml(log.message)}</span>
                `;
                consoleEl.appendChild(entry);

                // Add errors and warnings to error panel
                if (log.level === 'error' || log.level === 'warning') {
                    errorLogs.push(log);
                    const errorEntry = document.createElement('div');
                    errorEntry.className = 'log-entry';
                    errorEntry.innerHTML = `
                        <span class="log-time">[${log.time}]</span>
                        <span class="log-message ${log.level}">${escapeHtml(log.message)}</span>
                    `;
                    errorsEl.appendChild(errorEntry);

                    if (log.level === 'error') newErrorCount++;
                    if (log.level === 'warning') newWarningCount++;
                }
            }

            lastLogCount = logs.length;

            // Update counts
            const totalErrors = errorLogs.filter(l => l.level === 'error').length;
            const totalWarnings = errorLogs.filter(l => l.level === 'warning').length;
            document.getElementById('statErrors').textContent = totalErrors;
            document.getElementById('statWarnings').textContent = totalWarnings;

            if (wasAtBottom) {
                consoleEl.scrollTop = consoleEl.scrollHeight;
            }
            if (errorsWasAtBottom) {
                errorsEl.scrollTop = errorsEl.scrollHeight;
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function clearLogs() {
            fetch('/api/clear-logs', { method: 'POST' });
            document.getElementById('consoleBody').innerHTML = '';
            lastLogCount = 0;
            allLogs = [];
        }

        function clearErrors() {
            document.getElementById('errorsBody').innerHTML = '<div class="empty-state">No errors or warnings yet</div>';
            errorLogs = [];
            document.getElementById('statErrors').textContent = '0';
            document.getElementById('statWarnings').textContent = '0';
        }

        function copyLogs(type) {
            let text = '';
            const logs = type === 'errors' ? errorLogs : allLogs;

            for (const log of logs) {
                text += `[${log.time}] ${log.message}\\n`;
            }

            navigator.clipboard.writeText(text).then(() => {
                showToast(type === 'errors' ? 'Errors copied to clipboard!' : 'Logs copied to clipboard!');
            });
        }

        function showToast(message) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 2000);
        }

        let comfyUIWindow = null;

        function toggleComfyUI() {
            const action = isRunning ? 'stop' : 'start';
            fetch('/api/' + action, { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    if (!data.success) {
                        alert(data.message);
                    } else if (action === 'stop') {
                        // Try to close the ComfyUI browser tab if we opened it
                        if (comfyUIWindow && !comfyUIWindow.closed) {
                            try {
                                comfyUIWindow.close();
                            } catch (e) {
                                // Can't close windows we didn't open
                            }
                        }
                        comfyUIWindow = null;
                    }
                });
        }

        function openComfyUI() {
            const port = PORT_PLACEHOLDER;
            comfyUIWindow = window.open('http://127.0.0.1:' + port, '_blank');
        }

        // Settings management
        let currentSettings = {};

        function initSettingsUI() {
            // Radio group click handlers
            document.querySelectorAll('.radio-group').forEach(group => {
                group.querySelectorAll('.radio-option').forEach(option => {
                    option.addEventListener('click', () => {
                        group.querySelectorAll('.radio-option').forEach(o => o.classList.remove('selected'));
                        option.classList.add('selected');
                        option.querySelector('input').checked = true;
                    });
                });
            });

            // Checkbox click handlers
            document.querySelectorAll('.checkbox-option').forEach(option => {
                option.addEventListener('click', () => {
                    const checkbox = option.querySelector('input[type="checkbox"]');
                    checkbox.checked = !checkbox.checked;
                    option.classList.toggle('checked', checkbox.checked);
                });
            });
        }

        function showSettings() {
            // Fetch current settings and update UI
            fetch('/api/settings')
                .then(r => r.json())
                .then(settings => {
                    currentSettings = settings;

                    // Update basic fields
                    document.getElementById('settingListen').value = settings.listen || '0.0.0.0';
                    document.getElementById('settingPort').value = settings.port || 8188;
                    document.getElementById('settingArgs').value = settings.extra_args || '';

                    // Update auto-launch checkbox
                    const autoLaunch = document.getElementById('settingAutoLaunch');
                    autoLaunch.checked = settings.auto_launch_browser !== false;
                    autoLaunch.closest('.checkbox-option').classList.toggle('checked', autoLaunch.checked);

                    // Parse saved args to set radio/checkbox states
                    const savedArgs = (settings.comfy_args || '').split(' ').filter(a => a);

                    // Reset all options first
                    document.querySelectorAll('.radio-option').forEach(o => o.classList.remove('selected'));
                    document.querySelectorAll('.radio-option[data-value=""]').forEach(o => o.classList.add('selected'));
                    document.querySelectorAll('.checkbox-option:not([data-value="auto_launch"])').forEach(o => {
                        o.classList.remove('checked');
                        o.querySelector('input').checked = false;
                    });

                    // Set VRAM mode
                    const vramModes = ['--gpu-only', '--highvram', '--normalvram', '--lowvram', '--novram', '--cpu'];
                    for (const mode of vramModes) {
                        if (savedArgs.includes(mode)) {
                            const opt = document.querySelector(`.radio-option[data-value="${mode}"]`);
                            if (opt) {
                                document.querySelectorAll('#vramModeGroup .radio-option').forEach(o => o.classList.remove('selected'));
                                opt.classList.add('selected');
                            }
                            break;
                        }
                    }

                    // Set Precision
                    const precisions = ['--force-fp32', '--force-fp16'];
                    for (const p of precisions) {
                        if (savedArgs.includes(p)) {
                            const opt = document.querySelector(`.radio-option[data-value="${p}"]`);
                            if (opt) {
                                document.querySelectorAll('#precisionGroup .radio-option').forEach(o => o.classList.remove('selected'));
                                opt.classList.add('selected');
                            }
                            break;
                        }
                    }

                    // Set Preview method
                    const previews = ['--preview-method none', '--preview-method latent2rgb', '--preview-method taesd'];
                    const argsStr = savedArgs.join(' ');
                    for (const p of previews) {
                        if (argsStr.includes(p)) {
                            const opt = document.querySelector(`.radio-option[data-value="${p}"]`);
                            if (opt) {
                                document.querySelectorAll('#previewGroup .radio-option').forEach(o => o.classList.remove('selected'));
                                opt.classList.add('selected');
                            }
                            break;
                        }
                    }

                    // Set checkboxes
                    const checkboxArgs = [
                        '--disable-xformers', '--use-pytorch-cross-attention', '--disable-smart-memory',
                        '--dont-upcast-attention', '--use-split-cross-attention', '--disable-all-custom-nodes',
                        '--fast', '--cpu-vae'
                    ];
                    for (const arg of checkboxArgs) {
                        if (savedArgs.includes(arg)) {
                            const opt = document.querySelector(`.checkbox-option[data-value="${arg}"]`);
                            if (opt) {
                                opt.classList.add('checked');
                                opt.querySelector('input').checked = true;
                            }
                        }
                    }

                    document.getElementById('settingsModal').classList.add('active');
                });
        }

        function hideSettings() {
            document.getElementById('settingsModal').classList.remove('active');
        }

        function saveSettings() {
            // Build comfy_args from checkboxes and radio buttons
            let args = [];

            // Get VRAM mode
            const vramMode = document.querySelector('#vramModeGroup .radio-option.selected')?.dataset.value;
            if (vramMode) args.push(vramMode);

            // Get Precision
            const precision = document.querySelector('#precisionGroup .radio-option.selected')?.dataset.value;
            if (precision) args.push(precision);

            // Get Preview method
            const preview = document.querySelector('#previewGroup .radio-option.selected')?.dataset.value;
            if (preview) args.push(preview);

            // Get checkbox options
            document.querySelectorAll('.checkbox-option.checked:not([data-value="auto_launch"])').forEach(opt => {
                const val = opt.dataset.value;
                if (val) args.push(val);
            });

            const data = {
                listen: document.getElementById('settingListen').value,
                port: parseInt(document.getElementById('settingPort').value),
                extra_args: document.getElementById('settingArgs').value,
                auto_launch_browser: document.getElementById('settingAutoLaunch').checked,
                comfy_args: args.join(' ')
            };

            fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            }).then(() => {
                hideSettings();
                showToast('Settings saved! Restart ComfyUI for changes to take effect.');
            });
        }

        // Initialize settings UI handlers
        initSettingsUI();

        // Activity tracker
        let currentGenProgress = {current: 0, total: 0, percent: 0};

        function updateActivities(activities, genProgress) {
            const tracker = document.getElementById('activityTracker');
            const list = document.getElementById('activityList');

            if (genProgress) {
                currentGenProgress = genProgress;
            }

            if (!activities || activities.length === 0) {
                tracker.style.display = 'none';
                return;
            }

            tracker.style.display = 'block';
            list.innerHTML = '';

            for (const activity of activities) {
                const item = document.createElement('div');
                item.className = 'activity-item';

                let iconHtml = '';
                if (activity.status === 'done') {
                    iconHtml = '<div class="activity-icon done">✓</div>';
                } else if (activity.status === 'active') {
                    iconHtml = '<div class="activity-icon active"><div class="activity-spinner"></div></div>';
                } else {
                    iconHtml = '<div class="activity-icon pending">○</div>';
                }

                // Check if this is the Generating activity and show progress bar
                if (activity.name === 'Generating' && activity.status === 'active' && currentGenProgress.total > 0) {
                    item.innerHTML = iconHtml +
                        '<div class="activity-progress-container">' +
                            '<div class="activity-progress-header">' +
                                '<span class="activity-name">Generating</span>' +
                                '<span class="activity-progress-text">' + currentGenProgress.current + '/' + currentGenProgress.total + ' (' + currentGenProgress.percent + '%)</span>' +
                            '</div>' +
                            '<div class="activity-progress-bar">' +
                                '<div class="activity-progress-fill" style="width: ' + currentGenProgress.percent + '%"></div>' +
                            '</div>' +
                        '</div>';
                } else {
                    const nameClass = activity.status === 'done' ? 'activity-name done' : 'activity-name';
                    item.innerHTML = iconHtml + '<span class="' + nameClass + '">' + escapeHtml(activity.name) + '</span>';
                }
                list.appendChild(item);
            }
        }

        // Poll for updates
        function poll() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    updateUI(data.startup_info);
                    updateLogs(data.logs);
                    updateActivities(data.activities, data.generation_progress);
                })
                .catch(() => {});
        }

        setInterval(poll, 500);
        poll();

        // Update modal functions
        let isUpdating = false;
        let updatePollInterval = null;
        let lastUpdateLogCount = 0;

        function showUpdateModal() {
            document.getElementById('updateModal').classList.add('active');
            // Reset state
            document.getElementById('updateOptions').style.display = 'flex';
            document.getElementById('updateConsole').classList.remove('active');
            document.getElementById('updateStatus').classList.remove('active');
            lastUpdateLogCount = 0;
        }

        function hideUpdateModal() {
            if (isUpdating) {
                if (!confirm('Update is still in progress. Close anyway?')) {
                    return;
                }
            }
            document.getElementById('updateModal').classList.remove('active');
            if (updatePollInterval) {
                clearInterval(updatePollInterval);
                updatePollInterval = null;
            }
        }

        function runUpdate(type) {
            if (isRunning) {
                alert('Please stop ComfyUI before updating.');
                return;
            }

            if (isUpdating) {
                alert('Update already in progress.');
                return;
            }

            // Show console and status
            document.getElementById('updateOptions').style.display = 'none';
            document.getElementById('updateConsole').classList.add('active');
            document.getElementById('updateConsole').innerHTML = '';
            document.getElementById('updateStatus').classList.add('active');
            document.getElementById('updateStatusText').textContent = 'Starting update...';
            lastUpdateLogCount = 0;

            fetch('/api/update/' + type, { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    if (!data.success) {
                        alert(data.message);
                        document.getElementById('updateOptions').style.display = 'flex';
                        document.getElementById('updateConsole').classList.remove('active');
                        document.getElementById('updateStatus').classList.remove('active');
                        return;
                    }

                    isUpdating = true;
                    document.getElementById('updateStatusText').textContent = 'Updating...';

                    // Start polling for update status
                    updatePollInterval = setInterval(pollUpdateStatus, 500);
                });
        }

        function pollUpdateStatus() {
            fetch('/api/update/status')
                .then(r => r.json())
                .then(data => {
                    updateUpdateLogs(data.logs);

                    if (!data.is_updating && isUpdating) {
                        // Update finished
                        isUpdating = false;
                        document.getElementById('updateStatus').classList.remove('active');
                        if (updatePollInterval) {
                            clearInterval(updatePollInterval);
                            updatePollInterval = null;
                        }
                        showToast('Update completed!');
                    }
                })
                .catch(() => {});
        }

        function updateUpdateLogs(logs) {
            if (logs.length === lastUpdateLogCount) return;

            const consoleEl = document.getElementById('updateConsole');
            const newLogs = logs.slice(lastUpdateLogCount);

            for (const log of newLogs) {
                const entry = document.createElement('div');
                entry.className = 'log-entry';
                entry.innerHTML = `
                    <span class="log-time">[${log.time}]</span>
                    <span class="log-message ${log.level}">${escapeHtml(log.message)}</span>
                `;
                consoleEl.appendChild(entry);
            }

            lastUpdateLogCount = logs.length;
            consoleEl.scrollTop = consoleEl.scrollHeight;
        }
    </script>
</body>
</html>
'''


class LauncherHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress HTTP logs

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()

            # Fill in template values
            html = HTML_TEMPLATE
            html = html.replace('PORT_PLACEHOLDER', str(settings['port']))
            html = html.replace('LISTEN_PLACEHOLDER', settings['listen'])
            html = html.replace('ARGS_PLACEHOLDER', settings.get('extra_args', ''))
            html = html.replace('AUTOLAUNCH_PLACEHOLDER',
                              'checked' if settings.get('auto_launch_browser', True) else '')
            html = html.replace('AUTOLAUNCH_CLASS',
                              'checked' if settings.get('auto_launch_browser', True) else '')

            self.wfile.write(html.encode())

        elif self.path == '/api/status':
            with log_lock:
                logs_copy = list(log_buffer)
            self.send_json({
                "startup_info": startup_info,
                "logs": logs_copy,
                "is_running": is_running,
                "activities": get_activities(),
                "generation_progress": get_generation_progress()
            })

        elif self.path == '/api/update/status':
            self.send_json(get_update_status())

        elif self.path == '/api/settings':
            self.send_json(settings)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/start':
            result = start_comfyui()
            self.send_json(result)

        elif self.path == '/api/stop':
            result = stop_comfyui()
            self.send_json(result)

        elif self.path == '/api/clear-logs':
            with log_lock:
                log_buffer.clear()
            self.send_json({"success": True})

        elif self.path == '/api/settings':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            new_settings = json.loads(post_data)
            settings.update(new_settings)
            save_settings()
            self.send_json({"success": True})

        elif self.path.startswith('/api/update/'):
            update_type = self.path.split('/')[-1]
            if update_type == 'status':
                self.send_json(get_update_status())
            else:
                result = run_update(update_type)
                self.send_json(result)

        else:
            self.send_response(404)
            self.end_headers()


def find_free_port(start_port):
    """Find a free port starting from start_port"""
    port = start_port
    while port < start_port + 100:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            port += 1
    return start_port


def main():
    load_settings()

    # Find a free port for the launcher
    port = find_free_port(LAUNCHER_PORT)

    server = HTTPServer(('127.0.0.1', port), LauncherHandler)

    url = f"http://127.0.0.1:{port}"
    print(f"ComfyUI Launcher running at {url}")

    # Open browser after a short delay
    def open_browser():
        time.sleep(0.5)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        if is_running:
            stop_comfyui()
        server.shutdown()


if __name__ == "__main__":
    main()
