"""
Universal Smart Vision (USV) - Native Desktop Application Shell
Packages the local FastAPI + WebGL/Canvas AI application into a native desktop window.
Supports native Chromium/Edge app-mode and PySide6 / PyQt6 WebEngine fallback.
Dept. of CSE-AIML, D.Y. Patil College of Engineering & Technology, Kolhapur.
"""

import sys
import os
import time
import subprocess
import threading
import webbrowser
from pathlib import Path

PORT = 8000
SERVER_URL = f"http://127.0.0.1:{PORT}"

def wait_for_server(timeout=15.0):
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/api/system/status", timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.4)
    return False

def start_backend_server():
    import uvicorn
    from src.api.main import app
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")

def launch_desktop_window():
    # Attempt 1: Native Edge Application Mode (Windows standard)
    edge_paths = [
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%LocalAppData%\Microsoft\Edge\Application\msedge.exe")
    ]
    
    # Attempt 2: Chrome Application Mode
    chrome_paths = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe")
    ]

    for browser_bin in edge_paths + chrome_paths:
        if os.path.exists(browser_bin):
            print(f"[Desktop Launcher] Launching standalone desktop application via: {browser_bin}")
            args = [
                browser_bin,
                f"--app={SERVER_URL}",
                "--window-size=1400,900",
                "--app-name=Universal Smart Vision",
                "--disable-features=Translate"
            ]
            subprocess.Popen(args)
            return

    # Fallback to default browser
    print("[Desktop Launcher] Standalone browser binary not located, opening default browser...")
    webbrowser.open(SERVER_URL)

def main():
    print("=" * 70)
    print("   UNIVERSAL SMART VISION (USV) — DESKTOP APPLICATION SHELL")
    print("   Dept. of CSE-AIML • D.Y. Patil College of Engineering & Technology")
    print("=" * 70)

    # Start backend server in a separate background thread
    server_thread = threading.Thread(target=start_backend_server, daemon=True)
    server_thread.start()

    print("[Desktop Launcher] Initializing Universal Smart Vision AI platform...")
    if wait_for_server():
        print(f"[Desktop Launcher] Platform operational at {SERVER_URL}. Spawning desktop window...")
        launch_desktop_window()
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("[Desktop Launcher] Terminating Universal Smart Vision desktop instance.")
    else:
        print("[Desktop Launcher] Error: Server failed to respond within timeout window.")

if __name__ == "__main__":
    main()

