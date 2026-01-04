#!/usr/bin/env python
"""
Startup script to run both backend and frontend servers
Run this once and both will stay running
"""

import subprocess
import time
import sys
import os
import webbrowser

def main():
    print("="*60)
    print("STARTING DOCUMENT CLASSIFIER SERVERS")
    print("="*60)
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(script_dir, "backend")
    frontend_dir = os.path.join(script_dir, "frontend")
    
    print(f"\nBackend directory: {backend_dir}")
    print(f"Frontend directory: {frontend_dir}")
    
    # Start backend
    print("\n[1/2] Starting backend server on port 5000...")
    backend_cmd = [sys.executable, "app.py"]
    backend_process = subprocess.Popen(
        backend_cmd,
        cwd=backend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    time.sleep(3)
    
    if backend_process.poll() is not None:
        print("[ERROR] Backend failed to start!")
        return False
    
    print("[OK] Backend started!")
    
    # Start frontend
    print("\n[2/2] Starting frontend server on port 8000...")
    frontend_cmd = [sys.executable, "-m", "http.server", "8000"]
    frontend_process = subprocess.Popen(
        frontend_cmd,
        cwd=frontend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    time.sleep(2)
    
    if frontend_process.poll() is not None:
        print("[ERROR] Frontend failed to start!")
        return False
    
    print("[OK] Frontend started!")
    
    # Status
    print("\n" + "="*60)
    print("SERVERS RUNNING")
    print("="*60)
    print("\nFrontend: http://127.0.0.1:8000/index_new.html")
    print("Backend:  http://127.0.0.1:5000/api/health")
    print("\nBoth servers are running. Keep this window open.")
    print("Press Ctrl+C to stop both servers.")
    print("="*60 + "\n")
    
    # Open browser automatically
    print("Opening browser...")
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:8000/index_new.html")
    
    # Keep running forever
    try:
        # Just wait indefinitely for both processes
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n\nShutting down servers...")
        try:
            backend_process.terminate()
            frontend_process.terminate()
            time.sleep(2)
        except:
            pass
        try:
            backend_process.kill()
            frontend_process.kill()
        except:
            pass
        print("Servers stopped.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
