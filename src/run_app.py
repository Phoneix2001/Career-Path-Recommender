import subprocess
import sys
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os

class AppReloader(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.start_app()

    def start_app(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
        
        # Start Streamlit app
        self.process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "frontend/app.py"],
            cwd=os.path.dirname(__file__)
        )

    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            print(f"Detected change in {event.src_path}")
            self.start_app()

def main():
    path = os.path.dirname(__file__)
    event_handler = AppReloader()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        if event_handler.process:
            event_handler.process.terminate()
    observer.join()

if __name__ == "__main__":
    main() 