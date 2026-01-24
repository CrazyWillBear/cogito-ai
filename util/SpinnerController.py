import itertools
import sys
import threading
import time

class SpinnerController:
    def __init__(self, delay=0.3):
        self.text = ""
        self.delay = delay
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.frames = itertools.cycle(["⏳", "⏳", "⌛", "⌛"])
        self.ellipses_frames = itertools.cycle([".", "..", "..."])
        self.thread = None

    def start(self, text=""):
        # Reset stop event for a fresh run (important for restart after stop)
        if self.thread is None or not self.thread.is_alive():
            self.stop = threading.Event()
        self.text = text

        def run():
            max_len = 0
            while not self.stop.is_set():
                with self.lock:
                    msg = self.text
                frame = next(self.frames)
                dots = next(self.ellipses_frames)

                # build visible part (everything after \r)
                visible = f"{frame}{msg}{dots}"
                max_len = max(max_len, len(visible))

                # pad with spaces so shorter messages fully overwrite older ones
                padded = visible.ljust(max_len)
                sys.stdout.write("\r" + padded)
                sys.stdout.flush()
                time.sleep(self.delay)

            # erase entire line that was ever printed
            sys.stdout.write("\r" + (" " * max_len) + "\r")
            sys.stdout.flush()

        # If already running, don't spawn a new thread, just update text
        if self.thread is not None and self.thread.is_alive():
            return

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def set_text(self, text):
        with self.lock:
            self.text = text

    def stop_spinner(self):
        # Signal stop and wait for the thread to finish
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        # After stopping, clear thread reference so start() can relaunch later
        self.thread = None
