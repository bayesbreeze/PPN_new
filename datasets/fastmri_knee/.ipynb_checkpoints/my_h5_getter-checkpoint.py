import os
import time
import threading
import queue
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import h5py

class H5FileHandler(FileSystemEventHandler):
    def __init__(self, queue, file_end):
        self.queue = queue
        self.file_end = file_end
        self.current = None

    def on_created(self, event):
        if self.current is not None:
                self.queue.put(self.current)
        if not event.is_directory and event.src_path.endswith(self.file_end):
            self.current = event.src_path
        elif os.path.basename(event.src_path) == "end":
            os.remove(event.src_path)
            self.current = None
            self.queue.put("end")

All_Proccessed = set()
def get_h5_imgs(directory, file_end=".h5", needDel=False):

    if not os.path.exists(directory):
        os.makedirs(directory)

    q = queue.Queue()
    event_handler = H5FileHandler(q, file_end)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=False)
    observer.start()

    h5_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(file_end)]
    # print(h5_files)
    for f in h5_files:
        q.put(f)

    try:
        processed_num = 0
        while True:
            file_path = q.get()
            if file_path == "end":
                print("End file detected.")
                break
            else:
                if file_path in All_Proccessed:
                    continue
                
                yield file_path

                if needDel: 
                    os.remove(file_path)
    finally:
        observer.stop()
    observer.join()

