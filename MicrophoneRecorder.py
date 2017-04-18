# class taken from the SciPy 2015 Vispy talk opening example 
# see https://github.com/vispy/vispy/pull/928
import pyaudio
import threading
import atexit
import numpy as np

class MicrophoneRecorder(object):
    def __init__(self, rate=4000, chunksize=1024):
        self.rate = rate
        self.chunksize = chunksize
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunksize,
                                  stream_callback=self.new_frame)
        self.lock = threading.Lock()
        self.stopped = False
        self.running = False
        self.frames = []
        atexit.register(self.close)

    def new_frame(self, data, frame_count, time_info, status):
        data = np.fromstring(data, 'int16')
        with self.lock:
            self.frames.append(data)
            if self.stopped:
                return None, pyaudio.paComplete
        return None, pyaudio.paContinue
    
    def get_frames(self):
        with self.lock:
            frames = self.frames
            self.frames = []
            return frames
    
    def start(self):
        if self.stream.is_stopped():
            self.stream.start_stream()
            self.running = True

    def stop(self):
        if self.stream.is_active():
            self.stream.stop_stream()
            self.running = False
        
    def close(self):
        with self.lock:
            self.stopped = True
        self.stream.close()
        self.p.terminate()
        
