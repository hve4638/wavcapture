import time

class Timer:
    def __init__(self, messgage='Time'):
        self.message = messgage
    
    def show(self, message='Time'):
        t = time.time() - self.start
        print(f'{message} : {t:.6f}s')
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, type, value, traceback):
        self.show(self.message)
    
    @property
    def time(self):
        return self.end - self.start