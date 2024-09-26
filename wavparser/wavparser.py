import os, sys
from decimal import Decimal
import librosa
import numpy as np
from scipy.io import wavfile
from PIL import Image, ImageDraw
from multiprocessing import Pool, Process, Queue
from .timer import Timer
from numba import cuda
from . import waveform
import matplotlib.pyplot as plt

def capture_waveform(data, filename, width, height, use_gpu=False, use_legacy=False):
    if use_legacy:
        time_axis =  np.linspace(0, 10, num=len(data))
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, data)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.savefig(filename)
        return
    elif use_gpu:
        pixels = waveform.cuda_make(data, width, height)
    else:
        pixels = waveform.make(data, width, height)
    
    final_image = Image.fromarray(pixels)
    final_image.save(filename)

class WavCapture:
    def __init__(self, filename, *, width, height, export_directory = 'export', verbose=True, use_gpu=False, use_legacy=False):
        sample_rate, data = wavfile.read(filename)
        self.sample_rate = sample_rate
        self.data = data
        self.width = width
        self.height = height
        self.export_directory = export_directory
        self.processes = []
        self.duration = data.shape[0] / self.sample_rate
        self.verbose = verbose

        self.use_legacy = use_legacy

        if self.use_legacy:
            pass

        self.use_cuda = False
        if use_gpu:
            if cuda.is_available():
                self.use_cuda = True
            else:
                sys.stderr.write('CUDA is not available. Using CPU instead.\n')

        if self.verbose:
            print('sample_rate :', sample_rate)
            print('directory :', export_directory)
            print(f'duration : {self.duration}s')
            print()
        os.makedirs(export_directory, exist_ok=True)

    def resize(self, to_height):
        if self.use_cuda:
            self.data = self.__resize_gpu(self.data, to_height)
        else:
            self.data = self.__resize_cpu(self.data, to_height)
    
    def __resize_cpu(self, data, to_height):
        data = np.array(data, dtype=float)
        data *= to_height // 2
        data /= 65535 // 2
        return data.astype(np.int16)
    
    def __resize_gpu(self, data, to_height):
        size = data.shape[0]
        interval = 10000000

        result = np.empty(size, dtype=np.int16)
        localresult = np.empty(interval, dtype=np.int16)

        threads_per_block = 1024
        blocks_per_grid = (interval + threads_per_block) // threads_per_block

        data_device = cuda.to_device(data)
        result_device = cuda.to_device(localresult)
        
        for i in range(0, size, interval):
            endpos = min(i + interval, size)
            kernel_resize[blocks_per_grid, threads_per_block](data_device, i, endpos, to_height, result_device)
            if endpos == size:
                result[i:i+interval] = result_device.copy_to_host()[:size-i]
            else:
                result[i:i+interval] = result_device.copy_to_host()

        del data_device
        del result_device
        return result
        
    def analyze(self):
        print('[Analyze]')
        print(f'Sample Rate : {self.sample_rate}')
        print(f'Length : {len(self.data)}')
        print(f'Duration : {self.duration}')
        print('MIN | MAX')
        print(np.min(self.data), np.max(self.data))
    
    def capture_async(self, filename, start_time, end_time):
        cutdata = self.__cut(start_time, end_time)
        p = Process(target=capture_waveform, args=(cutdata, f'{self.export_directory}\\{filename}', self.width-1, self.height, self.use_cuda, self.use_legacy))
        p.start()
        self.processes.append(p)

    def wait(self):
        for p in self.processes:
            p.join()

    def __cut(self, start_time, end_time):
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        
        return self.data[start_sample:end_sample]

@cuda.jit
def kernel_resize(data, start_pos, end_pos, to_height, result):
    i = cuda.grid(1)
    if start_pos + i < end_pos:
        value = data[start_pos + i]
        value *= to_height // 2
        value /= 65535 // 2

        result[i] = value