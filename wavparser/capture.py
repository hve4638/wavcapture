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

def capture_cpu(data, filename, width, height, resize_to, fast):
    with Timer('capture_waveform: ' + filename, silence=True):
        if fast:
            pixels = waveform.make_experiment(data, width, height, resize_to)
        else:
            pixels = waveform.make(data, width, height, resize_to)
        final_image = Image.fromarray(pixels)
        final_image.save(filename)

def capture_matplotlib(data, filename, width, height, resize_to, fast):
    print('matplotlib')
    with Timer('capture_waveform: ' + filename, silence=True):
        time_axis =  np.linspace(0, 10, num=len(data))
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, data)
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks([])
        plt.yticks([])
        plt.grid()
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
