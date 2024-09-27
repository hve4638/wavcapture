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

def capture_cpu(data, filename, width, height, fast):
    with Timer('capture_waveform: ' + filename, silence=True):
        if fast:
            pixels = waveform.make_experiment(data, width, height, fast)
        else:
            pixels = waveform.make(data, width, height, fast)
        final_image = Image.fromarray(pixels)
        final_image.save(filename)