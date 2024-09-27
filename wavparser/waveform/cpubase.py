import numpy as np
import math

def make(data, width, height, resize_to):
    tomul = resize_to // 2
    todiv = 65535 // 2
    pivot = height // 2
    n = len(data)
    time = np.linspace(0, width-1, n).astype(int)

    pixels = np.full((height, width+1), 255, dtype=np.uint8)
    
    x = int(time[0])
    y = pivot - data[0] * tomul // todiv
    y = min(max(y, 0), height - 1)
    for i in range(0, len(time), 4):
        px, py = x, y
        x = int(time[i])
        y = pivot - data[i] * tomul // todiv
        y = min(max(y, 0), height - 1)
        pixels[y][x] = 0
        
        miny = min(y, py)
        maxy = max(y, py)
        if px == x:
            for i in range(miny, maxy+1):
                pixels[i, x] = 0
        else:
            midy = int(miny + abs(y-py) / 2)
            for i in range(miny, midy):
                pixels[i, px] = 0
            for i in range(midy, maxy+1):
                pixels[i, x] = 0
    
    return pixels

def make_experiment(data, width, height, resize_to:int):
    tomul = resize_to // 2
    todiv = 65535 // 2
    pivot = height // 2

    n = len(data)
    pixels = np.full((height, width+1), 255, dtype=np.uint8)
    for i in range(0, width):
        px = math.floor(i / width * n)
        x = math.floor((i+1) / width * n)

        sliced = data[px:x]
        maxy = sliced.max() * tomul // todiv
        miny = sliced.min() * tomul // todiv
        for h in range(miny, maxy+1):
            pixels[pivot+h, i] = 0

    return pixels