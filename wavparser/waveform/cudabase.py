import numpy as np
from numba import cuda

@cuda.jit
def draw_pixel(data, time, pixels, height):
    index = cuda.grid(1) * 2
    if index < len(data):
        px = int(time[index])
        py = int(height // 2 - data[index+1])
        py = min(max(py, 0), height-1)

        x = int(time[index])
        y = int(height // 2 - data[index])
        y = min(max(y, 0), height-1)
        
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

def cuda_make(data, width, height):
    n = len(data)
    time = np.linspace(0, width - 1, n).astype(int)

    data_device = cuda.to_device(data)
    time_device = cuda.to_device(time)
    
    pixels = np.full((height, width+1), 255, dtype=np.uint8)
    pixels_device = cuda.to_device(pixels)
    
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block) // threads_per_block
    draw_pixel[blocks_per_grid, threads_per_block](data_device, time_device, pixels_device, height)
    
    pixels_device.copy_to_host(pixels)
    return pixels