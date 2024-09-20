from .wavparser import WavCapture
import numpy as np

if __name__ == '__main__':
    wavcapture = WavCapture(
        'target.wav',
        width=1000,
        height=200,
        export_directory='.temporary',
        verbose=True
        )
    
    wavcapture.capture_async('test.png', 0, 1)