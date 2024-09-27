import argparse
from decimal import Decimal
import time
from wavparser import WavCapture
parser = argparse.ArgumentParser()

parser.add_argument('filename')
#parser.add_argument('--mfcc', action='store_true')
parser.add_argument('-c', '--cpu', action='store_true')
parser.add_argument('--legacy', action='store_true')
parser.add_argument('--fast', action='store_true')
parser.add_argument('--st', action='store_true') # single thread

parser.add_argument('-a', '--analyze', action='store_true')
parser.add_argument('-z', '--zoom', type=int, default=1)
parser.add_argument('-O', '--output', default='export')
parser.add_argument('-i', '--interval', type=int, default=1)
parser.add_argument('-o', '--overlap', type=int, default=0)
parser.add_argument('-s', '--start', type=int, default=0)
parser.add_argument('-e', '--end', type=int, default=0)
parser.add_argument('-W', '--width', type=int, default=1000)
parser.add_argument('-H', '--height', type=int, default=280)
parser.add_argument('--waveform', action='store_true')
parser.add_argument('--spectrogram', action='store_true')

args = parser.parse_args()

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

def main():
    print(f'Read {args.filename}')
    
    with Timer('Total'):
        with Timer('Read'):
            wav = WavCapture(args.filename,
                width=args.width,
                height=args.height,
                export_directory=args.output,
                use_gpu=not args.cpu,
                use_legacy=args.legacy,
                fast=args.fast,
                single_thread=args.st
            )
        
        if args.analyze:
            wav.analyze()
            return
        
        pos = Decimal(args.start)
        posEnd = Decimal(args.end if args.end > 0 else wav.duration)
        interval = Decimal(args.interval)
        increment = interval - Decimal(args.overlap)
        
        wav.cut(pos, posEnd)

        with Timer('Resize'):
            wav.resize(args.height*args.zoom)

        if wav.use_legacy:
            print('use Legacy (lib)')
        elif wav.use_cuda:
            print('use CUDA')
        else:
            print('use CPU')
        
        with Timer('Export'):
            while pos < posEnd:
                wav.capture_async(f'waveform{pos}.png', pos, pos+interval)
                pos += increment
            wav.wait()

def test():
    pos = Decimal(args.start)
    posEnd = Decimal(args.end if args.end > 0 else 10)
    interval = Decimal(args.interval if args.interval > 0 else 1)
    increment = interval - Decimal(args.overlap)
    while pos < posEnd:
        print(pos, pos+interval)
        pos += increment
    exit()

if __name__ == '__main__':
    #test()
    try:
        main()
    except KeyboardInterrupt:
        print("Inturrupt")
        exit()