import scipy.io.wavfile
import spafe.utils.vis as vis
from spafe.features.mfcc import mfcc, imfcc
from spafe.features.lfcc import lfcc
import numpy as np

from spafe.utils.converters import hz2mel, hz2bark, hz2erb

def export_spectrogram(filename,
    spectrogram,
    fs,
    xmin,
    xmax,
    ymin,
    ymax,
    dbf=80.0,
    xlabel="Time (s)",
    ylabel="Frequency (Hz)",
    title="Mel spectrogram (dB)",
    figsize=(14, 4),
    cmap="jet",
):
    import matplotlib.pyplot as plt

    # init vars
    amin = 1e-10
    magnitude = np.abs(spectrogram)
    ref_value = np.max(magnitude)

    # compute log spectrum (in dB)
    log_spec = 10.0 * np.log10(
        np.maximum(amin, magnitude) / np.maximum(amin, ref_value)
    )
    log_spec = np.maximum(log_spec, log_spec.max() - dbf)

    # Display the mel spectrogram in dB, seconds, and Hz
    plt.figure(figsize=figsize)
    mSpec_img = plt.imshow(
        log_spec,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        aspect="auto",
        cmap=cmap,
    )
    plt.tight_layout()
    ax = plt.gca()
    ax.set_xticks([])  # x축 눈금 제거
    ax.set_yticks([])  # y축 눈금 제거
    ax.set_xlabel('')  # x축 레이블 제거
    ax.set_ylabel('')  # y축 레이블 제거
    ax.spines['top'].set_visible(False)    # 위쪽 테두리 제거
    ax.spines['right'].set_visible(False)  # 오른쪽 테두리 제거
    ax.spines['left'].set_visible(False)   # 왼쪽 테두리 제거
    ax.spines['bottom'].set_visible(False) # 아래쪽 테두리 제거
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    #plt.show()

def export_mfcc(filename, sig, fs):
    mfccs = mfcc(
        sig,
        fs=fs,
        num_ceps=30,
        nfilts=128,
        low_freq=0,
        high_freq=5000000
    )
    export_spectrogram(filename, mfccs, fs, 0, 100, 0, 200, title="MFCC", ylabel="MFCC Index", xlabel="Frame Index")
    #vis.show_spectrogram(mfccs, fs, 0, 100, 0, 200, title="MFCC", ylabel="MFCC Index", xlabel="Frame Index")
def show_lfcc(sig, fs):
    lfccs = lfcc(
                sig,
                fs=fs,
                num_ceps=30,
                nfilts=128,
                low_freq=0,
                high_freq=5000000
    )
    #vis.show_spectrogram(lfccs, fs, 0, 100, 0, 200, title="LFCC", ylabel="MFCC Index", xlabel="Frame Index")
