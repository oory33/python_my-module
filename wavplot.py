import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa
import numpy as np
import time


def make_waveform_pyplot(filename):
    y, sr = librosa.load(filename)
    totaltime = len(y)/sr
    time_array = np.arange(0, totaltime, 1/sr)
    mpl.rcParams['agg.path.chunksize'] = 100000
    fig, ax = plt.subplots()
    formatter = mpl.ticker.FuncFormatter(
        lambda s, x: time.strftime('%M:%S', time.gmtime(s)))
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim(0, totaltime)
    ax.set_xlabel("Time")
    ax.plot(time_array, y)
    plt.show()
