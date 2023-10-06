import numpy as np
import pyloudnorm as pyln
from scipy.io.wavfile import write


def Generate(**kwargs):
    """
    Generate Binaural Beat signal with pure tones.
    Requires:
      pyloudnorm
      numpy
      scipy

    Parameters
    ----------
    srate : int
      Sampling rate.
    shift : int
      Shift frequency in Hz.
    duration : int
      Total duration in seconds.
    freq : int
      Frequency of pure tone in Hz.
    file_name : str
      Output file name.(optional)
    wav : bool
      Output wav file or not. If True, output wavfile.(optional)

    Returns
    -------
    Output signal in 32-bit float wav format at current directory.
    """

    lufs_targ = -14
    meter = pyln.Meter(kwargs["srate"])

    # sample数を決定
    length = kwargs["duration"] * kwargs["srate"]

    # 信号を生成
    sig_l = np.sin(np.linspace(0, 2 * np.pi * kwargs["freq"], length))
    sig_r = np.sin(np.linspace(0, 2 * np.pi * (kwargs["freq"] + kwargs["shift"]), length))

    # normalize
    lufs_sorc_l = meter.integrated_loudness(sig_l)
    lufs_sorc_r = meter.integrated_loudness(sig_r)

    sig_l_n = pyln.normalize.loudness(sig_l, lufs_sorc_l, lufs_targ)
    sig_r_n = pyln.normalize.loudness(sig_r, lufs_sorc_r, lufs_targ)

    # 信号を出力
    sig = np.vstack([sig_l_n, sig_r_n])
    if "wav" in kwargs:
        write(kwargs["file_name"], kwargs["srate"], sig.T)
    else:
        return sig.T
