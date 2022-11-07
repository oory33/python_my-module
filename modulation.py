import numpy as np
import pyloudnorm as pyln
from scipy.io.wavfile import write


def sinmod(signal, srate: int, freq: float, depth: float):
    """
    Sinosoidal Amplitude Modulation

    Parameters
    ----------
    signal : ndarray()
        shape: (n,2)
        input signal(stereo)
    srate : int
        sampling rate
    freq : float
        modulation frequency(Hz)
    depth : float
        modulation depth(0-1)

    Returns
    -------
    Output signal in stereo wav flie at current directory.
    """
    duration = int(len(signal) / srate)

    lufs_targ = -14
    meter = pyln.Meter(srate)

    alpha = 1
    beta = depth * alpha

    # 正弦波生成
    phi = freq / srate
    index = np.array(range(srate*duration))
    sin_sig = np.zeros(len(index))

    for i in range(len(index)):
        sin_sig[i] = np.sin(2 * np.pi * phi * index[i] + (3/2)*np.pi)

    # AM変調
    mod_sig_l = (alpha + beta * sin_sig) * signal.T[0] * 100

    mod_sig_r = (alpha + beta * sin_sig) * signal.T[1] * 100

    # 正規化
    lufs_sorc_l = meter.integrated_loudness(mod_sig_l)
    lufs_sorc_r = meter.integrated_loudness(mod_sig_r)
    modsig_ln = pyln.normalize.loudness(mod_sig_l, lufs_sorc_l, lufs_targ)
    modsig_rn = pyln.normalize.loudness(mod_sig_r, lufs_sorc_r, lufs_targ)

    mod_sig_n = np.vstack([modsig_ln, modsig_rn])

    write('SAM.wav', srate, mod_sig_n.T)


# raised-cosのOnset/Offset
# def
