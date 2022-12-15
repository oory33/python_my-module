import numpy as np
import pyloudnorm as pyln


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
    Output signal in ndarray() .
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

    mod_sig_n = np.vstack([modsig_ln, modsig_rn]).T

    return mod_sig_n


# raised-cosのOnset/Offset
def RaisedCos(signal, srate: int, beta: float, length: float):
    """
    Raised-cosine window. Cuttoff would be half of the Nyquist frequency.

    Parameters
    ----------
    signal : ndarray()
        shape: (n,2)
        input signal(stereo)
    srate : int
        sampling rate in Hz.
    beta : float
        window parameter.
    length : float
        window length in miliseconds.

    Returns
    -------
    Output signal in ndarray().
    """
    lufs_targ = -14
    meter = pyln.Meter(srate)

    window_length = int(srate * length / 1000)  # in bins
    T = 1 / window_length

    a_1 = int((1-beta)/(2 * T))
    a_2 = int((1+beta)/(2 * T))

    H_f = np.zeros(window_length, dtype=float)

    for i in range(a_1):
        H_f[i] = 1

    for i in range(a_1, a_2):
        H_f[i] = 0.5 * (1 + np.cos((np.pi*T/beta) * (i - a_1)))

    H_on = np.conj(np.flip(H_f))

    sig_l = signal.T[0]
    sig_r = signal.T[1]

    sig_l_raise = sig_l[0:window_length] * H_on
    sig_l_mid = sig_l[window_length:len(sig_l)-window_length]
    sig_l_release = sig_l[len(sig_l)-window_length:len(sig_l)] * H_f

    sig_r_raise = sig_r[0:window_length] * H_on
    sig_r_mid = sig_r[window_length:len(sig_r)-window_length]
    sig_r_release = sig_r[len(sig_r)-window_length:len(sig_r)] * H_f

    sig_l = np.hstack([sig_l_raise, sig_l_mid, sig_l_release])
    sig_r = np.hstack([sig_r_raise, sig_r_mid, sig_r_release])

    out = np.vstack([sig_l, sig_r]).T

    return out
