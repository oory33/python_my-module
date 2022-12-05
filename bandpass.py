import numpy as np
import pyloudnorm as pyln
import math
from scipy.io.wavfile import write


def generate(srate: int, duration: int, bwd: int, centre: int, type: str):
    """
    Generate a Band Pass Noise signal.
    Requires:
      pyloudnorm
      numpy
      math
      scipy.io.wavfile.write

    Parameters
    ----------
    srate : int
        Sampling rate.
    duration : int
        Total duration in seconds.
    bwd : int
        Bandwidth in Hz.
    centre : int
        Centre frequency of bandpass filter in Hz.
    type : str
        Type of signal. Either "Stereo" or "Mono".

    Returns
    -------
    Output signal in wav format at current directory.
    """

    lufs_targ = -14
    meter = pyln.Meter(srate)

    # 周波数をbin数に直す
    total_bin = srate * duration
    nq_bin = int(total_bin / 2)
    bwd_bin = bwd * duration

    # 通過帯域の上限下限のbin番号
    bwdlow_bin = (centre - int(bwd/2)) * duration
    bwdhigh_bin = (centre + int(bwd/2)) * duration

    # 通過帯域内の信号生成
    arg = np.random.normal(0, math.pi, bwd_bin)
    fsig_inbwd = np.ndarray((0, np.size(bwd_bin)))
    for i in arg:
        x = math.cos(i)
        y = math.sin(i)
        num = x + 1j * y
        fsig_inbwd = np.append(fsig_inbwd, num)

    # ゼロ詰
    dc = 0
    btm_zero = np.zeros(bwdlow_bin, dtype=complex)
    top_zero = np.zeros(nq_bin-bwdhigh_bin, dtype=complex)
    fsig_left = np.hstack([dc, btm_zero, fsig_inbwd, top_zero])

    # 複素共役
    fsig_right = np.conj(np.flipud(fsig_left[1:nq_bin]))
    fsig = np.hstack([fsig_left, fsig_right])

    # IFFT
    sig = np.fft.ifft(fsig)

    if type == "Mono":
        lufs_sorc = meter.integrated_loudness(sig)

        sig_n = pyln.normalize.loudness(sig, lufs_sorc, lufs_targ)

        output = np.vstack([sig_n, sig_n])

        write("bandpass.wav", srate, output.T)
    elif type == "Stereo":
        arg = np.random.normal(0, math.pi, bwd_bin)
        for i in arg:
            x = math.cos(i)
            y = math.sin(i)
            num = x + 1j * y
            fsig_inbwd = np.append(fsig_inbwd, num)
        dc = 0
        btm_zero = np.zeros(bwdlow_bin, dtype=complex)
        top_zero = np.zeros(nq_bin-bwdhigh_bin, dtype=complex)
        fsig_left = np.hstack([dc, btm_zero, fsig_inbwd, top_zero])

        # 複素共役
        fsig_right = np.conj(np.flipud(fsig_left[1:nq_bin]))
        fsig_r = np.hstack([fsig_left, fsig_right])

        # IFFT
        sig_r = np.fft.ifft(fsig_r)

        lufs_sorc_l = meter.integrated_loudness(sig)
        lufs_sorc_r = meter.integrated_loudness(sig_r)

        sig_n_l = pyln.normalize.loudness(sig, lufs_sorc_l, lufs_targ)
        sig_n_r = pyln.normalize.loudness(sig_r, lufs_sorc_r, lufs_targ)

        output = np.vstack([sig_n_l, sig_n_r])

        write("bandpass.wav", srate, output.T)
