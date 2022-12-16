import numpy as np
import pyloudnorm as pyln
from scipy.io.wavfile import write


def generate(srate: int, shift: int, duration: int, bwd: int, centre: int, init_direction: str):
    """
    Generate a Akeroyd signal.
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
    bwd : int
        Bandwidth in Hz.
    centre : int
        Centre frequency of bandpass filter in Hz.
    init_direction : str
        Initial direction of shift. Either "left" or "right".

    Returns
    -------
    Output signal in wav format at current directory.
    """
    if init_direction == "left":
        ud = -1
    elif init_direction == "right":
        ud = 1

    lufs_targ = -14
    meter = pyln.Meter(srate)

    # 周波数をbin数に直す
    total_bin = srate * duration
    nq_bin = int(total_bin / 2)
    shift_bin = shift * duration
    bwd_bin = bwd * duration

    # 通過帯域の上限下限のbin番号
    bwdlow_bin = (centre - int(bwd/2)) * duration
    bwdhigh_bin = (centre + int(bwd/2)) * duration

    # 通過帯域内の信号生成
    fsig_inbwd = np.random.normal(size=bwd_bin) + 1j * \
        np.random.normal(size=bwd_bin)

    # ゼロ詰
    dc = 0
    btm_zero = np.zeros(bwdlow_bin, dtype=complex)
    top_zero = np.zeros(nq_bin - bwdhigh_bin, dtype=complex)
    fsig_left = np.hstack([dc, btm_zero, fsig_inbwd, top_zero])
    # 複素共役
    fsig_right = np.conj(np.flipud(fsig_left[1:nq_bin]))
    fsig = np.hstack([fsig_left, fsig_right])

    # shfit
    shft_bwdlow_bin = bwdlow_bin + int(ud * shift_bin)
    shft_bwdhigh_bin = bwdhigh_bin + int(ud * shift_bin)

    # ゼロ詰
    btm_zero = np.zeros(shft_bwdlow_bin, dtype=complex)
    top_zero = np.zeros(int(total_bin/2) - shft_bwdhigh_bin, dtype=complex)
    fshift_left = np.hstack([dc, btm_zero, fsig_inbwd, top_zero])
    fshift_right = np.conj(np.flipud(fshift_left[1:nq_bin]))
    # 複素共役
    fshift = np.hstack([fshift_left, fshift_right])

    # IFFT
    tsig = np.real(np.fft.ifft(fsig))*100
    tshift = np.real(np.fft.ifft(fshift))*100

    # normalize
    lufs_sorc_l = meter.integrated_loudness(tsig)
    lufs_sorc_r = meter.integrated_loudness(tshift)

    tsig_n = pyln.normalize.loudness(tsig, lufs_sorc_l, lufs_targ)
    tshift_n = pyln.normalize.loudness(tshift, lufs_sorc_r, lufs_targ)

    sig = np.vstack([tsig_n, tshift_n])

    write('akeroyd.wav', srate, sig.T)
