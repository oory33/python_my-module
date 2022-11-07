import numpy as np
import pyloudnorm as pyln
from scipy.io.wavfile import write


def generate(srate: int, delay: int, duration: int, move_to: str):
    """
    Generate a Phase-delayed signal.

    Parameters
    ----------
    srate : int
        Sampling rate in Hz.
    delay : int
        Delay in milliseconds.
    duration : int
        Total duration in seconds.
    move_to : str
        Direction of move. Either "left" or "right".

    Returns
    -------
    Output signal in wav format at current directory.
    """
    if move_to == "right":
        ud = 1
    else:
        ud = -1

    lufs_targ = -14
    meter = pyln.Meter(srate)

    nq_bin = int(srate * duration / 2)

    fsig = np.random.normal(size=int(srate*duration / 2)) + 1j * \
        np.random.normal(size=int(srate*duration / 2))

    fshift = np.zeros(fsig.size, dtype=complex)

    # nuber of bin to freq

    def b2f(nbin):
        return nbin / duration

    # delay to freq and shift
    for i in range(len(fsig)):
        fshift[i] = fsig[i] * \
            np.exp(1j * 2 * np.pi * b2f(i) * delay * ud / 1000)

    # 複素共役
    fsig_right = np.conj(np.flipud(fsig[1:nq_bin]))
    fsig_sum = np.hstack([fsig, fsig_right])
    fshift_right = np.conj(np.flipud(fshift[1:nq_bin]))
    fshift_sum = np.hstack([fshift, fshift_right])

    # IFFT
    tsig = np.real(np.fft.ifft(fsig_sum))*100
    tshift = np.real(np.fft.ifft(fshift_sum))*100

    lufs_sorc_l = meter.integrated_loudness(tsig)
    lufs_sorc_r = meter.integrated_loudness(tshift)

    tsig_n = pyln.normalize.loudness(tsig, lufs_sorc_l, lufs_targ)
    tshift_n = pyln.normalize.loudness(tshift, lufs_sorc_r, lufs_targ)

    sig = np.vstack([tsig_n, tshift_n])

    write('phase_delay.wav', srate, sig.T)
