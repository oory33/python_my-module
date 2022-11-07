import numpy as np
import math
import pyloudnorm as pyln
from scipy.io.wavfile import write


def makeBPN(srate, bwd, fcenter, duration):
    fs = srate * duration
    fc = fcenter * duration
    width = bwd * duration
    fdwn = fc - width / 2
    fup = fc + width / 2
    specwid = np.random.normal(size=width) + 1j * \
        np.random.normal(size=width)
    specdwn = np.zeros((1, int(fdwn)), dtype=complex)
    specup = np.zeros((1, int((fs/2) - fup)), dtype=complex)

    spec = np.block([specdwn, specwid, specup])
    specCon = spec.conjugate()
    specCo = np.flipud(specCon)

    spec_BPN = np.block([spec, specCo])
    return np.imag(np.fft.ifft(spec_BPN))


def generate(srate, fcs, bwds, shifts, duration):
    # fs = srate  # sampring rate
    # fc  # center frequency
    # bwd  # bandwidth
    # shift  # shift frequency
    # duration
    fs = srate * duration
    shft_freq = shifts * duration

    lufs_targ = -14
    meter = pyln.Meter(fs)

    sig_BPN = makeBPN(srate, bwds, fcs, duration)
    sig_BPN2 = makeBPN(srate, bwds, fcs, duration)

    # make pure tone
    ln = 2 * math.pi * shft_freq / fs  # 1サンプルの位相の大きさ
    index = np.arange(0, 2 * math.pi * shft_freq - ln / 2, ln,
                      dtype=float)  # 0から信号長分の位相, lnが割り切れない時サンプル長が狂うのでln/2を引く
    shft_index = np.arange((math.pi / 2),
                           2 * math.pi * shft_freq + (math.pi / 2) - ln / 2,
                           ln,
                           dtype=float)

    sin_sig = np.array([math.sin(i) for i in index])
    shft_sin_sig = np.array([math.sin(i) for i in shft_index])

    # modulation
    sig_l = sig_BPN * sin_sig * 2
    shift_sig = sig_BPN2 * shft_sin_sig * 2
    sig_r = (sig_l + shift_sig)

    # normalize
    lufs_sorc_l = meter.integrated_loudness(sig_l.T)
    lufs_sorc_r = meter.integrated_loudness(sig_r.T)

    sig_l_n = pyln.normalize.loudness(sig_l, lufs_sorc_l, lufs_targ)
    sig_r_n = pyln.normalize.loudness(sig_r, lufs_sorc_r, lufs_targ)

    sig = np.block([sig_l_n.T, sig_r_n.T])

    write('oscar.wav', int(fs / duration), sig)
