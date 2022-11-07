import numpy as np
import pyloudnorm as pyln
from scipy.io.wavfile import write
import math


def generate(srate, fc_i, bwd_i, shft_freq_i, duration):
    fs = srate * duration
    fc = fc_i * duration
    bwd = bwd_i * duration
    shft_freq = shft_freq_i * duration

    fn = fs / 2

    lufs_targ = -14
    meter = pyln.Meter(fs)

    fdwn = fc - bwd / 2
    fup = fc + bwd / 2

    specwid = np.random.normal(size=bwd) + 1j * np.random.normal(size=bwd)
    specdwn = np.zeros((1, int(fdwn)), dtype=complex)
    specup = np.zeros((1, int(fn - fup)), dtype=complex)

    spec = np.block([specdwn, specwid, specup])
    specCo = spec.conjugate()
    specCon = np.flipud(specCo)

    spec_base = np.block([spec, specCon])
    sig_base = np.imag(np.fft.ifft(spec_base))

    # make cos curve
    ln = 2 * math.pi * shft_freq / fs  # 1sampleの位相の大きさ

    index = np.arange(0, 2 * math.pi * shft_freq - ln/2, ln, dtype=float)
    sh_index = np.arange(math.pi,  math.pi+2 * math.pi * shft_freq - ln/2, ln)

    cos_sig_l = [(math.cos(i)+1)/2 for i in index]
    cos_sig_r = [(math.cos(i)+1)/2 for i in sh_index]

    sig_l = sig_base * cos_sig_l * 10
    sig_r = sig_base * cos_sig_r * 10

    # normalize
    lufs_sorc_l = meter.integrated_loudness(sig_l.T)
    lufs_sorc_r = meter.integrated_loudness(sig_r.T)

    sig_l_n = pyln.normalize.loudness(sig_l, lufs_sorc_l, lufs_targ)
    sig_r_n = pyln.normalize.loudness(sig_r, lufs_sorc_r, lufs_targ)

    sig = np.block([sig_l_n.T, sig_r_n.T])
    write('ILD.wav', int(fs / duration), sig)
