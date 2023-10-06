import numpy as np
import pyloudnorm as pyln
from scipy.io.wavfile import write


def GenerateNoise(**kwargs):
    """"
    Generate Band Pass Noise signal.
    Requires:
      pyloudnorm
      numpy
      scipy

    Parameters
    ----------
    srate : int
      Sampling rate.
    bwd : int
      Bandwidth in Hz.
    centre : int
      Centre frequency of bandpass filter in Hz.
    duration : int
      Total duration in seconds.
    phase : str
      Phase of noise. Either "same" or "anti" or "normal".
    file_name : str
      Output file name.(optional)
    wav : bool
      Output wav file or not. If True, output wavfile.(optional)
    --------
    Output signal in 32-bit float wav format at current directory.
    """
    lufs_targ = -14
    meter = pyln.Meter(kwargs["srate"])

    if "file_name" in kwargs:
        file_name = kwargs["file_name"]
    else:
        file_name = "%s.wav" % kwargs["phase"]

    # 周波数をbin数に直す
    total_bin = kwargs["srate"] * kwargs["duration"]
    nq_bin = int(total_bin / 2)
    bwd_bin = kwargs["bwd"] * kwargs["duration"]

    # 通過帯域の上限下限のbin番号
    bwdlow_bin = (kwargs["centre"] - int(kwargs["bwd"]/2)) * kwargs["duration"]
    bwdhigh_bin = (kwargs["centre"] + int(kwargs["bwd"]/2)
                   ) * kwargs["duration"]

    ## ---信号生成---##
    fsig_inbwd = np.random.normal(size=bwd_bin) + 1j * \
        np.random.normal(size=bwd_bin)

    # ゼロ詰め
    dc = 0
    btm_zero = np.zeros(bwdlow_bin, dtype=complex)
    top_zero = np.zeros(nq_bin - bwdhigh_bin, dtype=complex)
    fsig_left = np.hstack([dc, btm_zero, fsig_inbwd, top_zero])

    # 複素共役
    fsig_right = np.conj(np.flipud(fsig_left[1:nq_bin]))
    fsig = np.hstack([fsig_left, fsig_right])

    # IFFT
    tsig = np.real(np.fft.ifft(fsig))*100

    # Rチャンネル
    if kwargs["phase"] == "same":
        tsig_r = tsig
    elif kwargs["phase"] == "anti":
        tsig_r = -tsig
    elif kwargs["phase"] == "normal":
        fsig_r_inbwd = np.random.normal(size=bwd_bin) + 1j * \
            np.random.normal(size=bwd_bin)
        fsig_r = np.hstack([dc, btm_zero, fsig_r_inbwd, top_zero])
        tsig_r = np.real(np.fft.ifft(fsig_r))*100

    # cast
    tsig = tsig.astype(np.float32)
    tsig_r = tsig_r.astype(np.float32)

    # normalize
    lufs_sorc_l = meter.integrated_loudness(tsig)
    lufs_sorc_r = meter.integrated_loudness(tsig_r)

    tsig_n = pyln.normalize.loudness(tsig, lufs_sorc_l, lufs_targ)
    tsig_r_n = pyln.normalize.loudness(tsig_r, lufs_sorc_r, lufs_targ)

    sig = np.vstack([tsig_n, tsig_r_n])

    if "wav" in kwargs:
        write(file_name, kwargs["srate"], sig.T)
    else:
        return sig.T


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
    sig_l = np.sin(np.linspace(
        0, 2 * np.pi * kwargs["freq"] * kwargs["duration"], length))
    sig_r = np.sin(np.linspace(
        0, 2 * np.pi * (kwargs["freq"] + kwargs["shift"]) * kwargs["duration"], length))

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
