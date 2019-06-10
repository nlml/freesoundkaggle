import numpy as np
import librosa
from copy import deepcopy
import torch
from scipy import signal
import scipy.io.wavfile as wavfile
from freesound.data_utils.audio_transforms_tf import path_to_im_fn_tf


def fixdiv(v):
    if np.isnan(v) or v == 0:
        return 1.
    else:
        return v


def read_wav(wavpath):
    sample_rate, x = wavfile.read(wavpath)
    a = np.float32(x / 32768.)
    a = hp_filter(a)
    return np.clip(a / fixdiv(np.percentile(np.abs(a), 99.95)), 0, 1).astype(np.float32)


def mu_law_transform(audio, mu=10):
    return np.sign(audio) * np.log(1 + mu * np.abs(audio)) / np.log(1 + mu)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def compress(audio, window_in_ms=1000, mu=0, clip_pct=99.95):
    wo2 = window_in_ms // 2
    moving_avg = moving_average(np.abs(np.lib.pad(audio, (wo2, wo2 - 1), 'reflect')), window_in_ms)
    factor = moving_avg / fixdiv(np.percentile(moving_avg, 98))
    factor[factor < 1] = 1
    # Apply compression
    audio = audio / factor
    # Renormalize
    if clip_pct:
        audio = np.clip(audio / fixdiv(np.percentile(np.abs(audio), clip_pct)), -1.0, 1.0)
    else:
        audio = np.clip(audio / np.max(np.abs(audio)), -1.0, 1.0)
    if mu:
        audio = mu_law_transform(audio, mu)
    return audio


def hp_filter(audio, sr=44100):
    b = signal.firwin(101, cutoff=60, fs=sr, pass_zero=False)
    return signal.lfilter(b, [1.0], audio)


def read_and_compress_wav(wavpath, window_in_ms=1000, mu=0, clip_pct=99.95):
    audio = read_wav(wavpath)
    return compress(audio, window_in_ms, mu, clip_pct)


def crop_or_pad_to_size_of(a, size_from):
    diff = size_from.shape[0] - a.shape[0]
    out = a
    if diff > 0:  # pad
        if len(a) > 44100 * 4:  # more than 4s: repeat it
            out = np.concatenate([a] * ((len(size_from) // len(a)) + 1))[:len(size_from)]
        else:
            out = np.zeros(size_from.shape[0])
            st = len(a) // 2
            out[st:st + len(a)] = a
    elif diff < 0:              # crop
        st = np.random.randint(0, len(a) - len(size_from))
        out = a[st:st + len(size_from)]
    return out


def log_mel_spectrogram_as_img(audio, mel_transform):
    S_tch = mel_transform(torch.FloatTensor(audio).unsqueeze(0))
    S_tch = 10 * torch.log10(torch.clamp(S_tch / torch.max(S_tch), 1e-8, 1.0))
    # librosa.display.specshow(S_tch[0].numpy().T)
    return (255 * (1 + S_tch / 80.))[0].numpy().T.astype(np.uint8)


def wav_to_mel_spectrogram(signal, config=None):
    _config = dict(
        sr=44100,
        n_mels=128,
        hop_length=694,
        fmin=20,
    )
    if config is not None:
        for k in _config:
            if k in config:
                _config[k] = config[k]
    _config['n_fft'] = config['n_mels'] * config['n_fft_multiplier']
    _config['fmax'] = config['sr'] // 2
    S = librosa.feature.melspectrogram(signal, **_config)
    if config.get('normalise', True):
        S = librosa.power_to_db(S, ref=np.max)
        div = (S.max() - S.min())
        div = 1 if div == 0 else div
        S = (S - S.min()) / div
    else:
        S = np.maximum(-80., 10 * np.log10(S + 1e-80) - 10 * 3.4)
        S = (S + 80.) / 80.
    return (S * 255).astype(np.uint8)


def read_wavfile(path, sr):
    signal, sr = librosa.load(path, sr=sr)
    return signal


def path_to_im_fn_mel(wavpath, config):
    signal = read_wavfile(wavpath, config['sr'])
    spectro = wav_to_mel_spectrogram(signal, config)
    return np.expand_dims(spectro, 0)


def path_to_im_fn_mel_compress(wavpath, config):
    signal = compress(read_wav(wavpath))
    spectro = wav_to_mel_spectrogram(signal, config)
    return np.expand_dims(spectro, 0)


def path_to_im_fn_mel_and_splice(wavpath, config):
    spectro = path_to_im_fn_mel(wavpath, config)
    return spectro.reshape(config['splice_factor'], -1, spectro.shape[2])


def path_to_im_fn_mel_and_splice_multi(wavpath, config):
    spectro = path_to_im_fn_mel(wavpath, config)
    spectro = spectro.reshape(config['splice_factor'], -1, spectro.shape[2])
    config_ = deepcopy(config)
    config_['n_fft_multiplier'] = config_['n_fft_multiplier'] * 2
    config_['n_mels'] = config_['n_mels'] // config_['splice_factor']
    spectro_basic = path_to_im_fn_mel(wavpath, config_)
    config_['normalise'] = False
    spectro_soft = path_to_im_fn_mel(wavpath, config_)
    return np.concatenate([spectro, spectro_basic, spectro_soft], 0)


path_to_im_fns = {
    'to_mel': path_to_im_fn_mel,
    'tf': path_to_im_fn_tf,
    'to_mel_compress': path_to_im_fn_mel_compress,
    'to_mel_and_splice': path_to_im_fn_mel_and_splice,
    'to_mel_and_splice_multi': path_to_im_fn_mel_and_splice_multi,
    'JUST_MELS': lambda x: x
}
