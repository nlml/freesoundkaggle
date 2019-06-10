import os
from pathlib import Path
from scipy.io import wavfile
from fastprogress import progress_bar
import tensorflow as tf
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


class SpectGetter:

    def signal_to_melspect(self, signals, *args, **kwargs):
        stfts = tf.contrib.signal.stft(
            signals,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length
        )

        magnitude_spectrograms = tf.abs(stfts)
        self.num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
        self.linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            self.num_mel_bins, self.num_spectrogram_bins, 44100, self.lower_edge_hertz,
            self.upper_edge_hertz)

        mel_spectrograms = tf.tensordot(
            magnitude_spectrograms, self.linear_to_mel_weight_matrix, 1)

        return mel_spectrograms

    def __init__(self,
                 frame_length=343 * 2,
                 frame_step=343,
                 fft_length=1024,
                 lower_edge_hertz=50,
                 upper_edge_hertz=9000,
                 num_mel_bins=128,
                 *args, **kwargs):
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.lower_edge_hertz, self.upper_edge_hertz, self.num_mel_bins = (
            lower_edge_hertz, upper_edge_hertz, num_mel_bins)
        # Warp the linear-scale, magnitude spectrograms into the mel-scale.


def path_to_logmelspect(audio_file, config):
    sg = SpectGetter(**config)
    sr, samples = wavfile.read(audio_file)
    sig = tf.reshape((samples / 32768).astype(np.float32), [1, -1])
    S = sg.signal_to_melspect(sig).numpy()
    S = np.maximum(-80., 10 * np.log10(S + 1e-80) - 10 * 3.4)
    S = (S + 80.) / 80.
    S = np.transpose(S, [0, 2, 1])  # [1, freqbins, time]
    S = (S * 255).astype(np.uint8)
    whr = np.where(S[0].mean(0) > 0)[0]
    if len(whr) > 128:
        S = S[:, :, whr[0]:whr[-1]]
    return S


def path_to_im_fn_tf(wavpath, config):
    return path_to_logmelspect(wavpath, config)
