import logging

import numpy as np
import librosa

from .. import Fingerprint


logger = logging.getLogger(__name__)


class Keypoint(object):
    def __init__(self, x, y, scale=None, orientation=None, source=None,
                 descriptor=None):
        self.x = x
        self.y = y
        self.scale = scale
        self.orientation = orientation
        self.source = source
        self.descriptor = descriptor

    @property
    def kp(self):
        return np.array([
            self.x,
            self.y,
            self.scale,
            self.orientation
        ])


def from_file(audio_path, sr, settings, implementation='vlfeat'):
    if implementation == 'vlfeat':
        from .vlfeat import Sift_vlfeat
        return Sift_vlfeat(audio_path, sr, settings)
    else:
        raise NotImplementedError(implementation)


class Sift(Fingerprint):
    def __init__(self, audio_path, sr, settings, implementation='vlfeat'):
        self.audio_path = audio_path
        self.sr = sr
        self.settings = settings
        kp, desc, S = self.sift_file(**settings)
        self.keypoints = kp
        self.descriptors = desc
        self.spectrogram = S

    def sift_spectrogram(self, S, id, height, **kwargs):
        raise NotImplementedError

    def sift_file(self, hop_length=512, octave_bins=24, n_octaves=8, fmin=40,
                  **kwargs):
        logger.info(
            '{}: Loading signal into memory...'.format(self.audio_path)
        )
        y, sr = librosa.load(self.audio_path, sr=self.sr)
        # logger.info('{}: Trimming silence...'.format(audio_path))
        # y = np.concatenate([[0], np.trim_zeros(y), [0]])
        logger.info('{}: Generating Spectrogram...'.format(self.audio_path))
        S = self.cqtgram(
            y,
            hop_length=hop_length,
            octave_bins=octave_bins,
            n_octaves=n_octaves,
            fmin=fmin
        )
        # S = chromagram(y, hop_length=256, n_fft=4096, n_chroma=36)
        keypoints, descriptors = self.sift_spectrogram(
            S,
            self.audio_path,
            octave_bins*n_octaves,
            **kwargs
        )
        return keypoints, descriptors, S

    def cqtgram(self, y, hop_length=512, octave_bins=24, n_octaves=8, fmin=40,
                perceptual_weighting=False):
        S_complex = librosa.cqt(
            y,
            sr=self.sr,
            hop_length=hop_length,
            bins_per_octave=octave_bins,
            n_bins=octave_bins*n_octaves,
            fmin=fmin,
            real=False
        )
        S = np.abs(S_complex)
        if perceptual_weighting:
            freqs = librosa.cqt_frequencies(
                S.shape[0],
                fmin=fmin,
                bins_per_octave=octave_bins
            )
            S = librosa.perceptual_weighting(S**2, freqs, ref_power=np.max)
        else:
            S = librosa.logamplitude(S**2, ref_power=np.max)
        return S

    def chromagram(self, y, hop_length=512, n_fft=1024, n_chroma=12):
        S = librosa.feature.chroma_stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            n_chroma=n_chroma
        )
        return S

    def remove_edge_keypoints(self, keypoints, descriptors, S, height):
        logger.info('Removing edge keypoints...')
        min_value = np.min(S)
        start = next(
            (index for index, frame in enumerate(S.T)
             if sum(value > min_value for value in frame) > height/2), 0)
        end = S.shape[1] - next(
            (index for index, frame in enumerate(reversed(S.T))
             if sum(value > min_value for value in frame) > height/2), 0)
        start = start + 10
        end = end - 10
        out_kp = []
        out_desc = []
        for keypoint, descriptor in zip(keypoints, descriptors):
            # Skip keypoints on the left and right edges of spectrogram
            if start < keypoint[0] < end:
                out_kp.append(keypoint)
                out_desc.append(descriptor)
        logger.info('Edge keypoints removed: {}, remaining: {}'.format(
            len(keypoints)-len(out_kp), len(out_kp))
        )
        return out_kp, out_desc
