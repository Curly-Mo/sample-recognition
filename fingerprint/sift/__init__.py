import logging

import numpy as np
import seaborn
import vlfeat
import librosa


seaborn.set(style='ticks')
seaborn.set_context("paper")
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


class Match(object):
    def __init__(self, query, train, distance, distance2):
        self.query = query
        self.train = train
        self.distance = distance
        self.d2 = distance2


class Model(object):
    def __init__(self, matcher, keypoints, settings, spectrograms=None):
        self.matcher = matcher
        self.keypoints = keypoints
        self.settings = settings
        self.spectrograms = spectrograms


class Sift(object):
    def __init__(self, audio_path, sr, settings, implementation='vlfeat'):
        self.audio_path = audio_path
        self.sr = sr
        self.settings = settings
        print(settings)
        kp, desc, S = self.sift_file(**settings)
        self.keypoints = kp
        self.descriptors = desc
        self.spectrogram = S

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
        #S = librosa.core.spectrum.stft(y, win_length=2048, hop_length=hop_length)
        #S = librosa.feature.spectral.logfsgram(y, sr=sr, n_fft=2048, hop_length=hop_length, bins_per_octave=octave_bins)
        return S

    def chromagram(self, y, hop_length=512, n_fft=1024, n_chroma=12):
        S = librosa.feature.chroma_stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            n_chroma=n_chroma
        )
        return S

    def sift_spectrogram(self, S, id, height, **kwargs):
        #I = np.flipud(S)
        logger.info('{}: Extracting SIFT keypoints...'.format(id))
        keypoints, descriptors = self.sift(S, **kwargs)
        keypoints, descriptors = keypoints.T, descriptors.T
        logger.info('{}: {} keypoints found!'.format(id, len(keypoints)))
        keypoints, descriptors = self.remove_edge_keypoints(keypoints, descriptors, S, height)

        logger.info('{}: Creating keypoint objects...'.format(id))
        keypoint_objs = []
        for keypoint, descriptor in zip(keypoints, descriptors):
            keypoint_objs.append(Keypoint(*keypoint, source=id, descriptor=descriptor))

        return keypoint_objs, descriptors

    def sift_file(self, hop_length=512, octave_bins=24, n_octaves=8, fmin=40,
                  **kwargs):
        logger.info('{}: Loading signal into memory...'.format(self.audio_path))
        y, sr = librosa.load(self.audio_path, sr=self.sr)
        #logger.info('{}: Trimming silence...'.format(audio_path))
        #y = np.concatenate([[0], np.trim_zeros(y), [0]])
        logger.info('{}: Generating Spectrogram...'.format(self.audio_path))
        S = self.cqtgram(
            y,
            hop_length=hop_length,
            octave_bins=octave_bins,
            n_octaves=n_octaves,
            fmin=fmin
        )
        #S = chromagram(y, hop_length=256, n_fft=4096, n_chroma=36)
        keypoints, descriptors = self.sift_spectrogram(
            S,
            self.audio_path,
            octave_bins*n_octaves,
            **kwargs
        )
        return keypoints, descriptors, S

    def remove_edge_keypoints(self, keypoints, descriptors, S, height):
        logger.info('Removing edge keypoints...')
        min_value = np.min(S)
        start = next((index for index, frame in enumerate(S.T) if sum(value > min_value for value in frame) > height/2), 0)
        end = S.shape[1] - next((index for index, frame in enumerate(reversed(S.T)) if sum(value > min_value for value in frame) > height/2), 0)
        start = start + 10
        end = end - 10
        out_kp = []
        out_desc = []
        for keypoint, descriptor in zip(keypoints, descriptors):
            # Skip keypoints on the left and right edges of spectrogram
            if start < keypoint[0] < end:
                out_kp.append(keypoint)
                out_desc.append(descriptor)
        logger.info('Edge keypoints removed: {}, remaining: {}'.format(len(keypoints)-len(out_kp), len(out_kp)))
        return out_kp, out_desc

    def sift(self, S, contrast_thresh=0.1, edge_thresh=100, levels=3, magnif=3,
             window_size=2, first_octave=0, **kwargs):
        # Scale to 0-255
        I = 255 - (S - S.min()) / (S.max() - S.min()) * 255
        keypoints, descriptors = vlfeat.vl_sift(
            I.astype(np.float32),
            peak_thresh=contrast_thresh,
            edge_thresh=edge_thresh,
            magnif=magnif,
            window_size=window_size,
            first_octave=first_octave,
        )
        # Add each keypoint orientation back to descriptors
        # This effectively removes rotation invariance
        # TODO: Not sure yet if this is a good idea
        # descriptors = (descriptors + [kp[3] for kp in keypoints.T])
        return keypoints, descriptors
