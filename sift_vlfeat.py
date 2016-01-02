import os
import itertools
import datetime
import logging
if not __name__ in logging.Logger.manager.loggerDict:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s :-: %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import seaborn
seaborn.set(style='ticks')
#seaborn.set_context("paper")
import vlfeat
from vlfeat.plotop.vl_plotframe import vl_plotframe
import librosa
from sklearn.externals import joblib

import ann


class Keypoint(object):
    def __init__(self, x, y, scale=None, orientation=None, source=None, descriptor=None):
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
    def __init__(self, clf, keypoints, settings, spectrograms=None):
        self.clf = clf
        self.keypoints = keypoints
        self.settings = settings
        self.spectrograms = spectrograms


def cqtgram(y, hop_length=512, octave_bins=24, n_octaves=8, fmin=40, sr=22050):
    S = librosa.core.constantq.cqt(y, sr=sr, hop_length=hop_length, bins_per_octave=octave_bins, n_bins=octave_bins*n_octaves, fmin=fmin)
    S = librosa.logamplitude(S, ref_power=np.max)
    #S = librosa.core.spectrum.stft(y, win_length=2048, hop_length=hop_length)
    #S = librosa.feature.spectral.logfsgram(y, sr=sr, n_fft=2048, hop_length=hop_length, bins_per_octave=octave_bins)
    return S


def sift_spectrogram(S, id, **kwargs):
    #I = np.flipud(S)
    logger.info('{}: Extracting SIFT keypoints...'.format(id))
    keypoints, descriptors = sift(S, **kwargs)
    keypoints, descriptors = remove_edge_keypoints(keypoints, descriptors, S)
    logger.info('{}: {} keypoints found!'.format(id, len(keypoints)))

    logger.info('{}: Creating keypoint objects...'.format(id))
    keypoint_objs = []
    for keypoint, descriptor in itertools.izip(keypoints, descriptors):
        keypoint_objs.append(Keypoint(*keypoint, source=id, descriptor=descriptor))

    return keypoint_objs, descriptors


def sift_file(audio_path, hop_length, octave_bins=24, n_octaves=8, fmin=40, sr=22050, **kwargs):
    logger.info('{}: Loading signal into memory...'.format(audio_path))
    y, sr = librosa.load(audio_path, sr=sr)
    logger.info('{}: Generating Spectrogram...'.format(audio_path))
    S = cqtgram(y, hop_length=hop_length, octave_bins=octave_bins, n_octaves=n_octaves, fmin=fmin, sr=sr)
    keypoints, descriptors = sift_spectrogram(S, audio_path, **kwargs)
    return keypoints, descriptors, S


def remove_edge_keypoints(keypoints, descriptors, S):
    start = next((index for index, frame in enumerate(S.T) if all(value > -80 for value in frame)), 0) + 8
    end = S.shape[1] - next((index for index, frame in enumerate(reversed(S.T)) if all(value > -80 for value in frame)), S.shape[1]) - 8
    out_kp = []
    out_desc = []
    for keypoint, descriptor in itertools.izip(keypoints.T, descriptors.T):
        # Skip keypoints on the left and right edges of spectrogram
        if start <= keypoint[0] <= end:
            out_kp.append(keypoint)
            out_desc.append(descriptor)
    return out_kp, out_desc


def sift(S, contrast_thresh=0.1, edge_thresh=100, levels=3, magnif=3, window_size=2):
    # Scale to 0-255
    I = 255 - (S - S.min()) / (S.max() - S.min()) * 255
    keypoints, descriptors = vlfeat.vl_sift(
        I.astype(np.float32),
        peak_thresh=contrast_thresh,
        edge_thresh=edge_thresh,
        magnif=magnif,
        window_size=window_size,
    )
    # Add each keypoint orientation back to descriptors
    # This effectively removes rotation invariance
    # TODO: Not sure yet if this is a good idea
    #descriptors = (descriptors + [kp[3] for kp in keypoints.T])
    return keypoints, descriptors


def sift_match(path1, path2, hop_length, abs_thresh=None, ratio_thresh=None, octave_bins=24, n_octaves=8, fmin=40, timeframe=1.5, cluster_size=3, sr=22050):
    # Extract keypoints
    kp1, desc1, S1 = sift_file(path1, hop_length, octave_bins=octave_bins, n_octaves=n_octaves, fmin=fmin, sr=sr)
    kp2, desc2, S2 = sift_file(path2, hop_length, octave_bins=octave_bins, n_octaves=n_octaves, fmin=fmin, sr=sr)

    # Plot keypoint images
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    ax1 = fig.add_subplot(2, 1, 1)
    plot_spectrogram(S1, hop_length, octave_bins, fmin, path1, sr=sr)
    vl_plotframe(np.array([kp.kp for kp in kp1]).T, color='y', linewidth=1)
    ax2 = fig.add_subplot(2, 1, 2)
    plot_spectrogram(S2, hop_length, octave_bins, fmin, path2, sr=sr)
    vl_plotframe(np.array([kp.kp for kp in kp2]).T, color='y', linewidth=1)

    # ANN
    distances, indices = ann.nearest_neighbors(desc1, desc2, k=2)

    # Build match  objects
    logger.info('Building match objects')
    matches = []
    for i, distance in enumerate(distances):
        matches.append(Match(kp1[i], kp2[indices[i][0]], distance[0], distance[1]))

    # Filter nearest neighbors
    logger.info('Filtering nearest neighbors down to actual matched samples')
    timeframe = int((sr/ hop_length) * timeframe)
    matches = filter_matches(matches, abs_thresh, ratio_thresh, timeframe, cluster_size)

    # Draw matching lines
    plot_matches(ax1, ax2, matches)
    plt.show(block=False)
    return matches


def filter_matches(matches, abs_thresh=None, ratio_thresh=None, timeframe=20, cluster_size=1, match_orientation=True):
    logger.info('Filtering nearest neighbors down to actual matched samples')
    filtered = []
    if match_orientation:
        # Remove matches with differing orientations
        total = len(matches)
        matches = [match for match in matches if abs(match.query.orientation - match.train.orientation) < 0.2]
        logger.info('Differing orientations removed: {}, remaining: {}'.format(total-len(matches), len(matches)))
    if abs_thresh:
        # Apply absolute threshold
        total = len(matches)
        matches = [match for match in matches if match.distance < abs_thresh]
        logger.info('Absolute threshold removed: {}, remaining: {}'.format(total-len(matches), len(matches)))
    if ratio_thresh:
        # Apply ratio test
        total = len(matches)
        matches = [match for match in matches if match.distance < ratio_thresh*match.d2]
        logger.info('Ratio threshold removed: {}, remaining: {}'.format(total-len(matches), len(matches)))
    # Only keep when there are multiple within the timeframe
    for match in matches:
        relevant = [m for m in matches if m.train.source == match.train.source]
        cluster = [m for m in relevant if match.query.x < m.query.x < match.query.x+timeframe
                     and match.train.x < m.train.x < match.train.x+timeframe]
        if len(cluster) >= cluster_size - 1:
            filtered.append(match)
    logger.info('Clustering removed: {}, remaining: {}'.format(len(matches)-len(filtered), len(filtered)))
    return filtered


def plot_matches(ax1, ax2, matches):
    """Draw matches across axes"""
    logger.info('Drawing lines between matches')
    for match in matches:
        con = ConnectionPatch(
            xyA=(match.query.x, match.query.y), xyB=(match.train.x, match.train.y),
            coordsA='data', coordsB='data',
            axesA=ax1, axesB=ax2,
            arrowstyle='<-', linewidth=1,
            zorder=999
        )
        ax1.add_artist(con)
    ax2.set_zorder(-1)


def plot_all_matches(S, matches, model, title, plot_all_kp=False):
    """Draw matches across axes"""
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    if len(matches) == 0:
        logger.info('No matches found')
        plot_spectrogram(
            S,
            model.settings['hop_length'],
            model.settings['octave_bins'],
            model.settings['fmin'],
            title,
            sr=model.settings['sr'],
            cbar=True
        )
        return
    rows = 2.0
    cols = len({match.train.source for match in matches})
    ax1 = fig.add_subplot(rows, cols, (1, cols))
    plot_spectrogram(
        S,
        model.settings['hop_length'],
        model.settings['octave_bins'],
        model.settings['fmin'],
        title,
        sr=model.settings['sr'],
        cbar=True
    )

    logger.info('Drawing lines between matches')
    source_plots = {}
    for match in matches:
        ax2 = source_plots.get(match.train.source, None)
        if ax2 is None:
            ax2 = fig.add_subplot(rows, cols, cols + len(source_plots) + 1)
            plot_spectrogram(
                model.spectrograms[match.train.source],
                model.settings['hop_length'],
                model.settings['octave_bins'],
                model.settings['fmin'],
                match.train.source,
                sr=model.settings['sr'],
                xticks=20/cols
            )
            ax2.set_zorder(-1)
            source_plots[match.train.source] = ax2
        con = ConnectionPatch(
            xyA=(match.query.x, match.query.y), xyB=(match.train.x, match.train.y),
            coordsA='data', coordsB='data',
            axesA=ax1, axesB=ax2,
            arrowstyle='<-', linewidth=1,
            zorder=999
        )
        ax1.add_artist(con)
        if not plot_all_kp:
            # Plot keypoints
            plt.axes(ax1)
            vl_plotframe(np.matrix(match.query.kp).T, color='y', linewidth=1)
            plt.axes(ax2)
            vl_plotframe(np.matrix(match.train.kp).T, color='y', linewidth=1)
    if plot_all_kp:
        for plot in source_plots:
            frames = np.array([kp.kp for kp in model.keypoints if kp.source == plot])
            plt.axes(source_plots[plot])
            vl_plotframe(frames.T, color='y', linewidth=1)


def plot_spectrogram(S, hop_length, octave_bins, fmin, title, sr=22050, xticks=20, yticks=10, cbar=False):
    # Plot Spectrogram
    librosa.display.specshow(
        S,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=octave_bins,
        fmin=fmin,
        x_axis='time',
        y_axis='cqt_hz',
        n_xticks=xticks,
        n_yticks=yticks,
    )
    plt.title(title)
    if cbar:
        plt.colorbar(format='%+2.0f dB')


def train_keypoints(audio_paths, hop_length, octave_bins=24, n_octaves=9, fmin=40, sr=22050, save=None, **kwargs):
    settings = locals()
    settings.pop('audio_paths', None)
    spectrograms = {}
    keypoints = []
    descriptors = []
    for audio_path in audio_paths:
        kp, desc, S = sift_file(audio_path, hop_length, octave_bins, n_octaves, fmin, sr=sr, **kwargs)
        # Remove duplicate keypoints (important for ratio thresholding if source track has exact repeated segments)
        a = np.array(desc)
        b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        _, idx = np.unique(b, return_index=True)
        desc = a[sorted(idx)]
        kp = [k for i, k  in enumerate(kp) if i in idx]
        logger.info('Removed {} duplicate keypoints'.format(a.shape[0] - idx.shape[0]))

        spectrograms[audio_path] = S
        keypoints.extend(kp)
        descriptors.extend(desc)

    clf = ann.fit_ann(descriptors)
    model = Model(clf, keypoints, settings, spectrograms)
    if save:
        logger.info('Saving model to disk...')
        joblib.dump(model, save, compress=True)
    return model


def find_matches(audio_path, model):
    # Extract keypoints
    kp, desc, S = sift_file(
        audio_path,
        model.settings['hop_length'],
        model.settings['octave_bins'],
        model.settings['n_octaves'],
        model.settings['fmin'],
        sr=model.settings['sr'],
        **model.settings['kwargs']
    )

    # Find (approximate) nearest neighbors
    distances, indices = ann.find_neighbors(model.clf, desc, k=2)

    # Build match  objects
    logger.info('Building match objects')
    matches = []
    for i, distance in enumerate(distances):
        matches.append(Match(kp[i], model.keypoints[indices[i][0]], distance[0], distance[1]))
    return matches, S


def query_track(audio_path, model, abs_thresh=None, ratio_thresh=None, timeframe=1.0, cluster_size=1, plot=True, save=True):
    matches, S = find_matches(audio_path, model)

    timeframe = int((model.settings['sr'] / model.settings['hop_length']) * timeframe)
    matches = filter_matches(matches, abs_thresh, ratio_thresh, timeframe, cluster_size)
    logger.info('Matches found: {}'.format(len(matches)))

    # Plot keypoint images and Draw matching lines
    if plot:
        plot_all_matches(S, matches, model, audio_path)
        plt.show(block=False)
    if save:
        if not plot:
            plot_all_matches(S, matches, model, audio_path)
        plt.savefig('{}_{}.pdf'.format(os.path.join('plots', os.path.basename(audio_path)), datetime.datetime.now()), bbox_inches='tight')

    return matches
