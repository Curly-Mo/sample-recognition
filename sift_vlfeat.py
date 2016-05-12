#!/usr/bin/env python
import os
import itertools
import datetime
import logging
import logging.config
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import seaborn
import vlfeat
from vlfeat.plotop.vl_plotframe import vl_plotframe
import librosa
from sklearn.externals import joblib

import ann

seaborn.set(style='ticks')
seaborn.set_context("paper")
logger = logging.getLogger(__name__)
logging.config.fileConfig('logging.ini', disable_existing_loggers=False)


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
    def __init__(self, matcher, keypoints, settings, spectrograms=None):
        self.matcher = matcher
        self.keypoints = keypoints
        self.settings = settings
        self.spectrograms = spectrograms


def cqtgram(y, hop_length=512, octave_bins=24, n_octaves=8, fmin=40, sr=22050, perceptual_weighting=False):
    S_complex = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=octave_bins,
        n_bins=octave_bins*n_octaves,
        fmin=fmin,
        real=False
    )
    S = np.abs(S_complex)
    if perceptual_weighting:
        freqs = librosa.cqt_frequencies(S.shape[0], fmin=fmin, bins_per_octave=octave_bins)
        S = librosa.perceptual_weighting(S**2, freqs, ref_power=np.max)
    else:
        S = librosa.logamplitude(S**2, ref_power=np.max)
    #S = librosa.core.spectrum.stft(y, win_length=2048, hop_length=hop_length)
    #S = librosa.feature.spectral.logfsgram(y, sr=sr, n_fft=2048, hop_length=hop_length, bins_per_octave=octave_bins)
    return S


def chromagram(y, hop_length=512, n_fft=1024, n_chroma=12):
    S = librosa.feature.chroma_stft(y, n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma)
    return S


def sift_spectrogram(S, id, height, **kwargs):
    #I = np.flipud(S)
    logger.info('{}: Extracting SIFT keypoints...'.format(id))
    keypoints, descriptors = sift(S, **kwargs)
    keypoints, descriptors = keypoints.T, descriptors.T
    logger.info('{}: {} keypoints found!'.format(id, len(keypoints)))
    keypoints, descriptors = remove_edge_keypoints(keypoints, descriptors, S, height)

    logger.info('{}: Creating keypoint objects...'.format(id))
    keypoint_objs = []
    for keypoint, descriptor in zip(keypoints, descriptors):
        keypoint_objs.append(Keypoint(*keypoint, source=id, descriptor=descriptor))

    return keypoint_objs, descriptors


def sift_file(audio_path, hop_length, octave_bins=24, n_octaves=8, fmin=40, sr=22050, **kwargs):
    logger.info('{}: Loading signal into memory...'.format(audio_path))
    y, sr = librosa.load(audio_path, sr=sr)
    #logger.info('{}: Trimming silence...'.format(audio_path))
    #y = np.concatenate([[0], np.trim_zeros(y), [0]])
    logger.info('{}: Generating Spectrogram...'.format(audio_path))
    S = cqtgram(y, hop_length=hop_length, octave_bins=octave_bins, n_octaves=n_octaves, fmin=fmin, sr=sr)
    #S = chromagram(y, hop_length=256, n_fft=4096, n_chroma=36)
    keypoints, descriptors = sift_spectrogram(S, audio_path, octave_bins*n_octaves, **kwargs)
    return keypoints, descriptors, S


def remove_edge_keypoints(keypoints, descriptors, S, height):
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


def sift(S, contrast_thresh=0.1, edge_thresh=100, levels=3, magnif=3, window_size=2, first_octave=0):
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


def sift_match(path1, path2, hop_length, abs_thresh=None, ratio_thresh=None, octave_bins=24, n_octaves=8, fmin=40, cluster_dist=1.5, cluster_size=3, sr=22050):
    # Extract keypoints
    kp1, desc1, S1 = sift_file(path1, hop_length, octave_bins=octave_bins, n_octaves=n_octaves, fmin=fmin, sr=sr)
    kp2, desc2, S2 = sift_file(path2, hop_length, octave_bins=octave_bins, n_octaves=n_octaves, fmin=fmin, sr=sr)

    # Plot keypoint images
    fig = plt.figure()
    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()
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
    cluster_dist = int((sr/ hop_length) * cluster_dist)
    matches = filter_matches(matches, abs_thresh, ratio_thresh, cluster_dist, cluster_size)

    # Draw matching lines
    plot_matches(ax1, ax2, matches)
    plt.tight_layout()
    plt.show(block=False)
    return matches


def filter_matches(matches, abs_thresh=None, ratio_thresh=None, cluster_dist=20, cluster_size=1, match_orientation=True):
    logger.info('Filtering nearest neighbors down to actual matched samples')
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
    # Only keep when there are multiple within a time cluster
    clusters = list(cluster_matches(matches, cluster_dist))
    filtered_clusters = [cluster for cluster in clusters if len(cluster) >= cluster_size]
    logger.info('Total Clusters: {}, filtered clusters: {}'.format(len(clusters), len(filtered_clusters)))
    matches = [match for cluster in filtered_clusters for match in cluster]
    logger.info('Filtered matches: {}'.format(len(matches)))
    return filtered_clusters


def cluster_matches(matches, cluster_dist):
    logger.info('Clustering matches...')
    logger.info('cluster_dist: {}'.format(cluster_dist))
    matches = sorted(matches, key=lambda m: (m.train.source, m.query.x))
    clusters = {}
    for source, group in itertools.groupby(matches, lambda m: m.train.source):
        for match in group:
            cluster_found = False
            for cluster in clusters.get(source, []):
                if (
                  (abs(match.query.x - cluster[0].query.x) <= cluster_dist or
                   abs(match.query.x - cluster[-1].query.x) <= cluster_dist) and
                  (abs(match.train.x - cluster[0].train.x) <= cluster_dist or
                   abs(match.train.x - cluster[-1].train.x) <= cluster_dist)
                ):
                    cluster_found = True
                    cluster.append(match)
            if not cluster_found:
                clusters.setdefault(source, []).append([match])
    return [cluster for sources in clusters.values() for cluster in sources]


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
    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()
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
        logger.info('Drawing ALL keypoints (this may take some time)...')
        for plot in source_plots:
            frames = np.array([kp.kp for kp in model.keypoints if kp.source == plot])
            plt.axes(source_plots[plot])
            vl_plotframe(frames.T, color='y', linewidth=1)


def plot_clusters(S, clusters, model, title, plot_all_kp=False):
    """Draw matches across axes"""
    fig = plt.figure()
    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()
    if len(clusters) == 0:
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
    cols = len({cluster[0].train.source for cluster in clusters})
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
    colors = itertools.cycle('bgrck')
    source_plots = {}
    for cluster in clusters:
        color = next(colors)
        for match in cluster:
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
                    xticks=40/cols
                )
                ax2.set_zorder(-1)
                source_plots[match.train.source] = ax2
            con = ConnectionPatch(
                xyA=(match.query.x, match.query.y), xyB=(match.train.x, match.train.y),
                coordsA='data', coordsB='data',
                axesA=ax1, axesB=ax2,
                arrowstyle='<-', linewidth=1,
                zorder=999, color=color
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
    #plt.tight_layout()
    #plt.subplots_adjust(wspace=0.2, hspace=0.2)


def plot_keypoints(keypoints, color='y', linewidth=1):
    for keypoint in keypoints:
        c = plt.Circle((keypoint.x, keypoint.y), keypoint.scale, color=color)
        ax = plt.gca()
        ax.add_artist(c)


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


def train_keypoints(audio_paths, hop_length, octave_bins=24, n_octaves=7, fmin=50, sr=22050, algorithm='lshf', dedupe=False, save=None, **kwargs):
    settings = locals().copy()
    del settings['audio_paths']
    logger.info('Settings: {}'.format(settings))
    spectrograms = {}
    keypoints = []
    descriptors = []
    if isinstance(audio_paths, str) and os.path.isdir(audio_paths):
        audio_paths = [os.path.join(audio_paths, f) for f in
                       os.listdir(audio_paths) if
                       os.path.isfile(os.path.join(audio_paths, f))]
    for audio_path in audio_paths:
        kp, desc, S = sift_file(audio_path, hop_length, octave_bins, n_octaves, fmin, sr=sr, **kwargs)
        if dedupe:
            # Remove duplicate keypoints (important for ratio thresholding if source track has exact repeated segments)
            kp, desc = remove_similar_keypoints(kp, desc)

        spectrograms[audio_path] = S
        keypoints.extend(kp)
        descriptors.extend(desc)

    matcher = ann.train_matcher(descriptors, algorithm=algorithm)
    model = Model(matcher, keypoints, settings, spectrograms)
    if save:
        logger.info('Saving model to disk... ({})'.format(save))
        joblib.dump(model, save, compress=True)
    return model


def remove_similar_keypoints(kp, desc):
    if len(desc) > 0:
        logger.info('Removing duplicate/similar keypoints...')
        a = np.array(desc)
        rounding_factor = 10
        b = np.ascontiguousarray((a//rounding_factor)*rounding_factor).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        _, idx = np.unique(b, return_index=True)
        desc = a[sorted(idx)]
        kp = [k for i, k in enumerate(kp) if i in idx]
        logger.info('Removed {} duplicate keypoints'.format(a.shape[0] - idx.shape[0]))
    return kp, desc


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
    distances, indices = ann.find_neighbors(model.matcher, desc, algorithm=model.settings['algorithm'], k=2)

    # Build match  objects
    logger.info('Building match objects')
    matches = []
    for i, distance in enumerate(distances):
        matches.append(Match(kp[i], model.keypoints[indices[i][0]], distance[0], distance[1]))
    return matches, S


def query_track(audio_path, model, abs_thresh=None, ratio_thresh=None, cluster_dist=1.0, cluster_size=1, plot=True, plot_all_kp=False, save=True):
    if isinstance(model, str):
        logger.info('Loading model into memory: {}'.format(model))
        model = joblib.load(model)
    logger.info('Settings: {}'.format(model.settings))
    matches, S = find_matches(audio_path, model)

    cluster_dist = int((model.settings['sr'] / model.settings['hop_length']) * cluster_dist)
    clusters = filter_matches(matches, abs_thresh, ratio_thresh, cluster_dist, cluster_size)

    # Plot keypoint images and Draw matching lines
    if plot:
        plot_clusters(S, clusters, model, audio_path, plot_all_kp)
        plt.show(block=False)
    if save:
        if not plot:
            plot_clusters(S, clusters, model, audio_path, plot_all_kp)
        plt.savefig('{}_{}.svg'.format(os.path.join('plots', os.path.basename(audio_path)), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), format='svg', figsize=(1920, 1080), bbox_inches=None)

    display_results(clusters, model.settings)
    return clusters, matches


def query_tracks(audio_paths, model, abs_thresh=None, ratio_thresh=None, cluster_dist=1.0, cluster_size=1, plot=True, plot_all_kp=False, save=True):
    kwargs = locals().copy()
    kwargs.pop('audio_paths', None)
    kwargs.pop('model', None)
    if isinstance(audio_paths, str) and os.path.isdir(audio_paths):
        audio_paths = [os.path.join(audio_paths, f) for f in
                       os.listdir(audio_paths) if
                       os.path.isfile(os.path.join(audio_paths, f))]
    for track in audio_paths:
        yield query_track(track, model, **kwargs)


def display_results(clusters, settings):
    if clusters:
        print('{} sampled from:'.format(clusters[0][0].query.source))
        sources = {}
        for cluster in clusters:
            if cluster[0].train.source in sources:
                sources[cluster[0].train.source].append(cluster)
            else:
                sources[cluster[0].train.source] = [cluster]

        for source in sources:
            print('{} at '.format(source), end='')
            times = (str(int(match[0].train.x * settings['hop_length'] / settings['sr'])) for match in sources[source])
            print(', '.join(times))
    print('\n')


def main():
    print('main')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a binaural stereo wav file from a mono audio file."
    )
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print debug messages to stdout')
    subparsers = parser.add_subparsers(dest='command', help='command')
    subparsers.required = True
    # train
    train = subparsers.add_parser('train', help='train a new model')
    train.add_argument('audio_paths', type=str, nargs='+',
                       help='Either a directory or list of files')
    train.add_argument('--hop_length', type=int, default=256,
                       help='Hop length for computing CQTgram')
    train.add_argument('--octave_bins', type=int, default=36,
                       help='Number of bins per octave of CQT')
    train.add_argument('--n_octaves', type=int, default=7,
                       help='Number of octaves for CQT')
    train.add_argument('--fmin', type=int, default=50,
                       help='Starting frequency of CQT in Hz')
    train.add_argument('--sr', type=int, default=22050,
                       help='Sampling frequency (audio will be resampled)')
    train.add_argument('--algorithm', type=str, default='lshf',
                       help='Approximate nearest neighbor algorithm')
    train.add_argument('--dedupe', type=bool, default=True,
                       help='Remove similar keypoints per track')
    train.add_argument('--contrast_thresh', type=float, default=5,
                       help='Contrast threshold for SIFT detector')
    train.add_argument('--save', type=str, default=None,
                       help='Location to save model to disk')
    # query
    train = subparsers.add_parser('query', help='query tracks')
    train.add_argument('audio_paths', type=str, nargs='+',
                       help='Either a directory or list of files')
    train.add_argument('model', type=str,
                       help='Location of saved model to query against')
    train.add_argument('--abs_thresh', type=float, default=None,
                       help='Absolute threshold for filtering matches')
    train.add_argument('--ratio_thresh', type=float, default=None,
                       help='Ratio threshold for filtering matches')
    train.add_argument('--cluster_dist', type=float, default=1.0,
                       help='Time in seconds for clustering matches')
    train.add_argument('--cluster_size', type=float, default=3,
                       help='Minimum cluster size to be considered an instance of sampling')
    train.add_argument('--plot', type=bool, default=True,
                       help='Plot results')
    train.add_argument('--plot_all_kp', type=bool, default=False,
                       help='Plot all keypoints on spectrograms')
    train.add_argument('--save', type=bool, default=False,
                       help='Save plot')
    args = parser.parse_args()

    import logging.config
    logging.config.fileConfig('logging.ini', disable_existing_loggers=False)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose debugging activated")
    del args.verbose

    if len(args.audio_paths) == 1 and os.path.isdir(args.audio_paths[0]):
        args.audio_paths = args.audio_paths[0]

    if args.command == 'train':
        del args.command
        model = train_keypoints(**vars(args))
    if args.command == 'query':
        del args.command
        results = query_tracks(**vars(args))
        for clusters, matches in results:
            pass
        if args.plot:
            plt.show(block=True)
