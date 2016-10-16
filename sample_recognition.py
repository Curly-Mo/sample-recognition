#!/usr/bin/env python
import os
import itertools
import datetime
import logging
import logging.config
import argparse
from distutils.util import strtobool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import seaborn
import joblib
import librosa

import fingerprint
import ann
import util.tracks


seaborn.set(style='ticks')
seaborn.set_context("paper")
logger = logging.getLogger(__name__)
logging.config.fileConfig('logging.ini', disable_existing_loggers=False)


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


def filter_matches(matches, abs_thresh=None, ratio_thresh=None,
                   cluster_dist=20, cluster_size=1, match_orientation=True,
                   ordered=False):
    logger.info('Filtering nearest neighbors down to actual matched samples')
    if match_orientation:
        # Remove matches with differing orientations
        total = len(matches)
        matches = [
            match for match in matches if
            abs(match.query.orientation - match.train.orientation) < 0.2
        ]
        logger.info('Differing orientations removed: {}, remaining: {}'.format(
            total-len(matches), len(matches))
        )
    if abs_thresh:
        # Apply absolute threshold
        total = len(matches)
        matches = [match for match in matches if match.distance < abs_thresh]
        logger.info('Absolute threshold removed: {}, remaining: {}'.format(
            total-len(matches), len(matches))
        )
    if ratio_thresh:
        # Apply ratio test
        total = len(matches)
        matches = [
            match for match in matches if
            match.distance < ratio_thresh*match.d2
        ]
        logger.info('Ratio threshold removed: {}, remaining: {}'.format(
            total-len(matches), len(matches))
        )
    # Only keep when there are multiple within a time cluster
    clusters = list(cluster_matches(matches, cluster_dist))
    filtered_clusters = [
        cluster for cluster in clusters if len(cluster) >= cluster_size
    ]
    logger.info('Total Clusters: {}, filtered clusters: {}'.format(
        len(clusters), len(filtered_clusters))
    )
    if ordered:
        orderedx_clusters = []
        ordered_clusters = []
        for cluster in filtered_clusters:
            sorted_trainx = sorted(cluster, key=lambda m: m.train.x)
            sorted_queryx = sorted(cluster, key=lambda m: m.query.x)
            if sorted_trainx == sorted_queryx:
                orderedx_clusters.append(cluster)
        logger.info('Total Clusters: {}, orderedx clusters: {}'.format(
            len(clusters), len(orderedx_clusters))
        )
        for cluster in orderedx_clusters:
            sorted_trainy = sorted(cluster, key=lambda m: m.train.y)
            sorted_queryy = sorted(cluster, key=lambda m: m.query.y)
            if sorted_trainy == sorted_queryy:
                ordered_clusters.append(cluster)
        logger.info('Total Clusters: {}, ordered clusters: {}'.format(
            len(clusters), len(ordered_clusters))
        )
        filtered_clusters = ordered_clusters
    matches = [match for cluster in filtered_clusters for match in cluster]
    logger.info('Filtered matches: {}'.format(len(matches)))
    return filtered_clusters


def cluster_matches(matches, cluster_dist):
    class Cluster(object):
        def __init__(self, match):
            self.min_query = match.query.x
            self.max_query = match.query.x
            self.min_train = match.train.x
            self.max_train = match.train.x
            self.matches = [match]

        def add(self, match):
            if match.query.x > self.min_query:
                self.min_query = match.query.x
            if match.query.x > self.max_query:
                self.max_query = match.query.x
            if match.train.x < self.min_train:
                self.min_train = match.train.x
            if match.train.x > self.max_train:
                self.max_train = match.train.x
            self.matches.append(match)

        def merge(self, cluster):
            if cluster.min_query < self.min_query:
                self.min_query = cluster.min_query
            if cluster.max_query > self.max_query:
                self.max_query = cluster.max_query
            if cluster.min_train < self.min_train:
                self.min_train = cluster.min_train
            if cluster.max_train > self.max_train:
                self.max_train = cluster.max_train
            self.matches.extend(cluster.matches)

    logger.info('Clustering matches...')
    logger.info('cluster_dist: {}'.format(cluster_dist))
    matches = sorted(matches, key=lambda m: (m.train.source, m.query.x))
    clusters = {}
    for source, group in itertools.groupby(matches, lambda m: m.train.source):
        for match in group:
            cluster_found = False
            for cluster in clusters.get(source, []):
                if (
                  (match.query.x >= cluster.min_query - cluster_dist and
                   match.query.x <= cluster.max_query + cluster_dist) and
                  (match.train.x >= cluster.min_train - cluster_dist and
                   match.train.x <= cluster.max_train + cluster_dist)
                ):
                    if not any(match.train.x == c.train.x and
                               match.train.y == c.train.y
                               for c in cluster.matches):
                        cluster_found = True
                        cluster.add(match)
            if not cluster_found:
                clusters.setdefault(source, []).append(Cluster(match))
        # Merge nearby clusters
        merged_clusters = clusters.get(source, [])
        for cluster in clusters.get(source, []):
            for c in merged_clusters:
                if (
                  c != cluster and
                  (cluster.min_query >= c.min_query - cluster_dist and
                   cluster.max_query <= c.max_query + cluster_dist) and
                  (cluster.min_train >= c.min_train - cluster_dist and
                   cluster.max_train <= c.max_train + cluster_dist)
                ):
                    cluster_points = set(
                        (m.train.x, m.train.y) for m in cluster.matches
                    )
                    c_points = set((m.train.x, m.train.y) for m in c.matches)
                    if cluster_points & c_points:
                        break
                    c.merge(cluster)
                    logging.info(len(merged_clusters))
                    merged_clusters.remove(cluster)
                    logging.info(len(merged_clusters))
                    cluster = c
        clusters[source] = merged_clusters
    clusters = [
        cluster.matches for sources in clusters.values() for cluster in sources
    ]
    return clusters


def plot_matches(ax1, ax2, matches):
    """Draw matches across axes"""
    logger.info('Drawing lines between matches')
    for match in matches:
        con = ConnectionPatch(
            xyA=(match.query.x, match.query.y),
            xyB=(match.train.x, match.train.y),
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
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
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
            xyA=(match.query.x, match.query.y),
            xyB=(match.train.x, match.train.y),
            coordsA='data', coordsB='data',
            axesA=ax1, axesB=ax2,
            arrowstyle='<-', linewidth=1,
            zorder=999
        )
        ax1.add_artist(con)
        if not plot_all_kp:
            # Plot keypoints
            plt.axes(ax1)
            vl_plotframe(np.matrix(match.query.kp).T, color='g', linewidth=1)
            plt.axes(ax2)
            vl_plotframe(np.matrix(match.train.kp).T, color='g', linewidth=1)
    if plot_all_kp:
        logger.info('Drawing ALL keypoints (this may take some time)...')
        for plot in source_plots:
            frames = np.array([
                kp.kp for kp in model.keypoints if kp.source == plot
            ])
            plt.axes(source_plots[plot])
            vl_plotframe(frames.T, color='g', linewidth=1)


def plot_clusters(S, clusters, spectrograms, settings, title,
                  plot_all_kp=False, S_kp=None):
    """Draw matches across axes"""
    fig = plt.figure()
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    if len(clusters) == 0:
        logger.info('No matches found')
        plot_spectrogram(
            S,
            settings['hop_length'],
            settings['octave_bins'],
            settings['fmin'],
            title,
            sr=settings['sr'],
            cbar=True
        )
        return
    rows = 2.0
    cols = len({cluster[0].train.source for cluster in clusters})
    ax1 = fig.add_subplot(rows, cols, (1, cols))
    plot_spectrogram(
        S,
        settings['hop_length'],
        settings['octave_bins'],
        settings['fmin'],
        title,
        sr=settings['sr'],
        cbar=True
    )

    if isinstance(spectrograms, str):
        logger.info('Loading spectrograms into memory: {}'.format(
            spectrograms))
        loaded_spectrograms = joblib.load(spectrograms)
    else:
        loaded_spectrograms = {}
    logger.info('Drawing lines between matches')
    colors = itertools.cycle('bgrck')
    source_plots = {}
    for cluster in clusters:
        color = next(colors)
        for match in cluster:
            ax2 = source_plots.get(match.train.source, None)
            if ax2 is None:
                ax2 = fig.add_subplot(rows, cols, cols + len(source_plots) + 1)
                if loaded_spectrograms.get(match.train.source, None) is None:
                    logger.info('Loading spectrogram into memory: {}'.format(
                        spectrograms[match.train.source]))
                    spec = joblib.load(spectrograms[match.train.source])
                    loaded_spectrograms[match.train.source] = spec
                plot_spectrogram(
                    loaded_spectrograms[match.train.source],
                    settings['hop_length'],
                    settings['octave_bins'],
                    settings['fmin'],
                    match.train.source,
                    sr=settings['sr'],
                    xticks=40/cols
                )
                ax2.set_zorder(-1)
                source_plots[match.train.source] = ax2
            con = ConnectionPatch(
                xyA=(match.query.x, match.query.y),
                xyB=(match.train.x, match.train.y),
                coordsA='data', coordsB='data',
                axesA=ax1, axesB=ax2,
                arrowstyle='<-', linewidth=1,
                zorder=999, color=color
            )
            ax1.add_artist(con)
            if not plot_all_kp:
                # Plot keypoints
                plt.axes(ax1)
                plot_keypoint(match.query, color='g', linewidth=1, ax=ax1)
                plt.axes(ax2)
                plot_keypoint(match.train, color='g', linewidth=1, ax=ax2)

    if plot_all_kp:
        plt.axes(ax1)
        plot_keypoints(S_kp, color='g', linewidth=1)
        for plot in source_plots:
            kps = [kp for kp in model.keypoints if kp.source == plot]
            plt.axes(source_plots[plot])
            plot_keypoints(kps, color='g', linewidth=1)
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0.2, hspace=0.2)


def plot_keypoint(keypoint, color='g', linewidth=2, ax=None):
    if ax is None:
        ax = plt.gcf().gca()
    c = plt.Circle(
        (keypoint.x, keypoint.y),
        keypoint.scale,
        color=color,
        fill=False
    )
    ax.add_artist(c)


def plot_keypoints(keypoints, color='g', linewidth=1, ax=None):
    for keypoint in keypoints:
        plot_keypoint(keypoint, color, linewidth, ax)


def plot_spectrogram(S, hop_length, octave_bins, fmin, title, sr=22050,
                     xticks=20, yticks=10, cbar=False):
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


def train_keypoints(tracks, hop_length, octave_bins=24, n_octaves=7,
                    fmin=50, sr=22050, algorithm='lshf', dedupe=False,
                    save=None, **kwargs):
    settings = locals().copy()
    for key in settings['kwargs']:
        settings[key] = settings['kwargs'][key]
    del settings['kwargs']
    del settings['tracks']
    logger.info('Settings: {}'.format(settings))
    spectrograms = {}
    keypoints = []
    descriptors = []
    for track in tracks:
        track_id = str(track)
        fp = fingerprint.from_file(track.path, sr, track_id, settings)
        if dedupe:
            # Remove duplicate keypoints
            # (important for ratio thresholding if source track
            # has exact repeated segments)
            fp.remove_similar_keypoints()

        if save:
            path = save_spectrogram(fp.spectrogram, track_id, save)
            spectrograms[track_id] = path
        keypoints.extend(fp.keypoints)
        descriptors.extend(fp.descriptors)
    descriptors = np.vstack(descriptors)
    matcher = ann.train_matcher(descriptors, algorithm=algorithm)
    model = Model(matcher, keypoints, settings)
    model.spectrograms = spectrograms
    if save:
        save_model(model, save)
    return model


def save_spectrogram(S, title, directory):
    directory = os.path.join(directory, 'spectrograms')
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, '{}.p'.format(title))
    logger.info('Saving spectrogram to disk... ({})'.format(path))
    joblib.dump(S, path, compress=True)
    return path


def save_model(model, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, 'model.p')
    logger.debug(type(model.matcher))
    if model.settings['algorithm'] == 'annoy':
        annoy_path = os.path.join(directory, 'annoy.ann')
        logger.info('Saving annoy index to disk... ({})'.format(annoy_path))
        model.matcher.save(annoy_path)
        model.matcher = annoy_path
    logger.info('Saving model to disk... ({})'.format(path))
    joblib.dump(model, path, compress=True)


def find_matches(track, model):
    track_id = str(track)
    # Extract keypoints
    fp = fingerprint.from_file(
        track.path, model.settings['sr'], track_id, model.settings
    )

    # Find (approximate) nearest neighbors
    distances, indices = ann.find_neighbors(
        model.matcher,
        fp.descriptors,
        algorithm=model.settings['algorithm'],
        k=2
    )

    # Build match  objects
    logger.info('Building match objects')
    matches = []
    for i, distance in enumerate(distances):
        matches.append(Match(
            fp.keypoints[i],
            model.keypoints[indices[i][0]],
            distance[0],
            distance[1]
        ))
    return matches, fp.spectrogram, fp.keypoints


def query_track(track, model, abs_thresh=None, ratio_thresh=None,
                cluster_dist=1.0, cluster_size=1, plot=True,
                plot_all_kp=False, match_orientation=True, save=True):
    if isinstance(model, str):
        model_file = os.path.join(model, 'model.p')
        logger.info('Loading model into memory: {}'.format(model_file))
        model = joblib.load(model_file)
        if model.settings['algorithm'] == 'annoy':
            model.matcher = ann.load_annoy(model.matcher)
    logger.info('Settings: {}'.format(model.settings))
    matches, S, kp = find_matches(track, model)

    cluster_dist = int(
        (model.settings['sr'] / model.settings['hop_length']) * cluster_dist
    )
    clusters = filter_matches(
        matches,
        abs_thresh,
        ratio_thresh,
        cluster_dist,
        cluster_size,
        match_orientation,
    )

    # Plot keypoint images and Draw matching lines
    spectrograms = model.spectrograms
    settings = model.settings
    model = None
    if plot:
        plot_clusters(
            S, clusters, spectrograms, settings, str(track), plot_all_kp, kp
        )
        plt.show(block=False)
    if save:
        if not plot:
            plot_clusters(
                S, clusters, spectrograms, settings,
                str(track), plot_all_kp, kp
            )
        plt.savefig('{}_{}.svg'.format(
            os.path.join(
                'plots',
                str(track)
            ),
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')),
            format='svg', figsize=(1920, 1080), bbox_inches=None
        )

    display_results(clusters, settings)
    return clusters, matches


def query_tracks(tracks, model, abs_thresh=None, ratio_thresh=None,
                 cluster_dist=1.0, cluster_size=1, plot=True,
                 plot_all_kp=False, match_orientation=True, save=True):
    kwargs = locals().copy()
    kwargs.pop('tracks', None)
    kwargs.pop('model', None)
    for track in tracks:
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
            times = []
            for match in sources[source]:
                seconds = int(
                    match[0].train.x * settings['hop_length'] / settings['sr']
                )
                time = datetime.timedelta(seconds=seconds)
                times.append(str(time))
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
    train.add_argument('tracks', type=str, nargs='+',
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
    train.add_argument('tracks', type=str, nargs='+',
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
                       help='Minimum cluster size to be considered a sample')
    train.add_argument('--match_orientation', type=strtobool, default=True,
                       help='Remove matches with differing orientations')
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

    args.tracks = util.tracks.parse_track_parameter(args.tracks)

    if args.command == 'train':
        del args.command
        model = train_keypoints(**vars(args))
    elif args.command == 'query':
        del args.command
        results = query_tracks(**vars(args))
        for clusters, matches in results:
            pass
        if args.plot:
            plt.show(block=True)