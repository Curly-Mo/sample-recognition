import itertools
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
import cv2
import librosa
import joblib

import ann


class Keypoint():
    def __init__(self, x, y, source, descriptor=None):
        self.x = x
        self.y = y
        self.source = source
        self.descriptor = descriptor


class Match():
    def __init__(self, query, train, distance, distance2):
        self.query = query
        self.train = train
        self.distance = distance
        self.d2 = distance2


class Model():
    def __init__(self, clf, keypoints, settings, images=None):
        self.clf = clf
        self.keypoints = keypoints
        self.settings = settings
        self.images = images


def cqtgram(y, hop_length=512, octave_bins=24, n_octaves=8, fmin=40, sr=22050):
    S = librosa.core.constantq.cqt(y, sr=sr, hop_length=hop_length, bins_per_octave=octave_bins, n_bins=octave_bins*n_octaves, fmin=fmin)
    S = librosa.logamplitude(S, ref_power=np.max)
    #S = librosa.core.spectrum.stft(y, win_length=2048, hop_length=hop_length)
    #S = librosa.feature.spectral.logfsgram(y, sr=sr, n_fft=2048, hop_length=hop_length, bins_per_octave=octave_bins)
    return S


def sift_file(audio_path, hop_length, octave_bins=24, n_octaves=9, fmin=40, sr=22050, **kwargs):
    logger.info('{}: Loading signal into memory...'.format(audio_path))
    y, sr = librosa.load(audio_path, sr=sr)
    logger.info('{}: Generating Spectrogram...'.format(audio_path))
    S = cqtgram(y, hop_length=hop_length, octave_bins=octave_bins, n_octaves=n_octaves, fmin=fmin, sr=sr)

    #I = (S - S.min()) / (S.max() - S.min()) * 128
    #I = np.uint8(S)
    #I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    #I = np.flipud(I)

    logger.info('{}: Extracting SIFT keypoints...'.format(audio_path))
    I = cv2.convertScaleAbs(S, alpha=(255.0/(S.max()-S.min())))
    keypoints, descriptors = sift(I, **kwargs)
    keypoints, descriptors = remove_edge_keypoints(keypoints, descriptors, S)
    #out = np.zeros(I.shape)
    #keypoint_img = cv2.drawKeypoints(I, keypoints, outImage=out, flags=(cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS+cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG))
    #keypoint_img = np.ma.masked_where(keypoint_img < 0.2, keypoint_img)
    keypoint_img = cv2.drawKeypoints(I, keypoints, color=(199, 21, 133), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    logger.info('{}: Creating keypoint objects...'.format(audio_path))
    keypoint_objs = []
    for keypoint, descriptor in itertools.izip(keypoints, descriptors):
        keypoint_objs.append(Keypoint(*keypoint.pt, source=audio_path, descriptor=descriptor))

    return keypoint_objs, descriptors, keypoint_img


def remove_edge_keypoints(keypoints, descriptors, S):
    start = next((index for index, frame in enumerate(S.T) if all(value > -80 for value in frame)), 0) + 8
    end = S.shape[1] - next((index for index, frame in enumerate(reversed(S.T)) if all(value > -80 for value in frame)), S.shape[1]) - 8
    out_kp = []
    out_desc = []
    for keypoint, descriptor in itertools.izip(keypoints, descriptors):
        # Skip keypoints on the left and right edges of spectrogram
        if start <= keypoint.pt[0] <= end:
            out_kp.append(keypoint)
            out_desc.append(descriptor)
    return out_kp, out_desc


def sift(S, contrast_thresh=0.01, edge_thresh=5, n_octave_layers=3, sigma=1.6):
    sift = cv2.SIFT(contrastThreshold=contrast_thresh, edgeThreshold=edge_thresh, nOctaveLayers=n_octave_layers, sigma=sigma)
    keypoints, descriptors = sift.detectAndCompute(S, None)
    return keypoints, descriptors


def sift_match(path1, path2, hop_length, abs_thresh=None, ratio_thresh=None, octave_bins=24, n_octaves=9, fmin=40, timeframe=20, cluster_size=1, sr=22050):
    # Extract keypoints
    kp1, desc1, img1 = sift_file(path1, hop_length, octave_bins=octave_bins, n_octaves=n_octaves, fmin=fmin, sr=sr)
    kp2, desc2, img2 = sift_file(path2, hop_length, octave_bins=octave_bins, n_octaves=n_octaves, fmin=fmin, sr=sr)

    # Plot keypoint images
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    plot_keypoint_image(img1, hop_length, octave_bins, fmin, path1, sr=sr)
    ax2 = fig.add_subplot(2, 1, 2)
    plot_keypoint_image(img2, hop_length, octave_bins, fmin, path2, sr=sr)
   # mng = plt.get_current_fig_manager()
   # mng.full_screen_toggle()

    # BFMatcher with default params
    #bf = cv2.BFMatcher()
    #matches = bf.knnMatch(desc1, desc2, k=2)
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


def filter_matches(matches, abs_thresh=None, ratio_thresh=None, timeframe=20, cluster_size=1):
    logger.info('Filtering nearest neighbors down to actual matched samples')
    filtered = []
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


def plot_all_matches(img, matches, model, title):
    """Draw matches across axes"""
    fig = plt.figure()
    if len(matches) == 0:
        logger.info('No matches found')
        plot_keypoint_image(
            img,
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
    plot_keypoint_image(
        img,
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
            plot_keypoint_image(
                model.images[match.train.source],
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


def plot_keypoint_image(I, hop_length, octave_bins, fmin, title, sr=22050, xticks=20, yticks=10, cbar=False):
    # Plot keypoints + image
    librosa.display.specshow(
        I,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=octave_bins,
        fmin=fmin,
        x_axis='time',
        y_axis='cqt_hz',
        n_xticks=xticks,
        n_yticks=yticks,
        cmap='gray_r',
    )
    plt.title(title)
    if cbar:
        cbar = plt.colorbar(format='%+2.0f dB', ticks=np.linspace(0, 255, 11))
        cbar.ax.set_yticklabels(['-80 dB', '-72 dB', '-64 dB', '-56 dB', '-48 dB', '-40 dB', '-32 dB', '-24 dB', '-16 dB', '-8 dB', '+0 dB'])
    plt.show(block=False)


def plot_spectrogram(S, hop_length, octave_bins, fmin, title, sr=22050):
    # Plot Spectrogram
    librosa.display.specshow(
        S,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=octave_bins,
        fmin=fmin,
        x_axis='time',
        y_axis='cqt_hz',
        n_xticks=20,
        n_yticks=10,
    )
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')


def train_keypoints(audio_paths, hop_length, octave_bins=24, n_octaves=9, fmin=40, sr=22050, save=None, **kwargs):
    settings = locals()
    settings.pop('audio_paths', None)
    images = {}
    keypoints = []
    descriptors = []
    for audio_path in audio_paths:
        kp, desc, img = sift_file(audio_path, hop_length, octave_bins, n_octaves, fmin, sr=sr, **kwargs)
        images[audio_path] = img
        keypoints.extend(kp)
        descriptors.extend(desc)
    clf = ann.fit_ann(descriptors)
    model = Model(clf, keypoints, settings, images)
    if save:
        logging.info('Saving model to disk...')
        joblib.dump(model, save, compress=True)
    return model


def find_matches(audio_path, model):
    # Extract keypoints
    kp, desc, img = sift_file(
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
    return matches, img


def query_track(audio_path, model, abs_thresh=None, ratio_thresh=None, timeframe=1.0, cluster_size=1):
    matches, img = find_matches(audio_path, model)

    timeframe = int((model.settings['sr']/ model.settings['hop_length']) * timeframe)
    matches = filter_matches(matches, abs_thresh, ratio_thresh, timeframe, cluster_size)
    logger.info('Matches found: {}'.format(len(matches)))

    # Plot keypoint images and Draw matching lines
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plot_all_matches(img, matches, model, audio_path)
    plt.show(block=False)

    return matches
