import sys
import itertools
from collections import Counter
import logging
logger = logging.getLogger('spectral_peaks')
if not logger.handlers:
    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.INFO)
    logger.addHandler(stream)

import librosa
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import seaborn
seaborn.set(style='ticks')

import ann


def spectral_peaks(audio_path, hop_length, octave_bins=24, n_octaves=7, wn=0.5, num_peaks=5, plot=False):
    logger.info('{}: Loading signal into memory...'.format(audio_path))
    y, sr = librosa.load(audio_path)
    logger.info('{}: Generating Spectrogram...'.format(audio_path))
    S = librosa.core.constantq.cqt(y, sr=22050, hop_length=hop_length, bins_per_octave=octave_bins, n_bins=octave_bins*n_octaves, fmin=20)
    S = librosa.logamplitude(S, ref_power=np.max)

    # Find peaks
    logger.info('{}: Finding Peaks...'.format(audio_path))
    peakx, peaky = scipy.signal.argrelextrema(lowpass(S.T, wn=wn), np.greater, axis=1)

    # Convert peaks to 'matrix'
    logger.info('{}: Creating peak matrix...'.format(audio_path))
    peaks = np.full([num_peaks, S.shape[1]], -1, dtype=int)
    for x, points in itertools.groupby(zip(peakx, peaky), lambda x: x[0]):
        points = list(points)
        values = [S[point[1], x] for point in points]
        ys = [point[1] for (value, point) in sorted(zip(values, points), reverse=True)]
        if len(ys) >= num_peaks:
            peaks[:, x] = sorted(ys[:num_peaks])
    peaks = np.clip(peaks, -1, 168)

   # # Convert peaks to distances between peaks
   # logger.info('{}: Finding peak distances...'.format(audio_path))
   # peak_dists = np.empty([len(list(itertools.combinations(range(num_peaks), 2))), peaks.shape[1]], dtype=int)
   # for x, frame in enumerate(peaks.T):
   #     peak_dists[:, x] = [b - a for (a, b) in itertools.combinations(frame, 2)]

    # Plot
    if plot:
        # Plot spectrogram
        librosa.display.specshow(
            S,
            sr=sr,
            hop_length=hop_length,
            bins_per_octave=octave_bins,
            fmin=20,
            x_axis='time',
            y_axis='cqt_hz',
            n_xticks=10,
            n_yticks=20,
        )
        #plt.imshow(S, aspect='auto', origin='lower')
        plt.title(audio_path)
        plt.colorbar(format='%+2.0f dB')
        # Plot Peaks
        peakx = [np.repeat(x, len(frame)) for (x, frame) in enumerate(peaks.T)]
        peakx = [i for i in peakx]
        peaky = [i for frame in peaks.T for i in frame]
        plt.scatter(peakx, peaky, c='y')
        plt.tight_layout()
    return S, peaks


def find_peaks(matrix, axis=1, width=10):
    peaks = [[], []]
    if axis == 1:
        matrix = matrix.T
    for x, frame in enumerate(matrix):
        y_vals = scipy.signal.find_peaks_cwt(frame, np.array([width]))
        x_vals = [x] * len(y_vals)
        peaks[0].extend(x_vals)
        peaks[1].extend(y_vals)
    return peaks


def lowpass(x, order=8, wn=0.4, axis=1):
    b, a = scipy.signal.butter(order, wn, btype='low')
    y = scipy.signal.filtfilt(b, a, x, axis=axis)
    return y


def match(path1, path2, hop_length, wn=0.4, num_peaks=5, thresh=0.001, plot=True):
    target_peaks = num_peaks + 2
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    S1, peaks1 = spectral_peaks(path1, hop_length, wn=wn, num_peaks=target_peaks, plot=plot)
    ax2 = fig.add_subplot(2, 1, 2)
    S2, peaks2 = spectral_peaks(path2, hop_length, wn=wn, num_peaks=num_peaks, plot=plot)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

    logger.info('Finding peak distances...')
    peak_dists2 = np.diff(peaks2, axis=0)

    num_combos = len(list(itertools.combinations(range(target_peaks), num_peaks)))
    peak_dists1 = np.empty([num_peaks-1, peaks1.shape[1]*(num_combos+1)], dtype=int)
    for x, frame in enumerate(peaks1.T):
        start = x*(num_combos+1)
        end = (x+1)*(num_combos+1) - 1
        peak_dists1[:, start:end] = np.array([np.diff(combo) for combo in itertools.combinations(frame, num_peaks)]).T

    logger.info('Finding nearest neighbors')
    distances, nearest_neighbors = ann.nearest_neighbors(peak_dists2.T, peak_dists1.T, k=1)
   # nearest_neighbors = []
   # for frame in peak_dists1:
   #     score = 0
   #     nn = None
   #     for i, yframe in enumerate(peak_dists2):
   #         count = count_matches(frame, yframe)
   #         if count > score:
   #             score = count
   #             nn = i
   #     nearest_neighbors.append({'score': score, 'index': nn})

    logger.info('Drawing lines between matches')
    for x, nn in enumerate(nearest_neighbors[:-1]):
        if distances[x] < thresh:
            if any(n in nearest_neighbors[x+1:x+(num_combos*3)] for n in [nn+1, nn+2, nn+3]):
                con = ConnectionPatch(
                    xyA=(x/num_combos, 0), xyB=(nn, S2.shape[0]),
                    coordsA='data', coordsB='data',
                    axesA=ax1, axesB=ax2,
                    arrowstyle='<-', linewidth=1,
                    zorder=999
                )
                ax1.add_artist(con)
                ax2.set_zorder(-1)
    plt.show(block=False)
    return nearest_neighbors, distances


def nearest_neighbors(peaks1, peaks2):
    # Convert peaks to distances between peaks
    peak_dists = [[] for i in range(len(peaks1))]
    for x, frame in enumerate(peaks1):
        for a, b in itertools.combinations(frame, 2):
            peak_dists[x].append(b-a)


def count_matches(lista, listb):
    if len(lista) > len(listb):
       lista, listb = listb, lista

    a_count = Counter(lista)
    b_count = Counter(listb)
    return sum(min(b_count[ak], av) for ak,av in a_count.iteritems())
