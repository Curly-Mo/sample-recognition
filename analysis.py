#!/usr/bin/env python
import os
import argparse
import logging
import logging.config
import statistics
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import seaborn
import joblib
import numpy as np
from tabulate import tabulate

from sample_recognition import Result, Match, Neighbor

from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
seaborn.set(style='ticks')
seaborn.set_context("paper")
logger = logging.getLogger(__name__)
scriptdir = os.path.dirname(os.path.realpath(__file__))
logfile = os.path.join(scriptdir, 'logging.ini')
logging.config.fileConfig(logfile, disable_existing_loggers=False)


def load_results(path):
    logger.info('Loading result into memory: {}'.format(path))
    result = joblib.load(path)
    return result


def display_result(result):
    print('{} sampled from:'.format(
        str(result.track).encode('ascii', 'ignore'))
    )
    for source, times in result.times.items():
        print('{} at '.format(source.encode('ascii', 'ignore')))
        for i, time in enumerate(times):
            print('\t{} => {}'.format(*time), end='')
            print('\tPitch_shift: {}'.format(
                statistics.median(result.pitch_shift[source][i])
            ), end='')
            print('\tTime_stretch: {}'.format(
                statistics.median(result.time_stretch[source][i])
            ))
    print('True Positives: {}'.format(result.true_pos))
    print('False Positives: {}'.format(result.false_pos))
    print('False Negatives: {}'.format(result.false_neg))
    print('\n')


def parse_input(track_param):
    if len(track_param) == 1:
        track_param = track_param[0]
    if isinstance(track_param, str) and os.path.isdir(track_param):
        tracks = []
        for f in os.listdir(track_param):
            path = os.path.join(track_param, f)
            if os.path.isfile(path):
                tracks.append(path)
        return tracks
    else:
        if isinstance(track_param, str):
            track_param = [track_param]
    return track_param


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a binaural stereo wav file from a mono audio file."
    )
    parser.add_argument('results', type=str, nargs='+',
                        help='Either a directory or list of files')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print debug messages to stdout')
    args = parser.parse_args()

    import logging.config
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    logfile = os.path.join(scriptdir, 'logging.ini')
    logging.config.fileConfig(logfile, disable_existing_loggers=False)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose debugging activated")

    inputs = sorted(parse_input(args.results))

    train = joblib.load('data/tracks/train_full.p')
    precisions = []
    recalls = []
    for item in inputs:
        results = load_results(item)
        true_pos = np.float64(sum(len(r.true_pos) for r in results))
        false_pos = np.float64(sum(len(r.false_pos) for r in results))
        false_neg = np.float64(sum(len(r.false_neg) for r in results))
        print('Totals:')
        print('True Pos: {}'.format(true_pos))
        print('False Pos: {}'.format(false_pos))
        print('False Neg: {}'.format(false_neg))
        precision = true_pos / (true_pos + false_pos)
        print('precision: {}'.format(precision))
        recall = true_pos / (true_pos + false_neg)
        print('recall: {}'.format(recall))
        f_score = (precision * recall) / (precision + recall)
        print('F-score: {}'.format(f_score))
        precisions.append(precision)
        recalls.append(recall)
        print('\n')

        genres_orig = defaultdict(lambda: defaultdict(int))
        genres_deriv = defaultdict(lambda: defaultdict(int))
        for r in results:
            for s in r.true_pos:
                genres_deriv[s.derivative.genre]['true_pos'] += 1
                genres_orig[s.original.genre]['true_pos'] += 1
            for t in r.false_pos:
                genres_deriv[r.track.genre]['false_pos'] += 1
                genres_orig[t.genre]['false_pos'] += 1
            for s in r.false_neg:
                genres_deriv[s.derivative.genre]['false_neg'] += 1
                genres_orig[s.original.genre]['false_neg'] += 1
        genres = []
        for g in genres_deriv:
            true_pos = np.float64(genres_deriv[g]['true_pos'])
            false_pos = np.float64(genres_deriv[g]['false_pos'])
            false_neg = np.float64(genres_deriv[g]['false_neg'])
            recall = true_pos / (true_pos + false_neg)
            precision = true_pos / (true_pos + false_pos)
            f_score = (precision * recall) / (precision + recall)
            genres.append([g, recall, precision, f_score])
        print(tabulate(genres, headers=['recall', 'precision', 'f_score'], tablefmt='latex'))
        print('\n')

        genres = []
        for g in genres_orig:
            true_pos = np.float64(genres_orig[g]['true_pos'])
            false_pos = np.float64(genres_orig[g]['false_pos'])
            false_neg = np.float64(genres_orig[g]['false_neg'])
            recall = true_pos / (true_pos + false_neg)
            precision = true_pos / (true_pos + false_pos)
            f_score = (precision * recall) / (precision + recall)
            genres.append([g, recall, precision, f_score])
        print(tabulate(genres, headers=['recall', 'precision', 'f_score'], tablefmt='latex'))
        print('\n')

        instruments = defaultdict(lambda: defaultdict(int))
        for r in results:
            for s in r.true_pos:
                instruments[s.instrument]['true_pos'] += 1
            for i in r.false_pos:
                instruments[s.instrument]['false_pos'] += 1
            for s in r.false_neg:
                instruments[s.instrument]['false_neg'] += 1
        inst_recalls = []
        for i in instruments:
            pos = np.float64(instruments[i]['true_pos'])
            neg = np.float64(instruments[i]['false_neg'])
            inst_recalls.append([i, pos / (pos + neg)])
        print(tabulate(inst_recalls, headers=['recall'], tablefmt='latex'))

    if len(inputs) > 1:
        plt.plot(recalls, precisions, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.show()
