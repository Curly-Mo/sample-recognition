import logging
import numpy as np
from sklearn.neighbors import LSHForest
from cv2 import FlannBasedMatcher

logger = logging.getLogger(__name__)

FLANN_ALGS = ['kdtree', 'kmeans', 'composite', 'lsh', 'autotuned']
SKLEARN_ALGS = ['lshf']


def train_matcher(data, algorithm='kdtree'):
    if algorithm in FLANN_ALGS:
        matcher = fit_flann(data, algorithm)
    if algorithm in SKLEARN_ALGS:
        matcher = fit_lshf(data)
    if not matcher:
        raise ValueError('Invalid matching algorithm: {}'.format(algorithm))
    return matcher


def find_neighbors(matcher, data, algorithm='kdtree', k=2):
    logger.info('Finding (approximate) nearest neighbors...')
    if algorithm in FLANN_ALGS:
        matches = matcher.knnMatch(np.float32(data), k=k)
        distances, indices = zip(*(((n1.distance, n2.distance), (n1.trainIdx, n2.trainIdx)) for n1, n2 in matches))
    if algorithm in SKLEARN_ALGS:
        distances, indices = matcher.kneighbors(data, n_neighbors=k)
    return distances, indices


def nearest_neighbors(test, train, algorithm='kdtree', k=2):
    matcher = train_matcher(train, algorithm)
    distances, indices = find_neighbors(matcher, test, algorithm, k=k)
    return distances, indices


def fit_flann(data, algorithm):
    logger.info('Fitting FLANN...')
    KDTREE = 0
    index_params = {
        'algorithm': KDTREE,
        'trees': 5,
        #'target_precision': 0.9,
        #'build_weight': 0.01,
        #'memory_weight': 0,
        #'sample_fraction': 0.1,
    }
    search_params = {'checks': 16}
    flann = FlannBasedMatcher(index_params, search_params)
    flann.add(np.float32(data))
    flann.train()
    return flann


def fit_lshf(data):
    logger.info('Fitting LSH Forest...')
    lshf = LSHForest(
        n_estimators=10,
        min_hash_match=4,
        n_candidates=5,
        n_neighbors=2,
        radius=1.0,
        radius_cutoff_ratio=0.9,
        random_state=None,
    )
    lshf.fit(data)
    return lshf
