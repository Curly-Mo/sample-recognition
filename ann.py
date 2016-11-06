import logging
import numpy as np

logger = logging.getLogger(__name__)

FLANN_ALGS = ['kdtree', 'kmeans', 'composite', 'lsh', 'autotuned']
CV_ALGS = ['kdtree', 'kmeans', 'composite', 'lsh', 'autotuned']
SKLEARN_ALGS = ['kd_tree', 'ball_tree', 'brute', 'auto']


def train_matcher(data, algorithm='kdtree'):
#    if algorithm in FLANN_ALGS:
#        matcher = fit_flann(data, algorithm)
#    el
    if algorithm in CV_ALGS:
        matcher = fit_cv2(data, algorithm)
    elif algorithm in SKLEARN_ALGS:
        matcher = fit_sklearn(data, algorithm)
    elif algorithm == 'lshf':
        matcher = fit_lshf(data)
    elif algorithm == 'annoy':
        matcher = fit_annoy(data)
    if not matcher:
        raise ValueError('Invalid matching algorithm: {}'.format(algorithm))
    return matcher


def find_neighbors(matcher, data, algorithm='lshf', k=2):
    logger.info('Finding (approximate) nearest neighbors...')
    if algorithm in FLANN_ALGS:
        matches = matcher.nn_index(np.float32(data), k=k)
        distances, indices = zip(*(((n1.distance, n2.distance), (n1.trainIdx, n2.trainIdx)) for n1, n2 in matches))
    elif algorithm in CV_ALGS:
        matches = matcher.knnMatch(np.float32(data), k=k)
        distances, indices = zip(*(((n1.distance, n2.distance), (n1.trainIdx, n2.trainIdx)) for n1, n2 in matches))
    elif algorithm in SKLEARN_ALGS:
        distances, indices = matcher.kneighbors(data, n_neighbors=k)
    elif algorithm == 'lshf':
        distances, indices = matcher.kneighbors(data, n_neighbors=k)
    elif algorithm == 'annoy':
        indices = []
        distances = []
        for d in data:
            index, distance = matcher.get_nns_by_vector(d, k, include_distances=True)
            indices.append(index)
            distances.append(distance)
    return distances, indices


def nearest_neighbors(test, train, algorithm='lshf', k=2):
    matcher = train_matcher(train, algorithm)
    distances, indices = find_neighbors(matcher, test, algorithm, k=k)
    return distances, indices


def fit_cv2(data, algorithm):
    logger.info('Fitting cv2 FLANN...')
    from cv2 import FlannBasedMatcher
    KDTREE = 0
    index_params = {
        'algorithm': KDTREE,
        'trees': 5,
        #'target_precision': 0.9,
        #'build_weight': 0.01,
        #'memory_weight': 0,
        #'sample_fraction': 0.1,
    }
    search_params = {'checks': 5}
    flann = FlannBasedMatcher(index_params, search_params)
    flann.add(np.float32(data))
    flann.train()
    return flann


def fit_flann(data, algorithm):
    logger.info('Fitting  FLANN...')
    from pyflann import FLANN
    matcher = FLANN(
        algorithm=algorithm,
        checks=32,
        eps=0.0,
        cb_index=0.5,
        trees=1,
        leaf_max_size=4,
        branching=32,
        iterations=5,
        centers_init='random',
        target_precision=0.9,
        build_weight=0.01,
        memory_weight=0.0,
        sample_fraction=0.1,
        log_level="warning",
        random_seed=-1,
    )
    matcher.build_index(data)
    return matcher


def fit_sklearn(data, algorithm):
    logger.info('Fitting Sklearn Matcher: {}...'.format(algorithm))
    from sklearn.neighbors import NearestNeighbors
    matcher = NearestNeighbors(
        algorithm=algorithm,
        n_neighbors=2,
        radius=1.0,
        leaf_size=30,
        metric='minkowski',
        p=2,
        metric_params=None,
        n_jobs=-1,
    )
    matcher.fit(data)
    return matcher


def fit_annoy(data, n_trees=-1):
    logger.info('Fitting Annoy Matcher...')
    from annoy import AnnoyIndex
    logger.info('Building Annoy index...')
    matcher = AnnoyIndex(data.shape[1], metric='euclidean')
    for i, d in enumerate(data):
        matcher.add_item(i, d)
    logger.info('Building Annoy Matcher...')
    matcher.build(n_trees)
    return matcher


def load_annoy(path, n_features=128):
    logger.info('Loading Annoy Index {}...'.format(path))
    from annoy import AnnoyIndex
    matcher = AnnoyIndex(n_features, metric='euclidean')
    matcher.load(path)
    return matcher


def fit_lshf(data):
    logger.info('Fitting  LSHForest...')
    from sklearn.neighbors import LSHForest
    lshf = LSHForest(
        n_estimators=20,
        min_hash_match=4,
        n_candidates=200,
        n_neighbors=2,
        radius=1.0,
        radius_cutoff_ratio=0.9,
        random_state=None,
    )
    lshf.fit(data)
    return lshf
