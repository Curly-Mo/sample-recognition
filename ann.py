import logging
if not __name__ in logging.Logger.manager.loggerDict:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s :-: %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
logger = logging.getLogger(__name__)
from sklearn.neighbors import LSHForest


def fit_ann(data):
    logger.info('Fitting LSH Forest...')
    lshf = LSHForest()
    lshf.fit(data)
    return lshf


def find_neighbors(lshf, data, k=2):
    logger.info('Finding (approximate) nearest neighbors...')
    distances, indices = lshf.kneighbors(data, n_neighbors=k)
    return distances, indices


def nearest_neighbors(test, train, k=2):
    lshf = fit_ann(train)
    distances, indices = find_neighbors(lshf, test, k=k)
    return distances, indices
