import math
import itertools
from collections import defaultdict


def cluster(matches, cluster_size=3, cluster_dist=20):
    clusters = set()
    votes = ght(matches, cluster_dist)
    for source, bins in votes.items():
        for bin, cluster in bins.items():
            if len(cluster) >= cluster_size:
                clusters.add(frozenset(cluster))
    clusters = [list(c) for c in clusters]
    total_clusters = [c for bins in votes.values() for c in bins.values()]
    return clusters, total_clusters


def ght(matches, cluster_dist=20):
    votes = defaultdict(lambda: defaultdict(set))
    try:
        dim = max(m.neighbors[0].kp.scale for m in matches)
    except:
        dim = 2
    for match in matches:
        ds = round_to(match.query.scale / match.neighbors[0].kp.scale, 2)
        d_theta = round_to(match.query.orientation - match.neighbors[0].kp.orientation, 0.4)
        dx = round_to(match.query.x - match.neighbors[0].kp.x, 1.5*dim)
        dy = round_to(match.query.y - match.neighbors[0].kp.y, 1.5*dim)
        bins = itertools.product(*(dx, dy))
        for bin in bins:
            train_kps = [tuple(m.neighbors[0].kp.kp[:2]) for m in votes[match.neighbors[0].kp.source][bin]]
            x = [m.neighbors[0].kp.x for m in  votes[match.neighbors[0].kp.source][bin]]
            try:
                min_x = min(x)
                max_x = max(x)
            except:
                min_x = max_x = match.neighbors[0].kp.x
            if tuple(match.neighbors[0].kp.kp[:2]) not in train_kps:
                if min_x - cluster_dist < match.neighbors[0].kp.x < max_x + cluster_dist:
                    votes[match.neighbors[0].kp.source][bin].add(match)
    return votes


def round_to(x, base=1, n=2):
    lo = base * math.floor(float(x)/base)
    hi = base * math.ceil(float(x)/base)
    return (lo, hi)
