import os
from sklearn.externals import joblib


class Track(object):
    def __init__(self, path, artist, title):
        self.path = path
        self.artist = artist
        self.title = title
        self.samples = []
        self.sampled = []

    def __repr__(self):
        return '{} - {}'.format(self.artist, self.title)


def load_tracks(tracks_file):
    tracks = joblib.load(tracks_file)
    return tracks


def tracks_from_dir(directory):
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        if os.path.isfile(path):
            track = track_from_path(path)
            yield track


def track_from_path(path):
    try:
        return track_from_tags(path)
    except:
        pass
    try:
        return track_from_filename(path)
    except:
        pass
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    track = Track(path, name, name)
    return track


def track_from_tags(path):
    import mutagen
    mutagen_track = mutagen.File(path)
    track = Track(
        path,
        mutagen_track['artist'][0],
        mutagen_track['title'][0],
    )
    return track


def track_from_filename(path):
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    parts = name.split(' - ')
    artist = parts[2].strip()
    title = parts[3].strip()
    track = Track(path, artist, title)
    return track


def parse_dir(directory):
    for root, subdirs, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            track = track_from_path(path)
            yield track


def parse_billboard(directory):
    for root, subdirs, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            name = os.path.splitext(name)[0]
            parts = name.split(' - ')
            artist = parts[2].strip()
            title = parts[3].strip()
            track = Track(path, artist, title)
            yield track
