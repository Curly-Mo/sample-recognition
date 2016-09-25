import os
import joblib


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
        track = track_from_tags(path)
        if track.artist and track.title:
            return track
    except:
        pass
    try:
        track = track_from_filename(path)
        if track.artist and track.title:
            return track
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


def get_source_and_query_tracks(tracks):
    source = []
    query = []
    for track in tracks:
        for sample in track.samples:
            details = sample.get('details')
            if details and details['type'] == 'direct':
                for t in tracks:
                    if t.artist == sample['artist'] and t.title == sample['title']:
                        query.append(track)
                        source.append(t)
    return query, source


def parse_track_parameter(track_param):
    """
    tracks could be a list of files, a directory, or pickle file containing
    a list of Track objects

    return an iterable of Track objects
    """
    if len(track_param) == 1:
        track_param = track_param[0]
    try:
        tracks = joblib.load(track_param)
        return tracks
    except:
        pass
    if isinstance(track_param, str) and os.path.isdir(track_param):
        tracks = tracks_from_dir(track_param)
        return tracks
    else:
        if isinstance(track_param, str):
            track_param = [track_param]
        tracks = (track_from_path(path) for path in track_param)
        return tracks
