import re
import subprocess
import os
import datetime
import requests
import logging
from collections import defaultdict

from lxml import html

from .tracks import Track, Sample


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


USERAGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'

GENRE_TOKENS = {
    'Jazz / Blues': 'Jazz-Blues',
    'Hip-Hop / Rap / R&B': 'Hip-Hop',
    'Rock / Pop': 'Rock-Pop',
    'Soul / Funk / Disco':  'Soul-Funk-Disco',
    'Electronic / Dance': 'Electronic-Dance',
    'Reggae': 'Reggae',
    'Country / Folk': 'Country-Folk',
    'World': 'World',
    'Soundtrack': 'Soundtrack',
    'Classical': 'Classical',
}


def build_pad_set(tracks, query, n=5000):
    pad_tracks = []
    year_genres = defaultdict(int)
    for track in tracks:
        year_genres[(track.year, track.genre)] += 1
    print('Year_genres: {}'.format(year_genres))
    for key, value in year_genres.items():
        year_genres[key] = round((value / len(tracks)) * n)
    print('Year_genre percents: {}'.format(year_genres))
    t = [(track.artist, track.title) for track in tracks]
    q = [(track.artist, track.title) for track in query]
    exclude = t + q
    for key, count in year_genres.items():
        if count > 0:
            year, genre = key
            t = tracks_by_year_genre(year, genre, count, exclude=exclude)
            pad_keys = [(track.artist, track.title) for track in pad_tracks]
            for pad_track in t:
                if (pad_track.artist, pad_track.title) in pad_keys:
                    logger.info('This track is already in the set, what happened?')
                    t.remove(pad_track)
            pad_tracks.extend(t)
            logger.info('Num Tracks: {}'.format(len(pad_tracks)))
    return pad_tracks


def tracks_by_year_genre(year, genre, n, exclude=[], page=1, type='sampled', useragent=USERAGENT):
    tracks = []
    url = 'http://www.whosampled.com/browse/year/{year}/{type}/{genre}/{page}/'.format(
        year=year,
        type=type,
        genre=GENRE_TOKENS[genre],
        page=page,
    )
    logger.info('Processing year-genre: {}'.format(url))
    p = requests.get(url, headers={'User-Agent': useragent})
    tree = html.fromstring(p.content)
    total = tree.find_class('browseYearMenus')[0].find_class('optionMenuLabel')[0]
    total = total.text_content().split(' ')[0]
    #if int(total) < n:
    #    raise 'Rut-roh, not enough tracks available in this year-genre: {}-{}'.format(year, genre)
    items = tree.find_class('trackItem')
    for item in items:
        link = item.find_class('trackName')[0].find('a').get('href')
        link = 'http://www.whosampled.com' + link
        track = get_track_data(
            link,
            num_sampled=20,
            require_youtube=False,
            match_type=False,
            simple=True,
        )
        track.link = link
        track.year = year
        track.genre = genre
        track.page = url
        if track.youtube is None:
            logger.info('Track has no youtube associated with it, skipping')
            continue
        invalid = False
        if (track.artist, track.title) in exclude:
            invalid = True
        if not invalid:
            for sample in track.samples:
                if (sample.original.artist, sample.original.title) in exclude:
                    invalid = True
                    break
        if not invalid:
            for sample in track.sampled:
                if (sample.derivative.artist, sample.derivative.title) in exclude:
                    invalid = True
                    break
        if invalid:
            logger.info('Track samples or is sampled by a track in the dataset')
            continue
        tracks.append(track)
        if len(tracks) >= n:
            return tracks
    if not tree.find_class('next'):
        if type == 'covers':
            return tracks
        if type == 'samples':
            type = 'covers'
        if type == 'remixed':
            type = 'samples'
        if type == 'covered':
            type = 'remixed'
        if type == 'sampled':
            type = 'covered'
        page = 0
    keys = [(track.artist, track.title) for track in tracks]
    return tracks + tracks_by_year_genre(
        year,
        genre,
        n-len(tracks),
        exclude=exclude+keys,
        page=page+1,
        type=type
    )


def top_n_sampled(n=50, num_sampled=10, url='http://www.whosampled.com/most-sampled-tracks/1/', tracks=[], useragent=USERAGENT):
    page = requests.get(url, headers={'User-Agent': useragent})
    tree = html.fromstring(page.content)
    items = tree.find_class('listEntry')
    for item in items:
        link = item.find('a').get('href')
        link = 'http://www.whosampled.com' + link
        logger.info('Processing original: {}'.format(link))
        track = get_track_data(link, num_sampled=num_sampled, recurse=True)
        # Skip if track samples from any track in tracks
        for sample in track.samples:
            if any(t.artist == sample.original.artist and
                   t.title == sample.original.title for t in tracks):
                logger.info('Skipping track, it samples from the training set')
                break
        else:
            logger.info('Adding track to list')
            tracks.append(track)
            logger.info('Tracks length: {}'.format(len(tracks)))
            if len(tracks) >= n:
                return tracks
    next_url = tree.find_class('next')[0].find('a').get('href')
    next_url = 'http://www.whosampled.com' + next_url
    return top_n_sampled(n=n, url=next_url, tracks=tracks)


def get_track_data(url, num_sampled=10, useragent=USERAGENT, recurse=False, require_youtube=True, match_type=True, simple=False):
    page = requests.get(url, headers={'User-Agent': useragent})
    tree = html.fromstring(page.content)
    trackinfo = tree.find_class('trackInfo')[0]
    title = trackinfo.find('h1').text_content()
    artist = trackinfo.find_class('trackArtists')[0].find('h1').find('a').text_content()
    headers = tree.find_class('sectionHeader')
    lists = tree.find_class('list bordered-list')
    track = Track(
        artist=artist,
        title=title,
        path='',
    )
    embed = tree.find('.//iframe')
    if embed is not None:
        track.youtube = youtube_url_from_embed(embed.get('src'))
    for header, elements in zip(headers, lists):
        type = header.find('span').text_content().split()[1]
        if type == 'samples':
            if header.find_class('moreButton'):
                more_url = header.find_class('moreButton')[0].get('href')
                track.samples = get_samples(
                    more_url,
                    derivative=track,
                    simple=simple,
                )
            else:
                items = elements.find_class('trackDetails')
                for item in items:
                    sample_url = item.find_class('trackName')[0].get('href')
                    if simple:
                        sample = Sample(
                            original=track,
                            derivative=None,
                            type=None,
                            instrument=None,
                        )
                        sample.derivative = Track(
                            item.find_class('trackArtist')[0].find('a').text_content(),
                            item.find_class('trackName')[0].get('title')
                        )
                    else:
                        sample = get_sample(sample_url, derivative=track)
                    track.samples.append(sample)
        if type == 'sampled':
            if header.find_class('moreButton'):
                more_url = header.find_class('moreButton')[0].get('href')
                track.sampled = get_samples(
                    more_url,
                    original=track,
                    n=num_sampled,
                    match_type=match_type,
                    require_youtube=require_youtube,
                    recurse=recurse,
                    simple=simple,
                )
            else:
                items = elements.find_class('trackDetails')
                for item in items:
                    if len(track.sampled) > num_sampled:
                        break
                    sample_url = item.find_class('trackName')[0].get('href')
                    if simple:
                        sample = Sample(
                            original=track,
                            derivative=None,
                            type=None,
                            instrument=None,
                        )
                        sample.derivative = Track(
                            item.find_class('trackArtist')[0].find('a').text_content(),
                            item.find_class('trackName')[0].get('title')
                        )
                    else:
                        sample = get_sample(sample_url, original=track)
                    track.sampled.append(sample)
    return track


def get_samples(url, original=None, derivative=None, n=10, useragent=USERAGENT, match_type=False, require_youtube=False, recurse=False, simple=False):
    url = 'http://www.whosampled.com' + url
    page = requests.get(url, headers={'User-Agent': useragent})
    tree = html.fromstring(page.content)
    items = tree.find_class('trackDetails')
    samples = []
    sampled_instrument = None
    for item in items:
        if len(samples) >= n:
            return samples
        sample_url = item.find_class('trackName')[0].get('href')
        if simple:
            sample = Sample(
                original=original,
                derivative=derivative,
                type=None,
                instrument=None,
            )
            if sample.original is None:
                sample.original = Track(
                    item.find_class('trackArtist')[0].find('a').text_content(),
                    item.find_class('trackName')[0].get('title')
                )
            if sample.derivative is None:
                sample.derivative = Track(
                    item.find_class('trackArtist')[0].find('a').text_content(),
                    item.find_class('trackName')[0].get('title')
                )
        else:
            sample = get_sample(
                sample_url,
                original=original,
                derivative=derivative,
                recurse=recurse,
            )
        if match_type:
            if sample.type != 'direct':
                continue
            if sample.instrument == 'multiple elements':
                continue
            if sample.instrument == 'direct sample':
                continue
            if sampled_instrument is None:
                sampled_instrument = sample.instrument
            if sample.instrument != sampled_instrument:
                continue
        if require_youtube:
            logger.info('deriv_youtube: {}'.format(sample.derivative.youtube))
            logger.info('orig_youtube: {}'.format(sample.original.youtube))
            if not (sample.derivative.youtube and sample.original.youtube):
                logger.info('Can\'t get youtube url, skipping')
                continue
        samples.append(sample)
        if len(samples) >= n:
            return samples
    if tree.find_class('next'):
        next_url = tree.find_class('next')[0].find('a').get('href')
        return samples + get_samples(
            next_url,
            original=original,
            derivative=derivative,
            n=n-len(samples),
            useragent=useragent,
            match_type=match_type,
            require_youtube=require_youtube,
            recurse=recurse,
            simple=simple,
        )
    return samples


def get_sample(url, original=None, derivative=None, useragent=USERAGENT, recurse=False):
    def to_seconds(timestring):
        try:
            t = datetime.datetime.strptime(timestring, '%M:%S')
            t = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
        except:
            t = None
        return t
    url = 'http://www.whosampled.com' + url
    logger.info('\tProcessing Sample: {}'.format(url))
    page = requests.get(url, headers={'User-Agent': useragent})
    tree = html.fromstring(page.content)
    if original is None:
        info = tree.find_class('sampleTrackInfo')[1]
        artist = info.find_class('sampleTrackArtists')[0].find('a').text_content()
        title = info.find_class('trackName')[0].find('span').text_content()
        original = Track(
            artist=artist,
            title=title,
        )
        if derivative:
            original.sampled.append(derivative)
    if derivative is None:
        info = tree.find_class('sampleTrackInfo')[0]
        artist = info.find_class('sampleTrackArtists')[0].find('a').text_content()
        title = info.find_class('trackName')[0].find('span').text_content()
        link = info.find_class('trackName')[0].get('href')
        link = 'http://www.whosampled.com' + link
        if recurse:
            derivative = get_track_data(link, num_sampled=0, recurse=False, require_youtube=False)
        else:
            derivative = Track(
                artist=artist,
                title=title,
            )
            derivative.samples.append(original)

    type = tree.find_class('sampleTitle')[0]
    sample_type = type.find_class('section-header-title')[0].text_content().split(' ')[0].lower()
    instrument = type.find_class('section-header-title')[0].text_content().split(' of ')[-1].lower()
    sample = Sample(
        original=original,
        derivative=derivative,
        type=sample_type,
        instrument=instrument,
    )
    deriv_times, orig_times = tree.find_class('timing-wrapper')
    deriv_text = deriv_times.find('strong').get('data-timings')
    orig_text = orig_times.find('strong').get('data-timings')
    if deriv_text:
        sample.derivative_times = deriv_text.split(',')
    if orig_text:
        sample.original_times = orig_text.split(',')

    iframe = tree.findall('.//iframe')
    if len(iframe) > 1:
        embed = iframe[0].get('src')
        derivative.youtube = youtube_url_from_embed(embed)
        embed = iframe[1].get('src')
        original.youtube = youtube_url_from_embed(embed)

    genres = tree.xpath('.//span[text()="Main genre"]')
    if len(genres) > 1:
        derivative.genre = genres[0].getnext().text_content()
        original.genre = genres[1].getnext().text_content()

    years = tree.xpath('.//span[@itemprop="datePublished"]')
    derivative.year = years[0].text_content()
    original.year = years[1].text_content()

    return sample


def youtube_url_from_embed(embed_url):
    try:
        result = re.search('embed\/(.*)\?', embed_url)
        id = result.group(1)
    except:
        return None
    if 'dailymotion.com' in embed_url:
        url = ''.join(['http://www.dailymotion.com/', id])
    else:
        url = ''.join(['https://www.youtube.com/watch?v=', id])
    return url


def download_tracks(tracks, directory, sampled=False, samples=False):
    errors = []
    for track in tracks:
        p = download_track(track, directory)
        output, err = p.communicate()  
        if p.returncode:
            logger.info('Error retrieving {}, retrying with proxy'.format(p))
            p = save_youtube_audio(track.youtube, track.path, proxy=True)
            output, err = p.communicate()  
        output = output.decode('utf-8')
        print(output)
        try:
            track.path = re.findall('(Destination: |Correcting container in )(.*)\n', output)[-1][-1]
            track.path = track.path.strip('\"')
        except:
            errors.append(track)
        if sampled:
            for sample in track.sampled:
                s = sample.derivative
                p = download_track(s, directory)
                output, err = p.communicate()  
                if p.returncode:
                    logger.info('Error retrieving {}, retrying with proxy'.format(p))
                    p = save_youtube_audio(s.youtube, s.path, proxy=True)
                    output, err = p.communicate()  
                output = output.decode('utf-8')
                print(output)
                s.path = re.findall('(Destination: |Correcting container in )(.*)\n', output)[-1][-1]
                s.path = s.path.strip('\"')
        if samples:
            for sample in track.samples:
                s = sample.original
                p = download_track(s, directory)
                output, err = p.communicate()  
                if p.returncode:
                    logger.info('Error retrieving {}, retrying with proxy'.format(p))
                    p = save_youtube_audio(s.youtube, s.path, proxy=True)
                    output, err = p.communicate()  
                output = output.decode('utf-8')
                print(output)
                s.path = re.findall('(Destination: |Correcting container in )(.*)\n', output)[-1][-1]
                s.path = s.path.strip('\"')
    return errors


def download_track(track, directory):
    title = track.title.replace('/', '').replace(':', '')
    path = os.path.join(
        directory,
        '{}-{}.%(ext)s'.format(track.artist, title),
    )
    p = save_youtube_audio(track.youtube, path)
    track.path = path
    return p


def save_youtube_audio(url, path, proxy=False):
    logger.info('Downloading {} to {}...'.format(url, path))
    path = path.replace('?', '')
    path = path.replace('*', '')
    path = path.replace('%', '%%')
    path = path.replace('"', "'").strip()
    args = [
        'youtube-dl',
        '--extract-audio',
        '--prefer-ffmpeg',
        '--audio-format',
        'best',
        '--socket-timeout',
        '300',
        '-o',
        path,
        url,
    ]
    if proxy:
        args.insert(0, 'proxychains')
    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    return p
