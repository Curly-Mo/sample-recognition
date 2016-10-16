import re
import subprocess
import os
import datetime
import requests
import logging

from lxml import html

from .tracks import Track, Sample


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


USERAGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'


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


def get_track_data(url, num_sampled=10, useragent=USERAGENT, recurse=False):
    page = requests.get(url, headers={'User-Agent': useragent})
    tree = html.fromstring(page.content)
    trackinfo = tree.find_class('trackInfo')[0]
    title = trackinfo.find('h1').text_content()
    artist = trackinfo.find_class('trackArtists')[0].find('h2').text_content()
    headers = tree.find_class('sectionHeader')
    lists = tree.find_class('list bordered-list')
    track = Track(
        artist=artist,
        title=title,
        path='',
    )
    embed = tree.find('.//iframe')
    if embed:
        track.youtube = youtube_url_from_embed(embed.get('src'))
    for header, elements in zip(headers, lists):
        type = header.find('span').text_content().split()[1]
        if type == 'samples':
            if header.find_class('moreButton'):
                more_url = header.find_class('moreButton')[0].get('href')
                track.samples = get_samples(
                    more_url,
                    derivative=track,
                )
            else:
                items = elements.find_class('trackDetails')
                for item in items:
                    sample_url = item.find_class('trackName')[0].get('href')
                    sample = get_sample(sample_url, derivative=track)
                    track.samples.append(sample)
        if type == 'sampled':
            if header.find_class('moreButton'):
                more_url = header.find_class('moreButton')[0].get('href')
                track.sampled = get_samples(
                    more_url,
                    original=track,
                    n=num_sampled,
                    match_type=True,
                    require_youtube=True,
                    recurse=recurse
                )
            else:
                items = elements.find_class('trackDetails')
                for item in items:
                    if len(track.sampled) > num_sampled:
                        break
                    sample_url = item.find_class('trackName')[0].get('href')
                    sample = get_sample(sample_url, original=track)
                    track.sampled.append(sample)
    return track


def get_samples(url, original=None, derivative=None, n=10, useragent=USERAGENT, match_type=False, require_youtube=False, recurse=False):
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
        sample = get_sample(sample_url, original=original, derivative=derivative, recurse=recurse)
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
            derivative = get_track_data(link, num_sampled=0, recurse=False)
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
    for track in tracks:
        p = download_track(track, directory)
        output, err = p.communicate()  
        if p.returncode:
            logger.info('Error retrieving {}, retrying with proxy'.format(p))
            p = save_youtube_audio(track.youtube, track.path, proxy=True)
            output, err = p.communicate()  
        output = output.decode('utf-8')
        print(output)
        track.path = re.findall('(Destination: |Correcting container in )(.*)\n', output)[-1][-1]
        track.path = track.path.strip('\"')
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


def download_track(track, directory):
    path = os.path.join(
        directory,
        '{}-{}.%(ext)s'.format(track.artist, track.title.replace('%', '%%')),
    )
    p = save_youtube_audio(track.youtube, path)
    track.path = path
    return p


def save_youtube_audio(url, path, proxy=False):
    logger.info('Downloading {} to {}...'.format(url, path))
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
