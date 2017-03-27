import json
import sys
import os
from fgnt.signal_processing import audioread, stft, audiowrite
import numpy as np
from os import listdir
from os.path import isfile, join


def gen_flist_simu(chime_data_dir, stage, ext=False):
    with open(os.path.join(
            chime_data_dir, 'annotations',
            '{}05_{}.json'.format(stage, 'simu'))) as fid:
        annotations = json.load(fid)
    if ext:
        isolated_dir = 'isolated_ext'
        # isolated_dir = 'clean_dt'
    else:
        isolated_dir = 'isolated'
    flist = [os.path.join(
        chime_data_dir, 'audio', '16kHz', isolated_dir,
        '{}05_{}_{}'.format(stage, a['environment'].lower(), 'simu'),
        '{}_{}_{}'.format(a['speaker'], a['wsj_name'], a['environment']))
             for a in annotations if a['environment'] == 'caf']
    print('audio', '16kHz', isolated_dir, '{}05_{}_{}'.format(stage, a['environment'].lower(), 'simu'))
    return flist


def audio_manipulation(self):
    print("longto")

    audio_file = audioread('new_dataset/chime_ex.wav', sample_rate=16000)
    babble_file = audioread('new_dataset/babble_16.wav', sample_rate=16000)

    print("len chime: ", audio_file.shape)
    print("len chime: ", babble_file.shape)

    audio_shape = audio_file.shape[0]
    babble_shape = babble_file.shape[0]
    split = int(babble_shape / audio_shape)
    # y = list()
    start = 0
    end = audio_file.shape[0]
    for i in range(1, 7):
        print("start = ", start, "end = ", end)
        y = babble_file[start:end]
        start = end + 1
        end = end + audio_file.shape[0]
        audiowrite(y, "new_dataset/babble_noise/babble.CH{}.wav".format(i))

    # audiowrite(y, "y.wav")
    # np.split(babble_file, 2)

    print("split into: ", split, "babble shape: ", babble_file.shape, "y: ", sys.getsizeof(y))

    audio_stft = stft(audio_file)
    babble_stft = stft(y)
    print(audio_stft.shape)
    print(babble_stft.shape)


def annotation_manipulation():
    annotation = json.load('annotation/tr05_simu.json')
    for a in annotation:
        if a['environment'] == 'caf':
            print(annotation)


def audio_joiner(path):
    chime_data_dir = path
    print(path)
    flist = [f for f in listdir(chime_data_dir) if isfile(join(chime_data_dir, f))]
    thefile = open('list.txt', 'w')
    y = list()
    counter = 0
    for item in flist:
        audio_file = audioread('{}/{}'.format(path, item), sample_rate=16000)
        print(item)
        if len(audio_file) < len(y):
            c = y.copy()
            c[:len(audio_file)] += audio_file
        else:
            c = audio_file.copy()
            c[:len(y)] += y

            # y = y + audio_file

    audiowrite(c, '/media/hipo/lento/Dataset/LibriSpeech/test/com.flac', samplerate=16000)


def audio_counter(path):
    audio_data = audioread(path)
    print(audio_data.shape)
    audio_data = np.concatenate(audio_data, axis=0)
    audio_data = audio_data.astype(np.float32)
    print(audio_data.shape)


if __name__ == '__main__':
    paths = "/home/hipo/workspace/BeamSaber/2m_mvdr.wav"
    audio_counter(paths)
