import json
import sys
import os
from fgnt.signal_processing import audioread, stft, audiowrite
import numpy as np


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


if __name__ == '__main__':
    noise_data = audioread('new_dataset/babble_7min.wav', sample_rate=44100)
    slie = noise_data[0:19000000]
    audiowrite(slie, 'new_dataset/babble_sliced.wav', samplerate=44100)