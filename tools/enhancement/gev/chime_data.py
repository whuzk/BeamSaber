import json
import os
import pickle

from time import sleep

import numpy as np
import tqdm
import sys

from fgnt.mask_estimation import estimate_IBM
from fgnt.signal_processing import audioread, audiowrite
from fgnt.signal_processing import stft
from fgnt.utils import mkdir_p


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
             for a in annotations]
    return flist


def gen_flist_real(chime_data_dir, stage):
    with open(os.path.join(
            chime_data_dir, 'annotations',
            '{}05_{}.json'.format(stage, 'real'))) as fid:
        annotations = json.load(fid)
    flist_tuples = [(os.path.join(
        chime_data_dir, 'audio', '16kHz', 'embedded', a['wavfile']),
                     a['start'], a['end'], a['wsj_name']) for a in annotations]
    return flist_tuples


def get_audio_data(file_template, postfix='', ch_range=range(1, 7)):
    audio_data = list()
    for ch in ch_range:
        audio_data.append(audioread(
            file_template + '.CH{}{}.wav'.format(ch, postfix))[None, :])
        # print("shape: ", audioread(file_template + '.CH{}{}.wav'.format(ch, postfix)).shape, "size: ",
        #       sys.getsizeof(audio_data))
    print(file_template + '.CH0.Clean.wav')
    audio_data = np.concatenate(audio_data, axis=0)
    audio_data = audio_data.astype(np.float32)
    return audio_data


def get_audio_nochime(file_template, postfix='', ch_range=range(1, 9), fs=16000):
    audio_data = list()
    for ch in ch_range:
        audio_data.append(audioread(
            file_template + '.CH{}{}.wav'.format(ch, postfix), sample_rate=fs)[None, :])
        print("shape: ", audioread(file_template + '.CH{}{}.wav'.format(ch, postfix)).shape, "size: ",
              sys.getsizeof(audio_data))
    # print(type(audio_data))
    audio_data = np.concatenate(audio_data, axis=0)
    audio_data = audio_data.astype(np.float32)
    return audio_data


def get_audio_babble(noise_data, chime_data, chan):
    audio_data = list()
    print("noise_data: ", noise_data.shape,
          "chime_data: ", chime_data.shape,
          "channel: ", chan, end="\n")
    start = noise_data
    end = chime_data.shape[0] + start
    for i in range(1, chan):
        y = noise_data[start:end]
        print("start: ", start, "end: ", end, end="\n")
        start = end
        end = end + chime_data.shape[0]
        audio_data.append(y[None, :])
    print("last_shape: ", chime_data.shape)
    audio_data = np.concatenate(audio_data, axis=0)
    audio_data = audio_data.astype(np.float32)
    return audio_data


def get_audio_data_with_context(embedded_template, t_start, t_end,
                                ch_range=range(1, 7)):
    start_context = max((t_start - 5), 0)
    context_samples = (t_start - start_context) * 16000
    audio_data = list()
    for ch in ch_range:
        audio_data.append(audioread(
            embedded_template + '.CH{}.wav'.format(ch),
            offset=start_context, duration=t_end - start_context)[None, :])
    audio_data = np.concatenate(audio_data, axis=0)
    audio_data = audio_data.astype(np.float32)
    return audio_data, context_samples


def prepare_custom_audio(noise_data, chime_data):
    print("new shape: ", chime_data.shape)
    # noise_data = audioread('new_dataset/babble.wav', sample_rate=16000)
    start = 0
    end = chime_data.shape[0]
    for i in range(1, 7):
        y = noise_data[start:end]
        print("start: ", start, "end: ", end, end="\n")
        start = end
        end = end + chime_data.shape[0]
        audiowrite(y, "new_dataset/babble_noise/babble.CH{}.wav".format(i))
    sleep(0.01)
    print("last_shape: ", chime_data.shape)


def prepare_training_data(chime_data_dir, dest_dir):
    start = 0
    print("sdsd")
    for stage in ['tr', 'dt']:
        flist = gen_flist_simu(chime_data_dir, stage, ext=True)
        print(type(flist))

        export_flist = list()
        mkdir_p(os.path.join(dest_dir, stage))
        noise_data = audioread('new_dataset/babble_7min.wav', sample_rate=44100)
        print("noise_data size:", noise_data.shape[0])
        for f in tqdm.tqdm(flist, desc='Generating data for {}'.format(stage)):
            clean_audio = get_audio_data(f, '.Clean')
            # noise_audio = get_audio_data(f, '.Noise')
            # print(chime_data_dir)

            chime_size = audioread('{}.CH{}{}.Clean.wav'.format(f, 1, ''))

            print(chime_size.shape[0])
            # noise_audio = get_audio_babble(noise_data, chime_size, 7)

            noise_files = list()
            # start = noise_data
            end = chime_size.shape[0] + start
            if end > 19000000:
                start = 0
                end = chime_size.shape[0] + start
                print("reset")
                print(chr(27) + "[2J")
            print("ksjdkjd", end)
            for i in range(1, 7):
                y = noise_data[start:end]
                start = end
                end = end + chime_size.shape[0]
                noise_files.append(y[None, :])
                print("start: ", start, "end: ", end, "size: {}".format(end - start), end="\n")
            print("last_shape: ", chime_size.shape)
            noise_files = np.concatenate(noise_files, axis=0)
            noise_files = noise_files.astype(np.float32)
            noise_audio = noise_files

            print("clean shape: ", clean_audio.shape, "noise shape: ", noise_audio.shape, end="\n")

            X = stft(clean_audio, time_dim=1).transpose((1, 0, 2))
            N = stft(noise_audio, time_dim=1).transpose((1, 0, 2))
            print("X shape: ", X.shape, "N shape: ", N.shape, end="\n")

            IBM_X, IBM_N = estimate_IBM(X, N)
            Y_abs = np.abs(X + N)
            export_dict = {
                'IBM_X': IBM_X.astype(np.float32),
                'IBM_N': IBM_N.astype(np.float32),
                'Y_abs': Y_abs.astype(np.float32)
            }
            export_name = os.path.join(dest_dir, stage, f.split('/')[-1])
            with open(export_name, 'wb') as fid:
                pickle.dump(export_dict, fid)
            export_flist.append(os.path.join(stage, f.split('/')[-1]))
        with open(os.path.join(dest_dir, 'flist_{}.json'.format(stage)),
                  'w') as fid:
            json.dump(export_flist, fid, indent=4)
