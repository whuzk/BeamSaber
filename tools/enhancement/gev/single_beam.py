import argparse
import os

import numpy as np
from chainer import Variable
from chainer import cuda
from chainer import serializers
from tqdm import tqdm

from chime_data import get_audio_nochime, get_audio_babble
from fgnt.beamforming import gev_wrapper_on_masks
from fgnt.signal_processing import audiowrite, stft, istft, audioread
from fgnt.utils import Timer
from fgnt.utils import mkdir_p
from nn_models import BLSTMMaskEstimator, SimpleFWMaskEstimator

"""
PS: the array of noise and speech should have same dimension,
    otherwise the code will be error.
    Current DEBUG: numpy einsum
"""

parser = argparse.ArgumentParser(description='NN GEV beamforming')
parser.add_argument('model',
                    help='Trained model file')
parser.add_argument('model_type',
                    help='Type of model (BLSTM or FW)')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')

args = parser.parse_args()

# Prepare model
if args.model_type == 'BLSTM':
    model = BLSTMMaskEstimator()
elif args.model_type == 'FW':
    model = SimpleFWMaskEstimator()
else:
    raise ValueError('Unknown model type. Possible are "BLSTM" and "FW"')

serializers.load_hdf5(args.model, model)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

t_io = 0
t_net = 0
t_beamform = 0


def single_noise():
    audio_data = get_audio_nochime('new_dataset/2m/2m_pub_new', ch_range=range(1, 9), fs=49000)
    # audio_data = get_audio_nochime('new_dataset/new_audio/AUDIO_RECORDING', ch_range=range(1, 9), fs=49000)
    noise_audio = audioread('new_dataset/babble.wav', sample_rate=16000)
    noise_data = get_audio_babble(noise_audio, audio_data, 9)
    print("Noise audio: ", noise_data.shape)
    context_samples = 0

    print("audio_data: ", audio_data.shape, "noise_data: ", noise_data.shape, end="\n")

    Y = stft(audio_data, time_dim=1).transpose((1, 0, 2))
    N = stft(noise_data, time_dim=1).transpose((1, 0, 2))
    print(audio_data.shape, type(audio_data))
    print("Y: ", len(Y), "N: ", len(N), end="\n")

    Y_var = Variable(np.abs(Y).astype(np.float32), True)
    N_var = Variable(np.abs(N).astype(np.float32), True)
    print("Y_var: ", Y_var.shape, "N_var: ", N_var.shape, end="\n")

    # mask estimation

    N_masks = model.calc_mask_noise(N_var)
    X_masks = model.calc_mask_speech(Y_var)
    print("calculate n_mask x_mask")
    N_masks.to_cpu()
    X_masks.to_cpu()

    N_mask = np.median(N_masks.data, axis=1)
    X_mask = np.median(X_masks.data, axis=1)
    print("N_mask: ", N_mask.shape, "X_mask: ", X_mask.shape, end="\n")
    Y_hat = gev_wrapper_on_masks(Y, N_mask)

    audiowrite(istft(Y_hat), "new_dataset_result/AUDIO_REC_old_model_babble.wav", 49000, True, True)

    print('Finished')


def single_normal():
    audio_data = get_audio_nochime('new_dataset/2m/2m_pub_new', ch_range=range(1, 9), fs=49000)
    # audio_data = get_audio_nochime('new_dataset/new_audio/AUDIO_RECORDING', ch_range=range(1, 9), fs=49000)
    context_samples = 0

    print("audio_data: ", audio_data.shape, end="\n")

    Y = stft(audio_data, time_dim=1).transpose((1, 0, 2))
    print("Y: ", len(Y), end="\n")

    Y_var = Variable(np.abs(Y).astype(np.float32), True)
    print("Y_var: ", Y_var.shape, end="\n")

    # mask estimation
    N_masks, X_masks = model.calc_masks(Y_var)
    print("calculate n_mask x_mask")
    N_masks.to_cpu()
    X_masks.to_cpu()

    N_mask = np.median(N_masks.data, axis=1)
    X_mask = np.median(X_masks.data, axis=1)
    print("N_mask: ", N_mask.shape, "X_mask: ", X_mask.shape, end="\n")
    Y_hat = gev_wrapper_on_masks(Y, N_mask)

    audiowrite(istft(Y_hat)[context_samples:], "new_dataset_result/2m_pub_7m.wav", 49000, True, True)

    print('Finished')


if __name__ == '__main__':
    single_normal()
    # single_noise()
