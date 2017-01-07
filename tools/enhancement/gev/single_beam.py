import argparse
import os

import numpy as np
from chainer import Variable
from chainer import cuda
from chainer import serializers
from tqdm import tqdm

from chime_data import gen_flist_simu, \
    gen_flist_real, get_audio_data, get_audio_data_with_context, get_audio_nochime
from fgnt.beamforming import gev_wrapper_on_masks
from fgnt.signal_processing import audiowrite, stft, istft, audioread
from fgnt.utils import Timer
from fgnt.utils import mkdir_p
from nn_models import BLSTMMaskEstimator, SimpleFWMaskEstimator

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

with Timer() as t:

    # audio_data = get_audio_nochime('new_dataset/2m/2m_pub_new', ch_range=range(1, 9), fs=49000)
    audio_data = get_audio_nochime('new_dataset/new_audio/AUDIO_RECORDING', ch_range=range(1, 9), fs=49000)
    noise_data = get_audio_nochime('new_dataset/babble_noise/babble', ch_range=range(1, 3), fs=19980)

# calculate the time for load the audio files
t_io += t.msecs

# change the audio files into frequency domain
Y = stft(audio_data, time_dim=1).transpose((1, 0, 2))
print(audio_data.shape, type(audio_data))
print(noise_data.shape, type(noise_data))

# change the noise file into frequency domain, the stft needs to perform single channel stft
N = stft(noise_data, time_dim=1).transpose((1, 0, 2))
print("Y: ", len(Y), "N: ", len(N), end="\n")

#
Y_var = Variable(np.abs(Y).astype(np.float32), True)
N_var = Variable(np.abs(N).astype(np.float32), True)
print("Y_var: ", Y_var.shape, "N_var: ", N_var.shape, end="\n")

# mask estimation
with Timer() as t:
    # N_masks, X_masks = model.calc_masks(Y_var)
    N_masks = model.calc_mask_noise(N_var)
    X_masks = model.calc_mask_speech(Y_var)
    # N_masks.to_cpu()
    # X_masks.to_cpu()

t_net += t.msecs

"""
PS: the array of noise and speech should have same dimension,
    otherwise the code will be error.
    Current DEBUG: numpy einsum
"""

with Timer() as t:
    N_mask = np.median(N_masks.data, axis=1)
    X_mask = np.median(X_masks.data, axis=1)
    print("N_mask: ", N_mask.shape, "X_mask: ", X_mask.shape, end="\n")
    Y_hat = gev_wrapper_on_masks(Y, N_mask)
t_beamform += t.msecs

with Timer() as t:
    audiowrite(istft(Y_hat), "2m_pub_enhancement.wav", 49000, True, True)
t_io += t.msecs

print('Finished')
print('Timings: I/O: {:.2f}s | Net: {:.2f}s | Beamformer: {:.2f}s | Total Time: {:.2f}s'.format(
    t_io / 1000, t_net / 1000, t_beamform / 1000,
    ((t_io / 1000) + (t_net / 1000)+(t_beamform / 1000))
))
