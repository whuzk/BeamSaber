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

# audio_data = audioread('new_dataset/AUDIO_RECORDING.wav', sample_rate=49000);
audio_data = get_audio_nochime('new_dataset/2m/2m_pub_new')
noise_data = audioread('new_dataset/babble_noise/babble-01.wav', sample_rate=19980)
noise_data = np.array(noise_data).astype(np.float32)

print(len(audio_data), len(noise_data))

Y = stft(audio_data, time_dim=1).transpose((1, 0, 2))
N = stft(noise_data)
print(len(Y), len(N))

Y_var = Variable(np.abs(Y).astype(np.float32), True)
N_var = Variable(np.abs(N).astype(np.float32), True)
print(type(Y_var), len(Y_var), type(N_var), len(N_var))

if args.gpu >= 0:
    Y_var.to_gpu(args.gpu)

X_masks = model.calc_mask_speech(Y_var)
N_masks = model.calc_mask_noise(N_var)

N_masks.to_cpu()
X_masks.to_cpu()

N_mask = np.median(N_masks.data, axis=1)
X_mask = np.median(X_masks.data, axis=1)
Y_hat = gev_wrapper_on_masks(Y, N_mask, X_mask, True)
audiowrite(istft(Y_hat), "2m_pub_enhancement.wav", 49000, True, True)

print('Finished')
