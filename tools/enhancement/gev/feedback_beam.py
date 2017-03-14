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
parser.add_argument('experiments',
                    help='Number of current experiment')
args = parser.parse_args()

# Prepare model
if args.model_type == 'BLSTM':
    model = BLSTMMaskEstimator()
elif args.model_type == 'FW':
    model = SimpleFWMaskEstimator()
else:
    raise ValueError('Unknown model type. Possible are "BLSTM" and "FW"')

"""
PS: feedback loop beamforming experiment, using beamforming result +
    specified channel. the beamforming as the channel
"""

# load nnet model
serializers.load_hdf5(args.model, model)

t_io = 0
t_net = 0
t_beamform = 0

with Timer() as t:

    audio_data = get_audio_nochime('new_dataset/2m/2m_pub_new', ch_range=range(1, 9), fs=48000)
    # audio_data = get_audio_nochime('new_dataset/new_audio/AUDIO_RECORDING', ch_range=range(1, 9), fs=49000)

# calculate the time for load the audio files
t_io += t.msecs

# change the audio files into frequency domain
Y = stft(audio_data, time_dim=1).transpose((1, 0, 2))
print(audio_data.shape, type(audio_data))

Y_var = Variable(np.abs(Y).astype(np.float32), True)

# mask estimation
with Timer() as t:
    N_masks, X_masks = model.calc_masks(Y_var)
    N_masks.to_cpu()
    X_masks.to_cpu()
t_net += t.msecs

with Timer() as t:
    N_mask = np.median(N_masks.data, axis=1)
    X_mask = np.median(X_masks.data, axis=1)
    print("Y: ", Y.shape,"N_mask: ", N_mask.shape, "X_mask: ", X_mask.shape, end="\n")
    Y_hat = gev_wrapper_on_masks(Y, N_mask, X_mask)
    # audiowrite(istft(Y_hat), "new_dataset_result/2m_feedback_.wav", 48000, True, True)
t_beamform += t.msecs

# second pass beamforming
# second_channel = audioread('AUDIO_RECORDING.CH2.wav', sample_rate=48000)
second_channel = audioread('new_dataset/2m/2m_pub_new.CH5.wav', sample_rate=48000)
second_channel = np.expand_dims(second_channel, axis=0)
print("second_size", second_channel.shape, end="\n")

second_channel = stft(second_channel, time_dim=1).transpose((1, 0, 2))
print("Y_hat: ", Y_hat.shape, "second_size", second_channel.shape, end="\n")

Y_hat = np.expand_dims(Y_hat, axis=1)
Y_var_second = Variable(np.abs(Y_hat).astype(np.float32), True)
print("Y_hat_second: ", Y_hat.shape)

Y_hat = np.add(Y_hat, second_channel)
print("Y_hat_combined: ", Y_hat.shape)

with Timer() as t:
    NN_masks, XX_masks = model.calc_masks(Y_var_second)
    NN_masks.to_cpu()
    XX_masks.to_cpu()

with Timer() as t:
    NN_mask = np.median(NN_masks.data, axis=1)
    XX_mask = np.median(XX_masks.data, axis=1)
    print("Y: ", Y_hat.shape, "N_mask: ", NN_mask.shape, "X_mask: ", XX_mask.shape, end="\n")
    # try:
    YY_hat = gev_wrapper_on_masks(Y_hat, NN_mask, XX_mask)
    # except AttributeError:
    #     YY_hat = gev_wrapper_on_masks(Y, NN_mask, XX_mask)

with Timer() as t:
    audiowrite(istft(YY_hat), "new_dataset_result/2m_feedback_{}.wav".format(args.experiments), 48000, True, True)
t_io += t.msecs

print('Finished')
print('Timings: I/O: {:.2f}s | Net: {:.2f}s | Beamformer: {:.2f}s | Total Time: {:.2f}s'.format(
    t_io / 1000, t_net / 1000, t_beamform / 1000,
    ((t_io / 1000) + (t_net / 1000)+(t_beamform / 1000))
))
