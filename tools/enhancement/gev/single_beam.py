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
# Beamform loop
# stft only for multichannel signal

# audio_data = audioread('new_dataset/AUDIO_RECORDING.wav', sample_rate=49000);
audio_data = get_audio_nochime('new_dataset/2m/2m_pub_new')
# audiowrite(audio_data, "beam.wav", 49000, True, True)
# audio_data = np.concatenate(audio_data, axis=0)
# audio_data = audio_data.astype(np.float32)

#audio_data, context_samples = get_audio_data_with_context(
#            cur_line[0], cur_line[1], cur_line[2])

Y = stft(audio_data, time_dim=1).transpose((1, 0, 2))
Y_var = Variable(np.abs(Y).astype(np.float32), True)

if args.gpu >= 0:
    Y_var.to_gpu(args.gpu)

N_masks, X_masks = model.calc_masks(Y_var)
N_masks.to_cpu()
X_masks.to_cpu()

N_mask = np.median(N_masks.data, axis=1)
X_mask = np.median(X_masks.data, axis=1)
Y_hat = gev_wrapper_on_masks(Y, N_mask, X_mask, True)
audiowrite(istft(Y_hat), "2m_pub_enhancement.wav", 49000, True, True)

# yHat = istft(Y_hat)
# yHat = stft(audioData, time_dim=1).transpose((1, 0, 2))
# yVar = Variable(np.abs(yHat).astype(np.float32), True)
# nMask, xMask = model.calc_masks(yVar)
# nMask = np.median(nMask.data, axis=1)
# xMask = np.median(xMask.data, axis=1)
# yHat = gev_wrapper_on_masks(yHat, nMask, xMask,True)
# audiowrite(istft(yHat), "2o.wav", 49000, True, True)
#


#audiowrite(istft(N_mask), "8ChanNoise", 48000, True, True)





print('Finished')
