import sys

from fgnt.signal_processing import audioread, stft, audiowrite
import numpy as np

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


