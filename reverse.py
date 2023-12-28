import librosa
import numpy as np
import IPython.display as ipd

filename = 'CSIT_DS_Mini-Challenge\Task_1\T1_audio.wav'
y, sr = librosa.load(filename, sr=11025)

reversedY = y[::-1]
ipd.Audio(reversedY, rate=sr)