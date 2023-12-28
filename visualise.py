import librosa
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display

filename = 'CSIT_DS_Mini-Challenge\Task_2\T2_audio_b.wav'
y, sr = librosa.load(filename, sr=11025)
fig, ax = plt.subplots(nrows=3, sharex=True)

X = librosa.stft(y)
Xdb = librosa.amplitude_to_db(abs(X))
mfccs = librosa.feature.mfcc(y=y, sr=sr)
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

plt.figure(figsize=(12, 5))
librosa.display.waveshow(y, sr=sr, ax=ax[1], color="blue")
ipd.Audio(y, rate=sr)
ax[0].set(title='Envelope view, mono')
ax[0].label_outer()

n0 = 7000
n1 = 7025
plt.figure(figsize=(14, 5))
plt.plot(y[n0:n1])

#chroma features
chromagram = librosa.feature.chroma_stft(y, sr=sr, hop_length=512)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=512, cmap='coolwarm')

