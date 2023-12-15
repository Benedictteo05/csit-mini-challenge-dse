import librosa
import numpy as np
import IPython.display as ipd

filename = 'CSIT_DS_Mini-Challenge\Task_1\T1_audio.wav'
y, sr = librosa.load(filename, sr=11025)

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print(tempo)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
print(beat_times)
ipd.Audio(filename, rate=sr)