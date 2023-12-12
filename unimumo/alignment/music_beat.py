import librosa
import numpy as np

def music_beat(seg, sr, fps=20):
    hop_length = round(sr/fps)
    onset_env = librosa.onset.onset_strength(y=seg, sr=sr, aggregate=np.median, hop_length=hop_length)
    # times = librosa.frames_to_time(frames=np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    return beats