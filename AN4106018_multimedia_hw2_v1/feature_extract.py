from math import ceil
import warnings
import librosa
import matplotlib.pyplot as plt
import librosa.display
# import IPython.display as ipd
# from os import path
from scipy.io import wavfile
import numpy as np
# from pydub import AudioSegment

warnings.filterwarnings("ignore")

### convert mp3 into wav
# src = "audio.mp3"
# dst = "output.wav"
# sound = AudioSegment.from_mp3(src)
# sound.export(dst,format="wav")

### load audio file
audio_path = "audio.wav"
y, sr = librosa.load(audio_path) # y = audio time series, a float32 numpy seq; sr = sampling rate

### play audio
# print("playing audio : ...")
# ipd.Audio(audio_path)
# print("finish playing")

### waveform ######################################################
# plt.figure(figsize=(10,3))
# librosa.display.waveshow(y, sr=sr, color="blue")
# plt.title("waveform")
# plt.savefig("waveform.png")

#######################################################################

### sampling ###############################################################
frame_len = int(25 * sr / 1000)
frame_step = int(10 * sr / 1000)
N = frame_step     # window size
eframes = np.array([y[i:i+frame_len] for i in range(0,len(y)-frame_len, frame_step)])
# rms = librosa.feature.rms(y=y)[0]
# zframes = range(len(rms))

### energy contour ##################################################
e = np.sum(eframes**2, axis=1) 
plt.figure(figsize=(10,3))
plt.plot(e)
plt.title("short time energy")
plt.savefig("short_time_energy.png")

#####################################################################

### zero-crossing rate contour ############################################
# t = librosa.frames_to_time(zframes)
z = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=frame_step)[0]
plt.figure(figsize=(10,3))
# plt.plot(t,z, label="line 1")
plt.plot(z, label="line 2")
plt.title("zero-crossing rate")
plt.savefig("zero_crossing_rate.png")

#########################################################################

### end point detection #################################################
# samples that equals 100ms
hundredmsec_rel = round((sr*.2)/frame_step)
chop = ceil((N/2-frame_step)/frame_step)
noise_e = e[2+chop:hundredmsec_rel-chop]
noise_z = z[2+chop:hundredmsec_rel-chop]
noise_e_mean = np.mean(noise_e)
noise_e_std  = np.std(noise_e)
noise_z_mean = np.mean(noise_z)
noise_z_std  = np.std(noise_z)
print(f"e mean : {noise_e_mean} , e std : {noise_e_std}, z mean : {noise_z_mean}, z std : {noise_z_std}")

# define ITU,ITL IZCT
i1 = 0.03*(max(e) - min(e)) + min(e)
i2 = 4 * min(e)
print(f"i1 : {i1} , i2 : {i2}")
IF = 25
ITL = min(i1, i2)
ITU = 5 * ITL
IZCT = min(IF, noise_z_mean + 2*noise_z_std)
# fudge = 5
# ITL = noise_e_mean + fudge*noise_e_std
# IZCT = noise_z_mean + fudge*noise_z_std
# ITU = 3.2*noise_e_mean      # since std << mean, twice the mean should cover it
print(f"ITL : {ITL} , ITU : {ITU}, IZCT : {IZCT}")

## front-end
start = 3
avg_last3pts = 0
# find the first point that exceeds ITU
while avg_last3pts < ITU:
    start = start + 1
    avg_last3pts = (e[start] + e[start-1] + e[start-2])/3

# check backwards until magnitude is below ITL
while e[start] > ITL:
    start = start - 1

# check forward until magnitude is below IZCT
below_izct_count = 0
first_below = -999

if start > 25:
    for i in range(start, start-25, -1):
        if z(i) < IZCT:
            below_izct_count = below_izct_count + 1
            if first_below == -999:
                first_below = i
    if below_izct_count >= 3:
        start = first_below

## back-end
endpt = len(e)-2
avg_last3pts = 0
# find the first point that exceeds ITU
while avg_last3pts < ITU:
    start = start - 1
    avg_last3pts = (e[start] + e[start+1] + e[start+2])/3

# check backwards until magnitude is below ITL
while e[start] > ITL:
    start = start + 1

# check forward until magnitude is below IZCT
below_izct_count = 0
first_below = -999

if (endpt - len(z)) > 25:
    for i in range(endpt, endpt+25, 1):
        if z(i) < IZCT:
            below_izct_count = below_izct_count + 1
            if first_below == -999:
                first_below = i
    if below_izct_count >= 3:
        endpt = first_below
        
mag = e[start:endpt]
pts = [start*frame_step,endpt*frame_step]

# print(f"start : {start}, endpt : {endpt}")
# print(mag)

# output the plot
# plt.figure(figsize=(10,3))
# plt.plot(mag)
# plt.title("end point detection")

#######################################################################

### pitch contour ######################################################
# pitches = librosa.core.yin(y=y, sr=sr, fmax=1000, fmin= 62)
# plt.figure(figsize=(10,3))
# plt.plot(pitches)
# plt.title("pitch contour")
# plt.savefig("pitch.png")

#######################################################################

### spectrogram in freq ######################################################
plt.figure(figsize=(10,3))
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr)
plt.colorbar(location="right")
plt.title("spectrogram")
plt.xlabel("time")
plt.ylabel("frequency")
plt.savefig("spectrogram.png")

plt.show()