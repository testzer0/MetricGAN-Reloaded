"""
Adithya Bhaskar, 2022.
This file contains functions that call the model functions and modify the
output in appropriate ways (e.g. replacing audio in a video file by the cleaned
version) before returing. Basically, all usage of the model other than training
goes here.
"""

from config import *
from utils.globals import *
from utils.data import wav_to_spectrogram, spectrogram_to_wav
import torch
import numpy as np
import librosa
import soundfile as sf
import subprocess
from pydub import AudioSegment

def run_generator_on_path(generator, in_path, out_path=None):
	"""
	Run the generator on the audio from 'in_path', and saved the denoised
	audio to 'out_path'. If the latter is None, out_path for <x>.wav is
	taken as <x>-out.wav
	"""
	if out_path is None:
		out_path = in_path.split('.wav')[0] + '-out.wav'
	noisy_wav, _ = librosa.load(in_path, sr=16000)
	noisy_spectrogram_normalized, _, _ = wav_to_spectrogram( \
		noisy_wav*SCALE_FACTOR, normalize=True)
	noisy_spectrogram, phase, length = wav_to_spectrogram( \
		noisy_wav*SCALE_FACTOR)

	# The spectrograms now have the shape, (num_frames, frame_dim)
	# which is what we want to give to the generator, since it expects 
	# (batch_size, seq_length, input_size) when batch_first=True is passed
	noisy_spectrogram_normalized = torch.from_numpy(noisy_spectrogram_normalized)
	noisy_spectrogram = torch.from_numpy(noisy_spectrogram)
	mask = MASK_MIN_VALUE * torch.ones((noisy_spectrogram.shape[0], 257))

	noisy_spectrogram_normalized = noisy_spectrogram_normalized.unsqueeze(0).to(device)
	noisy_spectrogram = noisy_spectrogram.unsqueeze(0).to(device)
	mask = mask.unsqueeze(0).to(device)
	output_generator = generator(noisy_spectrogram_normalized)
	mask = torch.maximum(output_generator, mask)
	cleaned = torch.mul(mask, noisy_spectrogram).squeeze().cpu().detach().numpy()
	cleaned_wav = spectrogram_to_wav(cleaned.T, phase, length) / SCALE_FACTOR
	sf.write(out_path, cleaned_wav, 16000)
	
def mix_audio_wavs(signal, noise, snr):
	"""
	Source: https://stackoverflow.com/questions/71915018/mix-second-audio-clip-at-specific-snr-to-original-audio-file-in-python
	This function has been taken from the above source, with one difference.
	In the source, signal and audio are combined so that the final energy is
	the same as the original signal. That is not how actual noise works, so
	we instead scale up the same so that now the noise is literally *added*
	to the signal at the specified SNR level.
	"""
	# if the audio is longer than the noise
	# play the noise in repeat for the duration of the audio
	noise = noise[np.arange(len(signal)) % len(noise)]

	# if the audio is shorter than the noi
	# this is important if loading resulted in 
	# uint8 or uint16 types, because it would cause overflow
	# when squaring and calculating mean
	noise = noise.astype(np.float32)
	signal = signal.astype(np.float32)

	# get the initial energy for reference
	signal_energy = np.mean(signal**2)
	noise_energy = np.mean(noise**2)
	# calculates the gain to be applied to the noise 
	# to achieve the given SNR
	g = np.sqrt(10.0 ** (-snr/10) * signal_energy / noise_energy)

	# Assumes signal and noise to be decorrelated
	# and calculate (a, b) such that energy of 
	# a*signal + b*noise matches the energy of the input signal
	a = np.sqrt(1 / (1 + g**2))
	b = np.sqrt(g**2 / (1 + g**2))
	
	# mix the signals
	return signal + g * noise

def add_noise(sound_file, noise_file, snr=0.0, out_file=None):
	"""
	Reads in the sound and noise from the specified files, adds
	noise to the sound at SNR level snr dB, then writes the result
	to out_file. If the latter is None, the name for when the
	sound_file is <x>.wav is <x>-noisy.wav.
	"""
	if out_file is None:
		out_file = sound_file.split('.wav')[0] + '-noisy.wav'
	sound_wav, _ = librosa.load(sound_file, sr=16000)
	noise_wav, _ = librosa.load(noise_file, sr=16000)
	mixed = mix_audio_wavs(sound_wav, noise_wav, snr)
	sf.write(out_file, mixed, 16000)

def convert_m4a_to_wav(m4a, wav=None):
	"""
	Converts <x>.m4a to <x>.wav if wav=None, otherwise to
	<wav>.wav.
	"""
	if wav is None:
		wav = m4a.split(".m4a")[0] + ".wav"
	track = AudioSegment.from_file(m4a,  format= 'm4a')
	file_handle = track.export(wav, format='wav')

def full_cycle(generator, file_path, noise_path, snr=0.0, from_m4a=True):
	"""
	Does a full cycle, i.e.
	1) Optionally converts the file from .m4a to .wav.
	2) Adds noise to the file and saves to <name>-noisy.wav
	3) Recovers the cleaned version using our model, and saves
	  it to <name>-noisy-out.wav.
	"""
	if from_m4a:
		convert_m4a_to_wav(file_path)
		file_path = file_path.split('.m4a')[0] + '.wav'
	add_noise(file_path, noise_path, snr)
	noisy_path = file_path.split('.wav')[0] + '-noisy.wav'
	run_generator_on_path(generator, noisy_path)

def clean(generator, file_path, out_path=None, from_m4a=True):
	"""
	Cleans a noisy audio. Can take in both .m4a and .wav files.
	"""
	if from_m4a:
		convert_m4a_to_wav(file_path)
		file_path = file_path.split('.m4a')[0] + '.wav'
	run_generator_on_path(generator, file_path, out_path)

def clean_video(generator, file_path, out_path=None):
	"""
	Extracts audio from the video, cleans it, and then pastes it back 
 	on the video. The video *must* be in mp4 format.
	Uses the ffmpeg command.
	"""
	audio_path = file_path.split(".mp4")[0] + ".wav"
	command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format( \
	 file_path, audio_path)
	p = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, \
     stdin=subprocess.PIPE, shell=True)
	p.stdin.write(b'y\n')
	clean(generator, audio_path)
	cleaned_path = audio_path.split(".wav")[0] + "-out.wav"
	new_path = file_path.split(".mp4")[0] + "-cleaned.mp4"
	command = "ffmpeg -i {} -i {} -c:v copy -map 0:v:0 -map 1:a:0 {}".format( \
	 file_path, cleaned_path, new_path)
	p = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, \
     stdin=subprocess.PIPE, shell=True)
	p.stdin.write(b'y\n')

if __name__ == '__main__':
		pass