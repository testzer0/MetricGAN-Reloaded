"""
Adithya Bhaskar, 2022.
This file contains all the helper functions that help with manipulation of data --
this includes initializing and updating the file pairs for training, reading of .mp4,
.m4a and .wav files, and their interconversion and such.
"""

from config import *
import os
import librosa
import numpy as np
import torch
import scipy
from pystoi import stoi

def get_MS_file_pairs(root_dir, split='train', snrs=[0.0, 10.0, 20.0], high=30.0):
    """
    Given the root directory for the data, reads in the training files 
    corresponding to the SNRs in 'snrs'. Pairs of (clean, clean) and 
    (clean, almost clean using dB of 'high') are also returned to aid 
    in training of the discriminator. 
    """
    clean_dir = os.path.join(root_dir, split+'/clean')
    noisy_dir = os.path.join(root_dir, split+'/noisy')
    data = []
    clean_data = []
    high_data = []
    for fname in os.listdir(clean_dir):
        if not (fname.startswith('clnsp') and fname.endswith('.wav')):
            continue
        example_number = int(fname[5:-4])
        for snr in snrs:
            noisy_name = "noisy{}_SNRdb_{:.1f}_clnsp{}.wav".format(example_number, snr, example_number)
            if os.path.isfile(os.path.join(noisy_dir, noisy_name)):
                data.append((os.path.join(clean_dir, fname), os.path.join(noisy_dir, noisy_name)))
                clean_data.append((os.path.join(clean_dir, fname), os.path.join(clean_dir, fname)))
        noisy_name = "noisy{}_SNRdb_{:.1f}_clnsp{}.wav".format(example_number, high, example_number)
        if os.path.isfile(os.path.join(noisy_dir, noisy_name)):
            high_data.append((os.path.join(clean_dir, fname), os.path.join(noisy_dir, noisy_name)))
    clean_data = list(set(clean_data))
    return data, clean_data, high_data

def wav_to_spectrogram(wav, normalize=False):
    """
    Given a wav file read in by librosa, performs STFT, then optionally 
    normalizes the result. Returns the magnitude, the phase of the STFT, 
    and signal length.
    """
    orig_length = wav.shape[0]
    n_fft = 512                 # Window size *after* padding with zeros
    wav_padded = librosa.util.fix_length(data=wav, size=(orig_length + (n_fft//2)))
                                # Pad the signal for FFT
    epsilon = 1e-12

    stft = librosa.stft(wav_padded, n_fft=n_fft, hop_length=(n_fft//2), win_length=n_fft, window=scipy.signal.hamming)
    result = np.abs(stft)
    phase = np.angle(stft)

    if normalize:
        mean = np.mean(result, axis=1).reshape((257,1))
        std = np.std(result, axis=1).reshape((257,1)) + epsilon
        result = (result-mean)/std
  
    result = np.reshape(result.T, (result.shape[1], 257))
    return result, phase, orig_length

def spectrogram_to_wav(stft, phase, signal_length):
    """
    Convert a spectrogram back to the original audio
    """
    scaled = np.multiply(stft, np.exp(1j*phase)) # Reconstruct the stft result from abs and phase
    result = librosa.istft(scaled, hop_length=256, win_length=512, window=scipy.signal.hamming, length=signal_length)
    return result

def get_path_for_generator(path, epoch, create=False):
    """
    Given a path to the noisy wav file, returns the name/path that should 
    be given to the generator's output wav file in the i-th training epoch.
    Unfortunately, while other places use 1-indexing for epoch, 
    this uses 0-indexing.
    """
    file_name = path.split('/')[-1]
    if create:
        if not os.path.exists('{}/epoch{}'.format(CHECKPT_DIR, epoch)):
            os.mkdir('{}/epoch{}'.format(CHECKPT_DIR, epoch))
    return '{}/epoch{}/{}'.format(CHECKPT_DIR, epoch, file_name)

def get_generator_sample(file_pair):
    """
    Given a file pair for (clean, noisy), reads the audio in and creates an 
    appropriate training/test sample for the Generator. It seems with noisy 
    audio, librosa clips off some audio because its conversion from mel to 
    stft is lossy. See here: 
    https://stackoverflow.com/questions/60365904/reconstructing-audio-from-a-melspectrogram-has-some-clipping-with-librosa
    Thus, we multiply the noisy audio by a constant (10), and later scale 
    the output down by the same amount.
    """
    clean_file, noisy_file = file_pair
    noisy_wav, _ = librosa.load(noisy_file, sr=16000)
    noisy_spectrogram_normalized, _, _ = wav_to_spectrogram( \
        noisy_wav*SCALE_FACTOR, normalize=True)
    noisy_spectrogram, phase, length = wav_to_spectrogram( \
        noisy_wav*SCALE_FACTOR)

    clean_wav, _ = librosa.load(clean_file, sr=16000)
    clean_spectrogram, _, _ = wav_to_spectrogram(clean_wav)

    # The spectrograms now have the shape, (num_frames, frame_dim)
    # which is what we want to give to the generator, since it expects (batch_size, seq_length, input_size)
    # when batch_first=True is passed
    noisy_spectrogram_normalized = torch.from_numpy( \
        noisy_spectrogram_normalized)
    noisy_spectrogram = torch.from_numpy(noisy_spectrogram)
    clean_spectrogram = torch.from_numpy(clean_spectrogram)
    mask = MASK_MIN_VALUE * torch.ones((noisy_spectrogram.shape[0], 257))

    return noisy_spectrogram_normalized, noisy_spectrogram, \
        clean_spectrogram, mask, phase, length
  
def get_discriminator_sample(file_pair):
    """
    The analogous function for the discriminator. Here we pass in a 
    'clean' sample and a corresponding 'noisy' sample -- except, the noisy 
    sample may also be clean. Irrespective, it needs to be scaled by the scale 
    factor as usual. We want to train the disciminator to give a 
    score close to 1 for clean samples and a score close to 0 for noisy ones. 
    Whether it is dirty may be found by checking whether 'SNRdb' appears in its name.
    """
    clean_file, noisy_file = file_pair
    noisy_wav, _ = librosa.load(noisy_file, sr=16000)
    noisy_spectrogram, _, _ = wav_to_spectrogram(noisy_wav*SCALE_FACTOR)
    clean_wav, _ = librosa.load(clean_file, sr=16000)
    clean_spectrogram, phase, sr = wav_to_spectrogram(clean_wav)
    true_stoi_noisy = torch.tensor([float(stoi(x=clean_wav, y=noisy_wav, fs_sig=16000, extended=False))])

    # both spectrograms are of the shape (1, n_frames, 257) now
    input_np_noisy = np.stack((noisy_spectrogram, clean_spectrogram), axis=-1)
    input_torch_noisy = torch.from_numpy(input_np_noisy)

    # Now the input is of shape (n_frames, 257, 2) - we need it to be (2, n_frames, 257)
    input_torch_noisy = input_torch_noisy.permute(2,0,1)
    return input_torch_noisy, true_stoi_noisy, phase, sr

if __name__ == '__main__':
    pass