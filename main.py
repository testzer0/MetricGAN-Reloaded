"""
Adithya Bhaskar, 2022.
This is the main file, where we take in commands and user inputs, and
call the respective modules/functions to do their job.
"""

from config import *
from utils.globals import *
from utils.data import get_MS_file_pairs
from model.model import load_latest_checkpt_only_generator, \
    get_models_and_maybe_optimizers, train
from model.utility import *
import argparse

generator = None

def do_training():
    """
    Read in the data, prepare datasets and the model, then train.
    """
    global generator
    file_pairs, clean_pairs, high_pairs = get_MS_file_pairs(DATASET_DIR)
    generator, discriminator, generator_optimizer, discriminator_optimizer = \
        get_models_and_maybe_optimizers()
    train(generator, discriminator, generator_optimizer, discriminator_optimizer, \
        file_pairs, clean_pairs, high_pairs)
    print("Training done.")
    
def denoise_audio_file(audio_file, out_path=None):
    """
    Simple wrapper around the audio denoising function, ensures that generator
    is not None before calling the 'real' function.
    """
    global generator
    if generator is None:
        generator, _ = get_models_and_maybe_optimizers(get_opts=False)
        load_latest_checkpt_only_generator(generator)
    if audio_file[-4:] not in ['.wav', '.m4a']:
        print("Audio format must be either .wav or .m4a !")
        exit(-1)
    is_m4a = audio_file.endswith('.m4a')
    clean(generator, audio_file, out_path, is_m4a)
    
def denoise_video_file(video_file, out_path):
    """
    Another wrapper.
    """
    global generator
    if generator is None:
        generator, _ = get_models_and_maybe_optimizers(get_opts=False)
        load_latest_checkpt_only_generator(generator)
    if video_file[-4:] != ".mp4":
        print("Video format must be .mp4 !")
        exit(-1)
    clean_video(generator, video_file, out_path)

def do_full_cycle(audio_file, noise_file, snr=0.0):
    """
    Wrapper on the full_cycle function, which optionally converts to .wav,
    then adds noise, and then denoises. Can be used for demonstration
    purposes.
    """
    global generator
    if generator is None:
        generator, _ = get_models_and_maybe_optimizers(get_opts=False)
        load_latest_checkpt_only_generator(generator)
    if audio_file[-4:] not in ['.wav', '.m4a']:
        print("Audio format must be either .wav or .m4a !")
        exit(-1)
    is_m4a = audio_file.endswith('.m4a')
    full_cycle(generator=generator, file_path=audio_file, noise_path=noise_file, \
        snr=snr, from_m4a=is_m4a)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Train the model", action="store_true")
    parser.add_argument("--full-cycle", help="If this is set, --audio will be interpreted "
        "as the input audio and --noise should point to the noise file.", action="store_true")
    parser.add_argument("--noise", help="The noise file for full-cycle", type=str)
    parser.add_argument("--snr", help="The snr value for full-cycle (default 0.0)", type=float)
    parser.add_argument("--audio", help="The audio file to be denoised. In the "
        "absence of --full-cycle this file will be denoised", type=str)
    parser.add_argument("--video", help="The video file whose audio should be denoised. In the "
        "absence of --full-cycle this file will be denoised", type=str)
    parser.add_argument("--out", help="Name of the output file (optional)", type=str)
    
    args = parser.parse_args()
    
    something = False
    if args.full_cycle and (args.audio is None or args.noise is None):
        print("Full cycle needs --audio as well as --noise to be specified!")
        exit(-1)
    if args.audio is not None and args.video is not None:
        print("Cannot denoise both audio and video in a single run!")
        exit(-1)
    
    if args.train:
        something = True
        do_training()
    if args.full_cycle:
        something = True
        if args.snr is not None:
            snr = args.snr
        else:
            snr = 0.0
        do_full_cycle(args.audio, args.noise, snr)
    elif args.audio:
        something = True
        denoise_audio_file(args.audio, args.out)
    if args.video:
        something = True
        denoise_video_file(args.video, args.out)
        
    if not something:
        parser.print_usage()