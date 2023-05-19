# import required module
import os
from multiprocessing import Pool, set_start_method
import argparse
from tqdm import *
from functools import partial
import logging
import librosa
import soundfile as sf
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
from random import randrange
import torch
from torch_audiomentations import Compose, AddBackgroundNoise, PolarityInversion
from pydub import AudioSegment
import random

def add_noise(audio_file_path, noise_file_path, min_SNR_dB=0, max_SNR_dB=20):
    # Load audio files
    audio_file = AudioSegment.from_file(audio_file_path)
    noise_file = AudioSegment.from_file(noise_file_path)

    # Set the desired SNR (signal-to-noise ratio) level in decibels
    SNR_dB = random.randint(min_SNR_dB, max_SNR_dB)

    # Calculate the power of the signal and noise
    signal_power = audio_file.dBFS
    noise_power = noise_file.dBFS

    # Calculate the scaling factor for the noise
    scaling_factor = 10 ** ((signal_power - SNR_dB - noise_power) / 20)

    # Apply the noise to the audio file
    augmented_audio = audio_file.overlay(noise_file - random.uniform(0.0, 1.0) * 0.05 * noise_file.dBFS, position=0)
    
    return augmented_audio

# Set up logging
logging.basicConfig(filename='running.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')


def parse_argument():
    parser = argparse.ArgumentParser(
        epilog=__doc__, 
        formatter_class=argparse.RawDescriptionHelpFormatter)
        

    parser.add_argument('--thread', type=int, default=16, help='Number of threads')

    parser.add_argument('--input_path', type=str, default="",required=True, help='Audio file path')
    
    parser.add_argument('--output_path', type=str, default="",required=True, help='Feature output path')

    parser.add_argument('--aug_type', type=str, default="env_noise", required=True, help='Augmentation type.\n'
                        +'Suported: background_noise, pitch, speed, volume, reverb, compression, time_stretch, pitch_shift')

    # env_noise
    parser.add_argument('--noise_path', type=str, default="./musan/",required=False, help='Noise file path')
    
    # load argument
    args = parser.parse_args()
        
    return args

def recursive_list_files(path):
    """Recursively lists all files in a directory and its subdirectories"""
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return files

def select_noise(noise_path):
    noise_list = recursive_list_files(noise_path)
    noise_file = random.choice(noise_list)
    return noise_file
    
def background_noise(args, filename):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    out_file = os.path.join(args.output_path, filename)

    noise_sample = select_noise(args.noise_path)
    augmented_audio = add_noise(in_file,noise_sample)
    # Export the augmented audio file
    augmented_audio.export(out_file, format='flac')
    
    # save to path
    # sf.write(out_file, Y, fs, subtype='PCM_16')


def main():
    args = parse_argument()
    # prepare config:
    set_start_method("spawn")
    filenames = os.listdir(args.input_path)
    num_files = len(filenames)
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    
    func = partial(args.aug_type, args)
    with Pool(processes=args.thread) as p:
        with tqdm(total=num_files) as pbar:
            for _ in p.imap_unordered(func, filenames):
                pbar.update()

if __name__ == '__main__':
    main()
