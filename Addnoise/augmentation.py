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

def add_noise(audio_file_path, noise_file_path):
    # Load audio files
    audio_file = AudioSegment.from_file(audio_file_path)
    noise_file = AudioSegment.from_file(noise_file_path)

    # Set the desired SNR (signal-to-noise ratio) level in decibels
    SNR_dB = 10

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
    
    mes = 'Config file'
    parser.add_argument('--config', type=str, default="./config.yaml", help=mes)
    
    mes = 'Number of threads'
    parser.add_argument('--thread', type=int, default=16, help=mes)

    mes = 'Audio file path'
    parser.add_argument('--input_path', type=str, default="",required=True, help=mes)

    mes = 'Feature output path'
    parser.add_argument('--output_path', type=str, default="",required=True, help=mes)
    
    # load argument
    args = parser.parse_args()
        
    return args

def addnoise(args, filename):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    out_file = os.path.join(args.output_path, filename)
    X,fs = librosa.load(in_file, sr=16000) 
    
    augmented_audio = add_noise(in_file,"pink_noise.wav")
    # Export the augmented audio file
    augmented_audio.export('augmented_audio.wav', format='wav')
    
    # save to path
    sf.write(out_file, Y, fs, subtype='PCM_24')


def main():
    args = parse_argument()
    # prepare config:
    set_start_method("spawn")
    filenames = os.listdir(args.input_path)
    num_files = len(filenames)
    
    func = partial(addnoise, args)
    with Pool(processes=args.thread) as p:
        with tqdm(total=num_files) as pbar:
            for _ in p.imap_unordered(func, filenames):
                pbar.update()

if __name__ == '__main__':
    main()
