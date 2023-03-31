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


# Set up logging
logging.basicConfig(filename='running.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')


# augmentation model config
transform = AddBackgroundNoise(
    sounds_path="/path/to/folder_with_sound_files",
    min_snr_in_db=3.0,
    max_snr_in_db=30.0,
    noise_transform=PolarityInversion(),
    p=1.0
)

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    Y=process_Rawboost_feature(X,fs,args,args.algo)
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
