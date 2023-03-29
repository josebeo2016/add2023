# import required module
import os
import sys
from denoise import AudioDeNoise
import os
import threading
from multiprocessing import Pool, set_start_method
import argparse
import numpy as np
from tqdm import *
from functools import partial
import logging
# Set up logging
logging.basicConfig(filename='running.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

# https://stackoverflow.com/questions/25553919/passing-multiple-parameters-to-pool-map-function-in-python

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
    
    mes = 'Padding length in number of samples, should be > 16000'
    parser.add_argument('--pad_length', type=int, default=64600, help=mes)
    
    mes = 'Trim non-speech segments'
    parser.add_argument('--trim', action="store_true", help=mes)
    
    mes = 'Type of feature: [lfcc, mfcc, wav2vec2] '
    parser.add_argument('--feature_type', type=str, default="lfcc", help=mes)

    # load argument
    args = parser.parse_args()
        
    return args

def denoise(args, filename):
    # load audio:
    fpath = os.path.join(args.input_path, filename)
    out_file = os.path.join(args.output_path, filename)
    audioDenoiser = AudioDeNoise(inputFile=fpath)
    audioDenoiser.deNoise(outputFile=out_file)
        
    # save to path


def main():
    args = parse_argument()
    # prepare config:
    set_start_method("spawn")
    filenames = os.listdir(args.input_path)
    num_files = len(filenames)
    
    func = partial(denoise, args)
    with Pool(processes=args.thread) as p:
        with tqdm(total=num_files) as pbar:
            for _ in p.imap_unordered(func, filenames):
                pbar.update()

if __name__ == '__main__':
    main()
