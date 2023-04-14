# import required module
import os
import sys
import noisereduce as nr
import os
import threading
from multiprocessing import Pool, set_start_method
import argparse
import numpy as np
from tqdm import *
from functools import partial
import logging
import soundfile as sf
import io
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
    

    # load argument
    args = parser.parse_args()
        
    return args

def denoise(args, filename):
    # load audio:
    fpath = os.path.join(args.input_path, filename)
    out_file = os.path.join(args.output_path, filename)
    
    data, rate = sf.read(fpath)
    audioDenoiser = nr.reduce_noise(y = data, sr=rate, n_std_thresh_stationary=1.5,stationary=True)
    # save to path
    sf.write(out_file, audioDenoiser, rate, subtype='PCM_24')


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
