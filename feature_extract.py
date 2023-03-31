import os
import threading
from multiprocessing import Pool, set_start_method
import argparse
import yaml
from lfcc import lfcc
from lpc import lpc
import soundfile as sf
import numpy as np
from tqdm import *
from functools import partial
import logging
# from wav2vec.wav2vec2_wrapper import wav2vec2
from speechbrain.pretrained import VAD


# Set up logging
logging.basicConfig(filename='running.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

# https://stackoverflow.com/questions/25553919/passing-multiple-parameters-to-pool-map-function-in-python

def load_audio(path):
    try:
        data, sample_rate = sf.read(path)
    except Exception as e:
        logging.error("file {} cannot read with error: {}".format(path,e))
        exit(0)
    if data.size == 0:
        logging.error("file {} has zero size".format(path))
        exit(0)
    # Convert to mono and normalize
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # convert to mono
    # data = data / np.max(np.abs(data))  # normalize to [-1, 1]
    return data, sample_rate

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

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

def extract_feat(args, filename):
    # load audio:
    fpath = os.path.join(args.input_path, filename)
    data, fs = load_audio(fpath)
    if (args.trim):
        
        VAD_model = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")
        boundaries = VAD_model.get_speech_segments(fpath)
        trimed_audio = np.empty([0])
        for t in boundaries:
            start = int(t[0]*fs)
            end = int(t[1]*fs)
            trimed_audio = np.concatenate((trimed_audio, data[start:end]))
        data = trimed_audio
    
    if (args.pad_length > 16000):
        data = pad(data, max_len=args.pad_length)
    
    # if (args.feature_type == "wav2vec"):
    #     feat = wav2vec2(data=data, sr=fs).detach().cpu().numpy()
    # extract LFCC
    if (args.feature_type == "lfcc"):
        feat = lfcc(sig=data,
                fs=fs,
                num_ceps=args.num_ceps,
                pre_emph=args.pre_emph,
                pre_emph_coeff=args.pre_emph_coeff,
                win_len=args.win_len,
                win_hop=args.win_hop,
                win_type=args.win_type,
                nfilts=args.nfilts,
                nfft=args.nfft,
                low_freq=args.low_freq,
                high_freq=args.high_freq,
                scale=args.scale,
                dct_type=args.dct_type,
                normalize=args.normalize
                )
    if (args.feature_type == "lpc"):
        feat = lpc (sig=data,
                fs=fs,
                num_ceps=args.num_ceps,
                pre_emph=args.pre_emph,
                pre_emph_coeff=args.pre_emph_coeff,
                win_type=args.win_type,
                win_len=args.win_len,
                win_hop=args.win_hop,
                do_rasta=args.do_rasta,
                dither=args.dither
                    )
        
    # save to path
    feat_file = os.path.join(args.output_path, filename.replace("wav","npy"))
    np.save(feat_file, feat)
    # pbar.update(1)

def main():
    args = parse_argument()
    # prepare config:
    set_start_method("spawn")
    with open(args.config, 'r') as f_yaml:
        parser1 = yaml.safe_load(f_yaml)
        vars(args).update(parser1)
    filenames = os.listdir(args.input_path)
    num_files = len(filenames)
    if(args.feature_type in ["lfcc", "lpc", "mfcc"]):
        func = partial(extract_feat, args)
    with Pool(processes=args.thread) as p:
        with tqdm(total=num_files) as pbar:
            for _ in p.imap_unordered(func, filenames):
                pbar.update()

if __name__ == '__main__':
    main()
