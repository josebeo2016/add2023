import argparse
import os
import librosa
import glob
import tqdm
import numpy as np
import torch
import yaml
import logging
from scipy.io import wavfile
from vocoder import *

# Set up logging
logging.basicConfig(filename='running.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
def build_vocoder(device, config):
    vocoder_name = config['vocoder']['type']
    VocoderClass = eval(vocoder_name)
    model = VocoderClass(**config['vocoder'][vocoder_name])
    return model

def normalize(wav):
    assert wav.dtype == np.float32
    eps = 1e-6
    sil = wav[1500:2000]
    #wav = wav - np.mean(sil)
    #wav = (wav - np.min(wav))/(np.max(wav)-np.min(wav)+eps)
    wav = wav / np.max(np.abs(wav))
    #wav = wav*2-1
    wav = wav * 32767
    return wav.astype('int16')


def to_int16(wav):
    wav = wav = wav * 32767
    wav = np.clamp(wav, -32767, 32768)
    return wav.astype('int16')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mel_dir', type=str, default='./feats/')
    parser.add_argument('--output_dir', type=str, default='./outputs/')
    parser.add_argument('-c', '--config', type=str, default='./config.yaml')
    parser.add_argument('-d', '--device', type=str, default='cuda')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
        logging.info(f.read())
    
    out_dir = os.path.join(args.output_dir, config['vocoder']['type'])
    logging.info("The result wav is in {}".format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    sr = config['fbank']['sample_rate']
    logging.info("Load Vocoder")
    vocoder = build_vocoder(args.device, config)
    logging.info("Done")
    torch.set_grad_enabled(False)
    
    mel_files = glob.glob(os.path.join(args.mel_dir, '*.npy'),recursive=False)
    logging.info(f'{len(mel_files)} found in {args.mel_dir}')
    mel_dir = args.mel_dir
    logging.info(f'mel files will be saved to {mel_dir}')

    # Create all folders
    # os.makedirs(mel_path, exist_ok=True)
    for melpath in tqdm.tqdm(mel_files, desc='preprocess wav to mel'):

        mel = np.load(melpath)
        id = os.path.basename(melpath).replace(".npy", "")
        if (config['vocoder']['type']=="MelGAN"):
            mel = torch.from_numpy(mel)
        wav = vocoder(mel)
        # resample
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        if config['synthesis']['normalize']:
            wav = normalize(wav)
        else:
            wav = to_int16(wav)
        dst_file = os.path.join(out_dir, f'{id}.wav')

        # logging.info(f'writing file to {dst_file}')
        wavfile.write(dst_file, 16000, wav)