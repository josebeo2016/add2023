from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from .env import AttrDict
from .meldataset import MAX_WAV_VALUE
from .models import Generator



def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

class HiFiGAN:
    def __init__(self,checkpoint:os.PathLike, h=None, device='cuda'):
        
        checkpoint = os.path.expanduser(checkpoint)
        config_file = os.path.join(os.path.split(checkpoint)[0], 'config.json')
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        self.generator = Generator(h).to(device)
        state_dict_g = load_checkpoint(checkpoint, device)
        self.generator.load_state_dict(state_dict_g['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()
        self.device = device
    

    def inference(self,x):
        with torch.no_grad():
            if isinstance(x,np.ndarray):
                x = torch.FloatTensor(x).to(self.device)
            else:
                x = x.to(self.device)
            y_g_hat = self.generator(x.unsqueeze(0))
            audio = y_g_hat.squeeze()
            audio = audio.cpu().numpy()

        return audio
    def __call__(self,x):
        return self.inference(x)



