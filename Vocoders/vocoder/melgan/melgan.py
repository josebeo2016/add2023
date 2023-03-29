import os
import glob
import tqdm
import torch
import argparse
from scipy.io.wavfile import write
from torch import Tensor
from .model.generator import Generator
from .utils.hparams import HParam, load_hparam_str

MAX_WAV_VALUE = 32768.0

class MelGAN:
    def __init__(self,checkpoint,device='cpu',config=None):
        checkpoint_path = os.path.expanduser(checkpoint)
        config = os.path.expanduser(config)
        
        ckpt = torch.load(checkpoint_path)
        if config is not None:
            hp = HParam(config)
        else:
            hp = load_hparam_str(ckpt['hp_str'])

        self.model = Generator(hp.audio.n_mel_channels).to(device)
        self.model.remove_weight_norm()
        self.device = device
        self.model.load_state_dict(ckpt)
        self.model.eval(inference=False)
    @torch.no_grad()
    def synthesize(self,mel:Tensor):
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.to(self.device)

        audio = self.model.inference(mel)
        audio = audio.cpu().detach().numpy()
        return audio
    def __call__(self,mel):
        return self.synthesize(mel)

