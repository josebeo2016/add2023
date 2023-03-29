# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
from scipy.io.wavfile import write
import torch
from .mel2samp import files_to_list, MAX_WAV_VALUE
from .denoiser import Denoiser
from .glow import WaveGlow

WN_config={
    "n_layers": 8,
    "n_channels": 256,
    "kernel_size": 3
}
            

class Waveglow:
    def __init__(self,checkpoint, sigma=1.0,denoiser_strength=0.0, device='cuda'):
        self.model = WaveGlow(80,12,8,4,2,WN_config)
        self.model.remove_weightnorm(self.model)

        self.model.load_state_dict(torch.load(checkpoint,map_location=device))
        self.model.to(device).eval()
        self.sigma = sigma
        self.device = device
        self.denoiser_strength = denoiser_strength
        if denoiser_strength > 0:
            self.denoiser = Denoiser(self.model).to(device)
        
    def __call__(self,mel):
        return self.synthesize(mel)
    
    @torch.no_grad()
    def synthesize(self, mel):
        if not isinstance(mel,torch.Tensor):
            mel = torch.tensor(mel)
        mel = mel.to(self.device)
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.to(self.device)
        audio = self.model.infer(mel, sigma=self.sigma)
        if self.denoiser_strength > 0:
            audio = self.denoiser(audio, self.denoiser_strength)
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        return audio
