from .base import BaseAugmentor
from .utils import recursive_list_files
from pydub import AudioSegment
import random
import numpy as np

class BackgroundNoiseAugmentor(BaseAugmentor):
    def __init__(self, input_path, config):
        """
        Background noise augmentation
        Config:
        noise_path: str, path to the folder containing noise files
        min_SNR_dB: int, min SNR in dB
        max_SNR_dB: int, max SNR in dB
        """
        super().__init__(input_path, config)
        self.noise_path = config["noise_path"]
        self.noise_list = self.select_noise(self.noise_path)
        self.min_SNR_dB = config["min_SNR_dB"]
        self.max_SNR_dB = config["max_SNR_dB"]
        
    def select_noise(self, noise_path):
        noise_list = recursive_list_files(noise_path)
        return noise_list
    
    def load(self):
        # load with librosa
        super().load()
        
        # Convert to pydub audio segment
        self.data = np.array(self.data * (1<<15), dtype=np.int16)
        self.audio_data = AudioSegment(self.data.tobytes(), 
                    frame_rate=self.sr,
                    sample_width=self.data.dtype.itemsize, 
                    channels=1
                    )
    
    def transform(self):
        # Load audio files
        noise_file = AudioSegment.from_file(random.choice(self.noise_list))
    
        # Set the desired SNR (signal-to-noise ratio) level in decibels
        SNR_dB = random.randint(self.min_SNR_dB, self.max_SNR_dB)
    
        # Calculate the power of the signal and noise
        signal_power = self.audio_data.dBFS
        noise_power = noise_file.dBFS
    
        # Calculate the scaling factor for the noise
        scaling_factor = SNR_dB * noise_power / signal_power
    
        # Apply the noise to the audio file
        scaled_audio = self.audio_data.apply_gain(scaling_factor)
        self.augmented_audio = scaled_audio.overlay(noise_file)
        
        
    
    