from .base import BaseAugmentor
from .utils import recursive_list_files
import scipy.signal as ss
import librosa
import random
import numpy as np
from pydub import AudioSegment

class ReverbAugmentor(BaseAugmentor):
    def __init__(self, input_path, config):
        """
        Reverb augmentation
        Config:
        rir_path: str, path to the folder containing RIR files 
        (RIR dataset example https://www.openslr.org/28/)
        """
        super().__init__(input_path, config)
        self.rir_path = config["rir_path"]
        self.rir_file = self.select_rir(self.rir_path)
        
    def select_rir(self, rir_path):
        rir_list = recursive_list_files(rir_path)
        return random.choice(rir_list)
        
    def transform(self):
        rir_data, _ = librosa.load(self.rir_file, sr=self.sr)
        # Compute convolution
        reverberate = np.convolve(self.data, rir_data)
        # Normalize output signal to avoid clipping
        reverberate /= (np.max(np.abs(reverberate)))
        
        # transform to pydub audio segment
        self.augmented_audio = AudioSegment(reverberate.tobytes(), 
                    frame_rate=self.sr,
                    sample_width=reverberate.dtype.itemsize, 
                    channels=1
                    )
    
        