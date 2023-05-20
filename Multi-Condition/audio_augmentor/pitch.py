from .base import BaseAugmentor
import random
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment

class PitchAugmentor(BaseAugmentor):
    def __init__(self, input_path, config):
        """
        Pitch augmentation
        Config:
        min_pitch_shift: int, min pitch shift factor
        max_pitch_shift: int, max pitch shift factor
        """
        super().__init__(input_path, config)
        self.min_pitch_shift = config["min_pitch_shift"]
        self.max_pitch_shift = config["max_pitch_shift"]
        self.pitch_shift = random.randint(self.min_pitch_shift, self.max_pitch_shift)
        
    
    def transform(self):
        augmented_audio = librosa.effects.pitch_shift(self.data, sr=self.sr, n_steps=self.pitch_shift)
        # Convert to pydub audio segment
        data = np.array(augmented_audio * (1<<15), dtype=np.int16)
        # transform to pydub audio segment
        self.augmented_audio = AudioSegment(data.tobytes(), 
                    frame_rate=self.sr,
                    sample_width=data.dtype.itemsize, 
                    channels=1
                    )
    