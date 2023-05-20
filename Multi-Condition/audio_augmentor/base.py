import librosa
import os

class BaseAugmentor():
    def __init__(self, input_path, config):
        """
        Basic augmentor class requires these config:
        aug_type: str, augmentation type
        output_path: str, output path
        out_format: str, output format
        """
        self.config = config
        self.aug_type = config["aug_type"]
        self.input_path = input_path
        self.output_path = config["output_path"]
        self.file_name = self.input_path.split("/")[-1].split(".")[0]
        self.out_format = config["out_format"]
        self.augmented_audio = None
        self.data = None
        self.sr = 16000

    def load(self):
        """
        Load audio file and normalize the data
        Librosa done this part
        self.data: audio data in numpy array (librosa load)
        """
        # load with librosa and auto resample to 16kHz
        self.data, self.sr = librosa.load(self.input_path, sr=self.sr)
        
        # Convert to mono channel
        self.data = librosa.to_mono(self.data)
    
    def transform(self):
        """
        Transform audio data (librosa load) to augmented audio data (pydub audio segment)
        Note that self.augmented_audio is pydub audio segment
        """
        raise NotImplementedError
    
    def save(self):
        """
        Save augmented audio data (pydub audio segment) to file
        self.out_format: output format
        This done the codec transform by pydub
        """
        self.augmented_audio.export(os.path.join(self.output_path,self.file_name +"."+ self.out_format), format=self.out_format)
    
    def run(self):
        """
        Run the augmentation pipeline
        """
        self.load()
        self.transform()
        self.save()
    