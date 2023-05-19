import librosa

class BaseAugmentor():
    def __init__(self, input_path, config):
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
        # load with librosa and auto resample to 16kHz
        self.data, self.sr = librosa.load(self.input_path, sr=self.sr)
        
        # Convert to mono channel
        self.data = librosa.to_mono(self.data)
    
    def transform(self):
        raise NotImplementedError
    
    def save(self):
        raise NotImplementedError
    
    def run(self):
        self.load()
        self.transform()
        self.save()
    