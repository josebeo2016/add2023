from audio_augmentor import ReverbAugmentor
import os
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def test_run():
    SAMPLE_WAV = os.path.join(BASE_DIR,"../test/LA_T_1000137.flac")
    CONFIG = {
        "aug_type": "reverb",
        "rir_path": os.path.join(BASE_DIR,"../RIRS_NOISES/simulated_rirs"),
        "output_path": os.path.join(BASE_DIR,"../test/augmented"),
        "out_format": "flac",
    }
    pa = ReverbAugmentor(SAMPLE_WAV, CONFIG)
    pa.run()
    assert os.path.exists(os.path.join(CONFIG["output_path"], pa.file_name +"."+ CONFIG["out_format"]))
    os.remove(os.path.join(CONFIG["output_path"], pa.file_name +"."+ CONFIG["out_format"]))
    