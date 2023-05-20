from audio_augmentor import BackgroundNoiseAugmentor
import os
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
def test_run():
    SAMPLE_WAV = os.path.join(BASE_DIR,"../test/LA_T_1000137.flac")
    CONFIG = {
        "aug_type": "background_noise",
        "output_path": os.path.join(BASE_DIR,"../test/augmented"),
        "out_format": "flac",
        "noise_path": "./musan",
        "min_SNR_dB": 0,
        "max_SNR_dB": 20
    }
    bga = BackgroundNoiseAugmentor(SAMPLE_WAV, CONFIG)
    bga.run()
    assert os.path.exists(os.path.join(CONFIG["output_path"], bga.file_name +"."+ CONFIG["out_format"]))
    os.remove(os.path.join(CONFIG["output_path"], bga.file_name +"."+ CONFIG["out_format"]))
    