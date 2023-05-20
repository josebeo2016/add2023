from .background_noise import BackgroundNoiseAugmentor
from .pitch import PitchAugmentor
from .reverb import ReverbAugmentor

from . import utils

from .__version__ import (
    
    __author__,
    __author_email__,
    __description__,
    __license__,
    __title__,
    __url__,
    __version__,
)

SUPPORTED_AUGMENTORS = ['background_noise', 'pitch', 'speed', 'volume', 'reverb', 'compression', 'time_stretch', 'pitch_shift']