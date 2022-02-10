import utils
import metrics
import models
from scipy.io import wavfile
from scipy import signal
from scipy.io.wavfile import write #remove this and use wavfile.write
import numpy as np


if __name__ == '__main__':
    utils.build_whamr_like_dataset(speakers_path="D:\SpeechSeperationData\chunks", wham_noise_path="D:\whamr\wham_noise\wham_noise", output_root="D:\SpeechSeperationData")
    

