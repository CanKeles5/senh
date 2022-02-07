<<<<<<< HEAD
import utils
import metrics
import models
from scipy.io import wavfile
from scipy import signal
from scipy.io.wavfile import write #remove this and use wavfile.write
import numpy as np


if __name__ == '__main__':
    utils.build_whamr_like_dataset(speakers_path="D:\SpeechSeperationData\chunks", wham_noise_path="D:\whamr\wham_noise\wham_noise", output_root="D:\SpeechSeperationData")
    
=======
import utils
import metrics
import models
from scipy.io import wavfile
from scipy import signal
from scipy.io.wavfile import write #remove this and use wavfile.write
import numpy as np


if __name__ == '__main__':
    utils.build_whamr_like_dataset(speakers_path="D:\SpeechSeperationData\chunks", wham_noise_path="D:\whamr\wham_noise\wham_noise", output_root="D:\SpeechSeperationData")
    
>>>>>>> 5d8c10f3f90e687c07f31772b03fc74ce26177f1
