import glob
import numpy as np
import torch
import torchaudio
import os
import utils
from scipy.io import wavfile
from metrics import compute_metrics_bsseval


window_sizes = ["full"]
SAMPLING_RATE = 16000

"""
TO-Do:
- Create new folder for each results, for example root+model_name+split+window_size
"""

"""
For latency estimation, try torch profiler

import torch
import torchvision.models as models

model = models.densenet121(pretrained=True)
x = torch.randn((1, 3, 224, 224), requires_grad=True)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    model(x)
print(prof) 

"""


def evaluate_model(root_pth, audio_pth, save_pth, model_type, model, device="cpu"):
    
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)

    mix_paths = sorted(glob.glob(audio_pth + "/*.wav"))
    audio_data = [torchaudio.load(aud_pth, normalize=True)[0].numpy() for aud_pth in mix_paths]
    
    for window_size in window_sizes:
        audio_index = 0
        for audio in audio_data:
            audio_len = audio.shape[1]
            if model_type == "asteroid":
                res = np.array([[]])
            else:
                res = torch.Tensor(1, 1).to(device)
            
            if window_size == "full":
              res = model.numpy_separate(audio)[0]
            else:
              prev_ind = 0
              for a in range(min(int(window_size/1000*SAMPLING_RATE), audio_len-1), audio_len, int(window_size/1000*SAMPLING_RATE)):
                
                if model_type == "speechbrain":
                    output = model.separate_batch(audio_ptorch).squeeze(2)
                    #convert output to numpy
                elif model_type == "asteroid":
                    output = model.numpy_separate(np.expand_dims(audio[0][prev_ind:a], axis=0))
                    
                res = np.concatenate( (res, np.squeeze(output, axis=0)), axis=1)
                prev_ind = a
                
                
            res_path = os.path.join(save_pth, os.path.split(mix_paths[audio_index])[-1].split(".")[0] +  ".wav")
            wavfile.write(res_path, SAMPLING_RATE, res[0].astype(np.float32))
            audio_index += 1
    
    return compute_metrics_bsseval(root_pth)

