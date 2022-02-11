import glob
import numpy as np
import torch
import torchaudio
import os
from scipy.io import wavfile
from asteroid.metrics import get_metrics


window_sizes = ["full"]
SAMPLING_RATE = 16000

"""
TO-Do:
- Move the compute metrics to the metrics file & use pb_bss_eval metrics if possible
- Test all models.
- Add model options to download
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


def compute_metrics(root_pth, save_pth):
    res_dict = {
      'input_pesq': 0.0,
      'input_sar': 0.0,
      'input_sdr': 0.0,
      'input_si_sdr': 0.0,
      'input_sir': 0.0,
      'input_stoi': 0.0,
      'pesq': 0.0,
      'sar': 0.0,
      'sdr': 0.0,
      'si_sdr': 0.0,
      'sir': 0.0,
      'stoi': 0.0
    }
    
    mix_paths = glob.glob(root_pth + "/mix/*.wav")
    clean_paths = glob.glob(root_pth + "/clean/*.wav") #Re factor this path
    res_paths = glob.glob(save_pth + "/*.wav")

    for mix_path, clean_path, res_path in zip(sorted(mix_paths), sorted(clean_paths), sorted(res_paths)):
      #compute metrics here
      mix = open_audio(mix_path)
      clean = open_audio(clean_path)
      est = open_audio(res_path)

      min_len = min(mix.shape[0], clean.shape[0], est.shape[0])
      mix = np.expand_dims(mix[0: min_len], axis=0) / 32768.0
      mix = np.clip(np.where(np.isnan(mix), 0, mix), 0, 1)

      clean = np.expand_dims(clean[0: min_len], axis=0) / 32768.0
      clean = np.clip(np.where(np.isnan(clean), 0, clean), 0, 1)

      est = np.expand_dims(est[0: min_len], axis=0)
      est = np.clip(np.where(np.isnan(est), 0, est), 0, 1)

      metrics_dict = get_metrics(mix, clean, est, sample_rate=SAMPLING_RATE, metrics_list='all')

      for key in res_dict:
        res_dict[key] += metrics_dict[key]

    #pprint.pprint(res_dict)
    return res_dict


def evaluate_model(root_pth, audio_pth, save_pth, model_type, model):
    
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
                res = torch.Tensor(1, 1).to('cuda')
            
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
    
    return compute_metrics(root_pth)

