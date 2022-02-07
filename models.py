from speechbrain.pretrained import SepformerSeparation as separator
import speechbrain as sb
import torchaudio

class Separator:
    def __init__(self, weight_path=None):
        self.model = None
        if weight_path==None:
            self.model = separator.from_hparams(source="speechbrain/sepformer-whamr16k", savedir='pretrained_models/sepformer-whamr16k')
    
    def separate(self, save=False):
        est_sources = model.separate_file(path=x)
        
        return est_sources

'''
import os
 
for x in os.listdir():
    if x.endswith(".wav"):
      est_sources = model.separate_file(path=x)
      print(f"{x[:-4]} done.")
      torchaudio.save(f"{x[:-4]}s1.wav", est_sources[:, :, 0].detach().cpu(), 16000) #np.int16 olarak kaydet
      torchaudio.save(f"{x[:-4]}s2.wav", est_sources[:, :, 1].detach().cpu(), 16000)
'''
