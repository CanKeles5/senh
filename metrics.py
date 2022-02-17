'''
To-Do:
- Add assertions to check the dimensions of the inputs
- Make the class/functions more intuitive to use, have them as seperate functions and have one function to compute all metrics
- Add MSE
- Add REAL-M-sisnr-estimator, from Cem Subakan
'''

import numpy as np
import math
import speechbrain
import scipy.io
#from pb_bss_eval import OutputMetrics
from asteroid.metrics import get_metrics



SAMPLE_RATE = 16000



def get_mask(source, source_lengths):
    """
    Arguments
    ---------
    source : [T, B, C]
    source_lengths : [B]

    Returns
    -------
    mask : [T, B, 1]
    """
    T, B, _ = source.shape
    mask = np.ones_like(source).astype(np.float32)
    
    for i in range(B):
        mask[source_lengths[i] :, i, :] = 0.0
    
    return mask


def cal_si_snr(source, estimate_source):
    """Calculate SI-SNR.

    Arguments:
    ---------
    source: [T, B, C],
        Where B is batch size, T is the length of the sources, C is the number of sources
        the ordering is made so that this loss is compatible with the class PitWrapper.

    estimate_source: [T, B, C]
        The estimated source.
    """
    EPS = 1e-8
    assert source.shape == estimate_source.shape
    
    source = np.expand_dims(source, axis=(1, 2)) #.astype(np.float32)
    estimate_source = np.expand_dims(estimate_source, axis=(1, 2)) #.astype(np.float32)
    
    source_lengths = np.array([estimate_source.shape[0]] * estimate_source.shape[1])
    mask = get_mask(source, source_lengths)
    
    estimate_source *= mask

    num_samples = (np.copy(source_lengths).reshape(1, -1, 1))  # [1, B, 1]
    mean_target = np.sum(source, axis=0, keepdims=True) / num_samples
    mean_estimate = (np.sum(estimate_source, axis=0, keepdims=True) / num_samples)
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT reshape to use broadcast
    s_target = zero_mean_target  # [T, B, C]
    s_estimate = zero_mean_estimate  # [T, B, C]

    dot = np.sum(s_estimate * s_target, axis=0, keepdims=True)  # [1, B, C]
    s_target_energy = (np.sum(s_target ** 2, axis=0, keepdims=True) + EPS)  # [1, B, C]
    proj = dot * s_target / s_target_energy  # [T, B, C]
    # e_noise = s' - s_target
    e_noise = s_estimate - proj  # [T, B, C]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    si_snr_beforelog = np.sum(proj ** 2, axis=0) / (np.sum(e_noise ** 2, axis=0) + EPS)
    si_snr = 10 * np.log10(si_snr_beforelog + EPS)  # [B, C]

    return -si_snr


def signaltonoise(s, n, eps=1e-5):
    min_len = min(s.shape[0], n.shape[0])
    s = np.where(s!=0, s, eps)[0:min_len]
    n = np.where(n!=0, n, eps)[0:min_len]
    log_sn = np.log((s/n)**2)
    
    return 10*np.sum(log_sn)/min_len


def compute_measures(estimated_signal, reference_signals, j=0, scaling=True):
    Rss= np.dot(reference_signals.transpose(), reference_signals)
    this_s= reference_signals[:,j]
    
    if scaling:
        # get the scaling factor for clean sources
        a= np.dot( this_s, estimated_signal) / Rss[j,j]
    else:
        a= 1

    e_true= a * this_s
    e_res= estimated_signal - e_true

    Sss= (e_true**2).sum()
    Snn= (e_res**2).sum()

    SDR= 10 * math.log10(Sss/Snn)
    
    # Get the SIR
    Rsr= np.dot(reference_signals.transpose(), e_res)
    b= np.linalg.solve(Rss, Rsr)

    e_interf= np.dot(reference_signals , b)
    e_artif= e_res - e_interf
    
    SIR= 10 * math.log10(Sss / (e_interf**2).sum())
    SAR= 10 * math.log10(Sss / (e_artif**2).sum())
    
    return SDR, SIR,SAR


def compute_all_metrics(target, result):
    res = compute_measures(result, np.expand_dims(target, axis=1))
    
    return {
        #'SNR': signaltonoise(result),
        'SDR': res[0],
        'SIR': res[1],
        'SAR': res[2]
        }

def pb_bss_metrics(target, estimate):
    """
    Copmutes si_sdr, sdr, sir, sar, stoi, pesq
    """
    output_metrics = OutputMetrics(
        speech_prediction=estimate,
        speech_source=target,
        enable_si_sdr=True,
        sample_rate=SAMPLE_RATE,
        compute_permutation=compute_permutation,
    )
    
    return output_metrics

def compute_metrics_bsseval(root_pth, save_pth):
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
      mix = utils.open_audio(mix_path)
      clean = utils.open_audio(clean_path)
      est = utils.open_audio(res_path)

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


'''
if __name__ == '__main__':
    #a1 = np.expand_dims(np.random.rand(16000*50), axis=0)
    a1 = np.random.rand(16000*5000)
    #a1 = np.ones(16000*5000)/2 + np.random.rand(16000*5000)
    a2 = np.expand_dims(np.random.rand(16000*5000), axis=1)
    #a2 = np.expand_dims(a1, axis=1)
    #a2 = np.expand_dims(a1, axis=1)
    
    print(f"a1.shape: {a1.shape}")
    
    cm = ComputeMetrics(a1, a2)
    
    
    print(cm.SDR)
    print(cm.SAR)
    print(cm.SIR)
'''

