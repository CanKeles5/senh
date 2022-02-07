'''
Utils to-do:
- Add type hints to functions
- Add descriptions to functions
- Add functions to go over the data verify it, ex. len!=0, num_channels==1, range of samples..., these might be unit tests also
- Add function to report the total data & other statistics
'''

from __future__ import unicode_literals
import numpy as np
import youtube_dl
import soundfile as sf
from pedalboard import Pedalboard, Reverb
from scipy.io import wavfile
from scipy import signal
from random import randint
import random
import math
import glob
import os
import soundfile as sf
import librosa
import metrics


SAMPLING_RATE = 16000


def convert_to_int16(float32_data):
    float32_data /=1.414
    float32_data *= 32767
    int16_data = float32_data.astype(np.int16)

    return int16_data


def open_audio(file_path, sr=16000):
    if "wham" in file_path: #These are IEEE 32-bit floats, convert them to PCM 16bit int
        audio, file_sr = sf.read(file_path)
        #audio = convert_to_int16(audio)
    else:
        file_sr, audio = wavfile.read(file_path)
    
    if len(audio.shape)==2:
        audio=audio[:,0]
    
    if file_sr!=sr:
        audio = signal.resample(audio, int(audio.shape[0]/file_sr*sr))
    
    return audio


#This is directly taken from SpeechBrain
def compute_amplitude(waveforms, lengths=None, amp_type="avg", scale="linear"):
    
    if len(waveforms.shape) == 1:
        waveforms = np.expand_dims(waveforms, axis=0)

    assert amp_type in ["avg", "peak"]
    assert scale in ["linear", "dB"]

    if amp_type == "avg":
        if lengths is None:
            out = np.mean(np.abs(waveforms), axis=1, keepdims=True)
        else:
            wav_sum = np.sum(np.abs(waveforms), axis=1, keepdims=True)
            out = wav_sum / lengths
    elif amp_type == "peak":
        out = np.max(np.abs(waveforms), axis=1, keepdims=True)[0]
    
    if scale == "linear":
        return out
    elif scale == "dB":
        return np.clip(20 * np.log10(out), a_min=-80)  # clamp zeros


def dB_to_amplitude(SNR):
    """Returns the amplitude ratio, converted from decibels.

    Arguments
    ---------
    SNR : float
        The ratio in decibels to convert.
    """
    return 10 ** (SNR / 20)

def range_of_vals(x, axis=0):
    return np.max(x, axis=axis) - np.min(x, axis=axis)


def add_noise_SB(source, noise_waveform, factor=1.0, addOpt='min', snr_high=12.0, snr_low=-3.0):
    min_len = min(noise_waveform.shape[0], source.shape[0])
    source = source[0:min_len].astype(np.float32) / 32767
    noise_waveform = noise_waveform[0:min_len].astype(np.float32)
    
    source = np.expand_dims(source, axis=(0,2))
    noisy_waveform = np.copy(source)
    lengths = np.expand_dims(np.array([1])*noisy_waveform.shape[1], axis=1)
    
    noise_waveform = np.expand_dims(noise_waveform, axis=(0, 2))
    
    # Compute the average amplitude of the clean waveforms
    clean_amplitude = compute_amplitude(source, lengths)
    
    # Pick an SNR and use it to compute the mixture amplitude factors
    SNR = np.random.rand(1, 1) #first dim is batch size
    SNR = SNR * (snr_high - snr_low) + snr_low
    
    noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
    
    new_noise_amplitude = noise_amplitude_factor * clean_amplitude
    
    # Scale clean signal appropriately
    noisy_waveform *= 1 - noise_amplitude_factor
    
    tensor_length = source.shape[1]
    noise_length = noise_waveform.shape[1]
    
    # Rescale and add
    noise_amplitude = compute_amplitude(noise_waveform, noise_length)
    noise_waveform *= new_noise_amplitude / (noise_amplitude + 1e-14)
    
    #Reshape
    noisy_waveform = np.squeeze(noisy_waveform, axis=(0,2))
    noise_waveform = np.squeeze(noise_waveform, axis=(0,2))
    noisy_waveform += noise_waveform
    
    return convert_to_int16(noisy_waveform), convert_to_int16(noise_waveform)


def add_noise(source, noise, factor=2.0, addOpt='min'):
    min_len = min(len(source), len(noise))
    res = source[:min_len] + noise[:min_len]*factor
    
    return res


def add_noise_wave(source, noise, factor=1.0):
    '''
    To-do:
    Generate a 1.5-2x length wave, select 1x portion of it randomly for our use case
    '''
    min_len = min(len(source), len(noise))
    
    Fs = 16000  #sampling rate
    f = 0.1     #frequency
    sample = min_len
    x = np.arange(sample)
    y = np.sin(2 * np.pi * f * x / Fs)/2 + 0.5 #sine wave
    
    res = source[:min_len] + noise[:min_len]*factor*y
    
    return res


def add_reverb(audio, sample_rate, room_size=0.25):
    
    room_size= 0.25
    
    board = Pedalboard([
        Reverb(room_size=room_size),
    ], sample_rate=sample_rate)
    
    effected = board(audio)
    
    return effected

'''
To-do:
- Change the chunk_num param to maxDuration & implement the details
- Cover the case that k>=chunk_dur
'''
def generate_chunks(audio_pth, save_pth, chunk_dur: int, k: int, chunk_num: int):
    i = 0
    audio = open_audio(audio_pth)[60*SAMPLING_RATE:] #skip the first min., intro might contaion music
    n = audio.shape[0]
    prev_id = 0
    
    for chunk_start in range(chunk_dur*SAMPLING_RATE, n, chunk_dur*SAMPLING_RATE): #change the prev_id int the range() to 0
        rand_k = randint(0, k)
        chunk = audio[prev_id : chunk_start + rand_k*SAMPLING_RATE]
        prev_id = chunk_start + rand_k*SAMPLING_RATE
        
        chunk_path = os.path.join(save_pth, os.path.split(audio_pth)[-1].split(".")[0] + f"{i}.wav") #Generate a path for the chunk, make an intuitive name
        
        wavfile.write(chunk_path, SAMPLING_RATE, chunk.astype(np.int16))
        i+=1
        
        if i>=chunk_num:
            break

'''
To-do:
Sort the speakers and the noise files acording to their lengths and append them in that order, so that we have the most data
'''
def add_augmentations(speakers_pth, noise_pth, options: dict): #To-do: make the paths Pathlib paths
    #Apply all the augmentations in the options to all speakers
    dataset = []
    speaker_files = glob.glob(speakers_pth + "\*.wav")
    noise_files = glob.glob(noise_pth + "\*.wav")
    #noise_data = [open_audio(noise_file) for noise_file in noise_files]
    
    for i in range(len(speaker_files)):
        speaker = open_audio(speaker_files[i])
        #noise = noise_data[i%len(noise_data)]
        noise = open_audio(noise_files[i%len(noise_files)])
        
        if options['add_noise_wave']:
            noise_added = add_noise_wave(speaker, noise)
            dataset.append(noise_added)
        else:
            noise_added = add_noise(speaker, noise, factor=1.0)
            dataset.append(noise_added)
        
        if options['reverb']:
            noise_added = add_reverb(noise_added, sample_rate=16000)
            dataset.append(noise_added)
        
        #print(f"speaker_file: {speaker_file}")
        if options['save_pth']: #if saved the augmented files will be saved with the name of their source file
            full_path = os.path.join(options['save_pth'], str(i)+os.path.split(speaker_files[i])[-1])
            wavfile.write(full_path, 16000, noise_added.astype(np.int16))
        
        if i%500==0:
            print(f"add_augmentations: Completed augmentation {i}.")
        
    
    return dataset

#Might not need to have this
def build_dataset(speaker_folder: str, noise_folder: str, options: dict, generateChunks=True):
    
    if generateChunks: #PUT THIS IN THE OPTIONS, PUT ALL THE PARAMS IN THE OPTIONS
        for folder_pth in os.walk(speaker_folder):
            for speaker_wav_path in glob.glob(os.path.join(folder_pth[0], '*.wav')):
                generate_chunks(audio_pth=speaker_wav_path, save_pth="D:\SpeechSeperationData\chunks", chunk_dur=12, k=5, chunk_num=275)
    
    options= {
    'reverb': True,
    'add_noise_wave': True,
    'save_pth': r"D:\SpeechSeperationData\try_data"
    }
    
    add_augmentations(speakers_pth="D:\SpeechSeperationData\chunks", noise_pth=r"D:\whamr\wham_noise\wham_noise\cv", options=options)


def build_whamr_like_dataset(speakers_path, wham_noise_path, output_root):
        SINGLE_DIR = 'mix_single'
        BOTH_DIR = 'mix_both'
        CLEAN_DIR = 'mix_clean'
        S1_DIR = 's1'
        S2_DIR = 's2'
        NOISE_DIR = 'noise'
        SUFFIXES = ['_anechoic', '_reverb']
        
        MONO = True  # Generate mono audio, change to false for stereo audio
        SPLITS = ['tr' ,'cv', 'tt']
        SAMPLE_RATES = ['16k'] #, '8k'] # Remove element from this list to generate less data
        DATA_LEN = ['min'] #, 'max'] # Remove element from this list to generate less data
        
        #create the folders
        for splt in SPLITS:
            noise_path = os.path.join(wham_noise_path, splt)
            data_path = os.path.join(speakers_path, splt)
            
            for wav_dir in ['wav' + sr for sr in SAMPLE_RATES]:
                for datalen_dir in DATA_LEN:
                    output_path = os.path.join(output_root, wav_dir, datalen_dir, splt)
                    for sfx in SUFFIXES:
                        os.makedirs(os.path.join(output_path, CLEAN_DIR+sfx), exist_ok=True)
                        os.makedirs(os.path.join(output_path, SINGLE_DIR+sfx), exist_ok=True)
                        os.makedirs(os.path.join(output_path, BOTH_DIR+sfx), exist_ok=True)
                        os.makedirs(os.path.join(output_path, S1_DIR+sfx), exist_ok=True)
                        os.makedirs(os.path.join(output_path, S2_DIR+sfx), exist_ok=True)
                    os.makedirs(os.path.join(output_path, NOISE_DIR), exist_ok=True)
            
            utt_wav_paths = glob.glob(data_path + "\*.wav")
            noise_paths = glob.glob(noise_path + "\*.wav")
            
            for id_num, output_name in enumerate(utt_wav_paths):
                s1 = open_audio(output_name)
                noise_samples = open_audio(noise_paths[id_num%len(noise_paths)])
                
                s1_anechoic = s1
                rand_room_size = random.uniform(0.05, 1.0)
                s1_reverb = add_reverb(s1_anechoic, SAMPLING_RATE)
                mix_single = s1_reverb
                
                s2 = s1
                s2_anechoic = s1
                s2_reverb = s1_reverb
                
                for sr_i, sr_dir in enumerate(SAMPLE_RATES):
                    wav_dir = 'wav' + sr_dir
                    
                    for datalen_dir in DATA_LEN:
                        output_path = os.path.join(output_root, wav_dir, datalen_dir, splt)
                        
                        sources = [(s1_anechoic, s2_anechoic), (s1_reverb, s2_reverb)]
                        for i_sfx, (sfx, source_pair) in enumerate(zip(SUFFIXES, sources)):
                            mix_clean = source_pair[0]
                            
                            #mix_single = mix_both = add_noise(source_pair[0], noise_samples, factor=rand_mix_factor) #add_noise_SB(source_pair[0], noise_samples)
                            mix_single, noise_samples =  add_noise_SB(source_pair[0], noise_samples)
                            mix_both = mix_single
                            #min_len = min(source_pair[0].shape, noise_samples.shape)[0]                            
                            
                            # write audio
                            samps = [mix_clean, mix_single, mix_both, source_pair[0], source_pair[1]]
                            dirs = [CLEAN_DIR, SINGLE_DIR, BOTH_DIR, S1_DIR, S2_DIR]
                            
                            for dir, samp in zip(dirs, samps):
                                output_name = output_name.split('\\')[-1]
                                sf.write(os.path.join(output_path, dir+sfx, output_name), samp.astype(np.int16), 16000)
                            
                            
                            if i_sfx == 0: # only write noise once as it doesn't change between anechoic and reverberant
                                sf.write(os.path.join(output_path, NOISE_DIR, output_name), noise_samples.astype(np.int16), 16000)

def validate_dataset(root_pth):
    '''
    This function checks the lengths of the audio files in the created dataset, assersts if len==0
    '''
    
    for root, _, files in os.walk(root_pth):
        for file in files:
            if not file.endswith(".wav"):
                continue

            # os.path.join() will create the path for the file
            file = os.path.join(root, file)
            
            audio = open_audio(file, 16000)
            #print(f"audio.shape: {audio.shape[0]}")
            if audio.shape[0]==0:
                print(f"audio len 0 for file {file}.")
            

def download_yt_audio(URL):
    '''
    To-do: See if we can specify folders to download
    Add no-playlist option, else it will download the whole playlist
    '''
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192'
        }],
        'postprocessor_args': [
            '-ar', '16000'
        ],
        'prefer_ffmpeg': True,
        'keepvideo': False
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([URL])

