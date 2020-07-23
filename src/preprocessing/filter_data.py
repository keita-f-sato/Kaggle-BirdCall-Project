import os
import pandas as pd
import numpy as np
import librosa
import math

import torch
import torchaudio
from torchaudio.transforms import Spectrogram, MelSpectrogram, AmplitudeToDB, ComplexNorm, Resample
from torchaudio.functional import lowpass_biquad, highpass_biquad

from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, cpu_count

def cov_pk(info):
    path, resample, rate = info
    
    waveform = torch.load(path)
    waveform = resample(waveform)
    ebird_code = path.parent.name
    torch.save(waveform, f'../../dataset/tensor_audio/{ebird_code}/re{rate}-{path.stem}.tensor')

NUM_WORKERS = cpu_count()
sr = 32_000

for i in [0.8, 0.9, 1.1, 1.2]:
    resample = Resample(sr, sr*i)
    for directory in tqdm(Path('../../dataset/tensor_audio').iterdir()):
        file_paths = list(directory.iterdir())
        with Pool(5) as p:
            p.map(cov_pk, (file_paths, resample, i))