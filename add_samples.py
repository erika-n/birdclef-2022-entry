import wave
import pandas as pd
import torchaudio
import os


metadata_file="birdclef-2022/train_metadata.csv"
md = pd.read_csv(metadata_file)

md["samples"] = 0
md["rate"] = 0
md["seconds"] = 0


for i in range(len(md)):
    waveform, rate = torchaudio.load(os.path.join('birdclef-2022/train_audio', md["filename"][i]))
    n_samples = list(waveform.size())[1]
    seconds = n_samples/rate
    md["samples"][i] = n_samples
    md["seconds"][i] = seconds
    md["rate"][i] = rate
    if i % 200 == 0:
        print("i", i,"of", len(md))



md.to_csv('train_metadata_updated.csv')
