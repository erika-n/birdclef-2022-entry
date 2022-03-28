from email.errors import NoBoundaryInMultipartDefect
import numpy as np
from scipy.io import wavfile
from load import BirdClefDataset
from torch.utils.data import DataLoader
import random
import pandas as pd
import torch

soundscape_seconds = 90
clip_seconds = 5
rate = 32000
soundscape = np.zeros(soundscape_seconds*rate)
num_clips = 15

soundscape_name = f"random_soundscape_{random.randint(0, 100000):06}"

dataset = BirdClefDataset(n_samples=rate*clip_seconds, n_per_file=1)
dataloader = DataLoader(dataset, batch_size=num_clips, shuffle=True)

features, labels = next(iter(dataloader))

row_id = []
target = []
for i in range(num_clips):
    clip = features[i].numpy()[0]
    label = labels[i]
    text_label = dataset.labels[int(torch.argmax(label))]
    start_seconds = 5*random.randint(0, int((soundscape_seconds - 5)/5))
    soundscape[start_seconds*rate:start_seconds*rate + clip_seconds*rate] += clip
    row_id.append(soundscape_name +"_" + text_label + "_" + str(start_seconds))
    target.append("TRUE")


d = {'row_id': row_id, 'target': target}
pdscore = pd.DataFrame(d)
pdscore.to_csv('random_soundscapes/' + soundscape_name + ".csv")
wavfile.write('random_soundscapes/' + soundscape_name + ".wav", rate, soundscape)







