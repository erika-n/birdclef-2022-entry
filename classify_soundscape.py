from models import BirdConv1d
from scipy.io import wavfile
import torch
import json
import pandas as pd

import os
from os import listdir


def classify_file(soundscape_dir, soundscape_file, row_id, target):

    soundscape_name = soundscape_file[:-4]
    rate, soundscape = wavfile.read(os.path.join(soundscape_dir, soundscape_file))

    segments = int((soundscape.shape[0]/rate)/seconds_per_segment)


    current_birds = []
    with open("current_birds.json") as f:
        current_birds = json.load(f)
    current_birds = sorted(current_birds)

    current_scored_birds = []
    with open("current_scored_birds.json") as f:
        current_scored_birds = json.load(f)
    current_scored_birds = sorted(current_scored_birds)


    model = BirdConv1d(n_input=1, n_output=len(current_birds))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    tests_per_segment = int(5*rate/n_samples)




    for i in range(segments):
        found = {}
        for j in range(tests_per_segment):
            start = i*seconds_per_segment*rate + j*n_samples
            end = start + n_samples
            track = soundscape[start:end]
            track = track.reshape((1,1, -1))
            track = torch.tensor(track).float()
            model = model.float()
            output = model(track)
            text_label = current_birds[int(torch.argmax(output))]
            found[text_label] = True

        for bird in current_scored_birds:
            row_id.append(soundscape_name + "_" + bird + "_" + str(i*5) )
            if bird in list(found.keys()):
                target.append(True)
            else:
                target.append(False)






model_path = 'models/birds_1d'
rate = 32000
n_samples = 10000
seconds_per_segment = 5

soundscape_dir = 'random_soundscapes/sounds'
soundscape_files = [f for f in listdir(soundscape_dir) if (f[-4:] == ".wav")]

row_id = []
target = []


for soundscape_file in soundscape_files:
    classify_file(soundscape_dir, soundscape_file, row_id, target)

d = {'row_id': row_id, 'target': target}
pdscore = pd.DataFrame(d)
pdscore.to_csv('random_soundscapes/run_data/run.csv', index=False)





        
