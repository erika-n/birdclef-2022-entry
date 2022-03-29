from models import BirdConv1d
from load import BirdClefDataset
from scipy.io import wavfile
import torch

model_path = 'models/birds_1d.pth'
rate = 32000
n_samples = 10000
seconds_per_segment = 5
rate, soundscape = wavfile.read('random_soundscapes/random_soundscape_069931.wav')

segments = int((soundscape.shape[0]/rate)/seconds_per_segment)


dataset = BirdClefDataset()

model = BirdConv1d(n_input=1, n_output=19)
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
        text_label = dataset.labels[int(torch.argmax(output))]
        found[text_label] = True
    print("at ", i*5, "seconds found", found.keys())

        
