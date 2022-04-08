
import ffmpeg
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path

input_folder =  'G:/birdclef-2022/train_audio'
output_folder = 'G:/birdclef-2022/train_audio_wav'

birds = [d for d in listdir(input_folder) ]
for b in birds:
    files = [f for f in listdir(join(input_folder, b) )]
    Path(join(output_folder, b)).mkdir(parents=True, exist_ok=True)

    for f in files:
        path = join(input_folder, b, f)
        new_filename = f[:-4] + '.wav'
        new_path = join(output_folder, b, new_filename)
        cmd = "ffmpeg -i " + path + " " + new_path
        print(cmd)
        os.system(cmd)

    