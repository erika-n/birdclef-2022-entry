# This plays tracks from the library at random with info


import pandas as pd
import random
import os
import pyglet
import time


def nextBird(player):
    i = random.randint(0, N)
    file = metadata_file["filename"][i]
    path = os.path.join('birdclef-2022/train_audio', file)
    print("---------------------------")
    print("Playing:")
    print(metadata_file["primary_label"][i])
    print(metadata_file["PRIMARY_COM_NAME"][i])
    print(metadata_file["SCI_NAME"][i])
    print(metadata_file["ORDER1"][i])
    print(metadata_file["FAMILY"][i])
    print("rating", metadata_file["rating"][i])
    player.queue(pyglet.media.load(path))


metadata_file = pd.read_csv('birdclef-2022/train_metadata.csv')

taxonomy_file = "birdclef-2022/eBird_Taxonomy_v2021.csv"
td = pd.read_csv(taxonomy_file)
metadata_file = pd.merge(metadata_file, td, left_on="primary_label", right_on="SPECIES_CODE")
metadata_file = metadata_file[metadata_file["rating"] > 3.5]
metadata_file = metadata_file.reset_index()
N = len(metadata_file)


player = pyglet.media.Player()
nextBird(player)
player.play()
#pyglet.app.run()
while True:
    for i in range(10):
        pyglet.clock.tick()
        time.sleep(1)

    nextBird(player)
    player.next_source()
#player.next_source()
player.play()
#

