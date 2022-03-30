
import pandas as pd

import os
from os import listdir
from os.path import isfile, join



target_data_dir = 'random_soundscapes/target_data'
target_data_files = [f for f in listdir(target_data_dir) if (f[-4:] == ".csv")]

run_file = 'random_soundscapes/run_data/run.csv'


run_data = pd.read_csv(run_file)
run_data["orig_target"] = False

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

for target_file in target_data_files:
    target_data = pd.read_csv(target_data_dir + "/" + target_file)
    target_data = target_data.reset_index()
    for index, row in target_data.iterrows():
        # run_row = run_data[run_data['row_id'] == row['row_id']]

        run_data.loc[run_data['row_id'] == row['row_id'], "orig_target"] = True
    


for index, row in run_data.iterrows():
    if row["target"] == True and row["orig_target"] == True:
        true_positive += 1
    if row["target"] == False and row["orig_target"] == False:
        true_negative += 1
    if row["target"] == True and row["orig_target"] == False:
        false_positive += 1
    if row["target"] == False and row["orig_target"] == True:
        false_negative += 1



print(f"true positive: {true_positive}")
print(f"true negative: {true_negative}")
print(f"false positive: {false_positive}")
print(f"false negative: {false_negative}")

total = true_positive + false_positive + true_negative + false_negative
accuracy = (true_positive + true_negative)/total
precision = true_positive/(true_positive + false_positive)
recall = true_positive/(true_positive + false_negative)
f1 = 2*precision*recall/(precision + recall)


print(f"total: {total:.0f}")
print(f"accuracy {100*accuracy:.1f}%")
print(f"precision {100*precision:.1f}%")
print(f"recall {100*recall:.1f}%")
print(f"f1 {100*f1:.1f}%")


run_data.to_csv("random_soundscapes/run_data/score_output.csv", index=False)