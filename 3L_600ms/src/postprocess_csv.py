import argparse
import os, glob
from tqdm import tqdm
from transformers import AutoModelForAudioFrameClassification, Wav2Vec2FeatureExtractor
import torch
import pandas as pd
from psds_eval import PSDSEval, plot_per_class_psd_roc, plot_psd_roc
import librosa
from torch import nn
import numpy as np
from sklearn.metrics import auc
import math

import argparse
import pandas as pd


def postprocess(input_list):
    output_list = []

    # aggregation step
    # lines are aggregated if the gap is lower than 0.1s
    previous = None
    pending = False
    for e in input_list:        
        if previous == None:
            previous = e
        else:
            if e[0] - previous[1] <= 0.1 and e[3] == previous[3]:
                previous[1] = e[1]
                pending = True
            else:
                output_list.append(previous)
                previous = e
                pending = False
    if pending:
        output_list.append(previous)

    # noise removal
    # lines are removed if the period length is shorter than 0.1s
    output_list = [e for e in output_list if e[1]-e[0]>0.1]

    return output_list



parser = argparse.ArgumentParser()

parser.add_argument(
    "--input", help="Where to load predictions CSV.", required=True
)

parser.add_argument(
    "--output", help="Where to store predictions CSV.", required=True
)

args = parser.parse_args()

head = os.path.split(args.output)[0]
if not os.path.exists(head):
    os.makedirs(head)

df = pd.read_csv(args.input, sep="\t")
df = df.drop(columns=['Unnamed: 0'])
original_list = df.values.tolist()

new_output = postprocess(original_list)

df_pred = pd.DataFrame(new_output, columns = ["onset", "offset", "event_label", "filename"])
df_pred.to_csv(args.output, sep="\t", index=False)