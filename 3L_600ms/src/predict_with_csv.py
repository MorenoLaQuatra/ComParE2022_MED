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
    "--pred_dir", help="Where to store predictions CSV.", required=True
)

parser.add_argument(
    "--checkpoint_dir", help="Path to model checkpoint.", required=True
)

parser.add_argument(
    "--feature_extractor_model_name", help="Path to model checkpoint.", default="microsoft/wavlm-base"
)

parser.add_argument(
    "--pred_file_prefix", default="baseline_"
)

parser.add_argument(
    "--test_folder", 
    help="Path to audio files.",
    required=True
)

parser.add_argument(
    "--max_seconds", help="Length of the window for streaming mode (in seconds).", required=True, type=float,
)

parser.add_argument(
    "--num_labels", help="Number of labels for the model.", required=True, type=int,
)

args = parser.parse_args()

pred_dir = args.pred_dir
checkpoint_dir = args.checkpoint_dir
feature_extractor_model_name = args.feature_extractor_model_name
pred_file_prefix = args.pred_file_prefix
test_folder = args.test_folder

use_cuda = True
BACKGROUND_LABEL = 0
SAMPLE_RATE = 16_000
FRAME_DURATION = 20.0001
LONG_FILES = True

MAX_SECONDS = args.max_seconds
num_labels = args.num_labels

# if pred_dir does not exist, create
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)
    print(f"{pred_dir} created!")


if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print (f"Device: {device}")


model = AutoModelForAudioFrameClassification.from_pretrained(
    checkpoint_dir, num_labels=num_labels
)
model = model.to(device)
model.eval()

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(feature_extractor_model_name)

d_res = {} 
for th in np.arange(0.1, 1.1, 0.1):
    d_res[th] = []


softmax_fn = torch.nn.Softmax()

with torch.no_grad():
    print ("Start")

    overall_res = []

    for dev_filename in tqdm(glob.glob(test_folder + "/*.wav")):
        #print (dev_filename)
        file_id = dev_filename.split("/")[-1].replace(".wav", "")
        audio, _ = librosa.load(dev_filename, sr=SAMPLE_RATE)

        try:
            if (len(audio) > (MAX_SECONDS * SAMPLE_RATE)) and LONG_FILES:
                output_logits = None
                list_audios = np.array_split(audio, math.ceil(len(audio) / (MAX_SECONDS * SAMPLE_RATE)))
                for ind_audio, audio_vector in enumerate(list_audios):
                    #print (f"{ind_audio} / {len(list_audios)}")
                    inputs = feature_extractor([audio_vector], 
                                                return_tensors="pt", 
                                                sampling_rate=SAMPLE_RATE, 
                                                max_length=MAX_SECONDS * SAMPLE_RATE,
                                                truncation=True)

                    inputs = inputs.to(device)
                    output_temp = model(**inputs).logits[0]
                    if output_logits is None:
                        output_logits = output_temp
                    else:
                        output_logits = torch.cat((output_logits, output_temp))
                    
                    del inputs
                    torch.cuda.empty_cache()
            else:
                inputs = feature_extractor([audio], 
                                            return_tensors="pt", 
                                            sampling_rate=SAMPLE_RATE, 
                                            max_length=MAX_SECONDS * SAMPLE_RATE,
                                            truncation=True)
                inputs = inputs.to(device)
                output_logits = model(**inputs).logits[0]
        except Exception as e:
            print ("ERROR:", e)
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory')
                print ("Duration:", len(audio)/SAMPLE_RATE, audio.shape, audio.dtype, file_id)
                del inputs
                torch.cuda.empty_cache()
                continue


            
        output_norm = softmax_fn(output_logits)
        #global_min = min(torch.flatten(output_logits))
        #output_norm = [[float((v - global_min)/(sum(e) - global_min * len(e))) for v in e] for e in output_logits]

        for th in np.arange(0.1, 1.1, 0.1):
            tmp_list = []
            prev_offset = None
            prev_onset = None
            temp_res = []
            for i, v in enumerate(output_norm):
                if sum(v[1:]) > th:
                    #print (f"{(i * 20)/1000} - M", end="\t")
                    onset = i * 20
                    offset = (i+1) * 20
                    
                    if prev_offset is None and prev_onset is None:
                        #init case
                        prev_offset = offset
                        prev_onset = onset
                    elif (prev_offset == onset) and (prev_offset is not None):
                        # continue
                        prev_offset = offset
                        # onset not change
                    else:
                        # end
                        d_res[th].append([prev_onset/1000, prev_offset/1000, "mosquito", file_id])
                        temp_res.append((prev_onset/1000, prev_offset/1000, "mosquito", file_id))
                        prev_offset = offset
                        prev_onset = onset
                else:
                    # end 2
                    if prev_offset is not None and prev_onset is not None:
                        d_res[th].append([prev_onset/1000, prev_offset/1000, "mosquito", file_id])
                        temp_res.append((prev_onset/1000, prev_offset/1000, "mosquito", file_id))
                    prev_offset = None
                    prev_onset = None
                    #print (f"{(i * 20)/1000} - O", end="\t")

            if prev_offset is not None and prev_onset is not None:
                # end 3
                d_res[th].append([prev_onset/1000, prev_offset/1000, "mosquito", file_id])
                temp_res.append((prev_onset/1000, prev_offset/1000, "mosquito", file_id))
            d_res[th] = postprocess(d_res[th])


        '''
        
        Save real predictions CSV

        '''
        prev_offset = None
        prev_onset = None
        for i, v in enumerate(output_logits):
            #print (torch.argmax(v, axis=0).item())
            if torch.argmax(v, axis=0).item() > 0:
                onset = i * 20
                offset = (i+1) * 20
                
                if prev_offset is None and prev_onset is None:
                    #init case
                    prev_offset = offset
                    prev_onset = onset
                elif (prev_offset == onset) and (prev_offset is not None):
                    # continue
                    prev_offset = offset
                    # onset not change
                else:
                    # end
                    overall_res.append([prev_onset/1000, prev_offset/1000, "mosquito", file_id])
                    prev_offset = offset
                    prev_onset = onset
            else:
                # end 2
                if prev_offset is not None and prev_onset is not None:
                    overall_res.append([prev_onset/1000, prev_offset/1000, "mosquito", file_id])
                prev_offset = None
                prev_onset = None

        if prev_offset is not None and prev_onset is not None:
            # end 3
            overall_res.append([prev_onset/1000, prev_offset/1000, "mosquito", file_id])
        overall_res = postprocess(overall_res)


    for i, th in enumerate(np.arange(0.1, 1.1, 0.1)):
        print (f"TH: {th}")
        df_pred = pd.DataFrame(d_res[th], columns = ["onset", "offset", "event_label", "filename"])
        df_pred.to_csv(f"{pred_dir}/{pred_file_prefix}{th:.1f}.csv", sep="\t")


    print (f"Overall predictions")
    df_pred = pd.DataFrame(overall_res, columns = ["onset", "offset", "event_label", "filename"])
    df_pred.to_csv(f"{pred_dir}/{pred_file_prefix}.csv", sep="\t")


'''
python src/predict_with_csv.py
		--pred_dir data/predictions/dev/a/4L_60s/
		--checkpoint_dir /data1/mlaquatra/compare/mosquitos/ComParE-2022/mosquito_subchallenge/checkpoints_4/best_checkpoint/
		--test_folder /data1/mlaquatra/compare/mosquitos/data/dev/a/
		--max_seconds 60
		--num_labels 4

python src/eval.py \
        --predictions_dir data/predictions/dev/a/4L_60s/ \
        --ground_truth_dir data/labels/dev/a/

python src/eval_DER.py --ground_truth_csv_path data/labels/dev/a/gt.csv \
    --prediction_csv_path data/predictions/dev/a/4L_60s/baseline_.csv

'''


