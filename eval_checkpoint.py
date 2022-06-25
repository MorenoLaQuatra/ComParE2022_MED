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
import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)


parser = argparse.ArgumentParser(description="Training XXForAudioFrameClassification")

parser.add_argument(
    "--ground_truth_csv_path", help="Path pointing to the ground-truth csv.", required=True
)

parser.add_argument(
    "--metadata_csv_path", help="Path pointing to the metadata csv.", required=True
)

parser.add_argument(
    "--pred_dir",
    help="Path pointing to the root folder containing the predictions of the model at different thresholds.",
    required=True,
)

parser.add_argument(
    "--checkpoint_dir",
    help="Path pointing to the checkpoint.",
    required=True,
)

parser.add_argument(
    "--feature_extractor_model_name",
    help="Model tag (as in HF hub).",
    required=True,
)

parser.add_argument(
    "--num_labels",
    help="Number of labels for training the model (2, 3, 4 supported).",
    required=True,
    type=int,
)

parser.add_argument(
    "--pred_file_prefix",
    help="Prefix of the prediction files (e.g., model_pred_0.1.csv -> model_pred_)",
    required=True,
)

parser.add_argument(
    "--max_window_duration",
    help="Max duration of the window to the stream mode.",
    type=float,
    default=60
)

parser.add_argument(
    "--plot_filename",
    help="Path to the filename of the plot (ROC).",
    default=None,
)

parser.add_argument(
    "--dev_folder",
    help="Path to the directory containing dev set files.",
    required = True
)

parser.add_argument(
    '--use_cuda', 
    help="Flag to use GPU.",
    action='store_true'
)

args = parser.parse_args()

if args.use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print (f"Device: {device}")
# PSDS parameters

dtc_threshold = 0.5
gtc_threshold = 0.5
beta = 1  # Emphasis on FP/TP -> beta: coefficient used to put more (beta > 1) or less (beta < 1) emphasis on false negatives.
cttc_threshold = 0.0
alpha_ct = 0.0
alpha_st = 0.0
# max_efpr = 170



ground_truth_csv_path = args.ground_truth_csv_path
metadata_csv_path = args.metadata_csv_path
gt_table = pd.read_csv(ground_truth_csv_path, sep="\t")
meta_table = pd.read_csv(metadata_csv_path, sep="\t")

pred_dir = args.pred_dir

psds_eval = PSDSEval(
    dtc_threshold,
    gtc_threshold,
    cttc_threshold,
    ground_truth=gt_table,
    metadata=meta_table,
    class_names=["mosquito"],
)



model = AutoModelForAudioFrameClassification.from_pretrained(
    args.checkpoint_dir, num_labels=args.num_labels
)

model = model.to(device)
model.eval()

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.feature_extractor_model_name)


# generate model predictions TODO
BACKGROUND_LABEL = 0
SAMPLE_RATE = 16_000
MAX_SECONDS = args.max_window_duration
FRAME_DURATION = 20.0001
LONG_FILES = True

softmax_fn = nn.Softmax(dim=1)

d_res = {} 
for th in np.arange(0.1, 1.1, 0.1):
    d_res[th] = []


with torch.no_grad():
    for dev_filename in tqdm(glob.glob(args.dev_folder + "/*.wav")):
        file_id = dev_filename.split("/")[-1].replace(".wav", "")
        audio, _ = librosa.load(dev_filename, sr=SAMPLE_RATE)

        try:
            if (len(audio) > (MAX_SECONDS * SAMPLE_RATE)) and LONG_FILES:
                output_logits = None
                print (f"Long file: {len(audio)} > {MAX_SECONDS * SAMPLE_RATE} -> {len(audio)/SAMPLE_RATE} > {MAX_SECONDS}")
                for audio_vector in np.array_split(audio, math.ceil(len(audio) / (MAX_SECONDS * SAMPLE_RATE))):
                    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    if info.used/(1024*1024*1024) > 10:
                        print("Used memory:", info.used/(1024*1024*1024))
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

            else:
                inputs = feature_extractor([audio], 
                                            return_tensors="pt", 
                                            sampling_rate=SAMPLE_RATE, 
                                            max_length=MAX_SECONDS * SAMPLE_RATE,
                                            truncation=True)
                inputs = inputs.to(device)
                output_logits = model(**inputs).logits[0]
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory')
                print ("Duration:", len(audio)/SAMPLE_RATE, audio.shape, audio.dtype, file_id)
                del inputs
                torch.cuda.empty_cache()
                continue
            else:
                print (e)


            
        #output_softmax = softmax_fn(output_logits)
        global_min = min(torch.flatten(output_logits))
        output_norm = [[float((v - global_min)/(sum(e) - global_min * len(e))) for v in e] for e in output_logits]

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
                        d_res[th].append([prev_onset/1000, prev_offset/1000, "mosquito", int(file_id)])
                        temp_res.append((prev_onset/1000, prev_offset/1000, "mosquito", int(file_id)))
                        prev_offset = offset
                        prev_onset = onset
                else:
                    # end 2
                    if prev_offset is not None and prev_onset is not None:
                        d_res[th].append([prev_onset/1000, prev_offset/1000, "mosquito", int(file_id)])
                        temp_res.append((prev_onset/1000, prev_offset/1000, "mosquito", int(file_id)))
                    prev_offset = None
                    prev_onset = None
                    #print (f"{(i * 20)/1000} - O", end="\t")

            if prev_offset is not None and prev_onset is not None:
                # end 3
                d_res[th].append([prev_onset/1000, prev_offset/1000, "mosquito", int(file_id)])
                temp_res.append((prev_onset/1000, prev_offset/1000, "mosquito", int(file_id)))
            '''
            print ("\n\n")
            for e in temp_res:
                print (e)
            '''


        

'''
for th in np.arange(0.1, 1.1, 0.1):
    print (th)
    for e in d_res[th]:
        print (e)
'''
# Add the operating points, with the attached information
for i, th in enumerate(np.arange(0.1, 1.1, 0.1)):
    print (f"TH: {th}")
    for e in d_res[th]:
        print(e)
    #csv_file = os.path.join(pred_dir, f"{args.pred_file_prefix}{th:.1f}.csv")
    #det_t = pd.read_csv(os.path.join(csv_file), sep="\t")
    det_t = pd.DataFrame(d_res[th], columns = ["onset", "offset", "event_label", "filename"])
    info = {"name": f"Op {i + 1}", "threshold": th}
    psds_eval.add_operating_point(det_t, info=info)
    macro_f, class_f = psds_eval.compute_macro_f_score(det_t, beta=beta)
    print(f"\nmacro F-score: {macro_f * 100:.2f}")

    print ("\n\n\n\n\n\n")

# Calculate the PSD-Score
psds = psds_eval.psds(alpha_ct, alpha_st)
print(f"\nPSD-Score: {psds.value:.5f}")

# Plot the PSD-ROC
if args.plot_filename is not None:
    plot_psd_roc(psds, filename = args.plot_filename)
else:
    plot_psd_roc(psds)

# Plot per class tpr vs fpr/efpr/ctr
tpr_vs_fpr, _, tpr_vs_efpr = psds_eval.psd_roc_curves(alpha_ct=alpha_ct)
plot_per_class_psd_roc(
    tpr_vs_fpr,
    psds_eval.class_names,
    title="Per-class TPR-vs-FPR PSDROC",
    xlabel="FPR",
)
print((tpr_vs_fpr[1]))
print((tpr_vs_fpr[0].squeeze()))
roc_auc = auc(tpr_vs_fpr[1], tpr_vs_fpr[0].squeeze())
roc_auc_norm = roc_auc / np.max(tpr_vs_fpr[1])
print("ROC", roc_auc)
print("ROC norm", roc_auc_norm)

"""
Command:

python eval_checkpoint.py \
        --ground_truth_csv_path eval_data/labels/dev/a/gt.csv \
        --metadata_csv_path eval_data/labels/dev/a/meta.csv \
        --pred_dir _temp/ \
        --checkpoint_dir checkpoints_3/best_checkpoint/ \
        --feature_extractor_model_name microsoft/wavlm-base \
        --num_labels 3 \
        --pred_file_prefix nl3_ \
        --plot_filename nl3_roc.png \
        --dev_folder ../../data/dev/a/

"""