import argparse
import os
from tqdm import tqdm
from transformers import AutoModelForAudioFrameClassification
from transformers import Trainer, TrainingArguments
import torch
from MosTimestampDataset import MosTimestampDataset
import pandas as pd
from datasets import load_metric
import numpy as np

parser = argparse.ArgumentParser(description="Training XXForAudioFrameClassification")

parser.add_argument(
    "--train_csv_path", help="Path pointing to the training set csv.", required=True
)

parser.add_argument(
    "--dev_csv_path", help="Path pointing to the evaluation set csv.", required=True
)

parser.add_argument(
    "--root_train_folder",
    help="Path pointing to the root folder containing wav examples for the train dataset.",
    required=True,
)

parser.add_argument(
    "--root_dev_folder",
    help="Path pointing to the root folder containing wav examples for the dev set.",
    required=True,
)

parser.add_argument(
    "--external_noise_folder",
    help="Path pointing to the root folder containing external noise background.",
    required=True,
)

parser.add_argument(
    "--model_name_or_path",
    help="Path or name of the model (as in HF Hub) that need to be finetuned.",
    required=True,
)

parser.add_argument(
    "--output_dir",
    help="Path pointing to the folder where checkpoints will be stored.",
    required=True,
)

parser.add_argument(
    "--log_dir",
    help="Path pointing to the folder where logs will be stored.",
    required=True,
)

parser.add_argument(
    "--num_labels",
    help="Number of labels for training the model (2, 3, 4 supported).",
    default=3,
    type=int,
)

parser.add_argument(
    "--logging_steps",
    help="Number of steps between two logging.",
    default=100,
    type=int,
)

parser.add_argument(
    "--train_batch_size", help="Batch size to use during training.", default=2, type=int
)

parser.add_argument(
    "--eval_batch_size",
    help="Batch size to use during evaluation.",
    default=2,
    type=int,
)

parser.add_argument(
    "--dataloader_num_workers",
    help="Num CPU threads for the dataloaders.",
    default=16,
    type=int,
)

parser.add_argument(
    "--weight_decay", help="Weight decay to used in training.", default=0.01, type=float
)

parser.add_argument(
    "--num_train_epochs", help="Number of training epochs.", default=25, type=int
)

parser.add_argument(
    "--learning_rate", help="Initial learning rate.", type=float, default=1e-5
)

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AutoModelForAudioFrameClassification.from_pretrained(
    args.model_name_or_path, num_labels=args.num_labels
)

train_df = pd.read_csv(args.train_csv_path)
train_ds = MosTimestampDataset(
    train_df,
    args.external_noise_folder,
    args.root_train_folder,
    model_name_or_path=args.model_name_or_path,
    num_labels=args.num_labels,
)
dev_df = pd.read_csv(args.dev_csv_path)
dev_ds = MosTimestampDataset(
    dev_df, args.external_noise_folder, args.root_dev_folder, num_labels=args.num_labels, augmentation=False
)

training_args = TrainingArguments(
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    fp16=True,
    fp16_full_eval=True,
    dataloader_num_workers=args.dataloader_num_workers,
    dataloader_pin_memory=True,
    output_dir=args.output_dir,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    logging_dir=args.log_dir,
    logging_steps=args.logging_steps,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=args.num_train_epochs,
    load_best_model_at_end=True,
    metric_for_best_model="eval_overall_f1",
)

seqeval = load_metric("seqeval")

def compute_metrics(pred):

    MAPPINGS ={
        0: "O",
        1: "B-MOS",
        2: "I-MOS",
        3: "E-MOS",
    }

    labels = []
    for j in range(len(pred.label_ids)):
        cur_labels = [] 
        for tok in pred.label_ids[j]:
            cur_labels.append(MAPPINGS[np.argmax(tok)])
        labels.append(cur_labels)

    preds = [] 
    for j in range(len(pred.predictions)):
        cur_preds = [] 
        for tok in pred.predictions[j]:
            cur_preds.append(MAPPINGS[np.argmax(tok)])
        preds.append(cur_preds)

    '''
    print ("LABELS")
    for i, l in enumerate(labels):
        print (i, l)

    print ("PREDS")
    for i, l in enumerate(preds):
        print (i, l)
    '''


    results = seqeval.compute(predictions=preds, references=labels)
    for k, v in results.items():
        print (k, v)

    return results


trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
)

trainer.train()
trainer.save_model(args.output_dir + "/best_checkpoint/")
trainer.evaluate()

"""
Command:

python train.py --train_csv_path df/train.csv \
                --dev_csv_path df/dev.csv \
                --root_train_folder /data1/mlaquatra/compare/mosquitos/data/train/ \
                --root_dev_folder /data1/mlaquatra/compare/mosquitos/data/dev/ab/ \
                --external_noise_folder /data1/mlaquatra/compare/mosquitos/noise_samples/ \
                --model_name_or_path microsoft/wavlm-base-plus-sd \
                --output_dir checkpoints/ \
                --log_dir logs/ 
"""
