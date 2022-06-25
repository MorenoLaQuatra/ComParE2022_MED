from transformers import (
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    AutoFeatureExtractor,
    EarlyStoppingCallback,
)
from sklearn.utils import class_weight
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
)
import pandas as pd
import torch
import torchaudio
import torch.nn as nn
import librosa
import numpy as np
import numpy as np
import argparse

import warnings

warnings.filterwarnings("ignore")


""" Dataset Class """


class Dataset(torch.utils.data.Dataset):
    def __init__(self, examples, feature_extractor, max_duration, path):
        self.examples = examples["id"]
        self.labels = examples["label"].astype(int)
        self.feature_extractor = feature_extractor
        self.max_duration = max_duration
        self.path = path

    def __getitem__(self, idx):
        inputs = self.feature_extractor(
            librosa.resample(
                np.asarray(
                    torchaudio.load(self.path + str(self.examples[idx]) + ".wav")[0]
                ).squeeze(0),
                48_000,
                16_000,
            ),
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            max_length=int(self.feature_extractor.sampling_rate * self.max_duration),
            truncation=True,
            padding="max_length",
        )  # .to(device)
        item = {"input_values": inputs["input_values"].squeeze(0)}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.examples)


""" Define Command Line Parser """


def parse_cmd_line_params():
    parser = argparse.ArgumentParser(description="batch_size")
    parser.add_argument(
        "--batch",
        help="Allows to specify values for the configuration entries to override default settings",
        default=8,
        type=int,
        required=False,
    )

    args = parser.parse_args()
    return args


""" Trainer Class """


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").long()
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


""" Define metrics"""


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    f1_macro = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    roc_auc_curve = roc_auc_score(labels, preds)

    print("roc_auc_score: " + str(roc_auc_curve))
    print("F1 Weighted Score: " + str(f1_weighted))
    print("F1 Macro Score: " + str(f1_macro))
    print("Accuracy: " + str(acc))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))

    return {
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall,
        "roc_auc_curve": roc_auc_curve,
    }


""" Main Program """
if __name__ == "__main__":

    torch.multiprocessing.set_start_method("spawn")

    ## Utils
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    path = "../../../data1/akoudounas/mosquito/"
    max_duration = 10.0

    """ Preprocess Data """
    df_train = pd.read_csv(path + "train.csv", index_col=None)
    df_train["label"] = df_train["label"].astype(str)
    df_dev = pd.read_csv(path + "dev.csv", index_col=None)
    df_dev["label"] = df_dev["label"].astype(str)

    ## Prepare the labels
    labels = df_train["label"].unique()
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    num_labels = len(id2label)
    print(id2label["0"])

    """ Define Model """
    model_checkpoint = "facebook/wav2vec2-xls-r-300m"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )  # .to(device)

    """ Build Dataset """
    path_train = path + "train/"
    train_dataset = Dataset(df_train, feature_extractor, max_duration, path_train)
    path_valid = path + "dev/a/"
    dev_dataset = Dataset(df_dev, feature_extractor, max_duration, path_valid)

    """ Training Model """
    model_name = model_checkpoint.split("/")[-1]
    batch_size = parse_cmd_line_params().batch
    output_dir = path + model_name + "-finetuned-med"

    # Define args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        warmup_ratio=0.1,
        logging_steps=30,
        eval_steps=30,
        save_steps=30,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
        fp16_full_eval=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    ## Class Weights
    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(df_train["label"]), y=np.array(df_train["label"])
    )
    class_weights = torch.tensor(class_weights, device="cuda", dtype=torch.float32)

    # Trainer
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        callbacks=[early_stopping],
        compute_metrics=compute_metrics,
    )

    # Train and Evaluate
    trainer.train()
    trainer.save_model(output_dir)
    trainer.evaluate()
